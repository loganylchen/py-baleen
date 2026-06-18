#!/usr/bin/env python3
"""Local benchmark with resource profiling for baleen pipeline.

Subcommands:
  run     — run pipeline on testdata, profile resources, compute accuracy
  compare — compare recent runs in a table

Timing granularity
------------------
Every run captures:
  - Total wall time (per stoichiometry level and across the whole run)
  - Per-stage timings parsed from pipeline logs:
      * indexing (FASTQ + BLOW5)
      * BAM validation + contig stats
      * contig filtering
      * streaming (DTW → HMM → aggregation)
  - Per-contig timings aggregated (mean / median / total):
      * BAM splitting
      * eventalign (f5c)
      * DTW
      * HMM smoothing
      * site aggregation + FDR

With ``--profile`` (optional, forces ``--threads 1`` for accurate attribution),
cProfile runs on the pipeline and time is bucketed by module:
  dtw, hmm, hmm_training, probability, aggregation, signal, f5c, bam,
  pipeline, io, tslearn, scipy, numpy, other.
Built-in C extensions (e.g. the CUDA DTW kernel) are matched by their
funcname since cProfile reports an empty filepath for them.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import logging
import os
import platform
import pstats
import re
import resource
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TESTDATA_DIR = PROJECT_ROOT / "testdata"
RESULTS_FILE = SCRIPT_DIR / "results.jsonl"
PROFILE_DIR = SCRIPT_DIR / "profiles"

ALL_STOICH = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5",
              "0.6", "0.7", "0.8", "0.9", "1.0"]

KNOWN_MODS_FILE = TESTDATA_DIR / "known_modifications.tsv"

METRICS = ["mod_ratio", "mean_p_mod", "effect_size", "stoichiometry"]
# p-value metrics need -log10 transform (lower p = more modified)
PVAL_METRICS = ["pvalue", "padj"]


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------

def _git_info() -> dict:
    def _run(args: list[str]) -> str:
        try:
            r = subprocess.run(args, capture_output=True, text=True,
                               cwd=PROJECT_ROOT, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    return {
        "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": _run(["git", "status", "--porcelain"]) != "",
    }


def _gpu_info() -> dict:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            return {"gpu_name": parts[0], "gpu_total_mb": int(parts[1])}
    except Exception:
        pass
    return {"gpu_name": None, "gpu_total_mb": None}


def _env_snapshot() -> dict:
    from baleen import _dtw
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "cuda_available": _dtw.CUDA_AVAILABLE,
        "dtw_backend": _dtw.backend(),  # 'gpu' or 'cpu' (krill)
    }
    env.update(_git_info())
    env.update(_gpu_info())
    return env


# ---------------------------------------------------------------------------
# GPU memory monitor (background thread)
# ---------------------------------------------------------------------------

class GpuMemoryMonitor:
    """Polls nvidia-smi for used GPU memory in a background thread."""

    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._peak_mb = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._peak_mb = 0
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self._peak_mb

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode == 0:
                    mb = int(r.stdout.strip().split("\n")[0])
                    self._peak_mb = max(self._peak_mb, mb)
            except Exception:
                pass
            self._stop.wait(self._interval)


# ---------------------------------------------------------------------------
# Log timing capture
# ---------------------------------------------------------------------------

_ELAPSED_RE = re.compile(r"(?:(\d+)m)?(\d+(?:\.\d+)?)s")


def _parse_elapsed(s: str) -> float:
    """Parse `_fmt_elapsed` output back into seconds. Accepts e.g. '1.5s' or '2m30.5s'."""
    m = _ELAPSED_RE.fullmatch(s.strip())
    if not m:
        return 0.0
    mins = int(m.group(1)) if m.group(1) else 0
    secs = float(m.group(2))
    return mins * 60.0 + secs


# Single-shot pipeline stages (one value per pipeline run)
_STAGE_PATTERNS = {
    "indexing": re.compile(r"\[Step 2/5\] Indexing complete \(([^)]+)\)"),
    "bam_stats": re.compile(r"\[Step 3/5\] BAM stats complete:.*\(([^)]+)\)"),
    "filtering": re.compile(r"\[Step 4/5\] \d+ contigs to process \(([^)]+)\)"),
    "streaming": re.compile(r"\[Step 5/5\] Streaming complete \(([^)]+)\)"),
}

# Per-contig patterns (collected as a list; aggregated later)
_CONTIG_PATTERNS = {
    "bam_split": re.compile(r"Extracted \d+ reads for contig \S+ into \S+ \(([^)]+)\)"),
    "eventalign": re.compile(r"Eventalign done \(([^)]+)\)"),
    "hmm": re.compile(r"\[Contig [^\]]+\] HMM done \(([^)]+)\)"),
    "aggregation": re.compile(r"\[Contig [^\]]+\] Aggregation done: \d+ sites \(([^)]+)\)"),
}
# DTW + total per contig come from one message: "[Contig X/Y] NAME done: ..., DTW in T1, total T2"
_CONTIG_DONE_RE = re.compile(
    r"\[Contig \d+/\d+\] \S+ done: \d+ positions \(\d+ skipped\), DTW in (\S+), total (\S+)"
)


class LogTimingCapture(logging.Handler):
    """Parse pipeline log messages into structured timing data.

    Hooks into the `baleen` root logger and regex-matches formatted messages.
    Works for both threaded and single-threaded pipeline runs because the
    pipeline already routes per-contig log messages back to the main process.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        # Single-shot pipeline stages (each captured once per stoich run)
        self.stages: dict[str, float] = {}
        # Per-contig aggregations (one list per contig, aggregated at end)
        self.per_contig: dict[str, list[float]] = {
            k: [] for k in list(_CONTIG_PATTERNS) + ["dtw", "contig_total"]
        }

    def reset(self) -> None:
        self.stages.clear()
        for k in self.per_contig:
            self.per_contig[k] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return

        for key, pat in _STAGE_PATTERNS.items():
            m = pat.search(msg)
            if m:
                self.stages[key] = _parse_elapsed(m.group(1))
                return

        for key, pat in _CONTIG_PATTERNS.items():
            m = pat.search(msg)
            if m:
                self.per_contig[key].append(_parse_elapsed(m.group(1)))
                return

        m = _CONTIG_DONE_RE.search(msg)
        if m:
            self.per_contig["dtw"].append(_parse_elapsed(m.group(1)))
            self.per_contig["contig_total"].append(_parse_elapsed(m.group(2)))


def _summarize_per_contig(values: list[float]) -> dict:
    """Return total/mean/median/max over a list of per-contig times."""
    if not values:
        return {"n": 0, "total_s": 0.0, "mean_s": 0.0,
                "median_s": 0.0, "max_s": 0.0}
    return {
        "n": len(values),
        "total_s": round(sum(values), 2),
        "mean_s": round(statistics.mean(values), 3),
        "median_s": round(statistics.median(values), 3),
        "max_s": round(max(values), 3),
    }


# ---------------------------------------------------------------------------
# cProfile bucketed breakdown
# ---------------------------------------------------------------------------

# Classification tiers. Order matters (first match wins).
# Each tier is (bucket_name, filepath_fragments, funcname_fragments).
# Funcname matching catches built-in C extensions where filepath is "~",
# e.g. "{built-in method krill._krill.dtw_multi_position_pairwise}".
_BUCKET_PATTERNS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    # Baleen project modules — match by filename fragment, also by funcname
    # so built-ins like krill's C DTW kernel land in the right bucket.
    ("dtw",          ("_dtw.py", "krill"),                    ("dtw_distance", "dtw_pairwise", "dtw_pairwise_varlen", "dtw_multi_position_pairwise")),
    ("hmm_training", ("_hmm_training.py",),                   ()),
    ("hmm",          ("_hierarchical.py",),                   ()),
    ("probability",  ("_probability.py",),                    ()),
    ("aggregation",  ("_aggregation.py",),                    ()),
    ("signal",       ("_signal.py",),                         ()),
    ("eventalign",   ("_eventalign.py",),                     ()),
    ("bam",          ("_bam.py", "pysam/"),                   ()),
    ("pipeline",     ("_pipeline.py",),                       ()),
    # IO / subprocess — typically f5c output reads and subprocess plumbing.
    ("io",           ("subprocess.py", "/selectors.py"),
                     ("TextIOWrapper", "BufferedReader", "_posixsubprocess")),
    # Third-party libraries
    ("tslearn",      ("tslearn/",),                           ()),
    ("scipy",        ("scipy/",),                             ()),
    ("numpy",        ("numpy/",),                             ("numpy.",)),
]


def _bucket_for(filepath: str, funcname: str) -> str:
    for name, path_frags, func_frags in _BUCKET_PATTERNS:
        if any(frag in filepath for frag in path_frags):
            return name
        if func_frags and any(frag in funcname for frag in func_frags):
            return name
    return "other"


def _profile_buckets(prof: cProfile.Profile) -> dict:
    """Aggregate cProfile stats into named buckets (self time per module)."""
    stats = pstats.Stats(prof)
    buckets: dict[str, float] = {}
    for (filepath, _lineno, funcname), (_cc, _nc, tt, _ct, _callers) in stats.stats.items():
        key = _bucket_for(str(filepath), str(funcname))
        buckets[key] = buckets.get(key, 0.0) + tt

    total = sum(buckets.values())
    return {
        "self_time_s": {k: round(v, 2) for k, v in sorted(buckets.items(),
                                                          key=lambda x: -x[1])},
        "pct": {k: round(v / total * 100, 1) if total > 0 else 0.0
                for k, v in sorted(buckets.items(), key=lambda x: -x[1])},
        "total_self_s": round(total, 2),
    }


def _profile_top_functions(prof: cProfile.Profile, n: int = 20) -> list[dict]:
    """Return top-N functions by cumulative time."""
    stats = pstats.Stats(prof)
    entries = []
    for (filepath, lineno, funcname), (_cc, nc, tt, ct, _callers) in stats.stats.items():
        entries.append({
            "func": funcname,
            "file": Path(str(filepath)).name,
            "line": lineno,
            "calls": nc,
            "self_s": round(tt, 3),
            "cum_s": round(ct, 3),
        })
    entries.sort(key=lambda e: -e["cum_s"])
    return entries[:n]


# ---------------------------------------------------------------------------
# Ground truth + accuracy
# ---------------------------------------------------------------------------

def _load_known_mods() -> set[tuple[str, int]]:
    """Load known modification positions from TSV."""
    mods: set[tuple[str, int]] = set()
    with open(KNOWN_MODS_FILE) as f:
        for line in f:
            if line.startswith("contig"):
                continue
            parts = line.strip().split("\t")
            mods.add((parts[0], int(parts[1])))
    return mods


def _compute_accuracy(sites: list, known_mods: set[tuple[str, int]]) -> dict:
    """Compute AUPRC and AUROC for each metric."""
    import math
    from sklearn.metrics import average_precision_score, roc_auc_score

    if not sites:
        return {}

    labels = []
    values: dict[str, list[float]] = {m: [] for m in METRICS}
    pvalues: dict[str, list[float]] = {m: [] for m in PVAL_METRICS}

    for site in sites:
        is_mod = 1 if (site.contig, site.position) in known_mods else 0
        labels.append(is_mod)
        for m in METRICS:
            values[m].append(getattr(site, m))
        for m in PVAL_METRICS:
            raw = getattr(site, m)
            # -log10 transform: clamp to avoid log10(0)
            pvalues[m].append(-math.log10(max(raw, 1e-300)))

    if len(set(labels)) < 2:
        return {}

    result: dict[str, float] = {}
    for m in METRICS:
        try:
            result[f"auprc_{m}"] = round(
                average_precision_score(labels, values[m]), 4)
            result[f"auroc_{m}"] = round(
                roc_auc_score(labels, values[m]), 4)
        except ValueError:
            pass
    for m in PVAL_METRICS:
        key = f"nlog10_{m}"
        try:
            result[f"auprc_{key}"] = round(
                average_precision_score(labels, pvalues[m]), 4)
            result[f"auroc_{key}"] = round(
                roc_auc_score(labels, pvalues[m]), 4)
        except ValueError:
            pass
    return result


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _load_sites_from_tsv(tsv_path: Path) -> list:
    """Re-parse merged site_results.tsv into objects with the attributes
    expected by :func:`_compute_accuracy`.
    """
    import csv
    from types import SimpleNamespace

    sites: list = []
    if not tsv_path.exists():
        return sites
    with tsv_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sites.append(SimpleNamespace(
                contig=row["contig"],
                position=int(row["position"]),
                mod_ratio=float(row["mod_ratio"]),
                mean_p_mod=float(row["mean_p_mod"]),
                effect_size=float(row["effect_size"]),
                stoichiometry=float(row["stoichiometry"]),
                pvalue=float(row["pvalue"]),
                padj=float(row["padj"]),
            ))
    return sites


def _run_single_stoich(
    stoich: str,
    *,
    use_cuda: bool | None,
    threads: int,
    mod_threshold: float,
    testdata: Path,
    profile: cProfile.Profile | None,
) -> tuple[list, float]:
    """Run pipeline for one stoichiometry level. Returns (sites, wall_s)."""
    import tempfile

    from baleen.eventalign._pipeline import run_pipeline_streaming

    stoich_dir = testdata / stoich / "data"
    native_dir = stoich_dir / "native_1"
    control_dir = stoich_dir / "control_1"

    with tempfile.TemporaryDirectory(prefix="baleen_bench_") as tmpdir:
        call_kwargs = dict(
            native_bam=native_dir / "native_1.bam",
            native_fastq=native_dir / "fastq" / "pass.fq.gz",
            native_blow5=native_dir / "blow5" / "nanopore.blow5",
            ivt_bam=control_dir / "control_1.bam",
            ivt_fastq=control_dir / "fastq" / "pass.fq.gz",
            ivt_blow5=control_dir / "blow5" / "nanopore.blow5",
            ref_fasta=testdata / "ref.fa",
            use_cuda=use_cuda,
            threads=threads,
            mod_threshold=mod_threshold,
            output_dir=tmpdir,
            write_bam=False,
        )

        t0 = time.perf_counter()
        if profile is not None:
            profile.enable()
            try:
                output_paths, _metadata = run_pipeline_streaming(**call_kwargs)
            finally:
                profile.disable()
        else:
            output_paths, _metadata = run_pipeline_streaming(**call_kwargs)
        wall_s = time.perf_counter() - t0

        sites = _load_sites_from_tsv(Path(output_paths["site_tsv"]))

    return sites, wall_s


def cmd_run(args: argparse.Namespace) -> None:
    """Run pipeline, profile resources, compute accuracy, store results."""
    stoich_levels = [s.strip() for s in args.stoich.split(",")]
    testdata = Path(args.testdata)

    for s in stoich_levels:
        d = testdata / s / "data"
        if not d.is_dir():
            print(f"ERROR: testdata not found: {d}", file=sys.stderr)
            sys.exit(1)

    use_cuda: bool | None = None
    if args.cuda:
        use_cuda = True
    elif args.no_cuda:
        use_cuda = False

    threads = args.threads
    if args.profile and threads != 1:
        print(f"WARNING: --profile forces --threads 1 "
              f"(cProfile cannot see worker processes; was {threads})")
        threads = 1

    print(f"Benchmark: {len(stoich_levels)} stoich levels, "
          f"threshold={args.threshold}, threads={threads}, "
          f"cuda={use_cuda}, profile={args.profile}")

    env = _env_snapshot()
    known_mods = _load_known_mods()

    # Install log timing capture on the baleen logger
    log_capture = LogTimingCapture()
    root_logger = logging.getLogger("baleen")
    prev_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(log_capture)

    gpu_mon = GpuMemoryMonitor()
    gpu_mon.start()

    profiler = cProfile.Profile() if args.profile else None

    all_sites: list = []
    per_stoich_timing: dict[str, dict] = {}
    per_stoich_accuracy: dict[str, dict] = {}

    total_t0 = time.perf_counter()

    try:
        for stoich in stoich_levels:
            suffix = f" x{args.repeat}" if args.repeat > 1 else ""
            print(f"  [{stoich}]{suffix} running...", end=" ", flush=True)
            log_capture.reset()

            wall_s_total = 0.0
            sites_first: list | None = None
            for _r in range(args.repeat):
                sites, wall_s = _run_single_stoich(
                    stoich,
                    use_cuda=use_cuda,
                    threads=threads,
                    mod_threshold=args.threshold,
                    testdata=testdata,
                    profile=profiler,
                )
                wall_s_total += wall_s
                if sites_first is None:
                    sites_first = sites
            n_sites = len(sites_first) if sites_first is not None else 0

            per_stoich_timing[stoich] = {
                "wall_s": round(wall_s_total, 2),
                "n_sites": n_sites,
                "n_repeats": args.repeat,
                "stages_s": {k: round(v, 2)
                             for k, v in log_capture.stages.items()},
                "per_contig": {
                    k: _summarize_per_contig(v)
                    for k, v in log_capture.per_contig.items()
                },
            }
            acc = _compute_accuracy(sites_first or [], known_mods)
            if acc:
                per_stoich_accuracy[stoich] = acc
            if sites_first is not None:
                all_sites.extend(sites_first)

            stages = log_capture.stages
            n_contigs = len(log_capture.per_contig["contig_total"])
            stage_summary = (
                f"idx={stages.get('indexing', 0):.1f}s "
                f"stats={stages.get('bam_stats', 0):.1f}s "
                f"filt={stages.get('filtering', 0):.1f}s "
                f"stream={stages.get('streaming', 0):.1f}s"
            )
            print(f"{wall_s_total:.1f}s, {n_sites} sites, {n_contigs} contigs "
                  f"[{stage_summary}]")
    finally:
        root_logger.removeHandler(log_capture)
        root_logger.setLevel(prev_level)

    total_wall = time.perf_counter() - total_t0
    peak_gpu_mb = gpu_mon.stop()

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_mb = int(rusage.ru_maxrss / 1024)  # Linux: ru_maxrss in KB

    summary_accuracy = _compute_accuracy(all_sites, known_mods)

    # Aggregate pipeline-level stages across stoich levels
    aggregate_stages: dict[str, float] = {}
    aggregate_per_contig: dict[str, list[float]] = {}
    for stoich, t in per_stoich_timing.items():
        for name, secs in t["stages_s"].items():
            aggregate_stages[name] = aggregate_stages.get(name, 0.0) + secs
        for name, summary in t["per_contig"].items():
            aggregate_per_contig.setdefault(name, []).append(summary["total_s"])

    result: dict = {
        "env": env,
        "params": {
            "threshold": args.threshold,
            "threads": threads,
            "stoich_levels": stoich_levels,
            "use_cuda": use_cuda,
            "profile": args.profile,
            "repeat": args.repeat,
        },
        "timing": {
            "total_wall_s": round(total_wall, 2),
            "per_stoich": per_stoich_timing,
            "aggregate_stages_s": {k: round(v, 2)
                                   for k, v in aggregate_stages.items()},
            "aggregate_per_contig_total_s": {
                k: round(sum(v), 2) for k, v in aggregate_per_contig.items()
            },
        },
        "resources": {
            "peak_rss_mb": peak_rss_mb,
            "peak_gpu_mb": peak_gpu_mb,
        },
        "accuracy": {
            "per_stoich": per_stoich_accuracy,
            "summary": summary_accuracy,
        },
        "label": args.label or "",
    }

    if profiler is not None:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        label_slug = (args.label or "run").replace("/", "_").replace(" ", "_")
        prof_path = PROFILE_DIR / f"{ts}_{label_slug}.prof"
        profiler.dump_stats(str(prof_path))
        result["profile"] = {
            "file": str(prof_path.relative_to(PROJECT_ROOT)),
            "buckets": _profile_buckets(profiler),
            "top_functions": _profile_top_functions(profiler, n=args.profile_top),
        }

    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\nDone in {total_wall:.1f}s  |  RSS {peak_rss_mb} MB  |  "
          f"GPU {peak_gpu_mb} MB")
    if summary_accuracy:
        auprc = summary_accuracy.get("auprc_nlog10_pvalue", "n/a")
        auroc = summary_accuracy.get("auroc_nlog10_pvalue", "n/a")
        print(f"AUPRC(-log10p)={auprc}  AUROC(-log10p)={auroc}")

    # Print stage breakdown
    if aggregate_stages:
        print("\nPipeline stages (summed across stoich levels):")
        for name in ("indexing", "bam_stats", "filtering", "streaming"):
            if name in aggregate_stages:
                print(f"  {name:<12} {aggregate_stages[name]:>8.2f}s")
    if aggregate_per_contig:
        print("\nPer-contig stage totals (summed):")
        rows = [
            ("bam_split", aggregate_per_contig.get("bam_split", [])),
            ("eventalign", aggregate_per_contig.get("eventalign", [])),
            ("dtw", aggregate_per_contig.get("dtw", [])),
            ("hmm", aggregate_per_contig.get("hmm", [])),
            ("aggregation", aggregate_per_contig.get("aggregation", [])),
            ("contig_total", aggregate_per_contig.get("contig_total", [])),
        ]
        for name, values in rows:
            if values:
                print(f"  {name:<13} {sum(values):>8.2f}s")

    if profiler is not None:
        print("\ncProfile self-time buckets:")
        buckets = result["profile"]["buckets"]
        for name, secs in buckets["self_time_s"].items():
            pct = buckets["pct"].get(name, 0.0)
            print(f"  {name:<13} {secs:>8.2f}s  ({pct:>5.1f}%)")
        print(f"  {'TOTAL':<13} {buckets['total_self_s']:>8.2f}s")

    print(f"\nSaved to {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> None:
    """Show comparison table of recent runs."""
    if not RESULTS_FILE.exists():
        print("No results yet. Run `bench.py run` first.")
        return

    runs: list[dict] = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))

    if not runs:
        print("No results yet.")
        return

    runs = runs[-args.last:]

    if args.detail:
        _print_detail_table(runs)
    elif args.steps:
        _print_steps_table(runs)
    else:
        _print_summary_table(runs)


def _print_summary_table(runs: list[dict]) -> None:
    header = (f"{'Commit':<9} {'Label':<20} {'Date':<12} {'Wall(s)':>8} "
              f"{'RSS(MB)':>8} {'GPU(MB)':>8} {'AUPRC(-lg10p)':>13} "
              f"{'AUROC(-lg10p)':>13} {'Delta':>7}")
    print(header)
    print("-" * len(header))

    baseline_auprc: float | None = None

    for run in runs:
        env = run["env"]
        commit = env.get("git_commit", "?")[:7]
        dirty = "*" if env.get("git_dirty") else ""
        label = (run.get("label") or "")[:20]
        ts = env.get("timestamp", "")[:10]
        wall = run["timing"]["total_wall_s"]
        rss = run["resources"]["peak_rss_mb"]
        gpu = run["resources"]["peak_gpu_mb"]
        summary = run["accuracy"].get("summary", {})
        auprc = summary.get("auprc_nlog10_pvalue")
        auroc = summary.get("auroc_nlog10_pvalue")

        auprc_s = f"{auprc:.4f}" if auprc is not None else "n/a"
        auroc_s = f"{auroc:.4f}" if auroc is not None else "n/a"

        if baseline_auprc is None and auprc is not None:
            baseline_auprc = auprc
            delta_s = "---"
        elif auprc is not None and baseline_auprc:
            pct = (auprc - baseline_auprc) / baseline_auprc * 100
            delta_s = f"{pct:+.1f}%"
        else:
            delta_s = "n/a"

        print(f"{commit}{dirty:<2} {label:<20} {ts:<12} {wall:>8.1f} "
              f"{rss:>8,} {gpu:>8,} {auprc_s:>13} {auroc_s:>13} "
              f"{delta_s:>7}")


def _print_steps_table(runs: list[dict]) -> None:
    """Print pipeline-stage + per-contig timing table across runs."""
    print(f"{'Commit':<9} {'Label':<18} {'Wall':>7}"
          f"  {'Idx':>6} {'Stats':>6} {'Filt':>6} {'Stream':>7}"
          f"  {'BAM':>6} {'Eval':>6} {'DTW':>6} {'HMM':>6} {'Agg':>6}")
    print("-" * 100)
    baseline: dict[str, float] | None = None

    for run in runs:
        env = run["env"]
        commit = env.get("git_commit", "?")[:7]
        dirty = "*" if env.get("git_dirty") else ""
        label = (run.get("label") or "")[:18]
        t = run["timing"]
        stages = t.get("aggregate_stages_s", {})
        pc = t.get("aggregate_per_contig_total_s", {})

        vals = {
            "wall": t.get("total_wall_s", 0.0),
            "indexing": stages.get("indexing", 0.0),
            "bam_stats": stages.get("bam_stats", 0.0),
            "filtering": stages.get("filtering", 0.0),
            "streaming": stages.get("streaming", 0.0),
            "bam_split": pc.get("bam_split", 0.0),
            "eventalign": pc.get("eventalign", 0.0),
            "dtw": pc.get("dtw", 0.0),
            "hmm": pc.get("hmm", 0.0),
            "aggregation": pc.get("aggregation", 0.0),
        }

        if baseline is None:
            baseline = vals

        print(f"{commit}{dirty:<2} {label:<18} "
              f"{vals['wall']:>7.1f}"
              f"  {vals['indexing']:>6.1f}"
              f"  {vals['bam_stats']:>5.1f}"
              f"  {vals['filtering']:>5.1f}"
              f"  {vals['streaming']:>6.1f}"
              f"  {vals['bam_split']:>5.1f}"
              f"  {vals['eventalign']:>5.1f}"
              f"  {vals['dtw']:>5.1f}"
              f"  {vals['hmm']:>5.1f}"
              f"  {vals['aggregation']:>5.1f}")

    # Delta row (latest vs first)
    if baseline is not None and len(runs) >= 2:
        latest = runs[-1]["timing"]
        latest_stages = latest.get("aggregate_stages_s", {})
        latest_pc = latest.get("aggregate_per_contig_total_s", {})
        latest_vals = {
            "wall": latest.get("total_wall_s", 0.0),
            "indexing": latest_stages.get("indexing", 0.0),
            "bam_stats": latest_stages.get("bam_stats", 0.0),
            "filtering": latest_stages.get("filtering", 0.0),
            "streaming": latest_stages.get("streaming", 0.0),
            "bam_split": latest_pc.get("bam_split", 0.0),
            "eventalign": latest_pc.get("eventalign", 0.0),
            "dtw": latest_pc.get("dtw", 0.0),
            "hmm": latest_pc.get("hmm", 0.0),
            "aggregation": latest_pc.get("aggregation", 0.0),
        }

        def _pct(new: float, old: float) -> str:
            if old <= 0:
                return "  n/a"
            return f"{(new - old) / old * 100:+5.1f}%"

        print("-" * 100)
        print(f"{'Δ vs 1st':<11} {'':<18} "
              f"{_pct(latest_vals['wall'], baseline['wall']):>7}"
              f"  {_pct(latest_vals['indexing'], baseline['indexing']):>6}"
              f"  {_pct(latest_vals['bam_stats'], baseline['bam_stats']):>5}"
              f"  {_pct(latest_vals['filtering'], baseline['filtering']):>5}"
              f"  {_pct(latest_vals['streaming'], baseline['streaming']):>6}"
              f"  {_pct(latest_vals['bam_split'], baseline['bam_split']):>5}"
              f"  {_pct(latest_vals['eventalign'], baseline['eventalign']):>5}"
              f"  {_pct(latest_vals['dtw'], baseline['dtw']):>5}"
              f"  {_pct(latest_vals['hmm'], baseline['hmm']):>5}"
              f"  {_pct(latest_vals['aggregation'], baseline['aggregation']):>5}")


def _print_detail_table(runs: list[dict]) -> None:
    """Print per-stoich breakdown + step timings + cProfile buckets for each run."""
    for i, run in enumerate(runs):
        env = run["env"]
        commit = env.get("git_commit", "?")[:7]
        label = run.get("label") or ""
        print(f"\n=== Run {i+1}: {commit} {label} ===")
        print(f"  Branch: {env.get('git_branch')}  "
              f"Dirty: {env.get('git_dirty')}  "
              f"Backend: {env.get('dtw_backend')}  "
              f"GPU: {env.get('gpu_name')}")
        print(f"  Total: {run['timing']['total_wall_s']}s  "
              f"RSS: {run['resources']['peak_rss_mb']} MB  "
              f"GPU: {run['resources']['peak_gpu_mb']} MB")

        agg_stages = run["timing"].get("aggregate_stages_s", {})
        if agg_stages:
            print("  Pipeline stages (summed):")
            for name in ("indexing", "bam_stats", "filtering", "streaming"):
                if name in agg_stages:
                    print(f"    {name:<12} {agg_stages[name]:>8.2f}s")

        agg_pc = run["timing"].get("aggregate_per_contig_total_s", {})
        if agg_pc:
            print("  Per-contig stage totals (summed):")
            for name in ("bam_split", "eventalign", "dtw", "hmm",
                         "aggregation", "contig_total"):
                if name in agg_pc:
                    print(f"    {name:<13} {agg_pc[name]:>8.2f}s")

        profile = run.get("profile", {})
        if profile:
            print(f"  cProfile ({profile.get('file', '?')}):")
            buckets = profile.get("buckets", {})
            self_times = buckets.get("self_time_s", {})
            pcts = buckets.get("pct", {})
            for name, secs in self_times.items():
                pct = pcts.get(name, 0.0)
                print(f"    {name:<13} {secs:>8.2f}s  ({pct:>5.1f}%)")

        per_stoich = run["timing"].get("per_stoich", {})
        per_acc = run["accuracy"].get("per_stoich", {})
        if per_stoich:
            print(f"\n  {'Stoich':<8} {'Wall(s)':>8} {'Sites':>6} "
                  f"{'AUPRC':>7} {'AUROC':>7}")
            for s in sorted(per_stoich.keys(), key=float):
                t = per_stoich[s]
                a = per_acc.get(s, {})
                auprc = a.get("auprc_mod_ratio")
                auroc = a.get("auroc_mod_ratio")
                auprc_s = f"{auprc:.4f}" if auprc is not None else "n/a"
                auroc_s = f"{auroc:.4f}" if auroc is not None else "n/a"
                print(f"  {s:<8} {t['wall_s']:>8.1f} {t['n_sites']:>6} "
                      f"{auprc_s:>7} {auroc_s:>7}")

        summary = run["accuracy"].get("summary", {})
        if summary:
            print(f"\n  Summary: " + "  ".join(
                f"{k}={v}" for k, v in sorted(summary.items())))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local benchmark with resource profiling")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- run --
    p_run = sub.add_parser("run", help="Run pipeline and profile")
    p_run.add_argument("--stoich", default=",".join(ALL_STOICH),
                       help="Comma-separated stoich levels (default: all 11)")
    p_run.add_argument("--threshold", type=float, default=0.75,
                       help="Modification threshold (default: 0.75)")
    p_run.add_argument("--threads", type=int, default=4,
                       help="Pipeline threads (default: 4)")
    cuda_grp = p_run.add_mutually_exclusive_group()
    cuda_grp.add_argument("--cuda", action="store_true",
                          help="Force CUDA backend")
    cuda_grp.add_argument("--no-cuda", action="store_true",
                          help="Force CPU backend")
    p_run.add_argument("--testdata", default=str(TESTDATA_DIR),
                       help="Path to testdata directory")
    p_run.add_argument("--label", default="",
                       help="Label for this run (e.g. 'before-refactor')")
    p_run.add_argument("--profile", action="store_true",
                       help="Enable cProfile (forces --threads 1 for accurate attribution). "
                            "Saves a .prof file under benchmarks/profiles/.")
    p_run.add_argument("--profile-top", type=int, default=20,
                       help="Top-N functions to capture when --profile is on (default: 20)")
    p_run.add_argument("--repeat", type=int, default=1,
                       help="Repeat each stoich N times back-to-back (multiplies effective "
                            "contig count for scaling tests). Timings are accumulated; accuracy "
                            "is measured once (outputs are deterministic). Default: 1.")

    # -- compare --
    p_cmp = sub.add_parser("compare", help="Compare recent runs")
    p_cmp.add_argument("--last", type=int, default=10,
                       help="Number of recent runs to show (default: 10)")
    p_cmp.add_argument("--detail", action="store_true",
                       help="Show per-stoich breakdown + step timings + cProfile buckets")
    p_cmp.add_argument("--steps", action="store_true",
                       help="Show per-stage timing breakdown table")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
