#!/usr/bin/env python3
"""Local benchmark with resource profiling for baleen pipeline.

Subcommands:
  run     — run pipeline on testdata, profile resources, compute accuracy
  compare — compare recent runs in a table
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
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
    from baleen._cuda_dtw import CUDA_AVAILABLE
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "cuda_available": CUDA_AVAILABLE,
        "dtw_backend": "cuda" if CUDA_AVAILABLE else "tslearn",
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

def _run_single_stoich(
    stoich: str,
    *,
    use_cuda: bool | None,
    threads: int,
    mod_threshold: float,
    testdata: Path,
) -> tuple[dict, list, float]:
    """Run pipeline for one stoichiometry level. Returns (hmm_results, sites, wall_s)."""
    from baleen.eventalign._pipeline import run_pipeline_streaming

    stoich_dir = testdata / stoich / "data"
    native_dir = stoich_dir / "native_1"
    control_dir = stoich_dir / "control_1"

    t0 = time.perf_counter()
    _hmm_results, sites, _metadata = run_pipeline_streaming(
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
        output_dir=None,
    )
    wall_s = time.perf_counter() - t0
    return _hmm_results, sites, wall_s


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

    print(f"Benchmark: {len(stoich_levels)} stoich levels, "
          f"threshold={args.threshold}, threads={args.threads}, "
          f"cuda={use_cuda}")

    env = _env_snapshot()
    known_mods = _load_known_mods()

    gpu_mon = GpuMemoryMonitor()
    gpu_mon.start()

    all_sites: list = []
    per_stoich_timing: dict[str, dict] = {}
    per_stoich_accuracy: dict[str, dict] = {}

    total_t0 = time.perf_counter()

    for stoich in stoich_levels:
        print(f"  [{stoich}] running...", end=" ", flush=True)
        _hmm, sites, wall_s = _run_single_stoich(
            stoich,
            use_cuda=use_cuda,
            threads=args.threads,
            mod_threshold=args.threshold,
            testdata=testdata,
        )
        n_sites = len(sites)
        per_stoich_timing[stoich] = {"wall_s": round(wall_s, 2),
                                     "n_sites": n_sites}
        acc = _compute_accuracy(sites, known_mods)
        if acc:
            per_stoich_accuracy[stoich] = acc
        all_sites.extend(sites)
        print(f"{wall_s:.1f}s, {n_sites} sites")

    total_wall = time.perf_counter() - total_t0
    peak_gpu_mb = gpu_mon.stop()

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_mb = int(rusage.ru_maxrss / 1024)  # Linux: ru_maxrss in KB

    summary_accuracy = _compute_accuracy(all_sites, known_mods)

    result = {
        "env": env,
        "params": {
            "threshold": args.threshold,
            "threads": args.threads,
            "stoich_levels": stoich_levels,
            "use_cuda": use_cuda,
        },
        "timing": {
            "total_wall_s": round(total_wall, 2),
            "per_stoich": per_stoich_timing,
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

    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\nDone in {total_wall:.1f}s  |  RSS {peak_rss_mb} MB  |  "
          f"GPU {peak_gpu_mb} MB")
    if summary_accuracy:
        auprc = summary_accuracy.get("auprc_mod_ratio", "n/a")
        auroc = summary_accuracy.get("auroc_mod_ratio", "n/a")
        print(f"AUPRC(mod_ratio)={auprc}  AUROC(mod_ratio)={auroc}")
    print(f"Saved to {RESULTS_FILE}")


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
    else:
        _print_summary_table(runs)


def _print_summary_table(runs: list[dict]) -> None:
    header = (f"{'Commit':<9} {'Label':<20} {'Date':<12} {'Wall(s)':>8} "
              f"{'RSS(MB)':>8} {'GPU(MB)':>8} {'AUPRC':>7} {'AUROC':>7} "
              f"{'Delta':>7}")
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
        auprc = summary.get("auprc_mod_ratio")
        auroc = summary.get("auroc_mod_ratio")

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
              f"{rss:>8,} {gpu:>8,} {auprc_s:>7} {auroc_s:>7} "
              f"{delta_s:>7}")


def _print_detail_table(runs: list[dict]) -> None:
    """Print per-stoich breakdown for each run."""
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

        per_stoich = run["timing"].get("per_stoich", {})
        per_acc = run["accuracy"].get("per_stoich", {})
        if per_stoich:
            print(f"  {'Stoich':<8} {'Wall(s)':>8} {'Sites':>6} "
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
            print(f"  Summary: " + "  ".join(
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
    p_run.add_argument("--threshold", type=float, default=0.9,
                       help="Modification threshold (default: 0.9)")
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

    # -- compare --
    p_cmp = sub.add_parser("compare", help="Compare recent runs")
    p_cmp.add_argument("--last", type=int, default=10,
                       help="Number of recent runs to show (default: 10)")
    p_cmp.add_argument("--detail", action="store_true",
                       help="Show per-stoichiometry breakdown")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
