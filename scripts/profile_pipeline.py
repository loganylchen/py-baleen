#!/usr/bin/env python3
"""Profile baleen pipeline stages on real data.

Usage:
    python scripts/profile_pipeline.py \
        --native-bam data/native.bam \
        --native-fastq data/native.fq.gz \
        --native-blow5 data/native.blow5 \
        --ivt-bam data/ivt.bam \
        --ivt-fastq data/ivt.fq.gz \
        --ivt-blow5 data/ivt.blow5 \
        --ref-fasta data/transcriptome.fa \
        [--contigs ENST00000202773 ENST00000214869] \
        [--max-contigs 3] \
        [--threads 1] \
        [--use-cuda]

Produces a JSON report with per-stage timings for each contig.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("profile")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@dataclass
class StageTimer:
    name: str
    start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.start


@dataclass
class PositionProfile:
    position: int
    n_native: int
    n_ivt: int
    n_total: int
    n_pairs: int
    signal_lengths_min: int
    signal_lengths_max: int
    signal_lengths_mean: float


@dataclass
class ContigProfile:
    contig: str
    native_depth: float
    ivt_depth: float
    n_common_positions: int
    n_positions_computed: int

    # Per-stage wall-clock seconds
    bam_split_native: float = 0.0
    bam_split_ivt: float = 0.0
    f5c_native: float = 0.0
    f5c_ivt: float = 0.0
    signal_parse_native: float = 0.0
    signal_parse_ivt: float = 0.0
    signal_extract: float = 0.0
    dtw_total: float = 0.0
    dtw_chunk_count: int = 0
    dtw_chunk_times: list[float] = field(default_factory=list)
    hierarchical_v1: float = 0.0
    hierarchical_v2: float = 0.0
    hierarchical_hmm: float = 0.0
    hierarchical_total: float = 0.0
    contig_total: float = 0.0

    # Position-level stats
    positions: list[PositionProfile] = field(default_factory=list)

    # Error if any
    error: Optional[str] = None


@dataclass
class ProfileReport:
    timestamp: str = ""
    cuda_available: bool = False
    cuda_used: bool = False
    dtw_backend: str = ""
    threads: int = 1
    subsample_n: int = 300
    padding: int = 1
    n_contigs_total: int = 0
    n_contigs_profiled: int = 0
    contigs: list[ContigProfile] = field(default_factory=list)

    # Aggregated
    total_wall_clock: float = 0.0
    total_bam_split: float = 0.0
    total_f5c: float = 0.0
    total_signal_parse: float = 0.0
    total_signal_extract: float = 0.0
    total_dtw: float = 0.0
    total_hierarchical: float = 0.0


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.1f}s"


# ---------------------------------------------------------------------------
# Profile a single contig
# ---------------------------------------------------------------------------

def profile_contig(
    contig: str,
    contig_idx: int,
    total_contigs: int,
    native_bam: Path,
    native_fastq: Path,
    native_blow5: Path,
    ivt_bam: Path,
    ivt_fastq: Path,
    ivt_blow5: Path,
    ref_fasta: Path,
    native_stats: dict,
    ivt_stats: dict,
    tmp_root: Path,
    use_cuda: Optional[bool],
    padding: int,
    subsample_n: int,
    num_cuda_streams: int,
    run_hmm: bool,
) -> ContigProfile:
    from baleen.eventalign import _bam, _f5c, _signal
    from baleen import _cuda_dtw

    prof = ContigProfile(
        contig=contig,
        native_depth=native_stats[contig].mean_depth,
        ivt_depth=ivt_stats[contig].mean_depth,
        n_common_positions=0,
        n_positions_computed=0,
    )
    contig_t0 = time.perf_counter()

    contig_tmp = tmp_root / contig
    contig_tmp.mkdir(parents=True, exist_ok=True)

    try:
        # --- BAM split ---
        with StageTimer("bam_split_native") as t:
            native_contig_bam = _bam.split_bam_contig(
                native_bam, contig, contig_tmp / "native",
                primary_only=True, min_mapq=0, max_reads=subsample_n,
            )
        prof.bam_split_native = t.elapsed

        with StageTimer("bam_split_ivt") as t:
            ivt_contig_bam = _bam.split_bam_contig(
                ivt_bam, contig, contig_tmp / "ivt",
                primary_only=True, min_mapq=0, max_reads=subsample_n,
            )
        prof.bam_split_ivt = t.elapsed

        logger.info(
            "  [%d/%d] %s  BAM split: native=%s ivt=%s",
            contig_idx, total_contigs, contig,
            _fmt(prof.bam_split_native), _fmt(prof.bam_split_ivt),
        )

        # --- f5c eventalign ---
        native_tsv = contig_tmp / "native.eventalign.tsv"
        ivt_tsv = contig_tmp / "ivt.eventalign.tsv"

        with StageTimer("f5c_native") as t:
            _f5c.run_eventalign(
                native_contig_bam, ref_fasta, native_fastq, native_blow5,
                native_tsv, rna=True,
            )
        prof.f5c_native = t.elapsed

        with StageTimer("f5c_ivt") as t:
            _f5c.run_eventalign(
                ivt_contig_bam, ref_fasta, ivt_fastq, ivt_blow5,
                ivt_tsv, rna=True,
            )
        prof.f5c_ivt = t.elapsed

        logger.info(
            "  [%d/%d] %s  f5c: native=%s ivt=%s",
            contig_idx, total_contigs, contig,
            _fmt(prof.f5c_native), _fmt(prof.f5c_ivt),
        )

        # --- Signal parsing ---
        with StageTimer("signal_parse_native") as t:
            native_by_pos = _signal.group_signals_by_position(native_tsv)
        prof.signal_parse_native = t.elapsed

        with StageTimer("signal_parse_ivt") as t:
            ivt_by_pos = _signal.group_signals_by_position(ivt_tsv)
        prof.signal_parse_ivt = t.elapsed

        common_positions = _signal.get_common_positions(native_by_pos, ivt_by_pos)
        prof.n_common_positions = len(common_positions)

        logger.info(
            "  [%d/%d] %s  Signals: parse_native=%s parse_ivt=%s  %d positions",
            contig_idx, total_contigs, contig,
            _fmt(prof.signal_parse_native), _fmt(prof.signal_parse_ivt),
            len(common_positions),
        )

        # --- Signal extraction ---
        position_data = []
        with StageTimer("signal_extract") as t:
            for pos in common_positions:
                if padding > 0:
                    nat_names, nat_sigs = _signal.extract_signals_for_dtw_padded(
                        native_by_pos, pos, padding,
                    )
                    ivt_names, ivt_sigs = _signal.extract_signals_for_dtw_padded(
                        ivt_by_pos, pos, padding,
                    )
                else:
                    native_pos = native_by_pos[pos]
                    ivt_pos = ivt_by_pos[pos]
                    nat_names, nat_sigs = _signal.extract_signals_for_dtw(native_pos)
                    ivt_names, ivt_sigs = _signal.extract_signals_for_dtw(ivt_pos)

                if not nat_sigs or not ivt_sigs:
                    continue

                all_sigs = nat_sigs + ivt_sigs
                kmer = native_by_pos[pos].reference_kmer
                position_data.append((pos, kmer, nat_names, ivt_names, all_sigs))

                # Record per-position stats
                lengths = [len(s) for s in all_sigs]
                n_total = len(all_sigs)
                prof.positions.append(PositionProfile(
                    position=pos,
                    n_native=len(nat_sigs),
                    n_ivt=len(ivt_sigs),
                    n_total=n_total,
                    n_pairs=n_total * (n_total - 1) // 2,
                    signal_lengths_min=min(lengths),
                    signal_lengths_max=max(lengths),
                    signal_lengths_mean=float(np.mean(lengths)),
                ))
        prof.signal_extract = t.elapsed
        prof.n_positions_computed = len(position_data)

        logger.info(
            "  [%d/%d] %s  Extract: %s  %d positions ready",
            contig_idx, total_contigs, contig,
            _fmt(prof.signal_extract), len(position_data),
        )

        # --- DTW ---
        if position_data:
            all_signal_lists = [d[4] for d in position_data]

            # Use the same chunking as the real pipeline
            total_gpu = _cuda_dtw.estimate_gpu_memory(all_signal_lists) if use_cuda else 0
            gpu_mem = 80 * 1024**3  # assume 80GB for chunk sizing
            chunk_mem_limit = int(gpu_mem * 0.8)

            chunks: list[list[int]] = []
            current_chunk: list[int] = []
            current_estimate = 0
            for i, sigs in enumerate(all_signal_lists):
                pos_estimate = _cuda_dtw.estimate_gpu_memory([sigs])
                if current_chunk and current_estimate + pos_estimate > chunk_mem_limit:
                    chunks.append(current_chunk)
                    current_chunk = [i]
                    current_estimate = pos_estimate
                else:
                    current_chunk.append(i)
                    current_estimate += pos_estimate
            if current_chunk:
                chunks.append(current_chunk)

            prof.dtw_chunk_count = len(chunks)

            dtw_t0 = time.perf_counter()
            all_matrices = []
            for chunk_idx, chunk_indices in enumerate(chunks):
                chunk_signals = [all_signal_lists[i] for i in chunk_indices]
                chunk_t0 = time.perf_counter()
                chunk_matrices = _cuda_dtw.dtw_multi_position_pairwise(
                    chunk_signals,
                    use_open_start=False,
                    use_open_end=False,
                    use_cuda=use_cuda,
                    num_streams=num_cuda_streams,
                )
                chunk_elapsed = time.perf_counter() - chunk_t0
                prof.dtw_chunk_times.append(chunk_elapsed)
                all_matrices.extend(chunk_matrices)

            prof.dtw_total = time.perf_counter() - dtw_t0

            logger.info(
                "  [%d/%d] %s  DTW: %s (%d chunks)",
                contig_idx, total_contigs, contig,
                _fmt(prof.dtw_total), len(chunks),
            )

            # Build ContigResult for hierarchical
            from baleen.eventalign._pipeline import ContigResult, PositionResult
            position_results = {}
            for (pos, kmer, nat_names, ivt_names, _), matrix in zip(position_data, all_matrices):
                position_results[pos] = PositionResult(
                    position=pos,
                    reference_kmer=kmer,
                    n_native_reads=len(nat_names),
                    n_ivt_reads=len(ivt_names),
                    native_read_names=nat_names,
                    ivt_read_names=ivt_names,
                    distance_matrix=matrix,
                )
            contig_result = ContigResult(
                contig=contig,
                native_depth=prof.native_depth,
                ivt_depth=prof.ivt_depth,
                positions=position_results,
            )

            # --- Hierarchical pipeline (V1 → V2 → V3) ---
            from baleen.eventalign._hierarchical import (
                compute_sequential_modification_probabilities,
            )

            # Profile sub-stages by monkey-patching timers
            hier_t0 = time.perf_counter()
            try:
                cmr = compute_sequential_modification_probabilities(
                    contig_result,
                    run_hmm=run_hmm,
                    show_progress=False,
                )
                prof.hierarchical_total = time.perf_counter() - hier_t0
            except Exception as exc:
                prof.hierarchical_total = time.perf_counter() - hier_t0
                logger.warning("  Hierarchical failed for %s: %s", contig, exc)

            logger.info(
                "  [%d/%d] %s  Hierarchical: %s",
                contig_idx, total_contigs, contig,
                _fmt(prof.hierarchical_total),
            )

    except Exception as exc:
        prof.error = str(exc)
        logger.error("  [%d/%d] %s  FAILED: %s", contig_idx, total_contigs, contig, exc)

    prof.contig_total = time.perf_counter() - contig_t0
    return prof


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile baleen pipeline stages")
    parser.add_argument("--native-bam", required=True)
    parser.add_argument("--native-fastq", required=True)
    parser.add_argument("--native-blow5", required=True)
    parser.add_argument("--ivt-bam", required=True)
    parser.add_argument("--ivt-fastq", required=True)
    parser.add_argument("--ivt-blow5", required=True)
    parser.add_argument("--ref-fasta", required=True)
    parser.add_argument("--contigs", nargs="*", default=None,
                        help="Specific contigs to profile (default: auto-pick by depth)")
    parser.add_argument("--max-contigs", type=int, default=3,
                        help="Max contigs to profile (default: 3)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--subsample-n", type=int, default=300)
    parser.add_argument("--num-cuda-streams", type=int, default=16)
    parser.add_argument("--no-hmm", action="store_true", default=False)
    parser.add_argument("--output", "-o", default="profile_report.json",
                        help="Output JSON path (default: profile_report.json)")
    args = parser.parse_args()

    from baleen.eventalign import _bam, _f5c
    from baleen import _cuda_dtw
    import tempfile
    from datetime import datetime

    report = ProfileReport(
        timestamp=datetime.now().isoformat(),
        cuda_available=_cuda_dtw.CUDA_AVAILABLE,
        cuda_used=args.use_cuda,
        dtw_backend=_cuda_dtw.backend(),
        threads=args.threads,
        subsample_n=args.subsample_n,
        padding=args.padding,
    )

    native_bam = Path(args.native_bam)
    ivt_bam = Path(args.ivt_bam)
    native_fastq = Path(args.native_fastq)
    ivt_fastq = Path(args.ivt_fastq)
    native_blow5 = Path(args.native_blow5)
    ivt_blow5 = Path(args.ivt_blow5)
    ref_fasta = Path(args.ref_fasta)

    # Index (idempotent)
    logger.info("Indexing...")
    _f5c.index_fastq_blow5(native_fastq, native_blow5)
    _f5c.index_fastq_blow5(ivt_fastq, ivt_blow5)
    _f5c.index_blow5(native_blow5)
    _f5c.index_blow5(ivt_blow5)

    # BAM stats
    logger.info("Computing BAM stats...")
    native_stats = _bam.get_contig_stats(native_bam, min_mapq=0, primary_only=True)
    ivt_stats = _bam.get_contig_stats(ivt_bam, min_mapq=0, primary_only=True)

    # Filter
    passed_contigs, _ = _bam.filter_contigs(native_stats, ivt_stats, min_depth=15.0)
    report.n_contigs_total = len(passed_contigs)

    # Select contigs to profile
    if args.contigs:
        profile_contigs = [c for c in args.contigs if c in passed_contigs]
        if not profile_contigs:
            logger.error("None of the specified contigs passed filtering!")
            sys.exit(1)
    else:
        # Pick contigs with diverse depths: high, medium, low
        depths = [(c, native_stats[c].mean_depth + ivt_stats[c].mean_depth)
                  for c in passed_contigs]
        depths.sort(key=lambda x: x[1], reverse=True)
        n = min(args.max_contigs, len(depths))
        if n <= 1:
            profile_contigs = [depths[0][0]] if depths else []
        else:
            # Spread across depth range
            indices = [int(i * (len(depths) - 1) / (n - 1)) for i in range(n)]
            profile_contigs = [depths[i][0] for i in indices]

    report.n_contigs_profiled = len(profile_contigs)
    logger.info("=" * 60)
    logger.info("Profiling %d contigs (of %d passed): %s",
                len(profile_contigs), len(passed_contigs), profile_contigs)
    logger.info("=" * 60)

    tmp_root = Path(tempfile.mkdtemp(prefix="baleen-profile-"))
    pipeline_t0 = time.perf_counter()

    for idx, contig in enumerate(profile_contigs, 1):
        logger.info("--- Profiling contig %d/%d: %s ---", idx, len(profile_contigs), contig)
        prof = profile_contig(
            contig=contig,
            contig_idx=idx,
            total_contigs=len(profile_contigs),
            native_bam=native_bam,
            native_fastq=native_fastq,
            native_blow5=native_blow5,
            ivt_bam=ivt_bam,
            ivt_fastq=ivt_fastq,
            ivt_blow5=ivt_blow5,
            ref_fasta=ref_fasta,
            native_stats=native_stats,
            ivt_stats=ivt_stats,
            tmp_root=tmp_root,
            use_cuda=args.use_cuda,
            padding=args.padding,
            subsample_n=args.subsample_n,
            num_cuda_streams=args.num_cuda_streams,
            run_hmm=not args.no_hmm,
        )
        report.contigs.append(prof)

    report.total_wall_clock = time.perf_counter() - pipeline_t0

    # Aggregate
    for c in report.contigs:
        if c.error:
            continue
        report.total_bam_split += c.bam_split_native + c.bam_split_ivt
        report.total_f5c += c.f5c_native + c.f5c_ivt
        report.total_signal_parse += c.signal_parse_native + c.signal_parse_ivt
        report.total_signal_extract += c.signal_extract
        report.total_dtw += c.dtw_total
        report.total_hierarchical += c.hierarchical_total

    # Print summary
    print("\n" + "=" * 60)
    print("PROFILE SUMMARY")
    print("=" * 60)
    print(f"Total wall clock:     {_fmt(report.total_wall_clock)}")
    print(f"Contigs profiled:     {report.n_contigs_profiled} / {report.n_contigs_total}")
    print(f"DTW backend:          {report.dtw_backend}")
    print(f"CUDA used:            {report.cuda_used}")
    print()

    stages = [
        ("BAM split",       report.total_bam_split),
        ("f5c eventalign",  report.total_f5c),
        ("Signal parsing",  report.total_signal_parse),
        ("Signal extract",  report.total_signal_extract),
        ("DTW computation", report.total_dtw),
        ("Hierarchical",    report.total_hierarchical),
    ]
    active_total = sum(s[1] for s in stages)
    if active_total > 0:
        for name, elapsed in stages:
            pct = elapsed / active_total * 100
            bar = "#" * int(pct / 2)
            print(f"  {name:20s}  {_fmt(elapsed):>8s}  ({pct:5.1f}%)  {bar}")

    print()
    for c in report.contigs:
        print(f"  Contig: {c.contig}")
        print(f"    depth: native={c.native_depth:.1f} ivt={c.ivt_depth:.1f}")
        print(f"    positions: {c.n_common_positions} common, {c.n_positions_computed} computed")
        if c.positions:
            total_pairs = sum(p.n_pairs for p in c.positions)
            avg_reads = np.mean([p.n_total for p in c.positions])
            avg_len = np.mean([p.signal_lengths_mean for p in c.positions])
            print(f"    reads/pos: avg={avg_reads:.1f}  total DTW pairs: {total_pairs:,}")
            print(f"    signal length: avg={avg_len:.0f} samples")
        if c.error:
            print(f"    ERROR: {c.error}")
        else:
            print(f"    bam_split={_fmt(c.bam_split_native + c.bam_split_ivt)}  "
                  f"f5c={_fmt(c.f5c_native + c.f5c_ivt)}  "
                  f"signals={_fmt(c.signal_parse_native + c.signal_parse_ivt + c.signal_extract)}  "
                  f"dtw={_fmt(c.dtw_total)}  "
                  f"hier={_fmt(c.hierarchical_total)}  "
                  f"total={_fmt(c.contig_total)}")
        print()

    # Save JSON (convert position details to summary to keep file manageable)
    def _to_dict(prof: ContigProfile) -> dict:
        d = {
            "contig": prof.contig,
            "native_depth": prof.native_depth,
            "ivt_depth": prof.ivt_depth,
            "n_common_positions": prof.n_common_positions,
            "n_positions_computed": prof.n_positions_computed,
            "bam_split_native_s": round(prof.bam_split_native, 3),
            "bam_split_ivt_s": round(prof.bam_split_ivt, 3),
            "f5c_native_s": round(prof.f5c_native, 3),
            "f5c_ivt_s": round(prof.f5c_ivt, 3),
            "signal_parse_native_s": round(prof.signal_parse_native, 3),
            "signal_parse_ivt_s": round(prof.signal_parse_ivt, 3),
            "signal_extract_s": round(prof.signal_extract, 3),
            "dtw_total_s": round(prof.dtw_total, 3),
            "dtw_chunk_count": prof.dtw_chunk_count,
            "dtw_chunk_times_s": [round(t, 3) for t in prof.dtw_chunk_times],
            "hierarchical_total_s": round(prof.hierarchical_total, 3),
            "contig_total_s": round(prof.contig_total, 3),
            "error": prof.error,
        }
        if prof.positions:
            reads_per_pos = [p.n_total for p in prof.positions]
            pairs_per_pos = [p.n_pairs for p in prof.positions]
            sig_lengths = [p.signal_lengths_mean for p in prof.positions]
            d["position_stats"] = {
                "count": len(prof.positions),
                "reads_per_pos_mean": round(float(np.mean(reads_per_pos)), 1),
                "reads_per_pos_median": round(float(np.median(reads_per_pos)), 1),
                "reads_per_pos_max": int(np.max(reads_per_pos)),
                "total_dtw_pairs": int(np.sum(pairs_per_pos)),
                "signal_length_mean": round(float(np.mean(sig_lengths)), 1),
            }
        return d

    out = {
        "timestamp": report.timestamp,
        "cuda_available": report.cuda_available,
        "cuda_used": report.cuda_used,
        "dtw_backend": report.dtw_backend,
        "threads": report.threads,
        "subsample_n": report.subsample_n,
        "padding": report.padding,
        "n_contigs_total": report.n_contigs_total,
        "n_contigs_profiled": report.n_contigs_profiled,
        "total_wall_clock_s": round(report.total_wall_clock, 3),
        "stage_totals_s": {
            "bam_split": round(report.total_bam_split, 3),
            "f5c": round(report.total_f5c, 3),
            "signal_parse": round(report.total_signal_parse, 3),
            "signal_extract": round(report.total_signal_extract, 3),
            "dtw": round(report.total_dtw, 3),
            "hierarchical": round(report.total_hierarchical, 3),
        },
        "contigs": [_to_dict(c) for c in report.contigs],
    }

    out_path = Path(args.output)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
