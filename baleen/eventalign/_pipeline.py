from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import multiprocessing as mp
import pickle
from pathlib import Path
import shutil
import tempfile
import time
from typing import Optional, Protocol, TypedDict, Union, cast

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from baleen import _cuda_dtw
from baleen.eventalign import _bam
from baleen.eventalign import _f5c
from baleen.eventalign import _signal

logger = logging.getLogger(__name__)


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m{secs:.1f}s"

PathLike = Union[str, Path]


class _DtwDistanceFn(Protocol):
    def __call__(
        self,
        seq1: NDArray[np.float32] | list[float],
        seq2: NDArray[np.float32] | list[float],
        use_open_start: bool = False,
        use_open_end: bool = False,
        use_cuda: Optional[bool] = None,
    ) -> float: ...


@dataclass
class PositionResult:
    position: int
    reference_kmer: str
    n_native_reads: int
    n_ivt_reads: int
    native_read_names: list[str]
    ivt_read_names: list[str]
    distance_matrix: NDArray[np.float64]


@dataclass
class ContigResult:
    contig: str
    native_depth: float
    ivt_depth: float
    positions: dict[int, PositionResult]


@dataclass
class PipelineMetadata:
    f5c_version: str
    min_depth: int
    use_cuda: Optional[bool]
    padding: int
    n_contigs_total: int
    n_contigs_passed_filter: int
    n_contigs_skipped: int
    filter_results: list[_bam.ContigFilterResult]


class _SerializedPayload(TypedDict):
    results: dict[str, ContigResult]
    metadata: PipelineMetadata


_dtw_distance = cast(_DtwDistanceFn, _cuda_dtw.dtw_distance)
_dtw_pairwise_varlen = _cuda_dtw.dtw_pairwise_varlen


def _compute_pairwise_distances(
    signals: list[NDArray[np.float32]],
    *,
    use_cuda: Optional[bool],
    use_open_start: bool,
    use_open_end: bool,
) -> NDArray[np.float64]:
    n = len(signals)
    n_pairs = n * (n - 1) // 2
    signal_lengths = [len(s) for s in signals]
    logger.debug(
        "  Computing %d pairwise DTW distances (%d signals, lengths %d–%d)",
        n_pairs, n, min(signal_lengths), max(signal_lengths),
    )
    t0 = time.perf_counter()

    want_cuda = use_cuda is True or (use_cuda is None and _cuda_dtw.CUDA_AVAILABLE)

    if want_cuda:
        matrix = _dtw_pairwise_varlen(
            signals,
            use_open_start=use_open_start,
            use_open_end=use_open_end,
            use_cuda=True,
        )
    elif not use_open_start and not use_open_end:
        matrix = _compute_pairwise_batch(signals)
    else:
        matrix = _compute_pairwise_loop(
            signals,
            use_cuda=False,
            use_open_start=use_open_start,
            use_open_end=use_open_end,
        )

    elapsed = time.perf_counter() - t0
    logger.debug("  DTW computation done: %d pairs in %s", n_pairs, _fmt_elapsed(elapsed))
    return matrix


def _compute_pairwise_batch(
    signals: list[NDArray[np.float32]],
) -> NDArray[np.float64]:
    from tslearn.metrics import dtw as _tslearn_dtw

    n = len(signals)
    prepped = [s.reshape(-1, 1) for s in signals]
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(_tslearn_dtw(prepped[i], prepped[j]))
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix


def _compute_pairwise_loop(
    signals: list[NDArray[np.float32]],
    *,
    use_cuda: Optional[bool],
    use_open_start: bool,
    use_open_end: bool,
) -> NDArray[np.float64]:
    n = len(signals)
    prepped = [
        np.ascontiguousarray(np.asarray(s, dtype=np.float32))
        for s in signals
    ]
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            distance = _dtw_distance(
                prepped[i],
                prepped[j],
                use_open_start=use_open_start,
                use_open_end=use_open_end,
                use_cuda=use_cuda,
            )
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def save_results(
    results: dict[str, ContigResult],
    metadata: PipelineMetadata,
    output_path: PathLike,
) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump({"results": results, "metadata": metadata}, handle)
    logger.info("Saved pipeline results to %s", out_path)
    return out_path


def load_results(output_path: PathLike) -> tuple[dict[str, ContigResult], PipelineMetadata]:
    in_path = Path(output_path)
    with in_path.open("rb") as handle:
        payload = cast(_SerializedPayload, pickle.load(handle))
    return payload["results"], payload["metadata"]


def _cleanup_paths(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _process_contig(
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
    native_stats: dict[str, _bam.ContigStats],
    ivt_stats: dict[str, _bam.ContigStats],
    tmp_root: Path,
    use_cuda: Optional[bool],
    use_open_start: bool,
    use_open_end: bool,
    padding: int,
    rna: bool,
    kmer_model: Optional[str],
    extra_f5c_args: Optional[list[str]],
    min_mapq: int,
    primary_only: bool,
    cleanup_temp: bool,
    num_cuda_streams: int,
) -> tuple[str, ContigResult]:
    """Process a single contig: BAM split → eventalign → signal extraction → DTW.

    This function is designed to be called in parallel by multiple worker processes.

    Parameters
    ----------
    contig : str
        Contig name to process.
    contig_idx : int
        Index of this contig (1-based, for logging).
    total_contigs : int
        Total number of contigs (for logging).
    ... (other params match run_pipeline)

    Returns
    -------
    tuple[str, ContigResult]
        (contig_name, result) tuple for aggregation.
    """
    contig_t0 = time.perf_counter()
    logger.info(
        "  [Contig %d/%d] %s  (native_depth=%.1f, ivt_depth=%.1f)",
        contig_idx, total_contigs, contig,
        native_stats[contig].mean_depth, ivt_stats[contig].mean_depth,
    )

    contig_tmp = tmp_root / contig
    contig_tmp.mkdir(parents=True, exist_ok=True)

    logger.info("    Splitting BAM → native contig BAM...")
    native_contig_bam = _bam.split_bam_contig(
        native_bam,
        contig,
        contig_tmp / "native",
        primary_only=primary_only,
        min_mapq=min_mapq,
    )
    logger.info("    Splitting BAM → IVT contig BAM...")
    ivt_contig_bam = _bam.split_bam_contig(
        ivt_bam,
        contig,
        contig_tmp / "ivt",
        primary_only=primary_only,
        min_mapq=min_mapq,
    )

    native_tsv = contig_tmp / "native.eventalign.tsv"
    ivt_tsv = contig_tmp / "ivt.eventalign.tsv"

    logger.info("    Running f5c eventalign (native)...")
    ea_t0 = time.perf_counter()
    _ = _f5c.run_eventalign(
        native_contig_bam,
        ref_fasta,
        native_fastq,
        native_blow5,
        native_tsv,
        rna=rna,
        kmer_model=kmer_model,
        extra_args=extra_f5c_args,
    )
    logger.info("    Running f5c eventalign (IVT)...")
    _ = _f5c.run_eventalign(
        ivt_contig_bam,
        ref_fasta,
        ivt_fastq,
        ivt_blow5,
        ivt_tsv,
        rna=rna,
        kmer_model=kmer_model,
        extra_args=extra_f5c_args,
    )
    logger.info("    Eventalign done (%s)", _fmt_elapsed(time.perf_counter() - ea_t0))

    logger.info("    Parsing signals and finding common positions...")
    native_by_pos = _signal.group_signals_by_position(native_tsv)
    ivt_by_pos = _signal.group_signals_by_position(ivt_tsv)
    common_positions = _signal.get_common_positions(native_by_pos, ivt_by_pos)
    logger.info("    %d common positions found", len(common_positions))

    position_results: dict[int, PositionResult] = {}
    n_skipped = 0
    dtw_t0 = time.perf_counter()

    # Phase 1: Collect all signals (CPU)
    position_data: list[tuple[int, str, list[str], list[str], list[NDArray[np.float32]]]] = []
    for pos in tqdm(
        common_positions,
        desc=f"    {contig} signals",
        unit="pos",
        leave=False,
    ):
        native_pos = native_by_pos[pos]
        ivt_pos = ivt_by_pos[pos]

        if padding > 0:
            native_read_names, native_signals = _signal.extract_signals_for_dtw_padded(
                native_by_pos, pos, padding,
            )
            ivt_read_names, ivt_signals = _signal.extract_signals_for_dtw_padded(
                ivt_by_pos, pos, padding,
            )
        else:
            native_read_names, native_signals = _signal.extract_signals_for_dtw(native_pos)
            ivt_read_names, ivt_signals = _signal.extract_signals_for_dtw(ivt_pos)

        if not native_signals or not ivt_signals:
            logger.debug(
                "    Skipping pos=%d: empty signals (native=%d, ivt=%d)",
                pos, len(native_signals), len(ivt_signals),
            )
            n_skipped += 1
            continue

        all_signals = native_signals + ivt_signals
        kmer = native_pos.reference_kmer
        logger.debug(
            "    [Position %d/%d] pos=%d kmer=%s  %d signals (%d native + %d ivt)",
            len(position_data) + 1, len(common_positions), pos,
            kmer,
            len(all_signals), len(native_signals), len(ivt_signals),
        )
        position_data.append((
            pos, kmer,
            native_read_names, ivt_read_names, all_signals,
        ))

    # Phase 2: Batch DTW (single GPU call for all positions)
    if position_data:
        all_signal_lists = [d[4] for d in position_data]
        all_matrices = _cuda_dtw.dtw_multi_position_pairwise(
            all_signal_lists,
            use_open_start=use_open_start,
            use_open_end=use_open_end,
            use_cuda=use_cuda,
            num_streams=num_cuda_streams,
        )

        # Phase 3: Package results
        for (pos, kmer, nat_names, ivt_names, _sigs), matrix in zip(position_data, all_matrices):
            position_results[pos] = PositionResult(
                position=pos,
                reference_kmer=kmer,
                n_native_reads=len(nat_names),
                n_ivt_reads=len(ivt_names),
                native_read_names=nat_names,
                ivt_read_names=ivt_names,
                distance_matrix=matrix,
            )

    dtw_elapsed = _fmt_elapsed(time.perf_counter() - dtw_t0)

    native_depth = native_stats[contig].mean_depth
    ivt_depth = ivt_stats[contig].mean_depth
    result = ContigResult(
        contig=contig,
        native_depth=native_depth,
        ivt_depth=ivt_depth,
        positions=position_results,
    )
    contig_elapsed = _fmt_elapsed(time.perf_counter() - contig_t0)
    logger.info(
        "  [Contig %d/%d] %s done: %d positions (%d skipped), DTW in %s, total %s",
        contig_idx, total_contigs, contig,
        len(position_results), n_skipped, dtw_elapsed, contig_elapsed,
    )

    if cleanup_temp:
        files_to_remove = [
            native_contig_bam,
            Path(f"{native_contig_bam}.bai"),
            native_contig_bam.with_suffix(".bai"),
            ivt_contig_bam,
            Path(f"{ivt_contig_bam}.bai"),
            ivt_contig_bam.with_suffix(".bai"),
            native_tsv,
            ivt_tsv,
        ]
        _cleanup_paths(files_to_remove)
        for subdir in [contig_tmp / "native", contig_tmp / "ivt", contig_tmp]:
            if subdir.exists():
                shutil.rmtree(subdir, ignore_errors=True)

    return contig, result


def run_pipeline(
    native_bam: PathLike,
    native_fastq: PathLike,
    native_blow5: PathLike,
    ivt_bam: PathLike,
    ivt_fastq: PathLike,
    ivt_blow5: PathLike,
    ref_fasta: PathLike,
    *,
    min_depth: int = 15,
    use_cuda: Optional[bool] = None,
    use_open_start: bool = False,
    use_open_end: bool = False,
    padding: int = 0,
    output_dir: Optional[PathLike] = None,
    cleanup_temp: bool = True,
    rna: bool = True,
    kmer_model: Optional[str] = None,
    extra_f5c_args: Optional[list[str]] = None,
    min_mapq: int = 0,
    primary_only: bool = True,
    threads: int = 1,
    num_cuda_streams: int = 16,
) -> tuple[dict[str, ContigResult], PipelineMetadata]:
    pipeline_t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Starting baleen eventalign pipeline")
    logger.info("  native_bam:   %s", native_bam)
    logger.info("  native_fastq: %s", native_fastq)
    logger.info("  native_blow5: %s", native_blow5)
    logger.info("  ivt_bam:      %s", ivt_bam)
    logger.info("  ivt_fastq:    %s", ivt_fastq)
    logger.info("  ivt_blow5:    %s", ivt_blow5)
    logger.info("  ref_fasta:    %s", ref_fasta)
    logger.info("  min_depth=%d  use_cuda=%s  rna=%s  padding=%d  threads=%d",
                min_depth, use_cuda, rna, padding, threads)
    logger.info("  open_start=%s  open_end=%s  min_mapq=%d  primary_only=%s  cuda_streams=%d",
                use_open_start, use_open_end, min_mapq, primary_only, num_cuda_streams)
    logger.info("=" * 60)

    # Validate threads parameter
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")

    native_bam = Path(native_bam)
    native_fastq = Path(native_fastq)
    native_blow5 = Path(native_blow5)
    ivt_bam = Path(ivt_bam)
    ivt_fastq = Path(ivt_fastq)
    ivt_blow5 = Path(ivt_blow5)
    ref_fasta = Path(ref_fasta)

    # ---- Step 1: f5c version check ----
    logger.info("[Step 1/6] Checking f5c availability...")
    f5c_version = _f5c.check_f5c()
    logger.info("[Step 1/6] f5c version %s OK", f5c_version)

    # ---- Step 2: Indexing ----
    logger.info("[Step 2/6] Indexing FASTQ and BLOW5 files...")
    step_t0 = time.perf_counter()
    logger.info("  Indexing native FASTQ against BLOW5...")
    _f5c.index_fastq_blow5(native_fastq, native_blow5)
    logger.info("  Indexing IVT FASTQ against BLOW5...")
    _f5c.index_fastq_blow5(ivt_fastq, ivt_blow5)
    logger.info("  Indexing native BLOW5...")
    _f5c.index_blow5(native_blow5)
    logger.info("  Indexing IVT BLOW5...")
    _f5c.index_blow5(ivt_blow5)
    logger.info("[Step 2/6] Indexing complete (%s)", _fmt_elapsed(time.perf_counter() - step_t0))

    # ---- Step 3: BAM validation & contig stats ----
    logger.info("[Step 3/6] Validating BAMs and computing contig statistics...")
    step_t0 = time.perf_counter()
    _bam.validate_bam(native_bam)
    _bam.validate_bam(ivt_bam)

    logger.info("  Computing native BAM contig stats...")
    native_stats = _bam.get_contig_stats(
        native_bam,
        min_mapq=min_mapq,
        primary_only=primary_only,
    )
    logger.info("  Computing IVT BAM contig stats...")
    ivt_stats = _bam.get_contig_stats(
        ivt_bam,
        min_mapq=min_mapq,
        primary_only=primary_only,
    )
    logger.info("[Step 3/6] BAM stats complete: %d native contigs, %d IVT contigs (%s)",
                len(native_stats), len(ivt_stats), _fmt_elapsed(time.perf_counter() - step_t0))

    # ---- Step 4: Contig filtering ----
    logger.info("[Step 4/6] Filtering contigs (min_depth=%d)...", min_depth)
    passed_contigs, filter_results = _bam.filter_contigs(
        native_stats,
        ivt_stats,
        min_depth=float(min_depth),
    )
    logger.info("[Step 4/6] %d/%d contigs passed filtering",
                len(passed_contigs), len(filter_results))
    for fr in filter_results:
        if fr.passed:
            logger.debug("  PASS: %s (native_depth=%.1f, ivt_depth=%.1f)",
                         fr.contig,
                         fr.native_stats.mean_depth if fr.native_stats else 0,
                         fr.ivt_stats.mean_depth if fr.ivt_stats else 0)
        else:
            logger.info("  SKIP: %s — %s", fr.contig, fr.reason.value)

    metadata = PipelineMetadata(
        f5c_version=f5c_version,
        min_depth=min_depth,
        use_cuda=use_cuda,
        padding=padding,
        n_contigs_total=len(filter_results),
        n_contigs_passed_filter=len(passed_contigs),
        n_contigs_skipped=len(filter_results) - len(passed_contigs),
        filter_results=filter_results,
    )

    results: dict[str, ContigResult] = {}

    if not passed_contigs:
        logger.warning("[Step 5/6] No contigs passed filtering; returning empty results.")
        if output_dir is not None:
            _ = save_results(results, metadata, Path(output_dir) / "pipeline_results.pkl")
        elapsed = _fmt_elapsed(time.perf_counter() - pipeline_t0)
        logger.info("Pipeline finished (no results) in %s", elapsed)
        return results, metadata

    # ---- Step 5: Per-contig eventalign + signal extraction + DTW ----
    logger.info("[Step 5/6] Processing %d contigs (eventalign → signals → DTW)...",
                len(passed_contigs))
    tmp_root = Path(tempfile.mkdtemp(prefix="baleen-eventalign-"))
    logger.debug("  Temporary directory: %s", tmp_root)

    try:
        if threads > 1:
            # Parallel processing with multiprocessing
            logger.info("  Using %d parallel workers (spawn context)", threads)
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=threads, mp_context=ctx) as executor:
                futures = {
                    executor.submit(
                        _process_contig,
                        contig=contig,
                        contig_idx=idx,
                        total_contigs=len(passed_contigs),
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
                        use_cuda=use_cuda,
                        use_open_start=use_open_start,
                        use_open_end=use_open_end,
                        padding=padding,
                        rna=rna,
                        kmer_model=kmer_model,
                        extra_f5c_args=extra_f5c_args,
                        min_mapq=min_mapq,
                        primary_only=primary_only,
                        cleanup_temp=cleanup_temp,
                        num_cuda_streams=num_cuda_streams,
                    ): contig
                    for idx, contig in enumerate(passed_contigs, 1)
                }
                with tqdm(
                    total=len(passed_contigs),
                    desc="Contigs",
                    unit="contig",
                ) as pbar:
                    for future in as_completed(futures):
                        contig_name, contig_result = future.result()
                        results[contig_name] = contig_result
                        pbar.set_postfix_str(contig_name)
                        pbar.update(1)
        else:
            # Sequential processing (original behavior)
            for contig_idx, contig in tqdm(
                list(enumerate(passed_contigs, 1)),
                desc="Contigs",
                unit="contig",
            ):
                contig_name, contig_result = _process_contig(
                    contig=contig,
                    contig_idx=contig_idx,
                    total_contigs=len(passed_contigs),
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
                    use_cuda=use_cuda,
                    use_open_start=use_open_start,
                    use_open_end=use_open_end,
                    padding=padding,
                    rna=rna,
                    kmer_model=kmer_model,
                    extra_f5c_args=extra_f5c_args,
                    min_mapq=min_mapq,
                    primary_only=primary_only,
                    cleanup_temp=cleanup_temp,
                    num_cuda_streams=num_cuda_streams,
                )
                results[contig_name] = contig_result
    finally:
        if cleanup_temp and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    # ---- Step 6: Save results ----
    if output_dir is not None:
        logger.info("[Step 6/6] Saving results to %s ...", output_dir)
        _ = save_results(results, metadata, Path(output_dir) / "pipeline_results.pkl")
    else:
        logger.info("[Step 6/6] No output_dir specified; results returned in memory only")

    total_positions = sum(len(cr.positions) for cr in results.values())
    pipeline_elapsed = _fmt_elapsed(time.perf_counter() - pipeline_t0)
    logger.info("=" * 60)
    logger.info("Pipeline complete: %d contigs, %d positions, %s",
                len(results), total_positions, pipeline_elapsed)
    logger.info("=" * 60)

    return results, metadata
