from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import multiprocessing as mp
import pickle
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Optional, Protocol, TypedDict, Union, cast

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

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



_MIN_GPU_PER_WORKER = 4 * 1024 ** 3  # 4 GB — minimum for efficient DTW chunks


def _gpu_concurrent_workers(
    threads: int,
    gpu_mem: int,
    use_cuda: Optional[bool],
) -> int:
    """Estimate how many workers can run GPU DTW concurrently.

    Returns the number of workers that can each hold a large DTW chunk
    (``_MIN_GPU_PER_WORKER``) on the GPU simultaneously.  This is used
    to size chunks (``chunk_mem_limit = gpu_mem * 0.8 / gpu_workers``)
    so each kernel launch keeps the GPU busy.

    Total *threads* is NOT reduced — extra workers run CPU phases
    (f5c, HMM, aggregation) in parallel and naturally stagger their
    DTW phases.
    """
    if threads <= 1:
        return 1
    want_cuda = use_cuda is True or (use_cuda is None and _cuda_dtw.CUDA_AVAILABLE)
    if not want_cuda:
        return threads  # CPU mode: no GPU constraint on chunk sizing
    gpu_workers = max(1, int(gpu_mem * 0.8 / _MIN_GPU_PER_WORKER))
    gpu_workers = min(gpu_workers, threads)  # never exceed actual threads
    if gpu_workers < threads:
        logger.info(
            "  GPU chunk sizing: %d concurrent GPU workers "
            "(%.0f GB GPU, %d total threads)",
            gpu_workers, gpu_mem / 1024 ** 3, threads,
        )
    return gpu_workers


def _get_gpu_memory() -> int:
    """Try to detect total GPU memory in bytes via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            mb = int(result.stdout.strip().split('\n')[0])
            return mb * 1024 * 1024
    except Exception:
        pass
    return 8 * 1024 ** 3  # default 8 GB


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
    subsample: bool = True,
    subsample_n: int = 300,
    gpu_memory_bytes: Optional[int] = None,
    num_workers: int = 1,
    show_progress: bool = True,
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
    subsample : bool
        If True, subsample reads per condition per contig.
    subsample_n : int
        Max reads per condition when subsampling.
    gpu_memory_bytes : int or None
        GPU memory available for chunking.  Auto-detected if *None*.
    num_workers : int
        Number of parallel workers sharing the GPU.  Chunk memory limit
        is divided by this to prevent GPU OOM.
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

    _max_reads = subsample_n if subsample else None
    logger.info("    Splitting BAM → native contig BAM...")
    native_contig_bam = _bam.split_bam_contig(
        native_bam,
        contig,
        contig_tmp / "native",
        primary_only=primary_only,
        min_mapq=min_mapq,
        max_reads=_max_reads,
    )
    logger.info("    Splitting BAM → IVT contig BAM...")
    ivt_contig_bam = _bam.split_bam_contig(
        ivt_bam,
        contig,
        contig_tmp / "ivt",
        primary_only=primary_only,
        min_mapq=min_mapq,
        max_reads=_max_reads,
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
    contig_short = contig if len(contig) <= 20 else contig[:17] + "..."
    pbar = tqdm(
        total=len(common_positions),
        desc=f"  {contig_short}",
        unit="pos",
        leave=False,
        disable=not show_progress,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    pbar.set_postfix_str("extracting signals")
    for pos in common_positions:
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
            pbar.update(1)
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
        pbar.update(1)

    # Phase 2: Chunked DTW (split positions into GPU-memory-sized batches)
    if position_data:
        all_signal_lists = [d[4] for d in position_data]
        all_matrices: list[NDArray[np.float64]] = []

        total_gpu = gpu_memory_bytes if gpu_memory_bytes is not None else _get_gpu_memory()
        chunk_mem_limit = int(total_gpu * 0.8 / max(num_workers, 1))

        # Greedy bin-packing by estimated GPU memory
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

        pbar.set_postfix_str(f"DTW {len(position_data)} pos in {len(chunks)} chunk(s)")
        pbar.refresh()

        for chunk_idx, chunk_indices in enumerate(chunks):
            chunk_signals = [all_signal_lists[i] for i in chunk_indices]
            estimated_bytes = _cuda_dtw.estimate_gpu_memory(chunk_signals)

            chunk_matrices = _cuda_dtw.dtw_multi_position_pairwise(
                chunk_signals,
                use_open_start=use_open_start,
                use_open_end=use_open_end,
                use_cuda=use_cuda,
                num_streams=num_cuda_streams,
            )

            all_matrices.extend(chunk_matrices)

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

    pbar.set_postfix_str(f"done ({len(position_results)} pos, {n_skipped} skipped)")
    pbar.close()

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


def _process_contig_streaming(
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
    run_hmm: bool = True,
    hmm_params: object = None,
    keep_intermediate: bool = False,
    intermediate_dir: Optional[Path] = None,
    subsample: bool = True,
    subsample_n: int = 300,
    gpu_memory_bytes: Optional[int] = None,
    legacy_scoring: bool = False,
    num_workers: int = 1,
    mod_threshold: float = 0.9,
    show_progress: bool = True,
) -> tuple[str, "ContigModificationResult", list["SiteResult"]]:
    """Process a single contig end-to-end: DTW → HMM → site aggregation.

    Unlike :func:`_process_contig`, this fuses all stages so distance matrices
    can be garbage-collected before returning.  Only lightweight results are
    returned to the caller.

    Parameters
    ----------
    run_hmm
        Whether to run HMM smoothing (V3).
    hmm_params
        Optional trained HMM parameters.
    keep_intermediate
        If True, save the per-contig ``ContigResult`` pickle.
    intermediate_dir
        Directory for intermediate files (used when *keep_intermediate* is True).
    mod_threshold
        Per-read probability threshold for counting a read as modified.

    Returns
    -------
    tuple[str, ContigModificationResult, list[SiteResult]]
        ``(contig_name, hmm_result, per_site_results)`` — distance matrices
        are **not** included.
    """
    from baleen.eventalign._aggregation import aggregate_contig
    from baleen.eventalign._hierarchical import compute_sequential_modification_probabilities

    # Stage 1: DTW
    contig_name, contig_result = _process_contig(
        contig=contig,
        contig_idx=contig_idx,
        total_contigs=total_contigs,
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
        subsample=subsample,
        subsample_n=subsample_n,
        gpu_memory_bytes=gpu_memory_bytes,
        num_workers=num_workers,
        show_progress=show_progress,
    )

    # Stage 2: HMM smoothing
    cmr = compute_sequential_modification_probabilities(
        contig_result, run_hmm=run_hmm, hmm_params=hmm_params,
        legacy_scoring=legacy_scoring,
        show_progress=show_progress,
    )

    # Stage 3: Site-level aggregation (no FDR — done globally later)
    sites = aggregate_contig(cmr, mod_threshold=mod_threshold)

    # Optionally save intermediate ContigResult
    if keep_intermediate and intermediate_dir is not None:
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = intermediate_dir / f"{contig_name}.pkl"
        with pkl_path.open("wb") as fh:
            pickle.dump(contig_result, fh)
        logger.info("  Saved intermediate: %s", pkl_path)

    return contig_name, cmr, sites


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
    padding: int = 1,
    output_dir: Optional[PathLike] = None,
    cleanup_temp: bool = True,
    rna: bool = True,
    kmer_model: Optional[str] = None,
    extra_f5c_args: Optional[list[str]] = None,
    min_mapq: int = 0,
    primary_only: bool = True,
    threads: int = 1,
    num_cuda_streams: int = 16,
    gpu_memory_limit: Optional[int] = None,
    subsample: bool = True,
    subsample_n: int = 300,
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
    logger.info("  subsample=%s  subsample_n=%d  gpu_memory_limit=%s",
                subsample, subsample_n, gpu_memory_limit)
    logger.info("  cleanup_temp=%s  kmer_model=%s  extra_f5c_args=%s",
                cleanup_temp, kmer_model, extra_f5c_args)
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

    resolved_gpu_mem = gpu_memory_limit if gpu_memory_limit is not None else _get_gpu_memory()
    gpu_workers = _gpu_concurrent_workers(threads, resolved_gpu_mem, use_cuda)

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
                        subsample=subsample,
                        subsample_n=subsample_n,
                        gpu_memory_bytes=resolved_gpu_mem,
                        num_workers=gpu_workers,
                        show_progress=False,
                    ): contig
                    for idx, contig in enumerate(passed_contigs, 1)
                }
                failed = []
                with tqdm(
                    total=len(passed_contigs),
                    desc="Pipeline",
                    unit="contig",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} contigs [{elapsed}<{remaining}] {postfix}",
                ) as pbar:
                    for future in as_completed(futures):
                        contig = futures[future]
                        try:
                            contig_name, contig_result = future.result()
                        except Exception:
                            logger.exception("Worker failed for contig %s", contig)
                            failed.append(contig)
                            pbar.update(1)
                            continue
                        results[contig_name] = contig_result
                        n_pos = len(contig_result.positions)
                        pbar.set_postfix_str(f"{contig_name} ({n_pos} pos)")
                        pbar.update(1)
                if failed:
                    logger.error("%d contig(s) failed: %s", len(failed), ", ".join(failed))
        else:
            # Sequential processing (original behavior)
            for contig_idx, contig in enumerate(passed_contigs, 1):
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
                    subsample=subsample,
                    subsample_n=subsample_n,
                    gpu_memory_bytes=resolved_gpu_mem,
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


def run_pipeline_streaming(
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
    padding: int = 1,
    output_dir: Optional[PathLike] = None,
    cleanup_temp: bool = True,
    rna: bool = True,
    kmer_model: Optional[str] = None,
    extra_f5c_args: Optional[list[str]] = None,
    min_mapq: int = 0,
    primary_only: bool = True,
    threads: int = 1,
    num_cuda_streams: int = 16,
    run_hmm: bool = True,
    hmm_params: object = None,
    target_contigs: Optional[list[str]] = None,
    keep_intermediate: bool = False,
    gpu_memory_limit: Optional[int] = None,
    subsample: bool = True,
    subsample_n: int = 300,
    legacy_scoring: bool = False,
    mod_threshold: float = 0.9,
) -> tuple[dict[str, "ContigModificationResult"], list["SiteResult"], PipelineMetadata]:
    """Memory-efficient streaming pipeline: DTW → HMM → aggregation per contig.

    Each contig is processed end-to-end in a single worker.  Distance matrices
    are discarded after HMM scoring, so only lightweight results are kept in
    memory.

    Parameters
    ----------
    target_contigs
        If given, only process these contig(s).  Contigs not passing depth
        filters are silently skipped.
    keep_intermediate
        Save per-contig ``ContigResult`` pickle files under
        ``output_dir/intermediate/``.
    run_hmm
        Whether to run HMM smoothing (V3).
    hmm_params
        Optional trained HMM parameters.

    Returns
    -------
    tuple[dict[str, ContigModificationResult], list[SiteResult], PipelineMetadata]
        ``(hmm_results, fdr_corrected_sites, metadata)``
    """
    from baleen.eventalign._aggregation import SiteResult, _benjamini_hochberg

    pipeline_t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Starting baleen streaming pipeline")
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
    logger.info("  subsample=%s  subsample_n=%d  gpu_memory_limit=%s",
                subsample, subsample_n, gpu_memory_limit)
    logger.info("  run_hmm=%s  legacy_scoring=%s  mod_threshold=%.2f",
                run_hmm, legacy_scoring, mod_threshold)
    logger.info("  target_contigs=%s  keep_intermediate=%s  cleanup_temp=%s",
                target_contigs, keep_intermediate, cleanup_temp)
    logger.info("  kmer_model=%s  extra_f5c_args=%s", kmer_model, extra_f5c_args)
    logger.info("=" * 60)

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
    logger.info("[Step 1/5] Checking f5c availability...")
    f5c_version = _f5c.check_f5c()
    logger.info("[Step 1/5] f5c version %s OK", f5c_version)

    # ---- Step 2: Indexing ----
    logger.info("[Step 2/5] Indexing FASTQ and BLOW5 files...")
    step_t0 = time.perf_counter()
    _f5c.index_fastq_blow5(native_fastq, native_blow5)
    _f5c.index_fastq_blow5(ivt_fastq, ivt_blow5)
    _f5c.index_blow5(native_blow5)
    _f5c.index_blow5(ivt_blow5)
    logger.info("[Step 2/5] Indexing complete (%s)", _fmt_elapsed(time.perf_counter() - step_t0))

    # ---- Step 3: BAM validation & contig stats ----
    logger.info("[Step 3/5] Validating BAMs and computing contig statistics...")
    step_t0 = time.perf_counter()
    _bam.validate_bam(native_bam)
    _bam.validate_bam(ivt_bam)
    native_stats = _bam.get_contig_stats(native_bam, min_mapq=min_mapq, primary_only=primary_only)
    ivt_stats = _bam.get_contig_stats(ivt_bam, min_mapq=min_mapq, primary_only=primary_only)
    logger.info("[Step 3/5] BAM stats complete: %d native contigs, %d IVT contigs (%s)",
                len(native_stats), len(ivt_stats), _fmt_elapsed(time.perf_counter() - step_t0))

    # ---- Step 4: Contig filtering ----
    logger.info("[Step 4/5] Filtering contigs (min_depth=%d)...", min_depth)
    passed_contigs, filter_results = _bam.filter_contigs(
        native_stats, ivt_stats, min_depth=float(min_depth),
    )

    # Apply target contig filter
    if target_contigs is not None:
        target_set = set(target_contigs)
        skipped_targets = target_set - set(passed_contigs)
        if skipped_targets:
            logger.warning("  Target contigs not passing filters: %s", sorted(skipped_targets))
        passed_contigs = [c for c in passed_contigs if c in target_set]

    logger.info("[Step 4/5] %d contigs to process", len(passed_contigs))

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

    hmm_results: dict[str, ContigModificationResult] = {}
    all_sites: list[SiteResult] = []

    if not passed_contigs:
        logger.warning("[Step 5/5] No contigs to process; returning empty results.")
        elapsed = _fmt_elapsed(time.perf_counter() - pipeline_t0)
        logger.info("Pipeline finished (no results) in %s", elapsed)
        return hmm_results, all_sites, metadata

    # ---- Step 5: Per-contig streaming (DTW → HMM → aggregation) ----
    logger.info("[Step 5/5] Processing %d contigs (streaming: DTW → HMM → aggregation)...",
                len(passed_contigs))
    tmp_root = Path(tempfile.mkdtemp(prefix="baleen-streaming-"))

    intermediate_dir = None
    if keep_intermediate and output_dir is not None:
        intermediate_dir = Path(output_dir) / "intermediate"

    resolved_gpu_mem = gpu_memory_limit if gpu_memory_limit is not None else _get_gpu_memory()
    gpu_workers = _gpu_concurrent_workers(threads, resolved_gpu_mem, use_cuda)

    try:
        worker_kwargs = dict(
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
            run_hmm=run_hmm,
            hmm_params=hmm_params,
            keep_intermediate=keep_intermediate,
            intermediate_dir=intermediate_dir,
            subsample=subsample,
            subsample_n=subsample_n,
            gpu_memory_bytes=resolved_gpu_mem,
            legacy_scoring=legacy_scoring,
            num_workers=gpu_workers,
            mod_threshold=mod_threshold,
            show_progress=(threads <= 1),
        )

        if threads > 1:
            logger.info("  Using %d parallel workers (spawn context)", threads)
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=threads, mp_context=ctx) as executor:
                futures = {
                    executor.submit(
                        _process_contig_streaming,
                        contig=contig,
                        contig_idx=idx,
                        total_contigs=len(passed_contigs),
                        **worker_kwargs,
                    ): contig
                    for idx, contig in enumerate(passed_contigs, 1)
                }
                failed = []
                with tqdm(
                    total=len(passed_contigs),
                    desc="Pipeline",
                    unit="contig",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} contigs [{elapsed}<{remaining}] {postfix}",
                ) as pbar:
                    for future in as_completed(futures):
                        contig = futures[future]
                        try:
                            contig_name, cmr, sites = future.result()
                        except Exception:
                            logger.exception("Worker failed for contig %s", contig)
                            failed.append(contig)
                            pbar.update(1)
                            continue
                        hmm_results[contig_name] = cmr
                        all_sites.extend(sites)
                        n_pos = len(cmr.position_stats)
                        pbar.set_postfix_str(f"{contig_name} ({n_pos} pos)")
                        pbar.update(1)
                if failed:
                    logger.error("%d contig(s) failed: %s", len(failed), ", ".join(failed))
        else:
            for contig_idx, contig in enumerate(passed_contigs, 1):
                contig_name, cmr, sites = _process_contig_streaming(
                    contig=contig,
                    contig_idx=contig_idx,
                    total_contigs=len(passed_contigs),
                    **worker_kwargs,
                )
                hmm_results[contig_name] = cmr
                all_sites.extend(sites)
    finally:
        if cleanup_temp and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    # ---- FDR correction across all sites ----
    if all_sites:
        pvalues = np.array([s.pvalue for s in all_sites], dtype=np.float64)
        padj = _benjamini_hochberg(pvalues)
        for site, adj in zip(all_sites, padj):
            site.padj = float(adj)

    total_positions = sum(len(cmr.position_stats) for cmr in hmm_results.values())
    pipeline_elapsed = _fmt_elapsed(time.perf_counter() - pipeline_t0)
    logger.info("=" * 60)
    logger.info("Streaming pipeline complete: %d contigs, %d positions, %d sites, %s",
                len(hmm_results), total_positions, len(all_sites), pipeline_elapsed)
    logger.info("=" * 60)

    return hmm_results, all_sites, metadata
