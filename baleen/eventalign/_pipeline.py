from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Literal, Optional, Protocol, TypedDict, Union, cast

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from baleen import _dtw
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


def _sanitize_contig_filename(name: str) -> str:
    """Map an arbitrary contig name to a safe filesystem stem.

    SAM spec permits characters like ``/``, ``\\``, ``..`` in reference names.
    To avoid path traversal or collisions when writing per-contig artifacts,
    we replace anything outside ``[A-Za-z0-9._-]`` with ``_`` and append a
    short hash of the original name as a disambiguator.
    """
    import hashlib
    import re

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if safe != name:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}-{digest}"
    return safe


# ---- Resume fingerprint ----------------------------------------------------
# A small JSON file under ``per_contig_dir`` records the inputs + parameters
# of the run that produced the existing per-contig slices.  ``--resume``
# refuses to proceed unless every fingerprint field matches the current run,
# so a half-finished run can never be silently mixed with outputs from a run
# using different ``--min-depth``, modified BAMs, etc.

_RESUME_PARAMS_FILENAME = ".run_params.json"
_RESUME_FINGERPRINT_SCHEMA = 1


def _file_fingerprint(path: Optional[PathLike]) -> Optional[dict]:
    """Cheap (size, mtime_ns) fingerprint — much faster than hashing GBs."""
    if path is None:
        return None
    try:
        st = os.stat(str(path))
    except OSError:
        return None
    return {
        "path": str(Path(path).resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _compute_resume_fingerprint(
    *,
    native_bam: PathLike,
    native_fastq: PathLike,
    native_blow5: PathLike,
    ivt_bam: PathLike,
    ivt_fastq: PathLike,
    ivt_blow5: PathLike,
    ref_fasta: PathLike,
    min_depth: float,
    depth_mode: str,
    padding: int,
    min_mapq: int,
    primary_only: bool,
    subsample: bool,
    subsample_n: int,
    legacy_scoring: bool,
    mod_threshold: float,
    write_bam: bool,
    run_hmm: bool,
    target_contigs: Optional[list[str]],
    read_intersection: bool,
) -> dict:
    """Build a JSON-serializable dict capturing everything that would
    invalidate a partial run.
    """
    return {
        "schema_version": _RESUME_FINGERPRINT_SCHEMA,
        "inputs": {
            "native_bam": _file_fingerprint(native_bam),
            "native_fastq": _file_fingerprint(native_fastq),
            "native_blow5": _file_fingerprint(native_blow5),
            "ivt_bam": _file_fingerprint(ivt_bam),
            "ivt_fastq": _file_fingerprint(ivt_fastq),
            "ivt_blow5": _file_fingerprint(ivt_blow5),
            "ref_fasta": _file_fingerprint(ref_fasta),
        },
        "params": {
            "min_depth": float(min_depth),
            "depth_mode": str(depth_mode),
            "padding": int(padding),
            "min_mapq": int(min_mapq),
            "primary_only": bool(primary_only),
            "subsample": bool(subsample),
            "subsample_n": int(subsample_n),
            "legacy_scoring": bool(legacy_scoring),
            "mod_threshold": float(mod_threshold),
            "write_bam": bool(write_bam),
            "run_hmm": bool(run_hmm),
            "target_contigs": (
                sorted(target_contigs) if target_contigs else None
            ),
            "read_intersection": bool(read_intersection),
        },
    }


def _validate_resume_compatibility(
    per_contig_dir: Path,
    current: dict,
) -> None:
    """Compare ``current`` against the fingerprint stored under
    ``per_contig_dir`` and raise on any mismatch.
    """
    import json as _json

    fp_path = per_contig_dir / _RESUME_PARAMS_FILENAME
    if not fp_path.exists():
        raise RuntimeError(
            f"Cannot resume: {fp_path} not found.  The directory "
            f"{per_contig_dir} exists but lacks the parameter "
            f"fingerprint required to verify compatibility.  "
            f"Delete {per_contig_dir} or run without --resume."
        )
    try:
        prior = _json.loads(fp_path.read_text())
    except (OSError, _json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot resume: failed to read {fp_path}: {exc}"
        ) from exc

    mismatches: list[str] = []
    for section in ("inputs", "params"):
        prior_section = prior.get(section, {}) or {}
        curr_section = current.get(section, {}) or {}
        for key in sorted(set(prior_section) | set(curr_section)):
            v_prev = prior_section.get(key)
            v_now = curr_section.get(key)
            if v_prev != v_now:
                mismatches.append(
                    f"  {section}.{key}: prior={v_prev!r} now={v_now!r}"
                )
    if mismatches:
        raise RuntimeError(
            "Cannot resume: parameter fingerprint mismatch.  "
            "Resuming would silently mix outputs from different runs.\n"
            + "\n".join(mismatches)
            + f"\nDelete {per_contig_dir} or run without --resume."
        )


def _scan_completed_contigs(
    per_contig_dir: Path,
    passed_contigs: list[str],
    write_bam: bool,
) -> dict[str, "ContigSummary"]:
    """Return contig→ContigSummary for contigs whose per-contig artifacts
    already exist under ``per_contig_dir``.

    A contig is "completed" iff its ``<safe_name>.tsv`` (and, when
    ``write_bam=True``, ``<safe_name>.bam``) is present.  Atomic
    rename in the worker guarantees these files are fully written.
    Counts are recovered by parsing the TSV.
    """
    import csv as _csv

    completed: dict[str, ContigSummary] = {}
    for contig in passed_contigs:
        safe = _sanitize_contig_filename(contig)
        tsv = per_contig_dir / f"{safe}.tsv"
        if not tsv.exists():
            continue
        bam = per_contig_dir / f"{safe}.bam"
        if write_bam and not bam.exists():
            continue

        n_sites = 0
        n_significant = 0
        try:
            with tsv.open(newline="") as fh:
                reader = _csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    n_sites += 1
                    padj_str = row.get("padj")
                    if padj_str is None:
                        continue
                    try:
                        if float(padj_str) < 0.05:
                            n_significant += 1
                    except (TypeError, ValueError):
                        pass
        except OSError:
            # Treat unreadable slice as "not done"; worker will redo it.
            continue

        completed[contig] = ContigSummary(
            contig_name=contig,
            n_sites=n_sites,
            n_positions=0,  # Not recoverable from TSV; only used for logging.
            n_significant=n_significant,
            tsv_path=tsv,
            bam_path=bam if write_bam else None,
        )
    return completed


def _write_resume_fingerprint(per_contig_dir: Path, current: dict) -> None:
    """Persist the fingerprint atomically (tmp + rename)."""
    import json as _json

    fp_path = per_contig_dir / _RESUME_PARAMS_FILENAME
    tmp = fp_path.with_suffix(fp_path.suffix + ".tmp")
    tmp.write_text(_json.dumps(current, indent=2, sort_keys=True))
    os.replace(tmp, fp_path)


PathLike = Union[str, Path]


class _DtwDistanceFn(Protocol):
    def __call__(
        self,
        seq1: NDArray[np.float32] | list[float],
        seq2: NDArray[np.float32] | list[float],
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


@dataclass
class ContigSummary:
    """Lightweight per-contig outcome returned by the streaming worker.

    Holds counts and on-disk paths to the per-contig TSV/BAM slices —
    no per-read arrays — so the main process memory stays O(N_contigs).
    """
    contig_name: str
    n_sites: int
    n_positions: int
    n_significant: int
    tsv_path: Path
    bam_path: Optional[Path] = None


class _SerializedPayload(TypedDict):
    results: dict[str, ContigResult]
    metadata: PipelineMetadata


_dtw_distance = cast(_DtwDistanceFn, _dtw.dtw_distance)
_dtw_pairwise_varlen = _dtw.dtw_pairwise_varlen


def _compute_pairwise_distances(
    signals: list[NDArray[np.float32]],
    *,
    use_cuda: Optional[bool],
) -> NDArray[np.float64]:
    n = len(signals)
    n_pairs = n * (n - 1) // 2
    signal_lengths = [len(s) for s in signals]
    logger.debug(
        "  Computing %d pairwise DTW distances (%d signals, lengths %d–%d)",
        n_pairs, n, min(signal_lengths), max(signal_lengths),
    )
    t0 = time.perf_counter()

    want_cuda = use_cuda is True or (use_cuda is None and _dtw.CUDA_AVAILABLE)

    if want_cuda:
        matrix = _dtw_pairwise_varlen(
            signals,
            use_cuda=True,
        )
    else:
        matrix = _compute_pairwise_batch(signals)

    elapsed = time.perf_counter() - t0
    logger.debug("  DTW computation done: %d pairs in %s", n_pairs, _fmt_elapsed(elapsed))
    return matrix


def _compute_pairwise_batch(
    signals: list[NDArray[np.float32]],
) -> NDArray[np.float64]:
    return _dtw_pairwise_varlen(signals, use_cuda=False)


def _compute_pairwise_loop(
    signals: list[NDArray[np.float32]],
    *,
    use_cuda: Optional[bool],
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
                use_cuda=use_cuda,
            )
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix



_MIN_GPU_PER_WORKER = 4 * 1024 ** 3  # 4 GB — minimum for efficient DTW chunks


def _gpu_concurrent_workers(
    threads: int,
    gpu_mems: list[int],
    cuda_devices: Optional[list[int]],
) -> tuple[int, list[int]]:
    """Estimate how many workers can run GPU DTW concurrently.

    Returns ``(total_gpu_workers, device_for_worker)`` where
    ``device_for_worker[i]`` is the CUDA device index for worker *i*.

    Workers are distributed across devices proportional to each device's
    memory.  Total *threads* is NOT reduced — extra workers run CPU phases
    (f5c, HMM, aggregation) in parallel and naturally stagger their
    DTW phases.
    """
    if threads <= 1:
        devices = cuda_devices if cuda_devices else [0]
        return 1, [devices[0]]

    want_cuda = cuda_devices is None or len(cuda_devices) > 0
    if not want_cuda:
        return threads, []  # CPU mode: no GPU constraint on chunk sizing

    if not gpu_mems:
        gpu_mems = [8 * 1024 ** 3]

    devices = cuda_devices if cuda_devices else list(range(len(gpu_mems)))

    # Compute workers per device proportional to memory
    total_mem = sum(gpu_mems[d] if d < len(gpu_mems) else gpu_mems[0] for d in devices)
    device_for_worker: list[int] = []
    remaining_workers = threads

    for i, dev in enumerate(devices):
        mem = gpu_mems[dev] if dev < len(gpu_mems) else gpu_mems[0]
        if i == len(devices) - 1:
            n_workers = remaining_workers
        else:
            n_workers = max(1, round(threads * mem / total_mem))
            n_workers = min(n_workers, remaining_workers)
        remaining_workers -= n_workers
        device_for_worker.extend([dev] * n_workers)

    total_gpu_workers = len(device_for_worker)

    if len(devices) > 1:
        from collections import Counter
        dist = Counter(device_for_worker)
        logger.info(
            "  Multi-GPU: %d workers across %d devices %s",
            total_gpu_workers, len(devices),
            {d: dist[d] for d in devices},
        )

    return total_gpu_workers, device_for_worker


def _get_gpu_memory(cuda_devices: Optional[list[int]] = None) -> list[int]:
    """Return total GPU memory in bytes per device.

    Parameters
    ----------
    cuda_devices : list[int] or None
        If given, only return memory for these device indices.
        If None, return memory for all visible devices.

    Returns
    -------
    list[int]
        Memory in bytes per device.  Falls back to ``[8 GB]`` on failure.
    """
    all_mems = _dtw.get_per_device_memory()
    if not all_mems:
        return [8 * 1024 ** 3]
    if cuda_devices is not None:
        return [all_mems[d] for d in cuda_devices if d < len(all_mems)]
    return all_mems


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
    cuda_device: int = 0,
    allowed_native_reads_path: Optional[Path] = None,
    allowed_ivt_reads_path: Optional[Path] = None,
) -> tuple[str, ContigResult]:
    """Process a single contig: BAM split → eventalign → signal extraction → DTW.

    This function is designed to be called in parallel by multiple worker processes.

    Parameters
    ----------
    allowed_native_reads_path, allowed_ivt_reads_path : pathlib.Path or None
        Optional paths to newline-separated UUID files restricting which
        reads from the native / IVT BAM are included.  Used by the
        read-id intersection step in ``run_pipeline_streaming`` to keep
        every stage in sync with the BAM ∩ FASTQ ∩ BLOW5 set.  Worker
        loads these once per call (cheap; a few MB max).

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

    contig_tmp = tmp_root / _sanitize_contig_filename(contig)
    contig_tmp.mkdir(parents=True, exist_ok=True)

    # Resolve the per-condition read-id intersection (if any).  Lazy load
    # inside the worker so we don't pay the pickling cost across spawn.
    from baleen.eventalign._read_ids import load_read_ids
    allowed_native = load_read_ids(allowed_native_reads_path)
    allowed_ivt = load_read_ids(allowed_ivt_reads_path)

    _max_reads = subsample_n if subsample else None
    logger.info("    Splitting BAM → native contig BAM...")
    native_contig_bam = _bam.split_bam_contig(
        native_bam,
        contig,
        contig_tmp / "native",
        primary_only=primary_only,
        min_mapq=min_mapq,
        max_reads=_max_reads,
        allowed_reads=allowed_native,
        _validated=True,
    )
    logger.info("    Splitting BAM → IVT contig BAM...")
    ivt_contig_bam = _bam.split_bam_contig(
        ivt_bam,
        contig,
        contig_tmp / "ivt",
        primary_only=primary_only,
        min_mapq=min_mapq,
        max_reads=_max_reads,
        allowed_reads=allowed_ivt,
        _validated=True,
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
        min_mapq=min_mapq,
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
        min_mapq=min_mapq,
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
            pos_estimate = _dtw.estimate_gpu_memory([sigs])
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
            estimated_bytes = _dtw.estimate_gpu_memory(chunk_signals)

            chunk_matrices = _dtw.dtw_multi_position_pairwise(
                chunk_signals,
                use_cuda=use_cuda,
                num_streams=num_cuda_streams,
                device_id=cuda_device,
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
    padding: int,
    rna: bool,
    kmer_model: Optional[str],
    extra_f5c_args: Optional[list[str]],
    min_mapq: int,
    primary_only: bool,
    cleanup_temp: bool,
    num_cuda_streams: int,
    per_contig_dir: Path,
    bam_header_dict: Optional[dict] = None,
    write_bam: bool = True,
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
    cuda_device: int = 0,
    allowed_native_reads_path: Optional[Path] = None,
    allowed_ivt_reads_path: Optional[Path] = None,
) -> ContigSummary:
    """Process a single contig end-to-end and flush its outputs to disk.

    Stages: DTW → HMM → site aggregation → flush per-contig TSV (always)
    and per-contig BAM (when ``write_bam=True``).  The ``cmr`` is dropped
    before return so peak memory is bounded by a single contig's footprint.

    Parameters
    ----------
    per_contig_dir
        Directory under which ``<contig>.tsv`` and ``<contig>.bam`` are written.
    bam_header
        Pre-built ``pysam.AlignmentHeader`` for output BAMs.  May be ``None``
        when ``write_bam=False``.
    write_bam
        Whether to flush a per-contig BAM slice.
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
    ContigSummary
        Lightweight summary of counts + on-disk slice paths.  The full
        ``cmr``/``sites`` payload is **not** returned.
    """
    from baleen.eventalign._aggregation import (
        _benjamini_hochberg,
        aggregate_contig,
        write_site_tsv_rows,
    )
    from baleen.eventalign._hierarchical import compute_sequential_modification_probabilities
    from baleen.eventalign._read_bam import flush_contig_to_bam

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
        cuda_device=cuda_device,
        allowed_native_reads_path=allowed_native_reads_path,
        allowed_ivt_reads_path=allowed_ivt_reads_path,
    )

    # Stage 2: HMM smoothing
    hmm_t0 = time.perf_counter()
    cmr = compute_sequential_modification_probabilities(
        contig_result, run_hmm=run_hmm, hmm_params=hmm_params,
        legacy_scoring=legacy_scoring,
        show_progress=show_progress,
    )
    hmm_elapsed = time.perf_counter() - hmm_t0
    logger.info("  [Contig %s] HMM done (%s)", contig_name, _fmt_elapsed(hmm_elapsed))

    # Stage 3: Site-level aggregation with per-transcript FDR
    agg_t0 = time.perf_counter()
    sites = aggregate_contig(cmr, mod_threshold=mod_threshold)
    if sites:
        pvalues = np.array([s.pvalue for s in sites], dtype=np.float64)
        padj = _benjamini_hochberg(pvalues)
        for site, adj in zip(sites, padj):
            site.padj = float(adj)
    agg_elapsed = time.perf_counter() - agg_t0
    logger.info(
        "  [Contig %s] Aggregation done: %d sites (%s)",
        contig_name, len(sites), _fmt_elapsed(agg_elapsed),
    )

    # Sanitize once — SAM permits unsafe characters (``/``, ``..``, ``\``)
    # in contig names; we use the sanitized stem for every on-disk artifact.
    safe_name = _sanitize_contig_filename(contig_name)

    # Optionally save intermediate ContigResult
    if keep_intermediate and intermediate_dir is not None:
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = intermediate_dir / f"{safe_name}.pkl"
        with pkl_path.open("wb") as fh:
            pickle.dump(contig_result, fh)
        logger.info("  Saved intermediate: %s", pkl_path)

    # Stage 4: Streaming flush of per-contig outputs
    per_contig_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = per_contig_dir / f"{safe_name}.tsv"
    tsv_tmp = tsv_path.with_suffix(tsv_path.suffix + ".tmp")
    success = False
    try:
        with tsv_tmp.open("w", newline="") as f:
            write_site_tsv_rows(f, sites)
        os.replace(tsv_tmp, tsv_path)
        success = True
    finally:
        if not success:
            tsv_tmp.unlink(missing_ok=True)

    bam_path: Optional[Path] = None
    if write_bam and bam_header_dict is not None:
        import pysam
        bam_header = pysam.AlignmentHeader.from_dict(bam_header_dict)
        bam_path = per_contig_dir / f"{safe_name}.bam"
        flush_contig_to_bam(cmr, native_bam, ivt_bam, bam_header, bam_path)

    n_significant = sum(1 for s in sites if s.padj < 0.05)
    n_positions = len(cmr.position_stats)
    n_sites = len(sites)

    # Drop the heavy payload before returning to the parent process.
    del cmr
    del sites
    del contig_result

    return ContigSummary(
        contig_name=contig_name,
        n_sites=n_sites,
        n_positions=n_positions,
        n_significant=n_significant,
        tsv_path=tsv_path,
        bam_path=bam_path,
    )


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
    depth_mode: Literal["mean_coverage", "read_count"] = "read_count",
    use_cuda: Optional[bool] = None,
    cuda_devices: Optional[list[int]] = None,
    padding: int = 1,
    output_dir: Optional[PathLike] = None,
    cleanup_temp: bool = True,
    rna: bool = True,
    kmer_model: Optional[str] = None,
    extra_f5c_args: Optional[list[str]] = None,
    min_mapq: int = 20,
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
    logger.info("  min_depth=%d  depth_mode=%s  use_cuda=%s  rna=%s  padding=%d  threads=%d",
                min_depth, depth_mode, use_cuda, rna, padding, threads)
    logger.info("  min_mapq=%d  primary_only=%s  cuda_streams=%d",
                min_mapq, primary_only, num_cuda_streams)
    logger.info("  subsample=%s  subsample_n=%d  gpu_memory_limit=%s",
                subsample, subsample_n, gpu_memory_limit)
    logger.info("  cleanup_temp=%s  kmer_model=%s  extra_f5c_args=%s",
                cleanup_temp, kmer_model, extra_f5c_args)
    logger.info("  DTW backend:  %s  (GPU=%s)",
                _dtw.backend(), _dtw.CUDA_AVAILABLE)
    logger.info("=" * 60)

    # Validate threads parameter
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")

    # Resolve cuda_devices from legacy use_cuda if needed
    if cuda_devices is None and use_cuda is not None:
        if use_cuda is True:
            cuda_devices = None  # auto-detect all GPUs
        elif use_cuda is False:
            cuda_devices = []  # CPU mode
    # Derive use_cuda bool for backward compat in internal code
    if cuda_devices is not None:
        use_cuda = len(cuda_devices) > 0 if cuda_devices else False
    # else: use_cuda stays None (auto-detect)

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
        _validated=True,
    )
    logger.info("  Computing IVT BAM contig stats...")
    ivt_stats = _bam.get_contig_stats(
        ivt_bam,
        min_mapq=min_mapq,
        primary_only=primary_only,
        _validated=True,
    )
    logger.info("[Step 3/6] BAM stats complete: %d native contigs, %d IVT contigs (%s)",
                len(native_stats), len(ivt_stats), _fmt_elapsed(time.perf_counter() - step_t0))

    # ---- Step 4: Contig filtering ----
    logger.info("[Step 4/6] Filtering contigs (min_depth=%d, depth_mode=%s)...",
                min_depth, depth_mode)
    passed_contigs, filter_results = _bam.filter_contigs(
        native_stats,
        ivt_stats,
        min_depth=float(min_depth),
        depth_mode=depth_mode,
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

    gpu_mems = _get_gpu_memory(cuda_devices) if gpu_memory_limit is None else [gpu_memory_limit]
    gpu_workers, device_for_worker = _gpu_concurrent_workers(threads, gpu_mems, cuda_devices)

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
                        gpu_memory_bytes=gpu_mems[0] if gpu_mems else 8 * 1024 ** 3,
                        num_workers=gpu_workers,
                        show_progress=False,
                        cuda_device=device_for_worker[idx - 1] if device_for_worker else 0,
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
                    gpu_memory_bytes=gpu_mems[0] if gpu_mems else 8 * 1024 ** 3,
                    cuda_device=device_for_worker[0] if device_for_worker else 0,
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
    depth_mode: Literal["mean_coverage", "read_count"] = "read_count",
    use_cuda: Optional[bool] = None,
    cuda_devices: Optional[list[int]] = None,
    padding: int = 1,
    output_dir: Optional[PathLike] = None,
    cleanup_temp: bool = True,
    rna: bool = True,
    kmer_model: Optional[str] = None,
    extra_f5c_args: Optional[list[str]] = None,
    min_mapq: int = 20,
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
    write_bam: bool = True,
    resume: bool = False,
    read_intersection: bool = True,
) -> tuple[dict[str, Any], PipelineMetadata]:
    """Memory-efficient streaming pipeline: DTW → HMM → aggregation per contig.

    Each contig is processed end-to-end in a worker, which writes its
    own ``<contig>.tsv`` and ``<contig>.bam`` slices to
    ``<output_dir>/per_contig/`` and drops the heavy ``cmr`` before
    returning.  The main process then merges slices into final outputs.

    Peak memory is bounded by ``O(single_contig + N_workers)`` rather than
    growing with the total number of contigs.

    Parameters
    ----------
    output_dir
        Required.  Final outputs are written to ``<output_dir>/site_results.tsv``
        and (when *write_bam*) ``<output_dir>/read_results.bam``.  If ``None``,
        a temporary directory is used and cleaned up on exit.
    target_contigs
        If given, only process these contig(s).  Contigs not passing depth
        filters are silently skipped.
    keep_intermediate
        Save per-contig ``ContigResult`` pickles under
        ``<output_dir>/intermediate/`` and keep ``<output_dir>/per_contig/``
        on disk after merging.
    write_bam
        Whether to produce a final ``read_results.bam`` (set to False to
        skip mod-BAM construction entirely).
    run_hmm
        Whether to run HMM smoothing (V3).
    hmm_params
        Optional trained HMM parameters.

    Returns
    -------
    tuple[dict[str, Any], PipelineMetadata]
        ``(output_paths, metadata)`` where ``output_paths`` is a dict with
        keys ``site_tsv`` (Path), ``read_bam`` (Path or None),
        ``per_contig_dir`` (Path or None), ``n_total_sites`` (int), and
        ``n_significant`` (int).
    """
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
    logger.info("  min_depth=%d  depth_mode=%s  use_cuda=%s  rna=%s  padding=%d  threads=%d",
                min_depth, depth_mode, use_cuda, rna, padding, threads)
    logger.info("  min_mapq=%d  primary_only=%s  cuda_streams=%d",
                min_mapq, primary_only, num_cuda_streams)
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

    # Resolve cuda_devices from legacy use_cuda if needed
    if cuda_devices is None and use_cuda is not None:
        if use_cuda is True:
            cuda_devices = None  # auto-detect all GPUs
        elif use_cuda is False:
            cuda_devices = []  # CPU mode
    if cuda_devices is not None:
        use_cuda = len(cuda_devices) > 0 if cuda_devices else False

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

    # ---- Step 2.5: Read-ID intersection (BAM ∩ FASTQ ∩ BLOW5) ----
    # f5c eventalign silently drops BAM reads whose UUIDs are not in
    # the BLOW5 signal file; computing the intersection up-front keeps
    # contig stats, ``min_depth`` filtering, and subsampling all in
    # sync with the read set that will actually produce signals.
    #
    # The intersection sets are written to disk and passed by path
    # (not by ``set[str]``) across the worker spawn boundary.
    allowed_native_reads_path: Optional[Path] = None
    allowed_ivt_reads_path: Optional[Path] = None
    allowed_native: Optional[set[str]] = None
    allowed_ivt: Optional[set[str]] = None
    if read_intersection:
        from baleen.eventalign._read_ids import (
            compute_condition_intersection,
            write_read_ids,
        )

        logger.info("[Step 2.5/5] Computing read-id intersection (BAM ∩ FASTQ ∩ BLOW5)...")
        step_t0 = time.perf_counter()
        allowed_native = compute_condition_intersection(
            bam=native_bam,
            fastq=native_fastq,
            blow5=native_blow5,
            primary_only=primary_only,
            min_mapq=min_mapq,
            label="native",
        )
        allowed_ivt = compute_condition_intersection(
            bam=ivt_bam,
            fastq=ivt_fastq,
            blow5=ivt_blow5,
            primary_only=primary_only,
            min_mapq=min_mapq,
            label="ivt",
        )
        logger.info(
            "[Step 2.5/5] Intersection complete: native=%d, ivt=%d reads (%s)",
            len(allowed_native), len(allowed_ivt),
            _fmt_elapsed(time.perf_counter() - step_t0),
        )

        # Persist intersection sets to disk so workers can lazy-load via
        # path instead of receiving multi-MB ``set[str]`` payloads across
        # the spawn pickle boundary (2924 contigs × millions of reads
        # would otherwise dominate dispatch cost).
        _intersection_dir = Path(
            tempfile.mkdtemp(prefix="baleen-intersection-")
        )
        allowed_native_reads_path = write_read_ids(
            allowed_native, _intersection_dir / "allowed_native_reads.txt"
        )
        allowed_ivt_reads_path = write_read_ids(
            allowed_ivt, _intersection_dir / "allowed_ivt_reads.txt"
        )

    # ---- Step 3: BAM validation & contig stats ----
    logger.info("[Step 3/5] Validating BAMs and computing contig statistics...")
    step_t0 = time.perf_counter()
    _bam.validate_bam(native_bam)
    _bam.validate_bam(ivt_bam)
    native_stats = _bam.get_contig_stats(
        native_bam, min_mapq=min_mapq, primary_only=primary_only,
        allowed_reads=allowed_native, _validated=True,
    )
    ivt_stats = _bam.get_contig_stats(
        ivt_bam, min_mapq=min_mapq, primary_only=primary_only,
        allowed_reads=allowed_ivt, _validated=True,
    )
    logger.info("[Step 3/5] BAM stats complete: %d native contigs, %d IVT contigs (%s)",
                len(native_stats), len(ivt_stats), _fmt_elapsed(time.perf_counter() - step_t0))

    # ---- Step 4: Contig filtering ----
    logger.info("[Step 4/5] Filtering contigs (min_depth=%d, depth_mode=%s)...",
                min_depth, depth_mode)
    step_t0 = time.perf_counter()
    passed_contigs, filter_results = _bam.filter_contigs(
        native_stats, ivt_stats, min_depth=float(min_depth), depth_mode=depth_mode,
    )

    # Apply target contig filter
    if target_contigs is not None:
        target_set = set(target_contigs)
        skipped_targets = target_set - set(passed_contigs)
        if skipped_targets:
            logger.warning("  Target contigs not passing filters: %s", sorted(skipped_targets))
        passed_contigs = [c for c in passed_contigs if c in target_set]

    logger.info("[Step 4/5] %d contigs to process (%s)",
                len(passed_contigs), _fmt_elapsed(time.perf_counter() - step_t0))

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

    # Resolve output_dir (use a tempdir if not given, cleaned up on exit).
    output_dir_temp: Optional[Path] = None
    if output_dir is None:
        output_dir_temp = Path(tempfile.mkdtemp(prefix="baleen-output-"))
        output_dir_path = output_dir_temp
        logger.warning(
            "  output_dir not specified; using temporary %s (will be removed)",
            output_dir_path,
        )
    else:
        output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    per_contig_dir = output_dir_path / "per_contig"
    # Detect prior state BEFORE we create the directory.  Missing dir or
    # missing fingerprint both mean "no usable prior run" — even with
    # --resume, treat it as a fresh run (nothing to skip, nothing to
    # validate against).  This matters for legacy interrupted runs that
    # pre-date the fingerprint file.
    fp_path_pre = per_contig_dir / _RESUME_PARAMS_FILENAME
    has_prior_fingerprint = per_contig_dir.exists() and fp_path_pre.exists()
    per_contig_dir.mkdir(parents=True, exist_ok=True)

    site_tsv_path = output_dir_path / "site_results.tsv"
    final_bam_path: Optional[Path] = (
        output_dir_path / "read_results.bam" if write_bam else None
    )

    # ---- Resume: scan & validate before dispatching workers ----
    fingerprint = _compute_resume_fingerprint(
        native_bam=native_bam,
        native_fastq=native_fastq,
        native_blow5=native_blow5,
        ivt_bam=ivt_bam,
        ivt_fastq=ivt_fastq,
        ivt_blow5=ivt_blow5,
        ref_fasta=ref_fasta,
        min_depth=min_depth,
        depth_mode=depth_mode,
        padding=padding,
        min_mapq=min_mapq,
        primary_only=primary_only,
        subsample=subsample,
        subsample_n=subsample_n,
        legacy_scoring=legacy_scoring,
        mod_threshold=mod_threshold,
        write_bam=write_bam,
        run_hmm=run_hmm,
        target_contigs=target_contigs,
        read_intersection=read_intersection,
    )
    resumed_summaries: list[ContigSummary] = []
    if resume:
        if has_prior_fingerprint:
            _validate_resume_compatibility(per_contig_dir, fingerprint)
            resumed_map = _scan_completed_contigs(
                per_contig_dir, passed_contigs, write_bam,
            )
            resumed_summaries = list(resumed_map.values())
            if resumed_summaries:
                logger.info(
                    "[Resume] Skipping %d/%d contigs already on disk under %s",
                    len(resumed_summaries), len(passed_contigs), per_contig_dir,
                )
            passed_contigs = [c for c in passed_contigs if c not in resumed_map]
        else:
            logger.info(
                "[Resume] No %s under %s; treating as a fresh run.",
                _RESUME_PARAMS_FILENAME, per_contig_dir,
            )
    _write_resume_fingerprint(per_contig_dir, fingerprint)

    if not passed_contigs and not resumed_summaries:
        logger.warning("[Step 5/5] No contigs to process; returning empty results.")
        # Still write an empty TSV (header only) for consistency, unless we
        # are going to delete the parent tempdir on the way out.
        from baleen.eventalign._aggregation import write_site_tsv
        write_site_tsv([], site_tsv_path)
        # Clean up empty per_contig dir we just created.
        shutil.rmtree(per_contig_dir, ignore_errors=True)
        returned_site_tsv: Optional[Path] = site_tsv_path
        if output_dir_temp is not None:
            shutil.rmtree(output_dir_temp, ignore_errors=True)
            returned_site_tsv = None
        elapsed = _fmt_elapsed(time.perf_counter() - pipeline_t0)
        logger.info("Pipeline finished (no results) in %s", elapsed)
        return (
            {
                "site_tsv": returned_site_tsv,
                "read_bam": None,
                "per_contig_dir": None,
                "n_total_sites": 0,
                "n_significant": 0,
            },
            metadata,
        )

    # ---- Step 5: Per-contig streaming (DTW → HMM → aggregation → flush) ----
    logger.info("[Step 5/5] Processing %d contigs (streaming flush: DTW → HMM → aggregation → disk)...",
                len(passed_contigs))
    step5_t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="baleen-streaming-"))

    intermediate_dir: Optional[Path] = None
    if keep_intermediate:
        intermediate_dir = output_dir_path / "intermediate"

    # Build BAM header once (heavy I/O).  pysam.AlignmentHeader is NOT
    # picklable (cdef-class with non-trivial __cinit__), so we ship its
    # ``to_dict()`` representation across the spawn boundary and rebuild
    # in each worker via ``pysam.AlignmentHeader.from_dict``.
    bam_header_dict: Optional[dict] = None
    if write_bam:
        # Ensure full input BAMs are indexed for per-contig fetch().
        from baleen.eventalign._read_bam import (
            _build_header_from_bam,
            _ensure_bam_indexed,
            merge_contig_bams,
        )
        _ensure_bam_indexed(native_bam)
        _ensure_bam_indexed(ivt_bam)
        bam_header_dict = _build_header_from_bam(native_bam, ref_fasta).to_dict()

    from baleen.eventalign._aggregation import merge_contig_tsvs

    gpu_mems = _get_gpu_memory(cuda_devices) if gpu_memory_limit is None else [gpu_memory_limit]
    gpu_workers, device_for_worker = _gpu_concurrent_workers(threads, gpu_mems, cuda_devices)

    summaries: list[ContigSummary] = list(resumed_summaries)
    failed: list[str] = []

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
            padding=padding,
            rna=rna,
            kmer_model=kmer_model,
            extra_f5c_args=extra_f5c_args,
            min_mapq=min_mapq,
            primary_only=primary_only,
            cleanup_temp=cleanup_temp,
            num_cuda_streams=num_cuda_streams,
            per_contig_dir=per_contig_dir,
            bam_header_dict=bam_header_dict,
            write_bam=write_bam,
            run_hmm=run_hmm,
            hmm_params=hmm_params,
            keep_intermediate=keep_intermediate,
            intermediate_dir=intermediate_dir,
            subsample=subsample,
            subsample_n=subsample_n,
            gpu_memory_bytes=gpu_mems[0] if gpu_mems else 8 * 1024 ** 3,
            legacy_scoring=legacy_scoring,
            num_workers=gpu_workers,
            mod_threshold=mod_threshold,
            show_progress=(threads <= 1),
            allowed_native_reads_path=allowed_native_reads_path,
            allowed_ivt_reads_path=allowed_ivt_reads_path,
        )

        if not passed_contigs:
            logger.info("  All contigs already on disk — no workers dispatched.")
        elif threads > 1:
            logger.info("  Using %d parallel workers (spawn context)", threads)
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=threads, mp_context=ctx) as executor:
                futures = {
                    executor.submit(
                        _process_contig_streaming,
                        contig=contig,
                        contig_idx=idx,
                        total_contigs=len(passed_contigs),
                        cuda_device=device_for_worker[(idx - 1) % len(device_for_worker)] if device_for_worker else 0,
                        **worker_kwargs,
                    ): contig
                    for idx, contig in enumerate(passed_contigs, 1)
                }
                with tqdm(
                    total=len(passed_contigs),
                    desc="Pipeline",
                    unit="contig",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} contigs [{elapsed}<{remaining}] {postfix}",
                ) as pbar:
                    for future in as_completed(futures):
                        contig = futures[future]
                        try:
                            summary = future.result()
                        except Exception:
                            logger.exception("Worker failed for contig %s", contig)
                            failed.append(contig)
                            pbar.update(1)
                            continue
                        summaries.append(summary)
                        pbar.set_postfix_str(
                            f"{summary.contig_name} ({summary.n_sites} sites)"
                        )
                        pbar.update(1)
                if failed:
                    logger.error("%d contig(s) failed: %s", len(failed), ", ".join(failed))
        else:
            for contig_idx, contig in enumerate(passed_contigs, 1):
                try:
                    summary = _process_contig_streaming(
                        contig=contig,
                        contig_idx=contig_idx,
                        total_contigs=len(passed_contigs),
                        cuda_device=device_for_worker[0] if device_for_worker else 0,
                        **worker_kwargs,
                    )
                except Exception:
                    logger.exception("Worker failed for contig %s", contig)
                    failed.append(contig)
                    continue
                summaries.append(summary)
            if failed:
                logger.error("%d contig(s) failed: %s", len(failed), ", ".join(failed))
    finally:
        if cleanup_temp and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        # Clean up intersection set files (only present when read_intersection=True).
        if allowed_native_reads_path is not None:
            inter_parent = Path(allowed_native_reads_path).parent
            if inter_parent.name.startswith("baleen-intersection-") and inter_parent.exists():
                shutil.rmtree(inter_parent, ignore_errors=True)

    # ---- Final merge step ----
    sorted_summaries = sorted(summaries, key=lambda s: s.contig_name)

    merge_contig_tsvs(
        [s.tsv_path for s in sorted_summaries],
        site_tsv_path,
    )

    if write_bam:
        bam_inputs = [s.bam_path for s in sorted_summaries if s.bam_path is not None]
        if bam_inputs:
            merge_contig_bams(bam_inputs, final_bam_path, threads=max(threads, 1))
        else:
            final_bam_path = None

    n_total_sites = sum(s.n_sites for s in sorted_summaries)
    n_significant = sum(s.n_significant for s in sorted_summaries)
    total_positions = sum(s.n_positions for s in sorted_summaries)

    if not keep_intermediate:
        shutil.rmtree(per_contig_dir, ignore_errors=True)
        per_contig_dir_out: Optional[Path] = None
    else:
        per_contig_dir_out = per_contig_dir

    step5_elapsed = _fmt_elapsed(time.perf_counter() - step5_t0)
    logger.info("[Step 5/5] Streaming complete (%s)", step5_elapsed)
    pipeline_elapsed = _fmt_elapsed(time.perf_counter() - pipeline_t0)
    logger.info("=" * 60)
    logger.info(
        "Streaming pipeline complete: %d contigs, %d positions, %d sites, %s",
        len(sorted_summaries), total_positions, n_total_sites, pipeline_elapsed,
    )
    logger.info("=" * 60)

    output_paths: dict[str, Any] = {
        "site_tsv": site_tsv_path,
        "read_bam": final_bam_path,
        "per_contig_dir": per_contig_dir_out,
        "n_total_sites": n_total_sites,
        "n_significant": n_significant,
    }

    if output_dir_temp is not None:
        # Caller did not pass an output_dir — clean up everything we wrote
        # and null the paths in the returned dict so callers do not chase
        # references to deleted files.  If the caller needs persistence,
        # they should pass output_dir.
        shutil.rmtree(output_dir_temp, ignore_errors=True)
        output_paths["site_tsv"] = None
        output_paths["read_bam"] = None
        output_paths["per_contig_dir"] = None

    return output_paths, metadata
