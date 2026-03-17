from __future__ import annotations

from dataclasses import dataclass
import logging
import pickle
from pathlib import Path
import shutil
import tempfile
from typing import Optional, Protocol, TypedDict, Union, cast

import numpy as np
from numpy.typing import NDArray

from baleen import _cuda_dtw
from baleen.eventalign import _bam
from baleen.eventalign import _f5c
from baleen.eventalign import _signal

logger = logging.getLogger(__name__)

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
    n_contigs_total: int
    n_contigs_passed_filter: int
    n_contigs_skipped: int
    filter_results: list[_bam.ContigFilterResult]


class _SerializedPayload(TypedDict):
    results: dict[str, ContigResult]
    metadata: PipelineMetadata


_dtw_distance = cast(_DtwDistanceFn, _cuda_dtw.dtw_distance)


def _compute_pairwise_distances(
    signals: list[NDArray[np.float32]],
    *,
    use_cuda: Optional[bool],
    use_open_start: bool,
    use_open_end: bool,
) -> NDArray[np.float64]:
    n = len(signals)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            distance = _dtw_distance(
                signals[i],
                signals[j],
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
    output_dir: Optional[PathLike] = None,
    cleanup_temp: bool = True,
    rna: bool = True,
    kmer_model: Optional[str] = None,
    extra_f5c_args: Optional[list[str]] = None,
    min_mapq: int = 0,
    primary_only: bool = True,
) -> tuple[dict[str, ContigResult], PipelineMetadata]:
    native_bam = Path(native_bam)
    native_fastq = Path(native_fastq)
    native_blow5 = Path(native_blow5)
    ivt_bam = Path(ivt_bam)
    ivt_fastq = Path(ivt_fastq)
    ivt_blow5 = Path(ivt_blow5)
    ref_fasta = Path(ref_fasta)

    f5c_version = _f5c.check_f5c()
    logger.info("Using f5c version %s", f5c_version)

    _f5c.index_fastq_blow5(native_fastq, native_blow5)
    _f5c.index_fastq_blow5(ivt_fastq, ivt_blow5)
    _f5c.index_blow5(native_blow5)
    _f5c.index_blow5(ivt_blow5)

    _bam.validate_bam(native_bam)
    _bam.validate_bam(ivt_bam)

    native_stats = _bam.get_contig_stats(
        native_bam,
        min_mapq=min_mapq,
        primary_only=primary_only,
    )
    ivt_stats = _bam.get_contig_stats(
        ivt_bam,
        min_mapq=min_mapq,
        primary_only=primary_only,
    )

    passed_contigs, filter_results = _bam.filter_contigs(
        native_stats,
        ivt_stats,
        min_depth=float(min_depth),
    )

    metadata = PipelineMetadata(
        f5c_version=f5c_version,
        min_depth=min_depth,
        use_cuda=use_cuda,
        n_contigs_total=len(filter_results),
        n_contigs_passed_filter=len(passed_contigs),
        n_contigs_skipped=len(filter_results) - len(passed_contigs),
        filter_results=filter_results,
    )

    results: dict[str, ContigResult] = {}

    if not passed_contigs:
        logger.warning("No contigs passed filtering; returning empty results.")
        if output_dir is not None:
            _ = save_results(results, metadata, Path(output_dir) / "pipeline_results.pkl")
        return results, metadata

    tmp_root = Path(tempfile.mkdtemp(prefix="baleen-eventalign-"))
    logger.debug("Created temporary pipeline directory: %s", tmp_root)

    try:
        for contig in passed_contigs:
            contig_tmp = tmp_root / contig
            contig_tmp.mkdir(parents=True, exist_ok=True)

            native_contig_bam = _bam.split_bam_contig(
                native_bam,
                contig,
                contig_tmp / "native",
                primary_only=primary_only,
                min_mapq=min_mapq,
            )
            ivt_contig_bam = _bam.split_bam_contig(
                ivt_bam,
                contig,
                contig_tmp / "ivt",
                primary_only=primary_only,
                min_mapq=min_mapq,
            )

            native_tsv = contig_tmp / "native.eventalign.tsv"
            ivt_tsv = contig_tmp / "ivt.eventalign.tsv"

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

            native_by_pos = _signal.group_signals_by_position(native_tsv)
            ivt_by_pos = _signal.group_signals_by_position(ivt_tsv)
            common_positions = _signal.get_common_positions(native_by_pos, ivt_by_pos)

            position_results: dict[int, PositionResult] = {}
            for pos in common_positions:
                native_pos = native_by_pos[pos]
                ivt_pos = ivt_by_pos[pos]

                native_read_names, native_signals = _signal.extract_signals_for_dtw(native_pos)
                ivt_read_names, ivt_signals = _signal.extract_signals_for_dtw(ivt_pos)

                if not native_signals or not ivt_signals:
                    logger.debug(
                        "Skipping contig=%s pos=%s due to empty signals (native=%d, ivt=%d)",
                        contig,
                        pos,
                        len(native_signals),
                        len(ivt_signals),
                    )
                    continue

                all_signals = native_signals + ivt_signals
                matrix = _compute_pairwise_distances(
                    all_signals,
                    use_cuda=use_cuda,
                    use_open_start=use_open_start,
                    use_open_end=use_open_end,
                )

                position_results[pos] = PositionResult(
                    position=pos,
                    reference_kmer=native_pos.reference_kmer,
                    n_native_reads=len(native_read_names),
                    n_ivt_reads=len(ivt_read_names),
                    native_read_names=native_read_names,
                    ivt_read_names=ivt_read_names,
                    distance_matrix=matrix,
                )

            native_depth = native_stats[contig].mean_depth
            ivt_depth = ivt_stats[contig].mean_depth
            results[contig] = ContigResult(
                contig=contig,
                native_depth=native_depth,
                ivt_depth=ivt_depth,
                positions=position_results,
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
    finally:
        if cleanup_temp and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    if output_dir is not None:
        _ = save_results(results, metadata, Path(output_dir) / "pipeline_results.pkl")

    return results, metadata
