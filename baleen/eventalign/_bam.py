"""BAM utilities for eventalign preprocessing.

This module provides helpers for validating indexed BAM files, collecting
per-contig mapping/depth statistics, filtering contigs by depth criteria, and
splitting BAMs into per-contig temporary BAM files.
"""

from __future__ import annotations

from collections.abc import Generator
import dataclasses
from enum import Enum
import importlib
import logging
from pathlib import Path
import tempfile
from typing import Optional, Protocol, Union, cast

import numpy as np


class _AlignedSegmentProtocol(Protocol):
    is_unmapped: bool
    is_secondary: bool
    is_supplementary: bool
    mapping_quality: int


class _IndexStatProtocol(Protocol):
    contig: str
    mapped: int


class _AlignmentHeaderProtocol(Protocol):
    def to_dict(self) -> dict[str, object]: ...


class _AlignmentFileProtocol(Protocol):
    references: tuple[str, ...]
    nreferences: int
    mapped: int
    unmapped: int
    header: _AlignmentHeaderProtocol

    def __enter__(self) -> _AlignmentFileProtocol: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...
    def has_index(self) -> bool: ...
    def get_index_statistics(self) -> list[_IndexStatProtocol]: ...
    def get_reference_length(self, contig: str) -> int: ...
    def fetch(self, contig: str) -> Generator[_AlignedSegmentProtocol, None, None]: ...
    def count_coverage(
        self,
        contig: str,
        start: int,
        end: int,
        quality_threshold: int = 0,
        read_callback: object = "all",
    ) -> tuple[object, object, object, object]: ...
    def write(self, read: _AlignedSegmentProtocol) -> None: ...


class _PysamModule(Protocol):
    def AlignmentFile(
        self,
        filename: str,
        mode: str,
        header: dict[str, object] | None = None,
    ) -> _AlignmentFileProtocol: ...

    def sort(self, *args: str) -> None: ...
    def index(self, filename: str) -> None: ...


pysam = cast(_PysamModule, cast(object, importlib.import_module("pysam")))

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class FilterReason(Enum):
    """Reason describing why a contig passed or failed filtering."""

    PASSED = "passed"
    LOW_DEPTH_NATIVE = "low_depth_native"
    LOW_DEPTH_IVT = "low_depth_ivt"
    LOW_DEPTH_BOTH = "low_depth_both"
    MISSING_IN_NATIVE = "missing_in_native"
    MISSING_IN_IVT = "missing_in_ivt"
    NO_MAPPED_READS_NATIVE = "no_mapped_reads_native"
    NO_MAPPED_READS_IVT = "no_mapped_reads_ivt"


@dataclasses.dataclass
class ContigStats:
    """Per-contig mapping statistics.

    Parameters
    ----------
    contig : str
        Contig/transcript identifier.
    mapped_reads : int
        Number of mapped reads contributing to the contig.
    mean_depth : float
        Mean base-level depth over the full contig length.
    """

    contig: str
    mapped_reads: int
    mean_depth: float


@dataclasses.dataclass
class ContigFilterResult:
    """Outcome of filtering a contig between native and IVT datasets.

    Parameters
    ----------
    contig : str
        Contig/transcript identifier.
    passed : bool
        Whether this contig passed all filter criteria.
    reason : FilterReason
        Detailed reason for pass/fail status.
    native_stats : ContigStats, optional
        Native BAM contig statistics when available.
    ivt_stats : ContigStats, optional
        IVT BAM contig statistics when available.
    """

    contig: str
    passed: bool
    reason: FilterReason
    native_stats: Optional[ContigStats] = None
    ivt_stats: Optional[ContigStats] = None


def validate_bam(bam_path: PathLike) -> None:
    """Validate that a BAM exists and is indexed.

    Parameters
    ----------
    bam_path : str or pathlib.Path
        Path to BAM file.

    Raises
    ------
    FileNotFoundError
        If the BAM file does not exist.
    ValueError
        If the BAM is not indexed.
    """
    bam_path = Path(bam_path)
    if not bam_path.exists():
        raise FileNotFoundError(f"BAM file not found: {bam_path}")

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        has_index = bam.has_index()
        if not has_index:
            bai_candidates = [
                Path(f"{bam_path}.bai"),
                bam_path.with_suffix(".bai"),
            ]
            has_index = any(candidate.exists() for candidate in bai_candidates)

        if not has_index:
            raise ValueError(f"BAM file is not indexed: {bam_path}")

        logger.info(
            "Validated BAM %s (references=%d, mapped=%d, unmapped=%d)",
            bam_path,
            bam.nreferences,
            bam.mapped,
            bam.unmapped,
        )


def _read_passes_filters(
    read: _AlignedSegmentProtocol,
    *,
    primary_only: bool,
    min_mapq: int,
) -> bool:
    """Return True if read satisfies primary/MAPQ filters."""
    if read.is_unmapped:
        return False
    if primary_only and (read.is_secondary or read.is_supplementary):
        return False
    if read.mapping_quality < min_mapq:
        return False
    return True


def get_contig_stats(
    bam_path: PathLike,
    *,
    min_mapq: int = 0,
    primary_only: bool = True,
) -> dict[str, ContigStats]:
    """Collect mapped-read and mean-depth statistics per contig.

    Parameters
    ----------
    bam_path : str or pathlib.Path
        Path to input BAM file.
    min_mapq : int, optional
        Minimum read mapping quality to include.
    primary_only : bool, optional
        If ``True``, include only primary alignments.

    Returns
    -------
    dict of str to ContigStats
        Mapping of contig name to statistics.
        Contigs with zero mapped reads after filtering are omitted.
    """
    bam_path = Path(bam_path)
    validate_bam(bam_path)

    stats: dict[str, ContigStats] = {}

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        index_stats = {item.contig: item for item in bam.get_index_statistics()}

        for contig in bam.references:
            contig_len = bam.get_reference_length(contig)
            if contig_len <= 0:
                continue

            idx_entry = index_stats.get(contig)
            total_mapped = idx_entry.mapped if idx_entry is not None else 0
            if total_mapped <= 0:
                continue

            if primary_only or min_mapq > 0:
                mapped_reads = 0
                for read in bam.fetch(contig=contig):
                    if _read_passes_filters(read, primary_only=primary_only, min_mapq=min_mapq):
                        mapped_reads += 1
            else:
                mapped_reads = total_mapped

            if mapped_reads <= 0:
                continue

            def _coverage_read_callback(read: _AlignedSegmentProtocol) -> bool:
                return _read_passes_filters(read, primary_only=primary_only, min_mapq=min_mapq)

            cov_a, cov_c, cov_g, cov_t = bam.count_coverage(
                contig=contig,
                start=0,
                end=contig_len,
                quality_threshold=0,
                read_callback=_coverage_read_callback,
            )
            depth = (
                np.asarray(cov_a, dtype=np.int64)
                + np.asarray(cov_c, dtype=np.int64)
                + np.asarray(cov_g, dtype=np.int64)
                + np.asarray(cov_t, dtype=np.int64)
            )
            mean_depth = float(np.mean(depth))

            stats[contig] = ContigStats(
                contig=contig,
                mapped_reads=mapped_reads,
                mean_depth=mean_depth,
            )

    logger.info("Computed contig stats for %d contigs in %s", len(stats), bam_path)
    return stats


def filter_contigs(
    native_stats: dict[str, ContigStats],
    ivt_stats: dict[str, ContigStats],
    *,
    min_depth: float = 15.0,
) -> tuple[list[str], list[ContigFilterResult]]:
    """Filter contigs using presence, mapped-read, and depth constraints.

    Parameters
    ----------
    native_stats : dict of str to ContigStats
        Per-contig stats computed from native BAM.
    ivt_stats : dict of str to ContigStats
        Per-contig stats computed from IVT BAM.
    min_depth : float, optional
        Minimum mean depth required in each dataset.

    Returns
    -------
    tuple[list[str], list[ContigFilterResult]]
        Passed contig names (sorted) and filter results for all contigs.
    """
    all_contigs = sorted(set(native_stats) | set(ivt_stats))
    results: list[ContigFilterResult] = []
    passed: list[str] = []

    for contig in all_contigs:
        native = native_stats.get(contig)
        ivt = ivt_stats.get(contig)

        if native is None:
            results.append(
                ContigFilterResult(
                    contig=contig,
                    passed=False,
                    reason=FilterReason.MISSING_IN_NATIVE,
                    native_stats=None,
                    ivt_stats=ivt,
                )
            )
            continue

        if ivt is None:
            results.append(
                ContigFilterResult(
                    contig=contig,
                    passed=False,
                    reason=FilterReason.MISSING_IN_IVT,
                    native_stats=native,
                    ivt_stats=None,
                )
            )
            continue

        if native.mapped_reads <= 0:
            results.append(
                ContigFilterResult(
                    contig=contig,
                    passed=False,
                    reason=FilterReason.NO_MAPPED_READS_NATIVE,
                    native_stats=native,
                    ivt_stats=ivt,
                )
            )
            continue

        if ivt.mapped_reads <= 0:
            results.append(
                ContigFilterResult(
                    contig=contig,
                    passed=False,
                    reason=FilterReason.NO_MAPPED_READS_IVT,
                    native_stats=native,
                    ivt_stats=ivt,
                )
            )
            continue

        native_low = native.mean_depth < min_depth
        ivt_low = ivt.mean_depth < min_depth
        if native_low and ivt_low:
            reason = FilterReason.LOW_DEPTH_BOTH
        elif native_low:
            reason = FilterReason.LOW_DEPTH_NATIVE
        elif ivt_low:
            reason = FilterReason.LOW_DEPTH_IVT
        else:
            reason = FilterReason.PASSED

        passed_flag = reason == FilterReason.PASSED
        if passed_flag:
            passed.append(contig)

        results.append(
            ContigFilterResult(
                contig=contig,
                passed=passed_flag,
                reason=reason,
                native_stats=native,
                ivt_stats=ivt,
            )
        )

    passed.sort()
    logger.info(
        "%d/%d contigs passed filtering (min_depth=%s)",
        len(passed),
        len(all_contigs),
        min_depth,
    )
    return passed, results


def split_bam_contig(
    bam_path: PathLike,
    contig: str,
    output_dir: PathLike,
    *,
    primary_only: bool = True,
    min_mapq: int = 0,
) -> Path:
    """Extract one contig into a sorted and indexed BAM.

    Parameters
    ----------
    bam_path : str or pathlib.Path
        Path to source BAM file.
    contig : str
        Contig name to extract.
    output_dir : str or pathlib.Path
        Output directory where ``<contig>.bam`` is written.
    primary_only : bool, optional
        If ``True``, include only primary alignments.
    min_mapq : int, optional
        Minimum read mapping quality to include.

    Returns
    -------
    pathlib.Path
        Path to sorted, indexed output BAM.
    """
    bam_path = Path(bam_path)
    output_dir = Path(output_dir)
    validate_bam(bam_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    unsorted_bam = output_dir / f"{contig}.unsorted.bam"
    out_bam = output_dir / f"{contig}.bam"

    n_written = 0
    with pysam.AlignmentFile(str(bam_path), "rb") as in_bam:
        if contig not in in_bam.references:
            raise ValueError(f"Contig '{contig}' not found in BAM: {bam_path}")

        header_dict = in_bam.header.to_dict()
        with pysam.AlignmentFile(str(unsorted_bam), "wb", header=header_dict) as out:
            for read in in_bam.fetch(contig=contig):
                if not _read_passes_filters(read, primary_only=primary_only, min_mapq=min_mapq):
                    continue
                out.write(read)
                n_written += 1

    pysam.sort("-o", str(out_bam), str(unsorted_bam))
    pysam.index(str(out_bam))

    if unsorted_bam.exists():
        unsorted_bam.unlink()

    logger.info("Extracted %d reads for contig %s into %s", n_written, contig, out_bam)
    return out_bam


def iter_contig_bams(
    bam_path: PathLike,
    contigs: list[str],
    *,
    primary_only: bool = True,
    min_mapq: int = 0,
) -> Generator[tuple[str, Path], None, None]:
    """Yield temporary sorted/indexed BAMs for each contig and clean up eagerly.

    Parameters
    ----------
    bam_path : str or pathlib.Path
        Path to source BAM file.
    contigs : list of str
        Contigs to split.
    primary_only : bool, optional
        If ``True``, include only primary alignments.
    min_mapq : int, optional
        Minimum read mapping quality to include.

    Yields
    ------
    tuple[str, pathlib.Path]
        Contig name and temporary BAM path.
    """
    bam_path = Path(bam_path)
    validate_bam(bam_path)

    with tempfile.TemporaryDirectory(prefix="baleen-contig-bams-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for contig in contigs:
            contig_bam = split_bam_contig(
                bam_path,
                contig,
                tmp_path,
                primary_only=primary_only,
                min_mapq=min_mapq,
            )

            try:
                yield contig, contig_bam
            finally:
                bai_candidates = [
                    Path(f"{contig_bam}.bai"),
                    contig_bam.with_suffix(".bai"),
                ]
                for path in [contig_bam, *bai_candidates]:
                    if path.exists():
                        path.unlink()
