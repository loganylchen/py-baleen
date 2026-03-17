"""Tests for BAM utility helpers used in eventalign workflows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Protocol, Union, cast

import importlib
import pytest


class _AlignedSegmentProtocol(Protocol):
    query_name: str
    query_sequence: str
    flag: int
    reference_id: int
    reference_start: int
    mapping_quality: int
    cigarstring: str
    query_qualities: object
    reference_name: str


class _AlignmentFileProtocol(Protocol):
    def __enter__(self) -> "_AlignmentFileProtocol": ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...
    def fetch(self, contig: str) -> Sequence[_AlignedSegmentProtocol]: ...
    def write(self, read: _AlignedSegmentProtocol) -> None: ...


class _PysamModule(Protocol):
    def AlignmentFile(
        self,
        filename: str,
        mode: str,
        header: object | None = None,
    ) -> _AlignmentFileProtocol: ...

    def AlignedSegment(self) -> _AlignedSegmentProtocol: ...
    def qualitystring_to_array(self, quality_string: str) -> object: ...
    def sort(self, *args: str) -> None: ...
    def index(self, filename: str) -> None: ...


pysam = cast(_PysamModule, cast(object, importlib.import_module("pysam")))

ReadTuple = Union[tuple[int, str], tuple[int, str, int, int]]

from baleen.eventalign._bam import (
    ContigStats,
    FilterReason,
    filter_contigs,
    get_contig_stats,
    iter_contig_bams,
    split_bam_contig,
    validate_bam,
)

Approx = cast(Callable[..., object], pytest.approx)


def create_test_bam(
    tmp_path: Path,
    contig_reads: dict[str, Sequence[ReadTuple]],
    contigs_info: Sequence[tuple[str, int]],
) -> Path:
    """Create a sorted, indexed BAM file for testing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    contig_reads : dict
        Mapping ``contig_name -> list`` of read tuples. A read tuple can be
        ``(pos, seq)`` or ``(pos, seq, mapq, flag)``.
    contigs_info : list
        List of ``(name, length)`` contig tuples used to build the header.

    Returns
    -------
    pathlib.Path
        Path to sorted BAM.
    """
    header = {
        "HD": {"VN": "1.0", "SO": "unsorted"},
        "SQ": [{"SN": name, "LN": length} for name, length in contigs_info],
    }
    contig_to_id = {name: i for i, (name, _) in enumerate(contigs_info)}

    unsorted_bam = Path(tmp_path) / "test.unsorted.bam"
    sorted_bam = Path(tmp_path) / "test.bam"

    with pysam.AlignmentFile(str(unsorted_bam), "wb", header=header) as bam:
        for contig, reads in contig_reads.items():
            for idx, read_info in enumerate(reads):
                if len(read_info) == 2:
                    pos, seq = read_info
                    mapq = 60
                    flag = 0
                elif len(read_info) == 4:
                    pos, seq, mapq, flag = read_info
                else:
                    raise ValueError("Read tuples must be (pos, seq) or (pos, seq, mapq, flag)")

                aln = pysam.AlignedSegment()
                aln.query_name = f"{contig}_read_{idx}"
                aln.query_sequence = seq
                aln.flag = int(flag)
                aln.reference_id = contig_to_id[contig]
                aln.reference_start = int(pos)
                aln.mapping_quality = int(mapq)
                aln.cigarstring = f"{len(seq)}M"
                aln.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
                bam.write(aln)

    pysam.sort("-o", str(sorted_bam), str(unsorted_bam))
    pysam.index(str(sorted_bam))
    unsorted_bam.unlink()
    return sorted_bam


class TestValidateBam:
    def test_valid_bam(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            contig_reads={"ctg1": [(0, "AAAAAAAAAA")]},
            contigs_info=[("ctg1", 100)],
        )
        validate_bam(bam)

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            validate_bam(tmp_path / "missing.bam")

    def test_no_index(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            contig_reads={"ctg1": [(0, "AAAAAAAAAA")]},
            contigs_info=[("ctg1", 100)],
        )
        bai1 = Path(f"{bam}.bai")
        bai2 = bam.with_suffix(".bai")
        if bai1.exists():
            bai1.unlink()
        if bai2.exists():
            bai2.unlink()

        with pytest.raises(ValueError, match="not indexed"):
            validate_bam(bam)


class TestGetContigStats:
    def test_single_contig(self, tmp_path: Path):
        reads = [(i * 5, "AAAAAAAAAA") for i in range(10)]
        bam = create_test_bam(tmp_path, {"ctg1": reads}, [("ctg1", 200)])

        stats = get_contig_stats(bam)
        assert set(stats) == {"ctg1"}
        assert stats["ctg1"].mapped_reads == 10
        assert stats["ctg1"].mean_depth > 0.0

    def test_multiple_contigs(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            {
                "ctg1": [(0, "AAAAAAAAAA")] * 3,
                "ctg2": [(0, "CCCCCCCCCC")] * 2,
                "ctg3": [(0, "GGGGGGGGGG")],
            },
            [("ctg1", 100), ("ctg2", 100), ("ctg3", 100)],
        )

        stats = get_contig_stats(bam)
        assert stats["ctg1"].mapped_reads == 3
        assert stats["ctg2"].mapped_reads == 2
        assert stats["ctg3"].mapped_reads == 1

    def test_empty_contig_skipped(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            {"ctg1": [(0, "AAAAAAAAAA")]},
            [("ctg1", 100), ("ctg2", 100)],
        )
        stats = get_contig_stats(bam)
        assert "ctg1" in stats
        assert "ctg2" not in stats

    def test_primary_only_filtering(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            {"ctg1": [(0, "AAAAAAAAAA", 60, 0), (10, "AAAAAAAAAA", 60, 256)]},
            [("ctg1", 100)],
        )
        stats_primary = get_contig_stats(bam, primary_only=True)
        stats_all = get_contig_stats(bam, primary_only=False)
        assert stats_primary["ctg1"].mapped_reads == 1
        assert stats_all["ctg1"].mapped_reads == 2

    def test_min_mapq_filtering(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            {"ctg1": [(0, "AAAAAAAAAA", 5, 0), (10, "AAAAAAAAAA", 30, 0)]},
            [("ctg1", 100)],
        )
        stats = get_contig_stats(bam, min_mapq=10)
        assert stats["ctg1"].mapped_reads == 1

    def test_mean_depth_calculation(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            {"ctg1": [(0, "AAAAAAAAAA"), (10, "AAAAAAAAAA")]},
            [("ctg1", 100)],
        )
        stats = get_contig_stats(bam)
        assert stats["ctg1"].mean_depth == Approx(0.2, abs=1e-6)


class TestFilterContigs:
    def test_both_pass(self):
        native = {"ctg1": ContigStats("ctg1", 5, 20.0)}
        ivt = {"ctg1": ContigStats("ctg1", 6, 21.0)}
        passed, results = filter_contigs(native, ivt)
        assert passed == ["ctg1"]
        assert results[0].reason == FilterReason.PASSED

    def test_native_low_depth(self):
        native = {"ctg1": ContigStats("ctg1", 5, 10.0)}
        ivt = {"ctg1": ContigStats("ctg1", 6, 20.0)}
        _, results = filter_contigs(native, ivt)
        assert results[0].reason == FilterReason.LOW_DEPTH_NATIVE

    def test_ivt_low_depth(self):
        native = {"ctg1": ContigStats("ctg1", 5, 20.0)}
        ivt = {"ctg1": ContigStats("ctg1", 6, 10.0)}
        _, results = filter_contigs(native, ivt)
        assert results[0].reason == FilterReason.LOW_DEPTH_IVT

    def test_both_low_depth(self):
        native = {"ctg1": ContigStats("ctg1", 5, 10.0)}
        ivt = {"ctg1": ContigStats("ctg1", 6, 11.0)}
        _, results = filter_contigs(native, ivt)
        assert results[0].reason == FilterReason.LOW_DEPTH_BOTH

    def test_missing_in_native(self):
        native: dict[str, ContigStats] = {}
        ivt = {"ctg1": ContigStats("ctg1", 6, 20.0)}
        _, results = filter_contigs(native, ivt)
        assert results[0].reason == FilterReason.MISSING_IN_NATIVE

    def test_missing_in_ivt(self):
        native = {"ctg1": ContigStats("ctg1", 5, 20.0)}
        ivt: dict[str, ContigStats] = {}
        _, results = filter_contigs(native, ivt)
        assert results[0].reason == FilterReason.MISSING_IN_IVT

    def test_filter_results_complete(self):
        native = {
            "pass": ContigStats("pass", 5, 20.0),
            "native_only": ContigStats("native_only", 5, 20.0),
        }
        ivt = {
            "pass": ContigStats("pass", 6, 21.0),
            "ivt_only": ContigStats("ivt_only", 6, 21.0),
        }
        passed, results = filter_contigs(native, ivt)
        assert passed == ["pass"]
        assert {r.contig for r in results} == {"pass", "native_only", "ivt_only"}

    def test_custom_min_depth(self):
        native = {
            "ctg1": ContigStats("ctg1", 5, 6.0),
            "ctg2": ContigStats("ctg2", 5, 4.0),
        }
        ivt = {
            "ctg1": ContigStats("ctg1", 6, 6.0),
            "ctg2": ContigStats("ctg2", 6, 6.0),
        }
        passed, _ = filter_contigs(native, ivt, min_depth=5.0)
        assert passed == ["ctg1"]


class TestSplitBamContig:
    def test_split_creates_bam(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            {"ctg1": [(0, "AAAAAAAAAA")], "ctg2": [(0, "CCCCCCCCCC")]},
            [("ctg1", 100), ("ctg2", 100)],
        )
        out_dir = tmp_path / "split"
        out_bam = split_bam_contig(bam, "ctg1", out_dir)
        assert out_bam.exists()
        assert Path(f"{out_bam}.bai").exists() or out_bam.with_suffix(".bai").exists()

    def test_split_correct_reads(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            {"ctg1": [(0, "AAAAAAAAAA")] * 2, "ctg2": [(0, "CCCCCCCCCC")] * 3},
            [("ctg1", 100), ("ctg2", 100)],
        )
        out_bam = split_bam_contig(bam, "ctg2", tmp_path / "split")
        with pysam.AlignmentFile(str(out_bam), "rb") as out:
            reads = list(out.fetch("ctg2"))
            assert len(reads) == 3
            assert all(read.reference_name == "ctg2" for read in reads)

    def test_split_primary_only(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            {"ctg1": [(0, "AAAAAAAAAA", 60, 0), (10, "AAAAAAAAAA", 60, 256)]},
            [("ctg1", 100)],
        )
        out_bam = split_bam_contig(bam, "ctg1", tmp_path / "split", primary_only=True)
        with pysam.AlignmentFile(str(out_bam), "rb") as out:
            reads = list(out.fetch("ctg1"))
            assert len(reads) == 1

    def test_split_min_mapq(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            {"ctg1": [(0, "AAAAAAAAAA", 5, 0), (10, "AAAAAAAAAA", 30, 0)]},
            [("ctg1", 100)],
        )
        out_bam = split_bam_contig(bam, "ctg1", tmp_path / "split", min_mapq=10)
        with pysam.AlignmentFile(str(out_bam), "rb") as out:
            reads = list(out.fetch("ctg1"))
            assert len(reads) == 1
            assert reads[0].mapping_quality == 30


class TestIterContigBams:
    def test_generator_yields_all(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            {
                "ctg1": [(0, "AAAAAAAAAA")],
                "ctg2": [(0, "CCCCCCCCCC")],
                "ctg3": [(0, "GGGGGGGGGG")],
            },
            [("ctg1", 100), ("ctg2", 100), ("ctg3", 100)],
        )
        generated = list(iter_contig_bams(bam, ["ctg1", "ctg2", "ctg3"]))
        assert [contig for contig, _ in generated] == ["ctg1", "ctg2", "ctg3"]

    def test_cleanup_after_yield(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            {"ctg1": [(0, "AAAAAAAAAA")], "ctg2": [(0, "CCCCCCCCCC")]},
            [("ctg1", 100), ("ctg2", 100)],
        )

        gen = iter_contig_bams(bam, ["ctg1", "ctg2"])
        _, bam1 = next(gen)
        assert bam1.exists()

        _, bam2 = next(gen)
        assert not bam1.exists()
        assert bam2.exists()

        with pytest.raises(StopIteration):
            _ = next(gen)
        assert not bam2.exists()

    def test_cleanup_on_break(self, tmp_path: Path):
        bam = create_test_bam(
            tmp_path,
            {"ctg1": [(0, "AAAAAAAAAA")], "ctg2": [(0, "CCCCCCCCCC")]},
            [("ctg1", 100), ("ctg2", 100)],
        )

        gen = iter_contig_bams(bam, ["ctg1", "ctg2"])
        _, bam1 = next(gen)
        assert bam1.exists()

        gen.close()
        assert not bam1.exists()
