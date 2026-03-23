from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import numpy as np
import pysam
import pytest

pipeline = importlib.import_module("baleen.eventalign._pipeline")
hier = importlib.import_module("baleen.eventalign._hierarchical")

PositionResult = pipeline.PositionResult
ContigResult = pipeline.ContigResult


def _make_synthetic_data():
    """Build minimal ContigResult + ContigModificationResult for testing."""
    n_native, n_ivt = 3, 2
    n_total = n_native + n_ivt

    # Distance matrix (not used by writer, but required by dataclass)
    mat = np.ones((n_total, n_total), dtype=np.float64)
    np.fill_diagonal(mat, 0.0)

    pr = PositionResult(
        position=100,
        reference_kmer="AACGU",
        n_native_reads=n_native,
        n_ivt_reads=n_ivt,
        native_read_names=["nat_0", "nat_1", "nat_2"],
        ivt_read_names=["ivt_0", "ivt_1"],
        distance_matrix=mat,
    )

    cr = ContigResult(
        contig="ecoli23S",
        native_depth=30.0,
        ivt_depth=20.0,
        positions={100: pr},
    )

    # Build PositionStats with known p_mod_hmm values
    ps = hier.PositionStats(
        position=100,
        reference_kmer="AACGU",
        coverage_class=hier.CoverageClass.HIGH,
        n_ivt=n_ivt,
        n_native=n_native,
        mu_raw=1.0, sigma_raw=0.5,
        mu_shrunk=1.0, sigma_shrunk=0.5,
        scores=np.zeros(n_total),
        z_scores=np.zeros(n_total),
        p_null=np.ones(n_total),
        p_mod_raw=np.array([0.8, 0.9, 0.7, 0.1, 0.05]),
        mixture_pi=0.5, mixture_null_gate=False, gate_weight=1.0,
        p_mod_knn=np.array([0.75, 0.85, 0.65, 0.1, 0.05]),
        p_mod_hmm=np.array([0.85, 0.92, 0.78, 0.05, 0.02]),
    )

    cmr = hier.ContigModificationResult(
        contig="ecoli23S",
        position_stats={100: ps},
        native_trajectories=[],
        ivt_trajectories=[],
        global_mu=1.0,
        global_sigma=0.5,
    )

    return {"ecoli23S": cr}, {"ecoli23S": cmr}


def test_write_read_bam_basic():
    """write_read_bam produces a sorted, indexed BAM with correct records."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    contig_results, hmm_results = _make_synthetic_data()

    with tempfile.TemporaryDirectory() as tmp:
        # Create a minimal FASTA for header
        ref_path = Path(tmp) / "ref.fa"
        ref_path.write_text(">ecoli23S\n" + "A" * 3000 + "\n")

        bam_path = Path(tmp) / "read_results.bam"
        result_path = read_bam.write_read_bam(
            hmm_results, contig_results, ref_path, bam_path,
        )

        # Verify BAM file and index exist
        assert result_path.exists()
        assert Path(str(result_path) + ".bai").exists()

        # Read back and verify records
        with pysam.AlignmentFile(str(result_path), "rb") as bam:
            records = list(bam.fetch())

        assert len(records) == 5  # 3 native + 2 ivt

        # Check first native read
        r0 = records[0]
        assert r0.query_name == "nat_0"
        assert r0.reference_name == "ecoli23S"
        assert r0.reference_start == 100
        assert abs(r0.get_tag("MP") - 0.85) < 1e-5
        assert r0.get_tag("RG") == "native"
        assert r0.get_tag("KM") == "AACGU"
        # SEQ should have U->T conversion
        assert "U" not in r0.query_sequence
        assert r0.query_sequence == "AACGT"
        # CIGAR should match kmer length
        assert r0.cigarstring == "5M"

        # Check last IVT read
        r4 = records[4]
        assert r4.query_name == "ivt_1"
        assert r4.get_tag("RG") == "ivt"
        assert abs(r4.get_tag("MP") - 0.02) < 1e-5


def test_write_read_bam_skips_nan():
    """Records with NaN p_mod_hmm are omitted from the BAM."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    contig_results, hmm_results = _make_synthetic_data()

    # Set one native read's p_mod_hmm to NaN
    ps = hmm_results["ecoli23S"].position_stats[100]
    ps.p_mod_hmm[1] = np.nan  # nat_1 becomes NaN

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.fa"
        ref_path.write_text(">ecoli23S\n" + "A" * 3000 + "\n")

        bam_path = Path(tmp) / "read_results.bam"
        read_bam.write_read_bam(hmm_results, contig_results, ref_path, bam_path)

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            records = list(bam.fetch())

        # 5 total - 1 NaN = 4
        assert len(records) == 4
        names = [r.query_name for r in records]
        assert "nat_1" not in names


def _write_test_bam(tmp_dir):
    """Helper: write a test BAM and return its path."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")
    contig_results, hmm_results = _make_synthetic_data()

    ref_path = Path(tmp_dir) / "ref.fa"
    ref_path.write_text(">ecoli23S\n" + "A" * 3000 + "\n")

    bam_path = Path(tmp_dir) / "read_results.bam"
    read_bam.write_read_bam(hmm_results, contig_results, ref_path, bam_path)
    return bam_path


def test_load_read_results_full():
    """load_read_results returns a DataFrame with all records."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    with tempfile.TemporaryDirectory() as tmp:
        bam_path = _write_test_bam(tmp)
        df = read_bam.load_read_results(bam_path)

    assert len(df) == 5
    assert set(df.columns) == {"contig", "position", "kmer", "read_name", "is_native", "p_mod_hmm"}
    assert df["is_native"].sum() == 3
    assert (~df["is_native"]).sum() == 2
    assert df["contig"].iloc[0] == "ecoli23S"
    assert df["kmer"].iloc[0] == "AACGU"  # Original RNA kmer from KM tag


def test_load_read_results_region_filter():
    """load_read_results with region filter returns subset."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    with tempfile.TemporaryDirectory() as tmp:
        bam_path = _write_test_bam(tmp)

        # Query region that includes position 100
        df = read_bam.load_read_results(bam_path, contig="ecoli23S", start=99, end=105)
        assert len(df) == 5

        # Query region that excludes position 100
        df_empty = read_bam.load_read_results(bam_path, contig="ecoli23S", start=200, end=300)
        assert len(df_empty) == 0
        # Empty DataFrame must have correct columns
        assert set(df_empty.columns) == {"contig", "position", "kmer", "read_name", "is_native", "p_mod_hmm"}


def test_load_read_results_iter():
    """load_read_results_iter yields dicts with correct keys."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    with tempfile.TemporaryDirectory() as tmp:
        bam_path = _write_test_bam(tmp)
        records = list(read_bam.load_read_results_iter(bam_path))

    assert len(records) == 5
    r0 = records[0]
    assert set(r0.keys()) == {"contig", "position", "kmer", "read_name", "is_native", "p_mod_hmm"}
    assert isinstance(r0["p_mod_hmm"], float)
    assert isinstance(r0["is_native"], bool)


def test_load_read_results_round_trip():
    """Values written by write_read_bam are preserved through load_read_results."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    with tempfile.TemporaryDirectory() as tmp:
        bam_path = _write_test_bam(tmp)
        df = read_bam.load_read_results(bam_path)

    # Check native reads have expected p_mod_hmm values
    native_df = df[df["is_native"]].sort_values("read_name").reset_index(drop=True)
    assert abs(native_df.loc[native_df["read_name"] == "nat_0", "p_mod_hmm"].values[0] - 0.85) < 1e-5
    assert abs(native_df.loc[native_df["read_name"] == "nat_1", "p_mod_hmm"].values[0] - 0.92) < 1e-5

    # Check IVT reads
    ivt_df = df[~df["is_native"]]
    assert len(ivt_df) == 2
    assert set(ivt_df["read_name"].tolist()) == {"ivt_0", "ivt_1"}


def test_write_read_bam_multi_contig():
    """BAM correctly handles multiple contigs sorted by (contig, position)."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")
    n = 3  # reads per group

    def _make_pos(pos, kmer="ACGTA"):
        mat = np.ones((n + n, n + n), dtype=np.float64)
        np.fill_diagonal(mat, 0.0)
        return PositionResult(
            position=pos, reference_kmer=kmer,
            n_native_reads=n, n_ivt_reads=n,
            native_read_names=[f"nat_{pos}_{i}" for i in range(n)],
            ivt_read_names=[f"ivt_{pos}_{i}" for i in range(n)],
            distance_matrix=mat,
        )

    def _make_ps(pos):
        return hier.PositionStats(
            position=pos, reference_kmer="ACGTA",
            coverage_class=hier.CoverageClass.HIGH,
            n_ivt=n, n_native=n,
            mu_raw=1.0, sigma_raw=0.5, mu_shrunk=1.0, sigma_shrunk=0.5,
            scores=np.zeros(2 * n), z_scores=np.zeros(2 * n),
            p_null=np.ones(2 * n),
            p_mod_raw=np.full(2 * n, 0.5),
            mixture_pi=0.5, mixture_null_gate=False, gate_weight=1.0,
            p_mod_knn=np.full(2 * n, 0.5),
            p_mod_hmm=np.full(2 * n, 0.5),
        )

    contig_results = {
        "chrB": ContigResult("chrB", 30, 20, {200: _make_pos(200), 300: _make_pos(300)}),
        "chrA": ContigResult("chrA", 30, 20, {100: _make_pos(100)}),
    }
    hmm_results = {
        "chrB": hier.ContigModificationResult("chrB", {200: _make_ps(200), 300: _make_ps(300)}, [], [], 1.0, 0.5),
        "chrA": hier.ContigModificationResult("chrA", {100: _make_ps(100)}, [], [], 1.0, 0.5),
    }

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.fa"
        ref_path.write_text(">chrA\n" + "A" * 1000 + "\n>chrB\n" + "A" * 1000 + "\n")

        bam_path = Path(tmp) / "read_results.bam"
        read_bam.write_read_bam(hmm_results, contig_results, ref_path, bam_path)

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            records = list(bam.fetch())

        # 3 positions × (3 native + 3 ivt) = 18 records
        assert len(records) == 18

        # Verify sorted by (contig, position)
        prev_contig, prev_pos = "", -1
        for r in records:
            if r.reference_name == prev_contig:
                assert r.reference_start >= prev_pos
            prev_contig = r.reference_name
            prev_pos = r.reference_start

        # Region query should work
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            chrA_records = list(bam.fetch("chrA"))
        assert len(chrA_records) == 6  # 1 position × 6 reads


def test_write_read_bam_no_fasta_fallback():
    """When ref_fasta doesn't exist, header is built from result contig names."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    contig_results, hmm_results = _make_synthetic_data()

    with tempfile.TemporaryDirectory() as tmp:
        # Use a non-existent FASTA path
        ref_path = Path(tmp) / "nonexistent.fa"

        bam_path = Path(tmp) / "read_results.bam"
        read_bam.write_read_bam(hmm_results, contig_results, ref_path, bam_path)

        # Should still write a valid BAM (with LN:0 in header)
        assert bam_path.exists()
        assert Path(str(bam_path) + ".bai").exists()

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            records = list(bam.fetch())

        assert len(records) == 5


def test_public_api_exports():
    """load_read_results and load_read_results_iter are importable from baleen.eventalign."""
    from baleen.eventalign import load_read_results, load_read_results_iter

    assert callable(load_read_results)
    assert callable(load_read_results_iter)
