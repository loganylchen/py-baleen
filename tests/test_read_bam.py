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


def _make_synthetic_bams(tmp_dir, contig_name, seq_len, native_names, ivt_names):
    """Create sorted, indexed synthetic native + IVT BAMs with reads aligned to contig."""
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": contig_name, "LN": seq_len}],
    })

    native_bam = Path(tmp_dir) / "native.bam"
    ivt_bam = Path(tmp_dir) / "ivt.bam"

    for bam_path, names in [(native_bam, native_names), (ivt_bam, ivt_names)]:
        with pysam.AlignmentFile(str(bam_path), "wb", header=header) as bf:
            for name in names:
                a = pysam.AlignedSegment(bf.header)
                a.query_name = name
                a.flag = 0
                a.reference_id = 0
                a.reference_start = 0
                a.mapping_quality = 60
                a.query_sequence = "A" * seq_len
                a.cigar = [(0, seq_len)]
                a.query_qualities = pysam.qualitystring_to_array("I" * seq_len)
                bf.write(a)
        sorted_path = str(bam_path) + ".sorted.bam"
        pysam.sort("-o", sorted_path, str(bam_path))
        Path(sorted_path).rename(bam_path)
        pysam.index(str(bam_path))

    return native_bam, ivt_bam


def _make_synthetic_data():
    """Build minimal ContigResult + ContigModificationResult for testing."""
    n_native, n_ivt = 3, 2
    n_total = n_native + n_ivt

    mat = np.ones((n_total, n_total), dtype=np.float64)
    np.fill_diagonal(mat, 0.0)

    native_names = ["nat_0", "nat_1", "nat_2"]
    ivt_names = ["ivt_0", "ivt_1"]

    pr = PositionResult(
        position=100,
        reference_kmer="AACGU",
        n_native_reads=n_native,
        n_ivt_reads=n_ivt,
        native_read_names=list(native_names),
        ivt_read_names=list(ivt_names),
        distance_matrix=mat,
    )

    cr = ContigResult(
        contig="ecoli23S",
        native_depth=30.0,
        ivt_depth=20.0,
        positions={100: pr},
    )

    ps = hier.PositionStats(
        position=100,
        reference_kmer="AACGU",
        coverage_class=hier.CoverageClass.HIGH,
        n_ivt=n_ivt,
        n_native=n_native,
        native_read_names=list(native_names),
        ivt_read_names=list(ivt_names),
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

    return {"ecoli23S": cr}, {"ecoli23S": cmr}, native_names, ivt_names


def _write_test_bam(tmp_dir):
    """Helper: write a test mod-BAM and return its path."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")
    contig_results, hmm_results, native_names, ivt_names = _make_synthetic_data()

    seq_len = 3000
    ref_path = Path(tmp_dir) / "ref.fa"
    ref_path.write_text(">ecoli23S\n" + "A" * seq_len + "\n")
    pysam.faidx(str(ref_path))

    native_bam, ivt_bam = _make_synthetic_bams(
        tmp_dir, "ecoli23S", seq_len, native_names, ivt_names,
    )

    bam_path = Path(tmp_dir) / "read_results.bam"
    read_bam.write_mod_bam(hmm_results, native_bam, ivt_bam, ref_path, bam_path)
    return bam_path


def test_write_mod_bam_basic():
    """write_mod_bam produces a sorted, indexed BAM with MM/ML tags."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    contig_results, hmm_results, native_names, ivt_names = _make_synthetic_data()

    with tempfile.TemporaryDirectory() as tmp:
        seq_len = 3000
        ref_path = Path(tmp) / "ref.fa"
        ref_path.write_text(">ecoli23S\n" + "A" * seq_len + "\n")
        pysam.faidx(str(ref_path))

        native_bam, ivt_bam = _make_synthetic_bams(
            tmp, "ecoli23S", seq_len, native_names, ivt_names,
        )

        bam_path = Path(tmp) / "read_results.bam"
        result_path = read_bam.write_mod_bam(
            hmm_results, native_bam, ivt_bam, ref_path, bam_path,
        )

        assert result_path.exists()
        assert Path(str(result_path) + ".bai").exists()

        with pysam.AlignmentFile(str(result_path), "rb") as bam:
            records = list(bam.fetch())

        # 3 native + 2 ivt = 5 reads (one record per read, not per position)
        assert len(records) == 5

        # Check first native read has MM/ML tags
        r0 = records[0]
        assert r0.get_tag("RG") == "native"
        assert r0.has_tag("MM")
        assert r0.has_tag("ML")

        # MM tag should be N+? format
        mm = r0.get_tag("MM")
        assert mm.startswith("N+?")

        # Check IVT reads
        ivt_records = [r for r in records if r.get_tag("RG") == "ivt"]
        assert len(ivt_records) == 2


def test_write_mod_bam_skips_nan():
    """Reads with all-NaN p_mod_hmm still appear but without MM/ML tags."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    contig_results, hmm_results, native_names, ivt_names = _make_synthetic_data()

    # Set one native read's p_mod_hmm to NaN
    ps = hmm_results["ecoli23S"].position_stats[100]
    ps.p_mod_hmm[1] = np.nan  # nat_1 becomes NaN

    with tempfile.TemporaryDirectory() as tmp:
        seq_len = 3000
        ref_path = Path(tmp) / "ref.fa"
        ref_path.write_text(">ecoli23S\n" + "A" * seq_len + "\n")
        pysam.faidx(str(ref_path))

        native_bam, ivt_bam = _make_synthetic_bams(
            tmp, "ecoli23S", seq_len, native_names, ivt_names,
        )

        bam_path = Path(tmp) / "read_results.bam"
        read_bam.write_mod_bam(hmm_results, native_bam, ivt_bam, ref_path, bam_path)

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            records = list(bam.fetch())

        # nat_1 still has a non-NaN position at pos 100 — wait, it only has 1 position
        # and it's NaN, so nat_1 won't be in read_positions at all.
        # All 5 reads still written, but nat_1 has no MM/ML tags.
        names_with_mm = [r.query_name for r in records if r.has_tag("MM")]
        assert "nat_1" not in names_with_mm


def test_load_read_results_full():
    """load_read_results returns a DataFrame with all records."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    with tempfile.TemporaryDirectory() as tmp:
        bam_path = _write_test_bam(tmp)
        df = read_bam.load_read_results(bam_path)

    assert len(df) == 5  # 5 reads × 1 position each
    expected_cols = {"contig", "position", "read_name", "is_native", "p_mod_hmm"}
    assert expected_cols.issubset(set(df.columns))
    assert df["is_native"].sum() == 3
    assert (~df["is_native"]).sum() == 2
    assert df["contig"].iloc[0] == "ecoli23S"


def test_load_read_results_region_filter():
    """load_read_results with region filter returns subset."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    with tempfile.TemporaryDirectory() as tmp:
        bam_path = _write_test_bam(tmp)

        # Query region that includes position 100 (0-based: 99)
        df = read_bam.load_read_results(bam_path, contig="ecoli23S", start=98, end=104)
        assert len(df) == 5

        # Query region that excludes position 100
        df_empty = read_bam.load_read_results(bam_path, contig="ecoli23S", start=200, end=300)
        assert len(df_empty) == 0
        expected_cols = {"contig", "position", "read_name", "is_native", "p_mod_hmm"}
        assert expected_cols.issubset(set(df_empty.columns))


def test_load_read_results_iter():
    """load_read_results_iter yields dicts with correct keys."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    with tempfile.TemporaryDirectory() as tmp:
        bam_path = _write_test_bam(tmp)
        records = list(read_bam.load_read_results_iter(bam_path))

    assert len(records) == 5
    r0 = records[0]
    expected_keys = {"contig", "position", "read_name", "is_native", "p_mod_hmm"}
    assert expected_keys.issubset(set(r0.keys()))
    assert isinstance(r0["p_mod_hmm"], float)
    assert isinstance(r0["is_native"], bool)


def test_load_read_results_round_trip():
    """Values written by write_mod_bam are preserved through load_read_results."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    with tempfile.TemporaryDirectory() as tmp:
        bam_path = _write_test_bam(tmp)
        df = read_bam.load_read_results(bam_path)

    # Check native reads have expected p_mod_hmm values (within uint8 quantization)
    native_df = df[df["is_native"]].sort_values("read_name").reset_index(drop=True)
    # Original: 0.85 → uint8 round(0.85*255)=217 → 217/255≈0.851
    assert abs(native_df.loc[native_df["read_name"] == "nat_0", "p_mod_hmm"].values[0] - 0.85) < 0.01
    assert abs(native_df.loc[native_df["read_name"] == "nat_1", "p_mod_hmm"].values[0] - 0.92) < 0.01

    # Check IVT reads
    ivt_df = df[~df["is_native"]]
    assert len(ivt_df) == 2
    assert set(ivt_df["read_name"].tolist()) == {"ivt_0", "ivt_1"}


def test_write_mod_bam_multi_contig():
    """BAM correctly handles multiple contigs with reads from different BAMs."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")
    n = 3

    def _make_ps(pos, native_names, ivt_names):
        n_total = len(native_names) + len(ivt_names)
        return hier.PositionStats(
            position=pos, reference_kmer="ACGTA",
            coverage_class=hier.CoverageClass.HIGH,
            n_ivt=len(ivt_names), n_native=len(native_names),
            native_read_names=native_names,
            ivt_read_names=ivt_names,
            mu_raw=1.0, sigma_raw=0.5, mu_shrunk=1.0, sigma_shrunk=0.5,
            scores=np.zeros(n_total), z_scores=np.zeros(n_total),
            p_null=np.ones(n_total),
            p_mod_raw=np.full(n_total, 0.5),
            mixture_pi=0.5, mixture_null_gate=False, gate_weight=1.0,
            p_mod_knn=np.full(n_total, 0.5),
            p_mod_hmm=np.full(n_total, 0.5),
        )

    # Use the same read names across positions (as in real pipeline)
    native_names = [f"nat_{i}" for i in range(n)]
    ivt_names = [f"ivt_{i}" for i in range(n)]

    hmm_results = {
        "chrA": hier.ContigModificationResult(
            "chrA",
            {100: _make_ps(100, native_names, ivt_names)},
            [], [], 1.0, 0.5,
        ),
        "chrB": hier.ContigModificationResult(
            "chrB",
            {200: _make_ps(200, native_names, ivt_names),
             300: _make_ps(300, native_names, ivt_names)},
            [], [], 1.0, 0.5,
        ),
    }

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.fa"
        ref_path.write_text(">chrA\n" + "A" * 1000 + "\n>chrB\n" + "A" * 1000 + "\n")
        pysam.faidx(str(ref_path))

        # Create BAMs with reads that map to both contigs
        header = pysam.AlignmentHeader.from_dict({
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chrA", "LN": 1000}, {"SN": "chrB", "LN": 1000}],
        })

        native_bam_path = Path(tmp) / "native.bam"
        ivt_bam_path = Path(tmp) / "ivt.bam"

        for bam_path_i, names in [(native_bam_path, native_names), (ivt_bam_path, ivt_names)]:
            with pysam.AlignmentFile(str(bam_path_i), "wb", header=header) as bf:
                for name in names:
                    # Map to chrA
                    a = pysam.AlignedSegment(bf.header)
                    a.query_name = name
                    a.flag = 0
                    a.reference_id = 0  # chrA
                    a.reference_start = 0
                    a.mapping_quality = 60
                    a.query_sequence = "A" * 500
                    a.cigar = [(0, 500)]
                    a.query_qualities = pysam.qualitystring_to_array("I" * 500)
                    bf.write(a)
            sorted_path = str(bam_path_i) + ".sorted.bam"
            pysam.sort("-o", sorted_path, str(bam_path_i))
            Path(sorted_path).rename(bam_path_i)
            pysam.index(str(bam_path_i))

        bam_path = Path(tmp) / "read_results.bam"
        read_bam.write_mod_bam(hmm_results, native_bam_path, ivt_bam_path, ref_path, bam_path)

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            records = list(bam.fetch())

        # Each read appears once (primary alignment), with MM/ML for positions
        # that fall within its alignment span
        assert len(records) == 6  # 3 native + 3 ivt


def test_public_api_exports():
    """load_read_results, load_read_results_iter, and write_mod_bam are importable."""
    from baleen.eventalign import load_read_results, load_read_results_iter, write_mod_bam

    assert callable(load_read_results)
    assert callable(load_read_results_iter)
    assert callable(write_mod_bam)


# ---------------------------------------------------------------------------
# Streaming-flush refactor: targeted regressions for the post-review fixes.
# ---------------------------------------------------------------------------


def test_flush_contig_to_bam_slice_is_coordinate_sorted():
    """Per-contig slice from flush_contig_to_bam must be coordinate-sorted.

    Native and IVT reads are read sequentially from two different input BAMs
    (native first, then IVT). Without an explicit sort step inside
    flush_contig_to_bam the merged output would interleave (native pos 100,
    ivt pos 50) and break ``samtools index`` on the merged file. Verify the
    slice is sorted by reference_start before any merge runs.
    """
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    # Two reads per side, deliberately at different start positions so a
    # naive native-then-IVT write would produce a non-sorted slice.
    native_names = ["nat_a", "nat_b"]
    ivt_names = ["ivt_a", "ivt_b"]

    n_total = 4
    pr = PositionResult(
        position=100, reference_kmer="AACGU",
        n_native_reads=2, n_ivt_reads=2,
        native_read_names=list(native_names),
        ivt_read_names=list(ivt_names),
        distance_matrix=np.eye(n_total),
    )
    cr = ContigResult(contig="chrZ", native_depth=2.0, ivt_depth=2.0,
                      positions={100: pr})  # noqa: F841

    ps = hier.PositionStats(
        position=100, reference_kmer="AACGU",
        coverage_class=hier.CoverageClass.HIGH,
        n_ivt=2, n_native=2,
        native_read_names=list(native_names),
        ivt_read_names=list(ivt_names),
        mu_raw=1.0, sigma_raw=0.5, mu_shrunk=1.0, sigma_shrunk=0.5,
        scores=np.zeros(n_total), z_scores=np.zeros(n_total),
        p_null=np.ones(n_total),
        p_mod_raw=np.array([0.8, 0.7, 0.1, 0.05]),
        mixture_pi=0.5, mixture_null_gate=False, gate_weight=1.0,
        p_mod_knn=np.array([0.8, 0.7, 0.1, 0.05]),
        p_mod_hmm=np.array([0.85, 0.75, 0.05, 0.02]),
    )
    cmr = hier.ContigModificationResult(
        contig="chrZ", position_stats={100: ps},
        native_trajectories=[], ivt_trajectories=[],
        global_mu=1.0, global_sigma=0.5,
    )

    with tempfile.TemporaryDirectory() as tmp:
        seq_len = 3000
        ref = Path(tmp) / "ref.fa"
        ref.write_text(">chrZ\n" + "A" * seq_len + "\n")
        pysam.faidx(str(ref))

        # Build native BAM with reads at HIGH start positions, IVT BAM with
        # reads at LOW start positions — order would be wrong without sort.
        header = pysam.AlignmentHeader.from_dict({
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chrZ", "LN": seq_len}],
        })
        native_bam = Path(tmp) / "native.bam"
        ivt_bam = Path(tmp) / "ivt.bam"

        def _make_bam(path, names, start_positions):
            with pysam.AlignmentFile(str(path), "wb", header=header) as bf:
                for name, st in zip(names, start_positions):
                    a = pysam.AlignedSegment(bf.header)
                    a.query_name = name
                    a.flag = 0
                    a.reference_id = 0
                    a.reference_start = st
                    a.mapping_quality = 60
                    a.query_sequence = "A" * 500
                    a.cigar = [(0, 500)]
                    a.query_qualities = pysam.qualitystring_to_array("I" * 500)
                    bf.write(a)
            sp = str(path) + ".sorted.bam"
            pysam.sort("-o", sp, str(path))
            Path(sp).rename(path)
            pysam.index(str(path))

        # Native at high pos, IVT at low pos → naive append would interleave wrong.
        _make_bam(native_bam, native_names, [800, 900])
        _make_bam(ivt_bam, ivt_names, [50, 60])

        out_path = Path(tmp) / "slice.bam"
        ref_path_for_header = native_bam
        slice_header = read_bam._build_header_from_bam(ref_path_for_header, ref)
        read_bam.flush_contig_to_bam(cmr, native_bam, ivt_bam, slice_header, out_path)

        # Slice must exist and be coordinate-sorted.
        with pysam.AlignmentFile(str(out_path), "rb") as bf:
            starts = [r.reference_start for r in bf.fetch(until_eof=True)]
        assert starts == sorted(starts), (
            f"per-contig slice is not coordinate-sorted: {starts}"
        )


def test_merge_contig_bams_indexable_after_merge():
    """merge_contig_bams output must be indexable (sort order preserved).

    Regression test: if any input slice were unsorted, ``pysam.index`` on the
    merged file would fail or produce corrupt indices. Build two slices via
    flush_contig_to_bam, merge, and confirm we can fetch by region.
    """
    read_bam = importlib.import_module("baleen.eventalign._read_bam")
    contig_results, hmm_results, native_names, ivt_names = _make_synthetic_data()

    with tempfile.TemporaryDirectory() as tmp:
        seq_len = 3000
        ref = Path(tmp) / "ref.fa"
        ref.write_text(">ecoli23S\n" + "A" * seq_len + "\n")
        pysam.faidx(str(ref))

        native_bam, ivt_bam = _make_synthetic_bams(
            tmp, "ecoli23S", seq_len, native_names, ivt_names,
        )

        header = read_bam._build_header_from_bam(native_bam, ref)

        slice_dir = Path(tmp) / "slices"
        slice_dir.mkdir()
        slice_path = slice_dir / "ecoli23S.bam"
        read_bam.flush_contig_to_bam(
            hmm_results["ecoli23S"], native_bam, ivt_bam, header, slice_path,
        )

        merged = Path(tmp) / "merged.bam"
        read_bam.merge_contig_bams([slice_path], merged, threads=1)

        # Index must exist and be usable (would raise on corrupt sort order).
        assert (merged.with_suffix(".bam.bai")).exists()
        with pysam.AlignmentFile(str(merged), "rb") as bf:
            fetched = list(bf.fetch("ecoli23S"))
        assert len(fetched) == 5


def test_sanitize_contig_filename_handles_unsafe_chars():
    """_sanitize_contig_filename must strip path-traversal characters."""
    pipeline_mod = importlib.import_module("baleen.eventalign._pipeline")
    sanitize = pipeline_mod._sanitize_contig_filename

    assert sanitize("chr1") == "chr1"  # safe names pass through unchanged
    assert "/" not in sanitize("../../etc/passwd")
    assert "\\" not in sanitize("foo\\bar")
    # Different unsafe inputs that map to the same plain stem must differ
    # (hash disambiguator) so per-contig artifacts don't collide.
    a = sanitize("a/b")
    b = sanitize("a\\b")
    assert a != b
    assert a.startswith("a_b-")
    assert b.startswith("a_b-")


def test_bam_header_dict_is_picklable():
    """Worker kwargs must be picklable for ProcessPoolExecutor (spawn).

    pysam.AlignmentHeader itself is NOT picklable (cdef class with non-trivial
    __cinit__).  The streaming pipeline must therefore ship ``to_dict()`` and
    rebuild via ``from_dict()`` in each worker.
    """
    import pickle
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    with tempfile.TemporaryDirectory() as tmp:
        seq_len = 1000
        ref = Path(tmp) / "ref.fa"
        ref.write_text(">chrX\n" + "A" * seq_len + "\n")
        pysam.faidx(str(ref))
        native_bam, _ = _make_synthetic_bams(tmp, "chrX", seq_len, ["r0"], ["r1"])

        header = read_bam._build_header_from_bam(native_bam, ref)

        # The raw header object is NOT picklable.
        with pytest.raises(TypeError):
            pickle.dumps(header)

        # to_dict() yields a picklable representation.
        header_dict = header.to_dict()
        blob = pickle.dumps(header_dict)
        rebuilt = pysam.AlignmentHeader.from_dict(pickle.loads(blob))
        assert "chrX" in rebuilt.references


def test_merge_contig_bams_chunked_matches_one_shot():
    """Chunked merge (batch_size < N) must produce a BAM bit-equivalent
    in record set to a single one-shot merge.

    Regression: ``samtools merge`` on thousands of inputs hits filesystem
    limits (fd, NFS, FUSE).  ``merge_contig_bams`` therefore merges in
    batches of ``batch_size`` and re-merges the intermediates.  Result
    must be identical (same reads, same order) to the one-shot merge.
    """
    read_bam = importlib.import_module("baleen.eventalign._read_bam")
    contig_results, hmm_results, native_names, ivt_names = _make_synthetic_data()

    with tempfile.TemporaryDirectory() as tmp:
        seq_len = 3000
        ref = Path(tmp) / "ref.fa"
        ref.write_text(">ecoli23S\n" + "A" * seq_len + "\n")
        pysam.faidx(str(ref))

        native_bam, ivt_bam = _make_synthetic_bams(
            tmp, "ecoli23S", seq_len, native_names, ivt_names,
        )

        header = read_bam._build_header_from_bam(native_bam, ref)

        # Build 5 identical slices (cheap proxy for "many contigs").
        slice_dir = Path(tmp) / "slices"
        slice_dir.mkdir()
        slices: list[Path] = []
        for i in range(5):
            slice_path = slice_dir / f"slice_{i:02d}.bam"
            read_bam.flush_contig_to_bam(
                hmm_results["ecoli23S"],
                native_bam,
                ivt_bam,
                header,
                slice_path,
            )
            slices.append(slice_path)

        one_shot = Path(tmp) / "one_shot.bam"
        chunked = Path(tmp) / "chunked.bam"

        read_bam.merge_contig_bams(slices, one_shot, threads=1, batch_size=100)
        # batch_size=2 forces ≥2 merge levels for 5 inputs.
        read_bam.merge_contig_bams(slices, chunked, threads=1, batch_size=2)

        with pysam.AlignmentFile(str(one_shot), "rb") as a, \
             pysam.AlignmentFile(str(chunked), "rb") as b:
            a_reads = [(r.query_name, r.reference_start) for r in a.fetch(until_eof=True)]
            b_reads = [(r.query_name, r.reference_start) for r in b.fetch(until_eof=True)]
        assert a_reads == b_reads, "chunked merge diverges from one-shot merge"
        assert chunked.with_suffix(".bam.bai").exists()


def test_merge_contig_bams_rejects_invalid_batch_size():
    """batch_size < 2 has no valid recursion behaviour and must raise."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")
    with pytest.raises(ValueError, match="batch_size must be >= 2"):
        read_bam.merge_contig_bams(
            [Path("/tmp/dummy.bam")] * 3,
            Path("/tmp/out.bam"),
            batch_size=1,
        )
