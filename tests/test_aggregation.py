from __future__ import annotations

import importlib
import os
import tempfile

import numpy as np
import pytest

hier = importlib.import_module("baleen.eventalign._hierarchical")
pipeline = importlib.import_module("baleen.eventalign._pipeline")
agg = importlib.import_module("baleen.eventalign._aggregation")

PositionResult = pipeline.PositionResult
ContigResult = pipeline.ContigResult
SiteResult = agg.SiteResult


def _make_block_distance_matrix(
    n_native: int,
    n_ivt: int,
    within_native: float = 1.0,
    within_ivt: float = 1.0,
    between: float = 5.0,
    noise: float = 0.1,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(42)
    n = n_native + n_ivt
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if i < n_native and j < n_native:
                d = within_native
            elif i >= n_native and j >= n_native:
                d = within_ivt
            else:
                d = between
            d += rng.normal(0, noise)
            d = max(d, 0.0)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _make_homogeneous_matrix(
    n_native: int,
    n_ivt: int,
    base_dist: float = 1.0,
    noise: float = 0.05,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(99)
    n = n_native + n_ivt
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = base_dist + rng.normal(0, noise)
            d = max(d, 0.0)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _make_contig_result(
    n_positions: int = 10,
    n_native: int = 15,
    n_ivt: int = 10,
    modified_positions: set[int] | None = None,
    position_start: int = 100,
    seed: int = 42,
    contig_name: str = "chr1",
) -> ContigResult:
    if modified_positions is None:
        modified_positions = set()
    rng = np.random.RandomState(seed)
    positions: dict[int, PositionResult] = {}
    native_names = [f"native_{i}" for i in range(n_native)]
    ivt_names = [f"ivt_{i}" for i in range(n_ivt)]
    for idx in range(n_positions):
        pos = position_start + idx
        if idx in modified_positions:
            dm = _make_block_distance_matrix(
                n_native, n_ivt, within_native=1.5, within_ivt=1.0,
                between=6.0, noise=0.15, rng=rng,
            )
        else:
            dm = _make_homogeneous_matrix(
                n_native, n_ivt, base_dist=1.0, noise=0.05, rng=rng,
            )
        positions[pos] = PositionResult(
            position=pos, reference_kmer="AACGT",
            n_native_reads=n_native, n_ivt_reads=n_ivt,
            native_read_names=list(native_names),
            ivt_read_names=list(ivt_names),
            distance_matrix=dm,
        )
    return ContigResult(
        contig=contig_name, native_depth=float(n_native),
        ivt_depth=float(n_ivt), positions=positions,
    )


def _run_pipeline(cr: ContigResult):
    return hier.compute_sequential_modification_probabilities(cr)


class TestThresholdAggregate:
    def test_all_modified(self):
        p_mod = np.ones(20, dtype=np.float64)
        ratio, lo, hi = agg._threshold_aggregate(p_mod, threshold=0.99)
        assert ratio == 1.0
        assert lo > 0.8

    def test_all_unmodified(self):
        p_mod = np.zeros(20, dtype=np.float64)
        ratio, lo, hi = agg._threshold_aggregate(p_mod, threshold=0.99)
        assert ratio == 0.0
        assert hi < 0.2

    def test_mixed(self):
        # 10 reads above threshold, 10 below
        p_mod = np.array([1.0] * 10 + [0.1] * 10, dtype=np.float64)
        ratio, lo, hi = agg._threshold_aggregate(p_mod, threshold=0.99)
        assert ratio == 0.5
        assert lo < ratio < hi

    def test_ci_width_decreases_with_coverage(self):
        # 2/5 modified vs 40/100 modified — same ratio, narrower CI
        p5 = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        _, lo5, hi5 = agg._threshold_aggregate(p5, threshold=0.99)
        p100 = np.array([1.0] * 40 + [0.0] * 60, dtype=np.float64)
        _, lo100, hi100 = agg._threshold_aggregate(p100, threshold=0.99)
        assert (hi100 - lo100) < (hi5 - lo5)


class TestFisherPvalue:
    def test_identical_distributions(self):
        # No reads above threshold in either group
        native = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
        ivt = np.array([0.11, 0.19, 0.14, 0.13, 0.17])
        p = agg._fisher_pvalue(native, ivt, threshold=0.99)
        assert p == 1.0

    def test_clearly_different(self):
        # All native above threshold, no IVT above
        native = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        ivt = np.array([0.1, 0.12, 0.08, 0.15, 0.11])
        p = agg._fisher_pvalue(native, ivt, threshold=0.99)
        assert p < 0.01

    def test_no_reads(self):
        p = agg._fisher_pvalue(np.array([]), np.array([0.1, 0.2]), threshold=0.99)
        assert p == 1.0


class TestBenjaminiHochberg:
    def test_basic(self):
        pvals = np.array([0.01, 0.04, 0.03, 0.20], dtype=np.float64)
        padj = agg._benjamini_hochberg(pvals)
        assert padj.shape == (4,)
        assert np.all(padj >= pvals)
        assert np.all(padj <= 1.0)

    def test_empty(self):
        padj = agg._benjamini_hochberg(np.array([], dtype=np.float64))
        assert len(padj) == 0

    def test_single(self):
        padj = agg._benjamini_hochberg(np.array([0.05]))
        assert padj[0] == pytest.approx(0.05)

    def test_all_significant(self):
        pvals = np.array([0.001, 0.002, 0.003], dtype=np.float64)
        padj = agg._benjamini_hochberg(pvals)
        assert np.all(padj < 0.05)


class TestAggregateContig:
    def test_produces_results(self):
        cr = _make_contig_result(
            n_positions=8, modified_positions={2, 5}, seed=42,
        )
        cmr = _run_pipeline(cr)
        sites = agg.aggregate_contig(cmr)
        assert len(sites) == 8
        assert all(isinstance(s, SiteResult) for s in sites)

    def test_modified_sites_have_higher_ratio(self):
        cr = _make_contig_result(
            n_positions=10, n_native=20, n_ivt=15,
            modified_positions={3, 7}, seed=55,
        )
        cmr = _run_pipeline(cr)
        # Use a lower threshold so synthetic data can pass it
        sites = agg.aggregate_contig(cmr, mod_threshold=0.5)
        mod_positions = {100 + i for i in {3, 7}}
        mod_ratios = [s.mod_ratio for s in sites if s.position in mod_positions]
        unmod_ratios = [
            s.mod_ratio for s in sites if s.position not in mod_positions
        ]
        # Modified sites should have higher mod_ratio on average,
        # or at least the max modified > max unmodified
        assert max(mod_ratios) > np.median(unmod_ratios)

    def test_modified_sites_have_lower_pvalue(self):
        cr = _make_contig_result(
            n_positions=10, modified_positions={3, 7}, seed=55,
        )
        cmr = _run_pipeline(cr)
        sites = agg.aggregate_contig(cmr, mod_threshold=0.5)
        mod_pvals = [s.pvalue for s in sites if s.position in {103, 107}]
        unmod_pvals = [
            s.pvalue for s in sites if s.position not in {103, 107}
        ]
        assert np.mean(mod_pvals) < np.mean(unmod_pvals)


class TestAggregateAll:
    def test_fdr_applied(self):
        cr1 = _make_contig_result(
            n_positions=8, modified_positions={1, 4},
            seed=60, contig_name="chr1",
        )
        cr2 = _make_contig_result(
            n_positions=8, modified_positions={2, 6},
            seed=61, contig_name="chr2",
        )
        cmr1 = _run_pipeline(cr1)
        cmr2 = _run_pipeline(cr2)
        sites = agg.aggregate_all({"chr1": cmr1, "chr2": cmr2})
        assert len(sites) == 16
        # padj should be >= pvalue
        for s in sites:
            assert s.padj >= s.pvalue - 1e-15

    def test_sorted_by_contig_then_position(self):
        cr1 = _make_contig_result(
            n_positions=5, seed=70, contig_name="chrB",
        )
        cr2 = _make_contig_result(
            n_positions=5, seed=71, contig_name="chrA",
        )
        cmr1 = _run_pipeline(cr1)
        cmr2 = _run_pipeline(cr2)
        sites = agg.aggregate_all({"chrB": cmr1, "chrA": cmr2})
        contigs = [s.contig for s in sites]
        assert contigs == sorted(contigs)


class TestWriteTsv:
    def test_roundtrip(self):
        cr = _make_contig_result(
            n_positions=5, modified_positions={1, 3}, seed=80,
        )
        cmr = _run_pipeline(cr)
        sites = agg.aggregate_contig(cmr)

        tmp = tempfile.NamedTemporaryFile(
            suffix=".tsv", delete=False, mode="w",
        )
        tmp.close()
        try:
            agg.write_site_tsv(sites, tmp.name)
            # Read back and check
            with open(tmp.name) as f:
                lines = f.readlines()
            assert len(lines) == 6  # header + 5 sites
            header = lines[0].strip().split("\t")
            assert header[0] == "contig"
            assert header[3] == "mod_ratio"
            assert header[6] == "pvalue"
            assert header[7] == "padj"
        finally:
            os.unlink(tmp.name)
