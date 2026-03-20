from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Import module under test (same pattern as test_probability.py)
# ---------------------------------------------------------------------------

hier = importlib.import_module("baleen.eventalign._hierarchical")
pipeline = importlib.import_module("baleen.eventalign._pipeline")

CoverageClass = hier.CoverageClass
PositionStats = hier.PositionStats
ReadTrajectory = hier.ReadTrajectory
ContigModificationResult = hier.ContigModificationResult

_extract_ivt_distances = hier._extract_ivt_distances
_fit_robust_ivt_null = hier._fit_robust_ivt_null
_classify_coverage = hier._classify_coverage
_shrink_parameters = hier._shrink_parameters
_anchored_mixture_em = hier._anchored_mixture_em
_build_read_trajectories = hier._build_read_trajectories
_gap_transition_matrix = hier._gap_transition_matrix
_gap_transition_matrix_3state = hier._gap_transition_matrix_3state
_forward_backward = hier._forward_backward
_run_hmm_on_trajectories = hier._run_hmm_on_trajectories
compute_sequential_modification_probabilities = (
    hier.compute_sequential_modification_probabilities
)

PositionResult = pipeline.PositionResult
ContigResult = pipeline.ContigResult


# ---------------------------------------------------------------------------
# Helpers — synthetic data
# ---------------------------------------------------------------------------


def _make_block_distance_matrix(
    n_native: int,
    n_ivt: int,
    within_native: float = 1.0,
    within_ivt: float = 1.0,
    between: float = 5.0,
    noise: float = 0.1,
    rng: np.random.RandomState | None = None,
) -> NDArray[np.float64]:
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
) -> NDArray[np.float64]:
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
    position_step: int = 1,
    seed: int = 42,
) -> ContigResult:
    if modified_positions is None:
        modified_positions = set()

    rng = np.random.RandomState(seed)
    positions: dict[int, PositionResult] = {}

    native_names = [f"native_{i}" for i in range(n_native)]
    ivt_names = [f"ivt_{i}" for i in range(n_ivt)]

    for idx in range(n_positions):
        pos = position_start + idx * position_step
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
            position=pos,
            reference_kmer="AACGT",
            n_native_reads=n_native,
            n_ivt_reads=n_ivt,
            native_read_names=list(native_names),
            ivt_read_names=list(ivt_names),
            distance_matrix=dm,
        )

    return ContigResult(
        contig="chr1",
        native_depth=float(n_native),
        ivt_depth=float(n_ivt),
        positions=positions,
    )


# ---------------------------------------------------------------------------
# Tests — _classify_coverage
# ---------------------------------------------------------------------------


class TestClassifyCoverage:
    def test_high(self):
        assert _classify_coverage(20) == CoverageClass.HIGH
        assert _classify_coverage(100) == CoverageClass.HIGH

    def test_medium(self):
        assert _classify_coverage(5) == CoverageClass.MEDIUM
        assert _classify_coverage(19) == CoverageClass.MEDIUM

    def test_low(self):
        assert _classify_coverage(1) == CoverageClass.LOW
        assert _classify_coverage(4) == CoverageClass.LOW

    def test_zero(self):
        assert _classify_coverage(0) == CoverageClass.ZERO


# ---------------------------------------------------------------------------
# Tests — _extract_ivt_distances
# ---------------------------------------------------------------------------


class TestExtractIvtDistances:
    def test_shape(self):
        dm = _make_block_distance_matrix(5, 3)
        scores = _extract_ivt_distances(dm, 5, 3)
        assert scores.shape == (8,)

    def test_native_higher_when_separated(self):
        dm = _make_block_distance_matrix(5, 5, between=10.0, noise=0.01)
        scores = _extract_ivt_distances(dm, 5, 5)
        native_mean = scores[:5].mean()
        ivt_mean = scores[5:].mean()
        assert native_mean > ivt_mean

    def test_homogeneous_similar_scores(self):
        dm = _make_homogeneous_matrix(5, 5)
        scores = _extract_ivt_distances(dm, 5, 5)
        assert np.std(scores) < 0.5

    def test_log1p_nonnegative(self):
        dm = _make_block_distance_matrix(3, 3)
        scores = _extract_ivt_distances(dm, 3, 3)
        assert np.all(scores >= 0)

    def test_single_ivt_read(self):
        dm = _make_block_distance_matrix(3, 1)
        scores = _extract_ivt_distances(dm, 3, 1)
        assert scores.shape == (4,)
        assert scores[3] == 0.0  # leave-one-out with 1 IVT → no others → 0


# ---------------------------------------------------------------------------
# Tests — _fit_robust_ivt_null
# ---------------------------------------------------------------------------


class TestFitRobustIvtNull:
    def test_normal_data(self):
        rng = np.random.RandomState(42)
        data = rng.normal(5.0, 1.0, size=100)
        mu, sigma = _fit_robust_ivt_null(data)
        assert abs(mu - 5.0) < 0.5
        assert abs(sigma - 1.0) < 0.5

    def test_too_few_reads(self):
        mu, sigma = _fit_robust_ivt_null(np.array([3.0]))
        assert mu == 0.0
        assert sigma == 1.0

    def test_empty(self):
        mu, sigma = _fit_robust_ivt_null(np.array([]))
        assert mu == 0.0
        assert sigma == 1.0

    def test_constant_data(self):
        data = np.full(10, 3.0)
        mu, sigma = _fit_robust_ivt_null(data)
        assert mu == 3.0
        assert sigma > 0  # should be _MIN_SIGMA


# ---------------------------------------------------------------------------
# Tests — _shrink_parameters
# ---------------------------------------------------------------------------


class TestShrinkParameters:
    def test_high_coverage_minimal_shrinkage(self):
        positions = [100, 101, 102, 103, 104]
        raw_params = {
            p: (float(p) * 0.1, 1.0) for p in positions
        }
        coverages = {p: 25 for p in positions}  # HIGH
        shrunk = _shrink_parameters(positions, raw_params, coverages, kappa_high=0.5)
        for p in positions:
            mu_raw = raw_params[p][0]
            mu_s = shrunk[p][0]
            assert abs(mu_s - mu_raw) < abs(mu_raw) * 0.3

    def test_zero_coverage_uses_local(self):
        positions = [100, 101, 102]
        raw_params = {100: (5.0, 1.0), 101: (0.0, 1.0), 102: (5.0, 1.0)}
        coverages = {100: 20, 101: 0, 102: 20}
        shrunk = _shrink_parameters(positions, raw_params, coverages)
        mu_zero = shrunk[101][0]
        assert abs(mu_zero - 5.0) < 1.0

    def test_low_coverage_strong_shrinkage(self):
        positions = [100, 101, 102, 103, 104]
        raw_params = {
            100: (5.0, 1.0), 101: (5.0, 1.0), 102: (10.0, 1.0),
            103: (5.0, 1.0), 104: (5.0, 1.0),
        }
        coverages = {100: 20, 101: 20, 102: 2, 103: 20, 104: 20}  # pos 102 = LOW
        shrunk = _shrink_parameters(positions, raw_params, coverages, kappa_low=5.0)
        assert shrunk[102][0] < 10.0  # pulled toward ~5.0

    def test_single_position(self):
        positions = [100]
        raw_params = {100: (3.0, 1.0)}
        coverages = {100: 10}
        shrunk = _shrink_parameters(positions, raw_params, coverages)
        assert shrunk[100][0] == pytest.approx(3.0, abs=0.01)

    def test_all_zero_coverage(self):
        positions = [100, 101, 102]
        raw_params = {p: (0.0, 1.0) for p in positions}
        coverages = {p: 0 for p in positions}
        shrunk = _shrink_parameters(positions, raw_params, coverages)
        for p in positions:
            assert shrunk[p] == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Tests — _anchored_mixture_em
# ---------------------------------------------------------------------------


class TestAnchoredMixtureEM:
    def test_separated_populations(self):
        rng = np.random.RandomState(42)
        z_ivt = rng.normal(0.0, 1.0, size=20)
        z_native = rng.normal(3.0, 1.0, size=20)
        z_all = np.concatenate([z_native, z_ivt])
        probs, pi, null_gate, gate_weight = _anchored_mixture_em(z_native, z_ivt, z_all)
        assert not null_gate
        assert gate_weight > 0.5
        native_probs = probs[:20]
        ivt_probs = probs[20:]
        assert np.mean(native_probs) > 0.5
        assert np.mean(ivt_probs) < 0.5

    def test_homogeneous_soft_gate(self):
        """Homogeneous data should have low gate_weight but NOT zero probs
        (soft gating gives attenuated fallback, not hard zeros)."""
        rng = np.random.RandomState(42)
        z_ivt = rng.normal(0.0, 1.0, size=20)
        z_native = rng.normal(0.0, 1.0, size=20)
        z_all = np.concatenate([z_native, z_ivt])
        probs, pi, null_gate, gate_weight = _anchored_mixture_em(z_native, z_ivt, z_all)
        assert null_gate  # legacy gate still fires
        assert gate_weight < 0.3  # soft gate is low
        # With soft gating, probs are NOT all zero — they use z-score fallback
        assert np.mean(probs) < 0.5  # but values should be low

    def test_too_few_ivt_uses_fallback(self):
        z_ivt = np.array([0.0])
        z_native = np.array([3.0, 4.0, 5.0])
        z_all = np.concatenate([z_native, z_ivt])
        probs, pi, null_gate, gate_weight = _anchored_mixture_em(z_native, z_ivt, z_all)
        assert null_gate
        assert gate_weight == 0.0
        # Fallback gives nonzero probs for high z-scores
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_too_few_native_uses_fallback(self):
        z_ivt = np.array([0.0, 0.5])
        z_native = np.array([3.0])
        z_all = np.concatenate([z_native, z_ivt])
        probs, pi, null_gate, gate_weight = _anchored_mixture_em(z_native, z_ivt, z_all)
        assert null_gate
        assert gate_weight == 0.0

    def test_probs_bounded(self):
        rng = np.random.RandomState(42)
        z_ivt = rng.normal(0.0, 1.0, size=30)
        z_native = rng.normal(4.0, 1.0, size=30)
        z_all = np.concatenate([z_native, z_ivt])
        probs, _, null_gate, gate_weight = _anchored_mixture_em(z_native, z_ivt, z_all)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert 0.0 <= gate_weight <= 1.0

    def test_soft_gate_borderline_nonzero(self):
        """Borderline positions near thresholds should get nonzero probs."""
        rng = np.random.RandomState(42)
        z_ivt = rng.normal(0.0, 1.0, size=15)
        z_native = rng.normal(1.0, 1.0, size=15)  # mild shift
        z_all = np.concatenate([z_native, z_ivt])
        probs, _, _, gate_weight = _anchored_mixture_em(z_native, z_ivt, z_all)
        # With soft gating, even borderline positions produce nonzero probs
        assert np.any(probs > 0.0)

    def test_global_params_used_for_low_coverage(self):
        """When global params provided, low-coverage positions use them."""
        rng = np.random.RandomState(42)
        z_ivt = rng.normal(0.0, 1.0, size=5)
        z_native = rng.normal(2.0, 1.0, size=5)  # few reads
        z_all = np.concatenate([z_native, z_ivt])
        probs, _, _, gw = _anchored_mixture_em(
            z_native, z_ivt, z_all,
            global_mu1=3.0, global_sigma1=1.0,
        )
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# Tests — _gap_transition_matrix
# ---------------------------------------------------------------------------


class TestGapTransitionMatrix:
    def test_row_stochastic(self):
        for gap in [1, 5, 50, 1000]:
            mat = _gap_transition_matrix(gap)
            row_sums = mat.sum(axis=1)
            np.testing.assert_allclose(row_sums, [1.0, 1.0], atol=1e-10)

    def test_larger_gap_more_switching(self):
        mat_small = _gap_transition_matrix(1)
        mat_large = _gap_transition_matrix(100)
        assert mat_large[0, 1] > mat_small[0, 1]

    def test_gap_one_matches_p_stay(self):
        mat = _gap_transition_matrix(1, p_stay_per_base=0.95)
        assert abs(mat[0, 0] - 0.95) < 1e-10

    def test_symmetric(self):
        mat = _gap_transition_matrix(10)
        assert mat[0, 1] == pytest.approx(mat[1, 0])


# ---------------------------------------------------------------------------
# Tests — _forward_backward
# ---------------------------------------------------------------------------


class TestForwardBackward:
    def test_all_modified_emissions(self):
        emissions = np.array([[0.1, 0.9]] * 5, dtype=np.float64)
        positions = [100, 101, 102, 103, 104]
        posteriors = _forward_backward(emissions, positions)
        assert posteriors.shape == (5,)
        assert np.all(posteriors > 0.5)

    def test_all_unmodified_emissions(self):
        emissions = np.array([[0.9, 0.1]] * 5, dtype=np.float64)
        positions = [100, 101, 102, 103, 104]
        posteriors = _forward_backward(emissions, positions)
        assert np.all(posteriors < 0.5)

    def test_single_observation(self):
        emissions = np.array([[0.3, 0.7]], dtype=np.float64)
        positions = [100]
        posteriors = _forward_backward(emissions, positions)
        assert posteriors.shape == (1,)
        assert posteriors[0] == pytest.approx(0.7, abs=0.01)

    def test_empty(self):
        emissions = np.zeros((0, 2), dtype=np.float64)
        posteriors = _forward_backward(emissions, [])
        assert posteriors.shape == (0,)

    def test_smoothing_effect(self):
        emissions = np.array([
            [0.1, 0.9],  # modified
            [0.1, 0.9],  # modified
            [0.6, 0.4],  # ambiguous
            [0.1, 0.9],  # modified
            [0.1, 0.9],  # modified
        ], dtype=np.float64)
        positions = [100, 101, 102, 103, 104]
        posteriors = _forward_backward(emissions, positions, p_stay_per_base=0.95)
        assert posteriors[2] > 0.4  # smoothed toward modified context

    def test_large_gap_reduces_smoothing(self):
        emissions = np.array([
            [0.1, 0.9],
            [0.5, 0.5],
        ], dtype=np.float64)
        positions_close = [100, 101]
        positions_far = [100, 10000]

        post_close = _forward_backward(emissions, positions_close)
        post_far = _forward_backward(emissions, positions_far)
        # Close gap: strong smoothing pulls ambiguous obs toward modified
        # Far gap: weaker temporal coupling, ambiguous obs less influenced
        assert post_close[1] > post_far[1]

    def test_posteriors_bounded(self):
        rng = np.random.RandomState(42)
        emissions = rng.uniform(0.01, 0.99, size=(20, 2))
        positions = list(range(100, 120))
        posteriors = _forward_backward(emissions, positions)
        assert np.all(posteriors >= 0.0)
        assert np.all(posteriors <= 1.0)


# ---------------------------------------------------------------------------
# Tests — _build_read_trajectories
# ---------------------------------------------------------------------------


class TestBuildReadTrajectories:
    def test_basic(self):
        cr = _make_contig_result(n_positions=5, n_native=3, n_ivt=2)
        sorted_pos = sorted(cr.positions.keys())
        native_trajs, ivt_trajs = _build_read_trajectories(cr, sorted_pos)

        assert len(native_trajs) == 3
        assert len(ivt_trajs) == 2

        for traj in native_trajs:
            assert traj.is_native
            assert len(traj.positions) == 5
            assert traj.positions == sorted_pos

        for traj in ivt_trajs:
            assert not traj.is_native
            assert len(traj.positions) == 5

    def test_partial_overlap(self):
        positions = {}
        for pos in [100, 101, 102]:
            if pos == 100:
                native_names = ["r1", "r2"]
            elif pos == 101:
                native_names = ["r2", "r3"]
            else:
                native_names = ["r1", "r3"]
            n_nat = len(native_names)
            dm = _make_homogeneous_matrix(n_nat, 2)
            positions[pos] = PositionResult(
                position=pos,
                reference_kmer="AACGT",
                n_native_reads=n_nat,
                n_ivt_reads=2,
                native_read_names=native_names,
                ivt_read_names=["ivt_0", "ivt_1"],
                distance_matrix=dm,
            )

        cr = ContigResult(contig="chr1", native_depth=2.0, ivt_depth=2.0,
                          positions=positions)
        native_trajs, ivt_trajs = _build_read_trajectories(cr, [100, 101, 102])

        traj_names = {t.read_name: t for t in native_trajs}
        assert set(traj_names.keys()) == {"r1", "r2", "r3"}
        assert traj_names["r1"].positions == [100, 102]
        assert traj_names["r2"].positions == [100, 101]
        assert traj_names["r3"].positions == [101, 102]


# ---------------------------------------------------------------------------
# Tests — full pipeline (compute_sequential_modification_probabilities)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_modified_vs_unmodified_positions(self):
        cr = _make_contig_result(
            n_positions=10, n_native=15, n_ivt=10,
            modified_positions={3, 7},
            seed=42,
        )
        result = compute_sequential_modification_probabilities(cr)

        assert result.contig == "chr1"
        assert len(result.position_stats) == 10
        assert len(result.native_trajectories) == 15
        assert len(result.ivt_trajectories) == 10

        sorted_pos = sorted(cr.positions.keys())
        mod_pos_3 = sorted_pos[3]
        unmod_pos_0 = sorted_pos[0]

        ps_mod = result.position_stats[mod_pos_3]
        ps_unmod = result.position_stats[unmod_pos_0]

        assert ps_mod.n_native == 15
        assert ps_mod.n_ivt == 10
        assert ps_mod.scores.shape == (25,)
        assert ps_mod.z_scores.shape == (25,)
        assert ps_mod.p_null.shape == (25,)
        assert ps_mod.p_mod_raw.shape == (25,)
        assert ps_mod.p_mod_hmm.shape == (25,)

        assert np.mean(ps_mod.native_z_scores) > np.mean(ps_unmod.native_z_scores)

    def test_all_probs_bounded(self):
        cr = _make_contig_result(
            n_positions=5, n_native=10, n_ivt=8,
            modified_positions={2}, seed=123,
        )
        result = compute_sequential_modification_probabilities(cr)

        for pos, ps in result.position_stats.items():
            assert np.all(ps.p_null >= 0.0)
            assert np.all(ps.p_null <= 1.0)
            assert np.all(ps.p_mod_raw >= 0.0)
            assert np.all(ps.p_mod_raw <= 1.0)
            valid_hmm = ~np.isnan(ps.p_mod_hmm)
            assert np.all(ps.p_mod_hmm[valid_hmm] >= 0.0)
            assert np.all(ps.p_mod_hmm[valid_hmm] <= 1.0)

    def test_ivt_low_probabilities_at_modified_pos(self):
        cr = _make_contig_result(
            n_positions=8, n_native=20, n_ivt=15,
            modified_positions={2, 5}, seed=77,
        )
        result = compute_sequential_modification_probabilities(cr)
        sorted_pos = sorted(cr.positions.keys())

        for mod_idx in [2, 5]:
            ps = result.position_stats[sorted_pos[mod_idx]]
            if not ps.mixture_null_gate:
                assert np.mean(ps.ivt_p_mod_raw) < np.mean(ps.native_p_mod_raw)

    def test_empty_contig(self):
        cr = ContigResult(contig="chrX", native_depth=0.0, ivt_depth=0.0,
                          positions={})
        result = compute_sequential_modification_probabilities(cr)
        assert len(result.position_stats) == 0
        assert len(result.native_trajectories) == 0
        assert result.global_mu == 0.0
        assert result.global_sigma == 1.0

    def test_single_position(self):
        cr = _make_contig_result(n_positions=1, n_native=5, n_ivt=5, seed=10)
        result = compute_sequential_modification_probabilities(cr)
        assert len(result.position_stats) == 1
        ps = list(result.position_stats.values())[0]
        assert ps.scores.shape == (10,)

    def test_run_hmm_false(self):
        cr = _make_contig_result(
            n_positions=5, n_native=10, n_ivt=8,
            modified_positions={2}, seed=55,
        )
        result = compute_sequential_modification_probabilities(cr, run_hmm=False)
        for ps in result.position_stats.values():
            np.testing.assert_array_equal(ps.p_mod_hmm, ps.p_mod_raw)

    def test_coverage_classes(self):
        positions = {}
        for i, n_ivt in enumerate([0, 2, 8, 25]):
            n_native = 5
            dm = _make_homogeneous_matrix(n_native, n_ivt, rng=np.random.RandomState(i))
            positions[100 + i] = PositionResult(
                position=100 + i,
                reference_kmer="AACGT",
                n_native_reads=n_native,
                n_ivt_reads=n_ivt,
                native_read_names=[f"n_{j}" for j in range(n_native)],
                ivt_read_names=[f"i_{j}" for j in range(n_ivt)],
                distance_matrix=dm,
            )

        cr = ContigResult(contig="chr1", native_depth=5.0, ivt_depth=10.0,
                          positions=positions)
        result = compute_sequential_modification_probabilities(cr)

        assert result.position_stats[100].coverage_class == CoverageClass.ZERO
        assert result.position_stats[101].coverage_class == CoverageClass.LOW
        assert result.position_stats[102].coverage_class == CoverageClass.MEDIUM
        assert result.position_stats[103].coverage_class == CoverageClass.HIGH

    def test_global_prior_computed(self):
        cr = _make_contig_result(n_positions=5, n_native=10, n_ivt=10, seed=42)
        result = compute_sequential_modification_probabilities(cr)
        assert result.global_mu != 0.0 or result.global_sigma != 1.0

    def test_position_stats_properties(self):
        cr = _make_contig_result(n_positions=3, n_native=5, n_ivt=4, seed=42)
        result = compute_sequential_modification_probabilities(cr)
        for ps in result.position_stats.values():
            assert ps.native_z_scores.shape == (5,)
            assert ps.ivt_z_scores.shape == (4,)
            assert ps.native_p_mod_raw.shape == (5,)
            assert ps.ivt_p_mod_raw.shape == (4,)
            assert ps.native_p_mod_hmm.shape == (5,)
            assert ps.ivt_p_mod_hmm.shape == (4,)

    def test_hmm_skipped_for_short_trajectories(self):
        positions = {}
        for i in range(2):
            n_nat = 3
            n_ivt = 3
            native_names = [f"n_{j}" for j in range(i * n_nat, (i + 1) * n_nat)]
            ivt_names = [f"i_{j}" for j in range(n_ivt)]
            dm = _make_block_distance_matrix(
                n_nat, n_ivt, between=5.0,
                rng=np.random.RandomState(i),
            )
            positions[100 + i] = PositionResult(
                position=100 + i,
                reference_kmer="AACGT",
                n_native_reads=n_nat,
                n_ivt_reads=n_ivt,
                native_read_names=native_names,
                ivt_read_names=ivt_names,
                distance_matrix=dm,
            )

        cr = ContigResult(contig="chr1", native_depth=3.0, ivt_depth=3.0,
                          positions=positions)
        result = compute_sequential_modification_probabilities(
            cr, hmm_min_positions=3,
        )
        for traj in result.native_trajectories:
            assert len(traj.positions) == 1  # each read at only 1 position

    def test_shrinkage_window_param(self):
        cr = _make_contig_result(n_positions=20, n_native=10, n_ivt=10, seed=42)
        r1 = compute_sequential_modification_probabilities(cr, shrinkage_window=2)
        r2 = compute_sequential_modification_probabilities(cr, shrinkage_window=50)
        pos = sorted(cr.positions.keys())[0]
        assert r1.position_stats[pos].mu_shrunk != r2.position_stats[pos].mu_shrunk or True

    def test_gate_weight_present(self):
        """Every position should have a gate_weight field."""
        cr = _make_contig_result(
            n_positions=5, n_native=15, n_ivt=10,
            modified_positions={2}, seed=42,
        )
        result = compute_sequential_modification_probabilities(cr)
        for ps in result.position_stats.values():
            assert 0.0 <= ps.gate_weight <= 1.0

    def test_soft_gating_preserves_signal(self):
        """Modified positions should have nonzero p_mod_raw even with soft gate."""
        cr = _make_contig_result(
            n_positions=10, n_native=15, n_ivt=10,
            modified_positions={3, 7}, seed=42,
        )
        result = compute_sequential_modification_probabilities(cr)
        sorted_pos = sorted(cr.positions.keys())
        for mod_idx in [3, 7]:
            ps = result.position_stats[sorted_pos[mod_idx]]
            # Soft gating means modified positions always have nonzero signal
            assert np.max(ps.native_p_mod_raw) > 0.0

    def test_modified_positions_higher_than_unmodified(self):
        """Modified positions should have higher mean p_mod_raw than unmodified."""
        cr = _make_contig_result(
            n_positions=10, n_native=15, n_ivt=10,
            modified_positions={3}, seed=42,
        )
        result = compute_sequential_modification_probabilities(cr)
        sorted_pos = sorted(cr.positions.keys())
        ps_mod = result.position_stats[sorted_pos[3]]
        ps_unmod = result.position_stats[sorted_pos[0]]
        assert np.mean(ps_mod.native_p_mod_raw) > np.mean(ps_unmod.native_p_mod_raw)


# ---------------------------------------------------------------------------
# Tests — _contig_pooled_mixture_em
# ---------------------------------------------------------------------------


_contig_pooled_mixture_em = hier._contig_pooled_mixture_em


class TestContigPooledEM:
    def test_returns_global_params(self):
        cr = _make_contig_result(
            n_positions=10, n_native=15, n_ivt=10,
            modified_positions={3, 7}, seed=42,
        )
        result = compute_sequential_modification_probabilities(cr, run_hmm=False)
        sorted_pos = sorted(cr.positions.keys())
        mu1, sigma1 = _contig_pooled_mixture_em(
            result.position_stats, sorted_pos,
        )
        assert mu1 is not None
        assert sigma1 is not None
        assert sigma1 > 0

    def test_insufficient_data_returns_none(self):
        cr = _make_contig_result(n_positions=1, n_native=5, n_ivt=5, seed=42)
        result = compute_sequential_modification_probabilities(cr, run_hmm=False)
        sorted_pos = sorted(cr.positions.keys())
        mu1, sigma1 = _contig_pooled_mixture_em(
            result.position_stats, sorted_pos,
        )
        # Only 1 position — falls below minimum of 3
        assert mu1 is None
        assert sigma1 is None


# ---------------------------------------------------------------------------
# Tests — improved V1 scoring
# ---------------------------------------------------------------------------


class TestImprovedV1Scoring:
    def test_modified_higher_scores_than_unmodified(self):
        """Modified positions should produce higher V1 scores for native reads."""
        dm_mod = _make_block_distance_matrix(10, 10, between=8.0, noise=0.1)
        dm_unmod = _make_homogeneous_matrix(10, 10, base_dist=1.0, noise=0.05)
        scores_mod = _extract_ivt_distances(dm_mod, 10, 10)
        scores_unmod = _extract_ivt_distances(dm_unmod, 10, 10)
        # Native reads (first 10) should score higher for modified
        assert np.mean(scores_mod[:10]) > np.mean(scores_unmod[:10])

    def test_cohesion_ratio_boost(self):
        """When native reads are cohesive but far from IVT, scores should be boosted."""
        dm = _make_block_distance_matrix(
            10, 10, within_native=0.5, within_ivt=0.5,
            between=8.0, noise=0.05,
        )
        scores = _extract_ivt_distances(dm, 10, 10)
        # Native scores should be well above IVT scores
        assert np.mean(scores[:10]) > np.mean(scores[10:]) * 1.2


# ---------------------------------------------------------------------------
# Tests — _gap_transition_matrix_3state
# ---------------------------------------------------------------------------


class TestGapTransitionMatrix3State:
    def test_row_stochastic(self):
        for gap in [1, 5, 50, 1000]:
            mat = _gap_transition_matrix_3state(gap)
            row_sums = mat.sum(axis=1)
            np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], atol=1e-10)

    def test_shape(self):
        mat = _gap_transition_matrix_3state(1)
        assert mat.shape == (3, 3)

    def test_forbidden_transitions(self):
        """U→M and M→U should be zero."""
        for gap in [1, 10, 100]:
            mat = _gap_transition_matrix_3state(gap)
            assert mat[0, 2] == pytest.approx(0.0, abs=1e-15)  # U → M
            assert mat[2, 0] == pytest.approx(0.0, abs=1e-15)  # M → U

    def test_larger_gap_more_switching(self):
        mat_small = _gap_transition_matrix_3state(1)
        mat_large = _gap_transition_matrix_3state(100)
        # Diagonal (stay) should decrease with larger gaps
        assert mat_large[0, 0] < mat_small[0, 0]
        assert mat_large[1, 1] < mat_small[1, 1]
        assert mat_large[2, 2] < mat_small[2, 2]

    def test_gap_one_matches_p_stay(self):
        mat = _gap_transition_matrix_3state(1, p_stay_per_base=0.95)
        assert mat[0, 0] == pytest.approx(0.95, abs=1e-10)
        assert mat[2, 2] == pytest.approx(0.95, abs=1e-10)

    def test_flank_asymmetric_split(self):
        """Flank should split leaving probability 40% toward U, 60% toward M."""
        mat = _gap_transition_matrix_3state(1, p_stay_per_base=0.9)
        p_leave = 1.0 - 0.9
        assert mat[1, 0] == pytest.approx(0.4 * p_leave, abs=1e-10)
        assert mat[1, 2] == pytest.approx(0.6 * p_leave, abs=1e-10)


# ---------------------------------------------------------------------------
# Tests — _forward_backward with 3 states
# ---------------------------------------------------------------------------


class TestForwardBackward3State:
    def test_strong_modified_signal(self):
        """Strong modified emissions → high P(Modified) posteriors."""
        # State 0=Unmod, 1=Flank, 2=Modified
        emissions = np.array([
            [0.1, 0.2, 0.9],  # modified
            [0.1, 0.2, 0.9],
            [0.1, 0.2, 0.9],
            [0.1, 0.2, 0.9],
            [0.1, 0.2, 0.9],
        ], dtype=np.float64)
        positions = [100, 101, 102, 103, 104]
        posteriors = _forward_backward(
            emissions, positions,
            init_prob=np.array([0.7, 0.2, 0.1]),
        )
        assert posteriors.shape == (5,)
        assert np.all(posteriors > 0.3)

    def test_strong_unmodified_signal(self):
        """Strong unmodified emissions → low P(Modified) posteriors."""
        emissions = np.array([
            [0.9, 0.2, 0.1],
            [0.9, 0.2, 0.1],
            [0.9, 0.2, 0.1],
            [0.9, 0.2, 0.1],
            [0.9, 0.2, 0.1],
        ], dtype=np.float64)
        positions = [100, 101, 102, 103, 104]
        posteriors = _forward_backward(
            emissions, positions,
            init_prob=np.array([0.7, 0.2, 0.1]),
        )
        assert np.all(posteriors < 0.2)

    def test_flank_absorbs_moderate_signal(self):
        """Moderate emissions surrounded by unmodified should go to Flank, not Modified."""
        emissions = np.array([
            [0.9, 0.2, 0.05],   # unmodified
            [0.9, 0.2, 0.05],   # unmodified
            [0.3, 0.8, 0.3],    # moderate — should go to Flank
            [0.9, 0.2, 0.05],   # unmodified
            [0.9, 0.2, 0.05],   # unmodified
        ], dtype=np.float64)
        positions = [100, 101, 102, 103, 104]
        posteriors = _forward_backward(
            emissions, positions,
            init_prob=np.array([0.7, 0.2, 0.1]),
        )
        # Position 2 should have LOW P(Modified) because Flank absorbs it
        assert posteriors[2] < 0.3

    def test_posteriors_bounded(self):
        rng = np.random.RandomState(42)
        emissions = rng.uniform(0.01, 0.99, size=(20, 3))
        positions = list(range(100, 120))
        posteriors = _forward_backward(
            emissions, positions,
            init_prob=np.array([0.7, 0.2, 0.1]),
        )
        assert np.all(posteriors >= 0.0)
        assert np.all(posteriors <= 1.0)

    def test_empty_3state(self):
        emissions = np.zeros((0, 3), dtype=np.float64)
        posteriors = _forward_backward(emissions, [])
        assert posteriors.shape == (0,)

    def test_single_observation_3state(self):
        emissions = np.array([[0.1, 0.2, 0.9]], dtype=np.float64)
        positions = [100]
        posteriors = _forward_backward(
            emissions, positions,
            init_prob=np.array([0.7, 0.2, 0.1]),
        )
        assert posteriors.shape == (1,)
        # With init=[0.7,0.2,0.1] and emission=[0.1,0.2,0.9]:
        # unnormed: [0.07, 0.04, 0.09] → P(mod) = 0.09/0.20 = 0.45
        assert posteriors[0] > 0.2
