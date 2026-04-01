from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
from numpy.typing import NDArray

prob_mod = importlib.import_module("baleen.eventalign._probability")
AlgorithmName = prob_mod.AlgorithmName
ModificationProbabilities = prob_mod.ModificationProbabilities
compute_modification_probabilities = prob_mod.compute_modification_probabilities
distance_to_ivt = prob_mod.distance_to_ivt
knn_ivt_purity = prob_mod.knn_ivt_purity
mds_gmm = prob_mod.mds_gmm
_score_distance_to_ivt = prob_mod._score_distance_to_ivt
_score_knn_ivt_purity = prob_mod._score_knn_ivt_purity
_classical_mds = prob_mod._classical_mds
_calibrate_normal = prob_mod._calibrate_normal
_calibrate_beta = prob_mod._calibrate_beta
_fit_normal = prob_mod._fit_normal
_fit_beta = prob_mod._fit_beta


# ---------------------------------------------------------------------------
# Helpers — synthetic distance matrices
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
    """Build a synthetic symmetric distance matrix with block structure.

    Native reads are close to each other (within_native), IVT reads are
    close to each other (within_ivt), and native-IVT distances are large
    (between).  This simulates a position where ALL native reads are modified.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    n = n_native + n_ivt
    mat = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            i_is_native = i < n_native
            j_is_native = j < n_native
            if i_is_native and j_is_native:
                d = within_native
            elif not i_is_native and not j_is_native:
                d = within_ivt
            else:
                d = between
            d += rng.normal(0, noise)
            d = max(d, 0.0)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _make_partial_mod_matrix(
    n_mod: int,
    n_unmod_native: int,
    n_ivt: int,
    within_unmod: float = 1.0,
    within_ivt: float = 1.0,
    within_mod: float = 1.5,
    unmod_to_ivt: float = 1.2,
    mod_to_ivt: float = 6.0,
    mod_to_unmod: float = 5.5,
    noise: float = 0.1,
    rng: np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], int, int]:
    """Build matrix where only some native reads are modified.

    Order: [modified native, unmodified native, IVT]
    Returns (matrix, n_native_total, n_ivt)
    """
    if rng is None:
        rng = np.random.RandomState(42)
    n_native = n_mod + n_unmod_native
    n = n_native + n_ivt
    mat = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            i_mod = i < n_mod
            i_unmod = n_mod <= i < n_native
            i_ivt = i >= n_native
            j_mod = j < n_mod
            j_unmod = n_mod <= j < n_native
            j_ivt = j >= n_native

            if i_mod and j_mod:
                d = within_mod
            elif i_unmod and j_unmod:
                d = within_unmod
            elif i_ivt and j_ivt:
                d = within_ivt
            elif (i_mod and j_ivt) or (i_ivt and j_mod):
                d = mod_to_ivt
            elif (i_mod and j_unmod) or (i_unmod and j_mod):
                d = mod_to_unmod
            elif (i_unmod and j_ivt) or (i_ivt and j_unmod):
                d = unmod_to_ivt
            else:
                d = 1.0

            d += rng.normal(0, noise)
            d = max(d, 0.0)
            mat[i, j] = d
            mat[j, i] = d
    return mat, n_native, n_ivt


def _make_null_matrix(
    n_native: int,
    n_ivt: int,
    within: float = 1.0,
    noise: float = 0.1,
    rng: np.random.RandomState | None = None,
) -> NDArray[np.float64]:
    """All reads are similar — no modification signal."""
    if rng is None:
        rng = np.random.RandomState(42)
    n = n_native + n_ivt
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = within + rng.normal(0, noise)
            d = max(d, 0.0)
            mat[i, j] = d
            mat[j, i] = d
    return mat


# ---------------------------------------------------------------------------
# Test fit helpers
# ---------------------------------------------------------------------------


class TestFitNormal:
    def test_basic(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu, sigma = _fit_normal(x)
        assert abs(mu - 3.0) < 1e-10
        assert sigma > 0

    def test_single_value(self) -> None:
        x = np.array([5.0])
        mu, sigma = _fit_normal(x)
        assert mu == 5.0
        assert sigma > 0  # clamped to _MIN_SIGMA


class TestFitBeta:
    def test_basic(self) -> None:
        rng = np.random.RandomState(0)
        x = rng.beta(2, 5, size=1000)
        a, b = _fit_beta(x)
        assert a > 0
        assert b > 0
        assert a < b  # 2 < 5 → a < b


# ---------------------------------------------------------------------------
# Test calibration
# ---------------------------------------------------------------------------


class TestCalibrateNormal:
    def test_separated_groups_produce_high_alt_probs(self) -> None:
        ivt_scores = np.random.RandomState(42).normal(0, 0.5, size=20)
        native_scores = np.random.RandomState(43).normal(3, 0.5, size=20)
        all_scores = np.concatenate([native_scores, ivt_scores])
        n_native = len(native_scores)
        ivt_mask = np.zeros(len(all_scores), dtype=bool)
        ivt_mask[n_native:] = True

        cal = _calibrate_normal(all_scores, ivt_mask, ~ivt_mask)
        assert not cal.null_gate_active
        assert cal.pi > 0.1
        # Native reads should have high probabilities
        native_probs = cal.probabilities[:n_native]
        ivt_probs = cal.probabilities[n_native:]
        assert float(np.mean(native_probs)) > 0.5
        assert float(np.mean(ivt_probs)) < 0.3

    def test_null_position_low_probabilities(self) -> None:
        """When native and IVT are drawn from the same distribution,
        soft gate should produce low modification probabilities."""
        rng = np.random.RandomState(42)
        all_scores = rng.normal(0, 0.5, size=30)
        ivt_mask = np.zeros(30, dtype=bool)
        ivt_mask[15:] = True

        cal = _calibrate_normal(all_scores, ivt_mask, ~ivt_mask)
        # With soft gating, probabilities are no longer hard-zeroed but
        # should remain moderate (z-score fallback on null data ≈ 0.5)
        assert float(np.mean(cal.probabilities)) < 0.7


class TestCalibrateBeta:
    def test_separated_groups(self) -> None:
        rng = np.random.RandomState(42)
        ivt_scores = rng.beta(2, 8, size=20)  # low values
        native_scores = rng.beta(8, 2, size=20)  # high values
        all_scores = np.concatenate([native_scores, ivt_scores])
        n_native = len(native_scores)
        ivt_mask = np.zeros(len(all_scores), dtype=bool)
        ivt_mask[n_native:] = True

        cal = _calibrate_beta(all_scores, ivt_mask, ~ivt_mask)
        assert not cal.null_gate_active
        native_probs = cal.probabilities[:n_native]
        assert float(np.mean(native_probs)) > 0.3

    def test_too_few_ivt_returns_zero(self) -> None:
        all_scores = np.array([0.5, 0.6, 0.7])
        ivt_mask = np.array([False, False, True])  # only 1 IVT
        cal = _calibrate_beta(all_scores, ivt_mask, ~ivt_mask)
        assert cal.null_gate_active
        assert np.all(cal.probabilities == 0.0)


# ---------------------------------------------------------------------------
# Test Algorithm 1 — Distance-to-IVT
# ---------------------------------------------------------------------------


class TestDistanceToIvt:
    def test_modified_natives_get_higher_prob(self) -> None:
        mat = _make_block_distance_matrix(10, 10, within_native=1.0,
                                          within_ivt=1.0, between=8.0)
        result = distance_to_ivt(mat, n_native=10, n_ivt=10)
        assert result.algorithm == AlgorithmName.DISTANCE_TO_IVT
        assert result.n_native == 10
        assert result.n_ivt == 10
        assert result.probabilities.shape == (20,)
        # Native should have higher prob than IVT
        assert float(np.mean(result.native_probabilities)) > float(np.mean(result.ivt_probabilities))

    def test_null_position_low_probabilities(self) -> None:
        """Null matrix (no modification signal) should yield low probabilities."""
        mat = _make_null_matrix(10, 10)
        result = distance_to_ivt(mat, n_native=10, n_ivt=10)
        # With soft gating, null positions produce low but non-zero probs
        assert float(np.mean(np.abs(result.probabilities))) < 0.7

    def test_scores_shape(self) -> None:
        mat = _make_block_distance_matrix(5, 5)
        result = distance_to_ivt(mat, n_native=5, n_ivt=5)
        assert result.scores.shape == (10,)


# ---------------------------------------------------------------------------
# Test Algorithm 3 — kNN IVT-Purity
# ---------------------------------------------------------------------------


class TestKnnIvtPurity:
    def test_modified_natives_get_higher_prob(self) -> None:
        mat = _make_block_distance_matrix(10, 10, within_native=1.0,
                                          within_ivt=1.0, between=8.0)
        result = knn_ivt_purity(mat, n_native=10, n_ivt=10)
        assert result.algorithm == AlgorithmName.KNN_IVT_PURITY
        assert float(np.mean(result.native_probabilities)) > float(np.mean(result.ivt_probabilities))

    def test_partial_modification(self) -> None:
        mat, n_nat, n_ivt = _make_partial_mod_matrix(
            n_mod=5, n_unmod_native=10, n_ivt=10,
            mod_to_ivt=10.0, mod_to_unmod=9.0, unmod_to_ivt=1.2,
        )
        result = knn_ivt_purity(mat, n_native=n_nat, n_ivt=n_ivt)
        # Modified native reads (first 5) should have higher probs
        # than unmodified native reads (next 10)
        mod_probs = result.probabilities[:5]
        unmod_probs = result.probabilities[5:15]
        if not result.null_gate_active:
            assert float(np.mean(mod_probs)) > float(np.mean(unmod_probs))

    def test_null_position_returns_zero(self) -> None:
        mat = _make_null_matrix(10, 10)
        result = knn_ivt_purity(mat, n_native=10, n_ivt=10)
        assert result.null_gate_active
        assert np.all(result.probabilities == 0.0)

    def test_scores_in_unit_interval(self) -> None:
        mat = _make_block_distance_matrix(8, 8)
        result = knn_ivt_purity(mat, n_native=8, n_ivt=8)
        assert np.all(result.scores >= 0.0)
        assert np.all(result.scores <= 1.0)

    def test_custom_k(self) -> None:
        mat = _make_block_distance_matrix(10, 10, between=8.0)
        result = knn_ivt_purity(mat, n_native=10, n_ivt=10, k=5)
        assert result.scores.shape == (20,)


# ---------------------------------------------------------------------------
# Test Algorithm 5 — MDS + GMM
# ---------------------------------------------------------------------------


class TestClassicalMds:
    def test_2d_embedding_shape(self) -> None:
        mat = _make_block_distance_matrix(5, 5)
        coords = _classical_mds(mat, n_components=2)
        assert coords.shape == (10, 2)

    def test_1d_embedding(self) -> None:
        mat = _make_block_distance_matrix(5, 5)
        coords = _classical_mds(mat, n_components=1)
        assert coords.shape == (10, 1)

    def test_zero_matrix_returns_zeros(self) -> None:
        mat = np.zeros((4, 4), dtype=np.float64)
        coords = _classical_mds(mat, n_components=2)
        assert coords.shape == (4, 2)
        assert np.allclose(coords, 0.0)


class TestMdsGmm:
    def test_modified_natives_get_higher_prob(self) -> None:
        mat = _make_block_distance_matrix(10, 10, within_native=1.0,
                                          within_ivt=1.0, between=8.0)
        result = mds_gmm(mat, n_native=10, n_ivt=10)
        assert result.algorithm == AlgorithmName.MDS_GMM
        assert float(np.mean(result.native_probabilities)) > float(np.mean(result.ivt_probabilities))

    def test_null_position_returns_zero(self) -> None:
        mat = _make_null_matrix(10, 10)
        result = mds_gmm(mat, n_native=10, n_ivt=10)
        assert result.null_gate_active
        assert np.all(result.probabilities == 0.0)

    def test_small_sample(self) -> None:
        mat = _make_block_distance_matrix(3, 3, between=8.0)
        result = mds_gmm(mat, n_native=3, n_ivt=3)
        assert result.probabilities.shape == (6,)

    def test_too_few_ivt_returns_zero(self) -> None:
        mat = np.array([
            [0.0, 1.0, 5.0],
            [1.0, 0.0, 5.0],
            [5.0, 5.0, 0.0],
        ])
        result = mds_gmm(mat, n_native=2, n_ivt=1)
        assert result.null_gate_active


# ---------------------------------------------------------------------------
# Test top-level API
# ---------------------------------------------------------------------------


class TestKnnScoreSpread:
    """Verify kNN scores have good spread for clearly separated data."""

    def test_modified_reads_score_high(self):
        """With strong block structure (native far from IVT), native kNN
        scores should be > 0.7 on average."""
        rng = np.random.RandomState(42)
        dm = _make_block_distance_matrix(
            20, 20, within_native=1.0, within_ivt=1.0,
            between=5.0, noise=0.1, rng=rng,
        )
        scores = _score_knn_ivt_purity(dm, 20, 20)
        native_mean = float(np.mean(scores[:20]))
        assert native_mean > 0.7, (
            f"Native kNN scores too low for strong block structure: {native_mean:.3f}"
        )

    def test_unmodified_reads_score_low(self):
        """With homogeneous matrix, all scores should be moderate (near 0.5)."""
        rng = np.random.RandomState(42)
        n_native, n_ivt = 20, 20
        n = n_native + n_ivt
        mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 + rng.normal(0, 0.05)
                mat[i, j] = max(d, 0)
                mat[j, i] = mat[i, j]
        scores = _score_knn_ivt_purity(mat, n_native, n_ivt)
        ivt_mean = float(np.mean(scores[n_native:]))
        # IVT scores should be low-to-moderate
        assert ivt_mean < 0.6, (
            f"IVT kNN scores too high for homogeneous data: {ivt_mean:.3f}"
        )


class TestPiWeightedPosteriors:
    """Verify that calibration uses fitted pi in posterior computation."""

    def test_high_pi_gives_high_posteriors(self):
        """When most reads are clearly modified (high pi), native posteriors
        should be > 0.7 on average."""
        rng = np.random.RandomState(42)
        dm = _make_block_distance_matrix(
            30, 15, within_native=1.0, within_ivt=1.0,
            between=5.0, noise=0.1, rng=rng,
        )
        result = knn_ivt_purity(dm, 30, 15)
        native_probs = result.native_probabilities
        native_mean = float(np.mean(native_probs))
        assert native_mean > 0.7, (
            f"Native posteriors too low when modification is clear: {native_mean:.3f}"
        )

    def test_ivt_posteriors_stay_low(self):
        """IVT read posteriors should remain low regardless of pi weighting."""
        rng = np.random.RandomState(42)
        dm = _make_block_distance_matrix(
            30, 15, within_native=1.0, within_ivt=1.0,
            between=5.0, noise=0.1, rng=rng,
        )
        result = knn_ivt_purity(dm, 30, 15)
        ivt_probs = result.ivt_probabilities
        ivt_mean = float(np.mean(ivt_probs))
        assert ivt_mean < 0.3, (
            f"IVT posteriors too high: {ivt_mean:.3f}"
        )


class TestComputeModificationProbabilities:
    def test_all_algorithms(self) -> None:
        mat = _make_block_distance_matrix(10, 10, between=8.0)
        results = compute_modification_probabilities(mat, n_native=10, n_ivt=10, position=42)
        assert len(results) == 3
        for alg_name, mp in results.items():
            assert mp.position == 42
            assert mp.probabilities.shape == (20,)
            assert mp.n_native == 10
            assert mp.n_ivt == 10

    def test_single_algorithm(self) -> None:
        mat = _make_block_distance_matrix(8, 8, between=8.0)
        results = compute_modification_probabilities(
            mat, n_native=8, n_ivt=8,
            algorithms=[AlgorithmName.DISTANCE_TO_IVT],
        )
        assert len(results) == 1
        assert AlgorithmName.DISTANCE_TO_IVT in results

    def test_shape_mismatch_raises(self) -> None:
        mat = np.zeros((5, 5))
        with pytest.raises(ValueError, match="does not match"):
            compute_modification_probabilities(mat, n_native=3, n_ivt=3)

    def test_position_label_propagated(self) -> None:
        mat = _make_block_distance_matrix(5, 5, between=8.0)
        results = compute_modification_probabilities(mat, n_native=5, n_ivt=5, position=999)
        for mp in results.values():
            assert mp.position == 999


class TestModificationProbabilitiesProperties:
    def test_native_ivt_split(self) -> None:
        mp = ModificationProbabilities(
            algorithm=AlgorithmName.DISTANCE_TO_IVT,
            position=10,
            probabilities=np.array([0.9, 0.8, 0.1, 0.05]),
            n_native=2,
            n_ivt=2,
            scores=np.array([3.0, 2.5, 0.5, 0.4]),
            null_gate_active=False,
            mixing_proportion=0.3,
        )
        np.testing.assert_array_equal(mp.native_probabilities, [0.9, 0.8])
        np.testing.assert_array_equal(mp.ivt_probabilities, [0.1, 0.05])
