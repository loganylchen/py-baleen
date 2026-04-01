"""Per-read modification probability estimation from DTW distance matrices.

This module provides multiple algorithms for converting pairwise DTW distance
matrices into per-read modification probabilities.  All algorithms share a
common calibration layer that fits a **null** distribution from IVT controls
and an **alternative** from native reads via Expectation-Maximisation (EM).

Algorithms
----------
1. ``distance_to_ivt`` — Robust median distance to IVT baseline.
3. ``knn_ivt_purity``  — k-NN IVT-purity score (best for partial modification).
5. ``mds_gmm``         — MDS embedding + anchored Gaussian mixture (full-matrix).

Public API
----------
compute_modification_probabilities
    Run one or all algorithms on a single :class:`PositionResult`.
ModificationProbabilities
    Container for per-read probabilities from a single algorithm.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm as _norm_dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_EPS = 1e-20  # avoid log(0) / division by zero
_MIN_SIGMA = 1e-6  # lower bound on fitted standard deviations


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class AlgorithmName(str, Enum):
    DISTANCE_TO_IVT = "distance_to_ivt"
    KNN_IVT_PURITY = "knn_ivt_purity"
    MDS_GMM = "mds_gmm"


@dataclass
class ModificationProbabilities:
    """Per-read modification probabilities from a single algorithm."""

    algorithm: AlgorithmName
    position: int
    probabilities: NDArray[np.float64]
    """Shape ``(n_native + n_ivt,)`` — native reads first, then IVT."""
    n_native: int
    n_ivt: int
    scores: NDArray[np.float64]
    """Raw scalar scores (before calibration).  Same ordering as *probabilities*."""
    null_gate_active: bool
    """True if the position was classified as unmodified (all probs set to 0)."""
    mixing_proportion: float
    """Fitted mixing proportion π for the alternative component."""

    @property
    def native_probabilities(self) -> NDArray[np.float64]:
        return self.probabilities[: self.n_native]

    @property
    def ivt_probabilities(self) -> NDArray[np.float64]:
        return self.probabilities[self.n_native :]


# ---------------------------------------------------------------------------
# Calibration layer — shared EM machinery
# ---------------------------------------------------------------------------


def _normal_pdf(x: NDArray[np.float64], mu: float, sigma: float) -> NDArray[np.float64]:
    """Vectorised Normal PDF."""
    z = (x - mu) / max(sigma, _MIN_SIGMA)
    return np.exp(-0.5 * z * z) / (max(sigma, _MIN_SIGMA) * math.sqrt(2 * math.pi))


def _normal_logpdf(x: NDArray[np.float64], mu: float, sigma: float) -> NDArray[np.float64]:
    """Vectorised Normal log-PDF (avoids underflow)."""
    sigma = max(sigma, _MIN_SIGMA)
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2 * math.pi)


def _mixture_log_likelihood(
    log_f0: NDArray[np.float64],
    log_f1: NDArray[np.float64],
    pi: float,
) -> float:
    """Compute sum(log((1-pi)*f0 + pi*f1)) in log-space via log-sum-exp.

    Avoids underflow when both f0 and f1 are extremely small.
    """
    log_w0 = math.log(max(1.0 - pi, _EPS))
    log_w1 = math.log(max(pi, _EPS))
    a = log_w0 + log_f0
    b = log_w1 + log_f1
    # log-sum-exp per element: log(exp(a) + exp(b))
    max_ab = np.maximum(a, b)
    log_sum = max_ab + np.log(np.exp(a - max_ab) + np.exp(b - max_ab))
    return float(np.sum(log_sum))


def _normal_log_likelihood(x: NDArray[np.float64], mu: float, sigma: float) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    sigma = max(sigma, _MIN_SIGMA)
    return float(-0.5 * n * math.log(2 * math.pi) - n * math.log(sigma)
                 - 0.5 * np.sum(((x - mu) / sigma) ** 2))


def _fit_normal(x: NDArray[np.float64]) -> tuple[float, float]:
    """Fit Normal(mu, sigma) from data."""
    mu = float(np.mean(x))
    sigma = max(float(np.std(x, ddof=0)), _MIN_SIGMA)
    return mu, sigma


def _beta_pdf(x: NDArray[np.float64], a: float, b: float) -> NDArray[np.float64]:
    """Vectorised Beta PDF with clamping."""
    from scipy.stats import beta as beta_dist
    x_safe = np.clip(x, 1e-10, 1.0 - 1e-10)
    return beta_dist.pdf(x_safe, a, b)


def _beta_logpdf(x: NDArray[np.float64], a: float, b: float) -> NDArray[np.float64]:
    """Vectorised Beta log-PDF (avoids underflow)."""
    from scipy.stats import beta as beta_dist
    x_safe = np.clip(x, 1e-10, 1.0 - 1e-10)
    return beta_dist.logpdf(x_safe, a, b)


def _beta_log_likelihood(x: NDArray[np.float64], a: float, b: float) -> float:
    return float(np.sum(_beta_logpdf(x, a, b)))


def _fit_beta(x: NDArray[np.float64]) -> tuple[float, float]:
    """Fit Beta(a, b) via method-of-moments."""
    x_safe = np.clip(x, 1e-10, 1.0 - 1e-10)
    m = float(np.mean(x_safe))
    v = float(np.var(x_safe, ddof=0))
    v = max(v, 1e-10)
    # Method-of-moments (capped to prevent Dirac-delta collapse)
    common = m * (1 - m) / v - 1.0
    common = max(min(common, 1000.0), 2.0)
    a = m * common
    b = (1 - m) * common
    return max(a, 0.1), max(b, 0.1)


@dataclass
class _CalibrationResult:
    probabilities: NDArray[np.float64]
    pi: float
    null_gate_active: bool


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid for scalar input."""
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _calibrate_normal(
    scores_all: NDArray[np.float64],
    ivt_mask: NDArray[np.bool_],
    native_mask: NDArray[np.bool_],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    pi_threshold: float = 0.05,
    separation_threshold: float = 0.5,
    tau_pi: float = 0.05,
    tau_bic: float = 10.0,
    tau_sep: float = 1.0,
) -> _CalibrationResult:
    """EM calibration with Normal null + Normal alternative.

    Parameters
    ----------
    scores_all : shape (N,)
    ivt_mask, native_mask : boolean masks into *scores_all*
    """
    ivt_scores = scores_all[ivt_mask]
    native_scores = scores_all[native_mask]

    if len(ivt_scores) < 2:
        return _CalibrationResult(
            probabilities=np.zeros_like(scores_all),
            pi=0.0,
            null_gate_active=True,
        )

    # Fit null from IVT
    mu0, sigma0 = _fit_normal(ivt_scores)

    if len(native_scores) < 2:
        probs = np.zeros_like(scores_all)
        return _CalibrationResult(probabilities=probs, pi=0.0, null_gate_active=True)

    # Initialise alternative
    pi = 0.1
    mu1 = max(float(np.mean(native_scores)), mu0 + 0.5 * sigma0)
    sigma1 = sigma0

    # EM on native scores only
    for _ in range(max_iter):
        f0 = _normal_pdf(native_scores, mu0, sigma0)
        f1 = _normal_pdf(native_scores, mu1, sigma1)
        denom = (1.0 - pi) * f0 + pi * f1 + _EPS
        r = (pi * f1) / denom  # responsibilities

        pi_new = float(np.mean(r))
        r_sum = float(np.sum(r)) + _EPS
        mu1_new = float(np.sum(r * native_scores)) / r_sum
        sigma1_new = math.sqrt(
            max(float(np.sum(r * (native_scores - mu1_new) ** 2)) / r_sum, _MIN_SIGMA**2)
        )

        converged = (
            abs(pi_new - pi) < tol
            and abs(mu1_new - mu1) < tol
            and abs(sigma1_new - sigma1) < tol
        )
        pi, mu1, sigma1 = pi_new, mu1_new, sigma1_new
        if converged:
            break

    # BIC gate (log-space to avoid underflow)
    n = len(native_scores)
    ll_null = _normal_log_likelihood(native_scores, mu0, sigma0)
    log_f0_mix = _normal_logpdf(native_scores, mu0, sigma0)
    log_f1_mix = _normal_logpdf(native_scores, mu1, sigma1)
    ll_mix = _mixture_log_likelihood(log_f0_mix, log_f1_mix, pi)
    bic_null = -2 * ll_null + 2 * math.log(max(n, 1))  # 2 params (mu0, sigma0)
    bic_mix = -2 * ll_mix + 5 * math.log(max(n, 1))  # 3 free params (pi, mu1, sigma1)

    # Effect-size separation
    separation = abs(mu1 - mu0) / max(sigma0, _MIN_SIGMA)

    # Soft gate: continuous weights ∈ [0, 1] via sigmoid (replaces hard gate)
    w_pi = _sigmoid((pi - pi_threshold) / tau_pi)
    w_bic = _sigmoid((bic_null - bic_mix) / tau_bic)
    w_sep = _sigmoid((separation - separation_threshold) / tau_sep)
    gate_weight = w_pi * w_bic * w_sep

    # Legacy hard gate flag (for reporting only)
    null_gate = pi < pi_threshold or bic_mix >= bic_null or separation < separation_threshold

    # Mixture posterior
    pi_post = min(max(pi, 0.01), 0.7)
    f0_all = _normal_pdf(scores_all, mu0, sigma0)
    f1_all = _normal_pdf(scores_all, mu1, sigma1)
    denom_all = (1.0 - pi_post) * f0_all + pi_post * f1_all + _EPS
    raw_posterior = (pi_post * f1_all) / denom_all

    # Z-score fallback for when gate is weak
    z_all = (scores_all - mu0) / max(sigma0, _MIN_SIGMA)
    fallback = 1.0 - _norm_dist.cdf(-z_all)

    # Blend: high gate_weight → trust mixture; low → fall back to z-score
    probs = gate_weight * raw_posterior + (1.0 - gate_weight) * fallback
    probs = np.clip(probs, 0.0, 1.0)

    return _CalibrationResult(probabilities=probs, pi=pi, null_gate_active=null_gate)


def _calibrate_beta(
    scores_all: NDArray[np.float64],
    ivt_mask: NDArray[np.bool_],
    native_mask: NDArray[np.bool_],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    pi_threshold: float = 0.05,
) -> _CalibrationResult:
    """EM calibration with Beta null + Beta alternative."""
    ivt_scores = scores_all[ivt_mask]
    native_scores = scores_all[native_mask]

    if len(ivt_scores) < 2:
        return _CalibrationResult(
            probabilities=np.zeros_like(scores_all),
            pi=0.0,
            null_gate_active=True,
        )

    a0, b0 = _fit_beta(ivt_scores)

    if len(native_scores) < 2:
        return _CalibrationResult(
            probabilities=np.zeros_like(scores_all),
            pi=0.0,
            null_gate_active=True,
        )

    # Initialise alternative — shifted toward higher scores
    pi = 0.1
    a1, b1 = _fit_beta(native_scores)
    # Nudge alternative mean higher than null mean if needed
    null_mean = a0 / (a0 + b0)
    alt_mean = a1 / (a1 + b1)
    if alt_mean <= null_mean + 0.05:
        a1 = max(a1, a0 + 0.5)

    for _ in range(max_iter):
        f0 = _beta_pdf(native_scores, a0, b0)
        f1 = _beta_pdf(native_scores, a1, b1)
        denom = (1.0 - pi) * f0 + pi * f1 + _EPS
        r = (pi * f1) / denom

        pi_new = float(np.mean(r))
        r_sum = float(np.sum(r)) + _EPS

        # Weighted method-of-moments for Beta alternative
        w_mean = float(np.sum(r * native_scores)) / r_sum
        w_var = float(np.sum(r * (native_scores - w_mean) ** 2)) / r_sum
        w_var = max(w_var, 1e-10)
        common = w_mean * (1 - w_mean) / w_var - 1.0
        common = max(min(common, 1000.0), 2.0)
        a1_new = max(w_mean * common, 0.1)
        b1_new = max((1 - w_mean) * common, 0.1)

        converged = (
            abs(pi_new - pi) < tol
            and abs(a1_new - a1) < tol
            and abs(b1_new - b1) < tol
        )
        pi, a1, b1 = pi_new, a1_new, b1_new
        if converged:
            break

    # BIC gate (log-space to avoid underflow)
    n = len(native_scores)
    ll_null = _beta_log_likelihood(native_scores, a0, b0)
    log_f0_n = _beta_logpdf(native_scores, a0, b0)
    log_f1_n = _beta_logpdf(native_scores, a1, b1)
    ll_mix = _mixture_log_likelihood(log_f0_n, log_f1_n, pi)
    bic_null = -2 * ll_null + 2 * math.log(max(n, 1))
    bic_mix = -2 * ll_mix + 5 * math.log(max(n, 1))

    null_mean = a0 / (a0 + b0)
    alt_mean_final = a1 / (a1 + b1)
    separation = abs(alt_mean_final - null_mean)
    null_gate = pi < pi_threshold or bic_mix >= bic_null or separation < 0.1

    if null_gate:
        return _CalibrationResult(
            probabilities=np.zeros_like(scores_all),
            pi=pi,
            null_gate_active=True,
        )

    # Clamp pi to prevent degenerate posteriors when pi→1.
    pi_post = min(max(pi, 0.01), 0.7)
    f0_all = _beta_pdf(scores_all, a0, b0)
    f1_all = _beta_pdf(scores_all, a1, b1)
    denom_all = (1.0 - pi_post) * f0_all + pi_post * f1_all + _EPS
    probs = (pi_post * f1_all) / denom_all
    probs = np.clip(probs, 0.0, 1.0)

    return _CalibrationResult(probabilities=probs, pi=pi, null_gate_active=False)


# ---------------------------------------------------------------------------
# Algorithm 1 — Robust Distance-to-IVT Baseline
# ---------------------------------------------------------------------------


def _score_distance_to_ivt(
    distance_matrix: NDArray[np.float64],
    n_native: int,
    n_ivt: int,
) -> NDArray[np.float64]:
    """Compute per-read scores: median distance to IVT controls.

    Returns log-transformed scores (larger = more likely modified).
    """
    n_total = n_native + n_ivt
    scores = np.zeros(n_total, dtype=np.float64)

    if n_ivt == 0:
        return np.log(scores + _EPS)

    # Native reads: median distance to all IVT columns
    if n_native > 0:
        scores[:n_native] = np.median(
            distance_matrix[:n_native, n_native:], axis=1,
        )

    # IVT reads: leave-one-out median distance to other IVT
    if n_ivt > 1:
        ivt_block = distance_matrix[n_native:, n_native:]
        # Set diagonal to NaN so nanmedian skips self-distance
        ivt_loo = ivt_block.copy()
        np.fill_diagonal(ivt_loo, np.nan)
        scores[n_native:] = np.nanmedian(ivt_loo, axis=1)
    # n_ivt == 1: leave-one-out has no others, score stays 0

    # Log-transform
    return np.log(scores + _EPS)


def distance_to_ivt(
    distance_matrix: NDArray[np.float64],
    n_native: int,
    n_ivt: int,
    *,
    max_iter: int = 100,
    pi_threshold: float = 0.05,
) -> ModificationProbabilities:
    """Algorithm 1: Robust Distance-to-IVT Baseline.

    Each read is scored by its median DTW distance to IVT controls,
    then calibrated via a Normal null + Normal alternative EM mixture.
    """
    n_total = n_native + n_ivt
    scores = _score_distance_to_ivt(distance_matrix, n_native, n_ivt)

    ivt_mask = np.zeros(n_total, dtype=bool)
    ivt_mask[n_native:] = True
    native_mask = ~ivt_mask

    cal = _calibrate_normal(scores, ivt_mask, native_mask,
                            max_iter=max_iter, pi_threshold=pi_threshold)

    return ModificationProbabilities(
        algorithm=AlgorithmName.DISTANCE_TO_IVT,
        position=-1,  # caller fills in
        probabilities=cal.probabilities,
        n_native=n_native,
        n_ivt=n_ivt,
        scores=scores,
        null_gate_active=cal.null_gate_active,
        mixing_proportion=cal.pi,
    )


# ---------------------------------------------------------------------------
# Algorithm 3 — kNN IVT-Purity Score
# ---------------------------------------------------------------------------


def _score_knn_ivt_purity(
    distance_matrix: NDArray[np.float64],
    n_native: int,
    n_ivt: int,
    k: Optional[int] = None,
    weighted: bool = False,
    ratio_correction: bool = True,
) -> NDArray[np.float64]:
    """Compute per-read kNN IVT-purity scores.

    Score = 1 - (IVT fraction among k nearest neighbors).
    Higher score = fewer IVT neighbors = more likely modified.
    Returns values in [0, 1].

    Parameters
    ----------
    weighted : bool
        If True, weight each neighbor by 1/distance instead of
        uniform counting.  Closer neighbors have more influence.
    ratio_correction : bool
        If True, correct for IVT/native sample size imbalance by
        adjusting the expected IVT fraction under null hypothesis.

    Uses rank-based normalization to ensure scores span [0, 1].
    """
    n_total = n_native + n_ivt
    if k is None:
        k = int(_clip(round(math.sqrt(n_total)), 3, 15))
    k = min(k, n_total - 1)

    is_ivt = np.zeros(n_total, dtype=bool)
    is_ivt[n_native:] = True

    # Work on a copy with diagonal set to inf (exclude self), once
    dm = distance_matrix.copy()
    np.fill_diagonal(dm, np.inf)

    # Find k nearest neighbors for all reads at once
    # argpartition along axis=1 gives indices of k smallest per row
    knn_idx = np.argpartition(dm, k, axis=1)[:, :k]

    raw_scores = np.empty(n_total, dtype=np.float64)

    if weighted:
        # Gather neighbor distances: shape (n_total, k)
        knn_dists = np.take_along_axis(dm, knn_idx, axis=1)
        weights = 1.0 / np.maximum(knn_dists, _EPS)
        # Check which neighbors are IVT: shape (n_total, k)
        neighbor_is_ivt = is_ivt[knn_idx]
        ivt_weight = np.sum(weights * neighbor_is_ivt, axis=1)
        total_weight = np.sum(weights, axis=1)
        raw_scores = 1.0 - ivt_weight / np.maximum(total_weight, _EPS)
    else:
        # Count IVT neighbors per read
        neighbor_is_ivt = is_ivt[knn_idx]
        ivt_count = np.sum(neighbor_is_ivt, axis=1)
        raw_scores = 1.0 - ivt_count / k

    # IVT-native ratio correction: adjust for sample size imbalance.
    # Under the null (no modification), the expected IVT fraction among
    # neighbors of read i is (n_ivt - 1{i is IVT}) / (n_total - 1).
    # We subtract this expected fraction so scores are centered around 0
    # under the null, then rescale back to [0, 1].
    if ratio_correction and n_total > 1:
        # Expected IVT fraction for each read (excluding self)
        expected_ivt_frac = np.where(
            is_ivt,
            (n_ivt - 1) / (n_total - 1),
            n_ivt / (n_total - 1),
        )
        # raw_scores = 1 - ivt_frac, so expected null score = 1 - expected_ivt_frac
        expected_null = 1.0 - expected_ivt_frac
        # Center around expected null, then shift to [0, 1]
        raw_scores = np.clip(
            0.5 + (raw_scores - expected_null), 0.0, 1.0,
        )

    # Rank-based normalization: map to [0, 1] using ranks
    # This ensures good spread regardless of the raw score distribution
    ranks = np.argsort(np.argsort(raw_scores)).astype(np.float64)
    scores = ranks / max(n_total - 1, 1)

    # Blend: 70% rank-based + 30% raw to preserve some absolute signal
    scores = 0.7 * scores + 0.3 * raw_scores

    return scores


def _score_lof(
    distance_matrix: NDArray[np.float64],
    n_native: int,
    n_ivt: int,
    k: Optional[int] = None,
) -> NDArray[np.float64]:
    """Compute Local Outlier Factor (LOF) anomaly scores.

    LOF measures how isolated a point is relative to its neighbors'
    local density.  LOF > 1 indicates the point is in a sparser region
    than its neighbors (i.e., more anomalous / more likely modified).

    The score is normalized to [0, 1] via rank-based normalization.

    Parameters
    ----------
    distance_matrix : (n_total, n_total)
        Pairwise DTW distance matrix.
    n_native, n_ivt : int
        Number of native / IVT reads.
    k : int, optional
        Number of neighbors.  Defaults to sqrt(n_total) clipped to [3, 15].

    Returns
    -------
    scores : (n_total,)
        LOF-based anomaly scores in [0, 1].  Higher = more anomalous.
    """
    n_total = n_native + n_ivt
    if k is None:
        k = int(_clip(round(math.sqrt(n_total)), 3, 15))
    k = min(k, n_total - 1)

    # Exclude self-distances
    dm = distance_matrix.copy()
    np.fill_diagonal(dm, np.inf)

    # k nearest neighbor indices and distances
    knn_idx = np.argpartition(dm, k, axis=1)[:, :k]  # (n_total, k)
    knn_dists = np.take_along_axis(dm, knn_idx, axis=1)  # (n_total, k)

    # k-distance: max distance among k nearest neighbors
    k_dist = np.max(knn_dists, axis=1)  # (n_total,)

    # Reachability distance: max(k_dist[neighbor], actual_dist)
    reach_dist = np.maximum(knn_dists, k_dist[knn_idx])  # (n_total, k)

    # Local reachability density: inverse of mean reachability distance
    lrd = 1.0 / np.maximum(np.mean(reach_dist, axis=1), _EPS)  # (n_total,)

    # LOF: mean ratio of neighbors' LRD to own LRD
    # LOF > 1 → point is in sparser region than neighbors
    neighbor_lrd = lrd[knn_idx]  # (n_total, k)
    lof = np.mean(neighbor_lrd, axis=1) / np.maximum(lrd, _EPS)  # (n_total,)

    # Rank-based normalization to [0, 1]
    ranks = np.argsort(np.argsort(lof)).astype(np.float64)
    scores = ranks / max(n_total - 1, 1)

    return scores


def knn_ivt_purity(
    distance_matrix: NDArray[np.float64],
    n_native: int,
    n_ivt: int,
    *,
    k: Optional[int] = None,
    lof_weight: float = 0.3,
    max_iter: int = 100,
    pi_threshold: float = 0.05,
) -> ModificationProbabilities:
    """Algorithm 3: kNN IVT-Purity Score with LOF blending.

    Each read is scored by how few of its k nearest neighbors are IVT
    controls, blended with a Local Outlier Factor (LOF) anomaly score,
    then calibrated via a Beta null + Beta alternative EM mixture.

    Parameters
    ----------
    lof_weight : float
        Weight for LOF score in the blend (0 = pure kNN, 1 = pure LOF).
        Default 0.3 gives 70% kNN purity + 30% LOF anomaly signal.
    """
    n_total = n_native + n_ivt
    knn_scores = _score_knn_ivt_purity(distance_matrix, n_native, n_ivt, k=k)

    # Blend with LOF anomaly score for density-aware detection
    if lof_weight > 0:
        lof_scores = _score_lof(distance_matrix, n_native, n_ivt, k=k)
        scores = (1.0 - lof_weight) * knn_scores + lof_weight * lof_scores
        # Clip to [0, 1] for Beta calibration
        scores = np.clip(scores, _EPS, 1.0 - _EPS)
    else:
        scores = knn_scores

    ivt_mask = np.zeros(n_total, dtype=bool)
    ivt_mask[n_native:] = True
    native_mask = ~ivt_mask

    cal = _calibrate_beta(scores, ivt_mask, native_mask,
                          max_iter=max_iter, pi_threshold=pi_threshold)

    return ModificationProbabilities(
        algorithm=AlgorithmName.KNN_IVT_PURITY,
        position=-1,
        probabilities=cal.probabilities,
        n_native=n_native,
        n_ivt=n_ivt,
        scores=scores,
        null_gate_active=cal.null_gate_active,
        mixing_proportion=cal.pi,
    )


# ---------------------------------------------------------------------------
# Algorithm 5 — MDS + Anchored Gaussian Mixture
# ---------------------------------------------------------------------------


def _classical_mds(
    distance_matrix: NDArray[np.float64],
    n_components: int = 2,
) -> NDArray[np.float64]:
    """Classical (metric) MDS from a symmetric distance matrix.

    Returns
    -------
    coords : shape (n, n_components)
    """
    n = distance_matrix.shape[0]
    D_sq = distance_matrix ** 2
    # Centering matrix J = I - 11'/n
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H

    # Eigendecomposition — take top positive eigenpairs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eigenvalues, eigenvectors = np.linalg.eigh(B)

    # eigh returns ascending; reverse to descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep top n_components positive eigenvalues
    n_pos = int(np.sum(eigenvalues > 0))
    r = min(n_components, n_pos, n - 1)
    if r == 0:
        return np.zeros((n, n_components), dtype=np.float64)

    coords = eigenvectors[:, :r] * np.sqrt(np.maximum(eigenvalues[:r], 0.0))[np.newaxis, :]

    # Pad with zeros if fewer than n_components
    if r < n_components:
        pad = np.zeros((n, n_components - r), dtype=np.float64)
        coords = np.hstack([coords, pad])

    return coords


def _fit_multivariate_normal(
    X: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit mean and covariance (with regularisation)."""
    mu = np.mean(X, axis=0)
    d = X.shape[1]
    if X.shape[0] < 2:
        return mu, np.eye(d) * 1.0
    cov = np.cov(X, rowvar=False, ddof=0)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    # Regularise
    cov += np.eye(d) * _MIN_SIGMA
    return mu, cov


def _mvn_pdf(
    X: NDArray[np.float64],
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Multivariate Normal PDF (log-space norm factor to avoid det overflow)."""
    d = len(mu)
    diff = X - mu
    try:
        cov_inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
    except np.linalg.LinAlgError:
        cov_reg = cov + np.eye(d) * 0.01
        cov_inv = np.linalg.inv(cov_reg)
        sign, logdet = np.linalg.slogdet(cov_reg)
    # If det is negative or zero, regularise
    if sign <= 0:
        cov_reg = cov + np.eye(d) * 0.01
        cov_inv = np.linalg.inv(cov_reg)
        _, logdet = np.linalg.slogdet(cov_reg)
    log_norm = -0.5 * (d * math.log(2 * math.pi) + logdet)
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    return np.exp(log_norm + exponent)


def mds_gmm(
    distance_matrix: NDArray[np.float64],
    n_native: int,
    n_ivt: int,
    *,
    n_components: int = 2,
    max_iter: int = 100,
    pi_threshold: float = 0.05,
) -> ModificationProbabilities:
    """Algorithm 5: MDS + Anchored Gaussian Mixture.

    Embed the full distance matrix into low-dimensional space via classical
    MDS, then fit an IVT-anchored null Gaussian and a native alternative
    Gaussian using EM.
    """
    n_total = n_native + n_ivt

    # Embed
    coords = _classical_mds(distance_matrix, n_components=n_components)

    ivt_coords = coords[n_native:]
    native_coords = coords[:n_native]

    # 1D score for the output (Euclidean distance to IVT centroid)
    ivt_centroid = np.mean(ivt_coords, axis=0) if n_ivt > 0 else np.zeros(n_components)
    scores_all = np.sqrt(np.sum((coords - ivt_centroid) ** 2, axis=1))

    if n_ivt < 2 or n_native < 2:
        return ModificationProbabilities(
            algorithm=AlgorithmName.MDS_GMM,
            position=-1,
            probabilities=np.zeros(n_total, dtype=np.float64),
            n_native=n_native,
            n_ivt=n_ivt,
            scores=scores_all,
            null_gate_active=True,
            mixing_proportion=0.0,
        )

    # Fit null from IVT
    mu0, sigma0 = _fit_multivariate_normal(ivt_coords)

    # Initialise alternative from native
    mu1, sigma1 = _fit_multivariate_normal(native_coords)
    pi = 0.1

    tol = 1e-6
    for _ in range(max_iter):
        f0 = _mvn_pdf(native_coords, mu0, sigma0)
        f1 = _mvn_pdf(native_coords, mu1, sigma1)
        denom = (1.0 - pi) * f0 + pi * f1 + _EPS
        r = (pi * f1) / denom

        pi_new = float(np.mean(r))
        r_sum = float(np.sum(r)) + _EPS
        mu1_new = np.sum(r[:, np.newaxis] * native_coords, axis=0) / r_sum

        diff = native_coords - mu1_new
        sigma1_new = (diff * r[:, np.newaxis]).T @ diff / r_sum
        sigma1_new += np.eye(n_components) * _MIN_SIGMA  # regularise

        if abs(pi_new - pi) < tol and np.max(np.abs(mu1_new - mu1)) < tol:
            pi, mu1, sigma1 = pi_new, mu1_new, sigma1_new
            break
        pi, mu1, sigma1 = pi_new, mu1_new, sigma1_new

    # BIC gate
    n = len(native_coords)
    d = n_components
    f0_n = _mvn_pdf(native_coords, mu0, sigma0)
    ll_null = float(np.sum(np.log(f0_n + _EPS)))
    f1_n = _mvn_pdf(native_coords, mu1, sigma1)
    ll_mix = float(np.sum(np.log((1 - pi) * f0_n + pi * f1_n + _EPS)))
    k_null = d + d * (d + 1) // 2
    k_mix = 1 + d + d * (d + 1) // 2
    bic_null = -2 * ll_null + k_null * math.log(max(n, 1))
    bic_mix = -2 * ll_mix + k_mix * math.log(max(n, 1))

    centroid_sep = float(np.sqrt(np.sum((mu1 - mu0) ** 2)))
    null_scale = float(np.sqrt(np.trace(sigma0) / d))
    separation = centroid_sep / max(null_scale, _MIN_SIGMA)
    null_gate = pi < pi_threshold or bic_mix >= bic_null or separation < 1.0

    if null_gate:
        return ModificationProbabilities(
            algorithm=AlgorithmName.MDS_GMM,
            position=-1,
            probabilities=np.zeros(n_total, dtype=np.float64),
            n_native=n_native,
            n_ivt=n_ivt,
            scores=scores_all,
            null_gate_active=True,
            mixing_proportion=pi,
        )

    # Clamp pi to prevent degenerate posteriors when pi→1.
    pi_post = min(max(pi, 0.01), 0.7)
    f0_all = _mvn_pdf(coords, mu0, sigma0)
    f1_all = _mvn_pdf(coords, mu1, sigma1)
    denom_all = (1.0 - pi_post) * f0_all + pi_post * f1_all + _EPS
    probs = (pi_post * f1_all) / denom_all
    probs = np.clip(probs, 0.0, 1.0)

    return ModificationProbabilities(
        algorithm=AlgorithmName.MDS_GMM,
        position=-1,
        probabilities=probs,
        n_native=n_native,
        n_ivt=n_ivt,
        scores=scores_all,
        null_gate_active=False,
        mixing_proportion=pi,
    )


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

# Avoid circular import — PositionResult is only needed for type checking.
# At runtime we duck-type to avoid importing _pipeline.
_ALGORITHM_DISPATCH = {
    AlgorithmName.DISTANCE_TO_IVT: distance_to_ivt,
    AlgorithmName.KNN_IVT_PURITY: knn_ivt_purity,
    AlgorithmName.MDS_GMM: mds_gmm,
}

ALL_ALGORITHMS: tuple[AlgorithmName, ...] = tuple(AlgorithmName)


def compute_modification_probabilities(
    distance_matrix: NDArray[np.float64],
    n_native: int,
    n_ivt: int,
    position: int = -1,
    *,
    algorithms: Optional[Sequence[AlgorithmName]] = None,
    knn_k: Optional[int] = None,
    mds_components: int = 2,
    max_iter: int = 100,
    pi_threshold: float = 0.05,
) -> dict[AlgorithmName, ModificationProbabilities]:
    """Run one or more algorithms on a distance matrix.

    Parameters
    ----------
    distance_matrix : shape (n_native + n_ivt, n_native + n_ivt)
        Symmetric DTW distance matrix.  Rows/columns ordered:
        native reads first, then IVT reads.
    n_native, n_ivt : int
        Number of native and IVT reads.
    position : int
        Genomic position (for labelling only).
    algorithms : sequence of AlgorithmName, optional
        Which algorithms to run.  Defaults to all three.
    knn_k : int, optional
        k for kNN algorithm.  ``None`` = auto.
    mds_components : int
        Embedding dimensionality for MDS+GMM.
    max_iter : int
        Maximum EM iterations.
    pi_threshold : float
        Null-gate threshold on mixing proportion.

    Returns
    -------
    dict mapping AlgorithmName → ModificationProbabilities
    """
    if algorithms is None:
        algorithms = ALL_ALGORITHMS

    n_total = n_native + n_ivt
    if distance_matrix.shape != (n_total, n_total):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} does not match "
            f"n_native={n_native} + n_ivt={n_ivt} = {n_total}"
        )

    results: dict[AlgorithmName, ModificationProbabilities] = {}

    for alg in algorithms:
        if alg == AlgorithmName.DISTANCE_TO_IVT:
            mp = distance_to_ivt(
                distance_matrix, n_native, n_ivt,
                max_iter=max_iter, pi_threshold=pi_threshold,
            )
        elif alg == AlgorithmName.KNN_IVT_PURITY:
            mp = knn_ivt_purity(
                distance_matrix, n_native, n_ivt,
                k=knn_k, max_iter=max_iter, pi_threshold=pi_threshold,
            )
        elif alg == AlgorithmName.MDS_GMM:
            mp = mds_gmm(
                distance_matrix, n_native, n_ivt,
                n_components=mds_components, max_iter=max_iter,
                pi_threshold=pi_threshold,
            )
        else:
            raise ValueError(f"Unknown algorithm: {alg}")

        mp.position = position
        results[alg] = mp

    return results
