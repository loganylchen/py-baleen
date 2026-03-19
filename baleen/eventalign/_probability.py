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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_EPS = 1e-300  # avoid log(0)
_MIN_SIGMA = 1e-6  # lower bound on fitted standard deviations
_NULL_GATE_CEILING = 0.1  # max probability when null gate fires


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


def _beta_log_likelihood(x: NDArray[np.float64], a: float, b: float) -> float:
    from scipy.stats import beta as beta_dist
    x_safe = np.clip(x, 1e-10, 1.0 - 1e-10)
    return float(np.sum(beta_dist.logpdf(x_safe, a, b)))


def _fit_beta(x: NDArray[np.float64]) -> tuple[float, float]:
    """Fit Beta(a, b) via method-of-moments."""
    x_safe = np.clip(x, 1e-10, 1.0 - 1e-10)
    m = float(np.mean(x_safe))
    v = float(np.var(x_safe, ddof=0))
    v = max(v, 1e-10)
    # Method-of-moments
    common = m * (1 - m) / v - 1.0
    common = max(common, 2.0)  # ensure a, b > 0
    a = m * common
    b = (1 - m) * common
    return max(a, 0.1), max(b, 0.1)


@dataclass
class _CalibrationResult:
    probabilities: NDArray[np.float64]
    pi: float
    null_gate_active: bool


def _calibrate_normal(
    scores_all: NDArray[np.float64],
    ivt_mask: NDArray[np.bool_],
    native_mask: NDArray[np.bool_],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    pi_threshold: float = 0.05,
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

        if abs(pi_new - pi) < tol and abs(mu1_new - mu1) < tol:
            pi, mu1, sigma1 = pi_new, mu1_new, sigma1_new
            break
        pi, mu1, sigma1 = pi_new, mu1_new, sigma1_new

    # BIC gate
    n = len(native_scores)
    ll_null = _normal_log_likelihood(native_scores, mu0, sigma0)
    f0_mix = _normal_pdf(native_scores, mu0, sigma0)
    f1_mix = _normal_pdf(native_scores, mu1, sigma1)
    ll_mix = float(np.sum(np.log((1 - pi) * f0_mix + pi * f1_mix + _EPS)))
    bic_null = -2 * ll_null + 2 * math.log(max(n, 1))  # 2 params (mu0, sigma0)
    bic_mix = -2 * ll_mix + 5 * math.log(max(n, 1))  # 3 free params (pi, mu1, sigma1)

    # Effect-size gate: the alternative mean must be meaningfully separated from null
    separation = abs(mu1 - mu0) / max(sigma0, _MIN_SIGMA)
    null_gate = pi < pi_threshold or bic_mix >= bic_null or separation < 1.0

    # Compute posteriors using pi-weighted prior from EM.
    f0_all = _normal_pdf(scores_all, mu0, sigma0)
    f1_all = _normal_pdf(scores_all, mu1, sigma1)
    denom_all = (1.0 - pi) * f0_all + pi * f1_all + _EPS
    probs = (pi * f1_all) / denom_all
    probs = np.clip(probs, 0.0, 1.0)

    if null_gate:
        # Soft gate: cap probabilities to preserve ranking information
        probs = np.minimum(probs, _NULL_GATE_CEILING)
        return _CalibrationResult(probabilities=probs, pi=pi, null_gate_active=True)

    return _CalibrationResult(probabilities=probs, pi=pi, null_gate_active=False)


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
        common = max(common, 2.0)
        a1_new = max(w_mean * common, 0.1)
        b1_new = max((1 - w_mean) * common, 0.1)

        if abs(pi_new - pi) < tol:
            pi, a1, b1 = pi_new, a1_new, b1_new
            break
        pi, a1, b1 = pi_new, a1_new, b1_new

    # BIC gate
    n = len(native_scores)
    ll_null = _beta_log_likelihood(native_scores, a0, b0)
    f0_n = _beta_pdf(native_scores, a0, b0)
    f1_n = _beta_pdf(native_scores, a1, b1)
    ll_mix = float(np.sum(np.log((1 - pi) * f0_n + pi * f1_n + _EPS)))
    bic_null = -2 * ll_null + 2 * math.log(max(n, 1))
    bic_mix = -2 * ll_mix + 5 * math.log(max(n, 1))

    null_mean = a0 / (a0 + b0)
    alt_mean_final = a1 / (a1 + b1)
    separation = abs(alt_mean_final - null_mean)
    null_gate = pi < pi_threshold or bic_mix >= bic_null or separation < 0.1

    f0_all = _beta_pdf(scores_all, a0, b0)
    f1_all = _beta_pdf(scores_all, a1, b1)
    denom_all = (1.0 - pi) * f0_all + pi * f1_all + _EPS
    probs = (pi * f1_all) / denom_all
    probs = np.clip(probs, 0.0, 1.0)

    if null_gate:
        probs = np.minimum(probs, _NULL_GATE_CEILING)
        return _CalibrationResult(probabilities=probs, pi=pi, null_gate_active=True)

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
    scores = np.empty(n_total, dtype=np.float64)
    ivt_indices = np.arange(n_native, n_total)

    for i in range(n_total):
        if i >= n_native:
            # IVT read: leave-one-out
            others = ivt_indices[ivt_indices != i]
        else:
            others = ivt_indices
        if len(others) == 0:
            scores[i] = 0.0
        else:
            scores[i] = float(np.median(distance_matrix[i, others]))

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
) -> NDArray[np.float64]:
    """Compute per-read kNN IVT-purity scores.

    Score = 1 - (weighted IVT fraction among k nearest neighbors).
    Higher score = fewer IVT neighbors = more likely modified.
    Returns values in [0, 1].
    """
    n_total = n_native + n_ivt
    if k is None:
        k = int(_clip(round(math.sqrt(n_total)), 3, 10))
    k = min(k, n_total - 1)  # can't have more neighbors than other reads

    # Bandwidth for weighting
    off_diag = distance_matrix[np.triu_indices(n_total, k=1)]
    eta = float(np.median(off_diag)) if len(off_diag) > 0 else 1.0
    eta = max(eta, _MIN_SIGMA)

    is_ivt = np.zeros(n_total, dtype=bool)
    is_ivt[n_native:] = True

    scores = np.empty(n_total, dtype=np.float64)

    for i in range(n_total):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf  # exclude self
        neighbor_idx = np.argsort(dists)[:k]

        weights = np.exp(-dists[neighbor_idx] / eta)
        ivt_weight = float(np.sum(weights[is_ivt[neighbor_idx]]))
        total_weight = float(np.sum(weights)) + _EPS

        a_i = ivt_weight / total_weight  # IVT affinity
        scores[i] = 1.0 - a_i  # high = few IVT neighbors

    return scores


def knn_ivt_purity(
    distance_matrix: NDArray[np.float64],
    n_native: int,
    n_ivt: int,
    *,
    k: Optional[int] = None,
    max_iter: int = 100,
    pi_threshold: float = 0.05,
) -> ModificationProbabilities:
    """Algorithm 3: kNN IVT-Purity Score.

    Each read is scored by how few of its k nearest neighbors are IVT
    controls, then calibrated via a Beta null + Beta alternative EM mixture.
    """
    n_total = n_native + n_ivt
    scores = _score_knn_ivt_purity(distance_matrix, n_native, n_ivt, k=k)

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
    """Multivariate Normal PDF."""
    d = len(mu)
    diff = X - mu
    try:
        cov_inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
    except np.linalg.LinAlgError:
        cov_reg = cov + np.eye(d) * 0.01
        cov_inv = np.linalg.inv(cov_reg)
        det = np.linalg.det(cov_reg)
    det = max(det, _EPS)
    norm = 1.0 / (math.sqrt((2 * math.pi) ** d * det))
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    return norm * np.exp(exponent)


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

    f0_all = _mvn_pdf(coords, mu0, sigma0)
    f1_all = _mvn_pdf(coords, mu1, sigma1)
    denom_all = (1.0 - pi) * f0_all + pi * f1_all + _EPS
    probs = (pi * f1_all) / denom_all
    probs = np.clip(probs, 0.0, 1.0)

    if null_gate:
        probs = np.minimum(probs, _NULL_GATE_CEILING)

    return ModificationProbabilities(
        algorithm=AlgorithmName.MDS_GMM,
        position=-1,
        probabilities=probs,
        n_native=n_native,
        n_ivt=n_ivt,
        scores=scores_all,
        null_gate_active=null_gate,
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
