"""Contig-level hierarchical empirical-Bayes modification probability estimation.

This module implements a **three-stage** pipeline that borrows strength across
genomic positions within a contig and smooths modification calls along
individual reads via a Hidden Markov Model.

Stages
------
V1  Empirical-Bayes null scoring
    Robust IVT null (median + MAD) per position, shrunk toward a local
    window and global prior.  Produces z-scores and one-sided p-values.

V2  Anchored two-component mixture
    EM on native z-scores with the null component fixed to IVT; a shifted
    alternative captures modified reads.  Produces raw P(mod).

V3  HMM spatial smoothing
    Two-state forward–backward along each read's genomic trajectory, with
    gap-aware transition probabilities.  Produces smoothed P(mod).

Public API
----------
compute_sequential_modification_probabilities
    Run the full V1→V2→V3 pipeline on a :class:`ContigResult`.
ContigModificationResult
    Container for all per-read, per-position outputs.
CoverageClass
    Enum labelling IVT coverage at each position.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm as _norm_dist

if TYPE_CHECKING:
    from baleen.eventalign._hmm_training import HMMParams
    from baleen.eventalign._pipeline import ContigResult, PositionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-300
_MIN_SIGMA = 1e-6
_MAD_SCALE = 1.4826  # MAD → σ for Normal


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class CoverageClass(str, Enum):
    """IVT coverage quality at a position."""
    HIGH = "high"        # n_ivt >= 20
    MEDIUM = "medium"    # 5 <= n_ivt < 20
    LOW = "low"          # 1 <= n_ivt < 5
    ZERO = "zero"        # n_ivt == 0


@dataclass
class PositionStats:
    """Per-position summary from the hierarchical pipeline."""

    position: int
    reference_kmer: str
    coverage_class: CoverageClass
    n_ivt: int
    n_native: int

    # V1 — robust IVT null (before and after shrinkage)
    mu_raw: float
    sigma_raw: float
    mu_shrunk: float
    sigma_shrunk: float

    # Per-read arrays (native reads first, then IVT — same ordering as
    # PositionResult.distance_matrix)
    scores: NDArray[np.float64]
    """log1p(median distance to IVT), shape (n_native + n_ivt,)."""
    z_scores: NDArray[np.float64]
    """(score − μ_shrunk) / σ_shrunk, shape (n_native + n_ivt,)."""
    p_null: NDArray[np.float64]
    """One-sided tail probability under null, shape (n_native + n_ivt,)."""

    # V2 — mixture posterior (may be all zeros if null gate fires)
    p_mod_raw: NDArray[np.float64]
    """Raw P(mod) from anchored mixture, shape (n_native + n_ivt,)."""
    mixture_pi: float
    """Fitted mixing proportion for alt component."""
    mixture_null_gate: bool
    """True if position classified as unmodified by mixture."""

    # V3 — HMM smoothed (populated only if HMM runs)
    p_mod_hmm: NDArray[np.float64]
    """HMM-smoothed P(mod), shape (n_native + n_ivt,).  NaN if HMM skipped."""

    @property
    def native_z_scores(self) -> NDArray[np.float64]:
        return self.z_scores[: self.n_native]

    @property
    def ivt_z_scores(self) -> NDArray[np.float64]:
        return self.z_scores[self.n_native :]

    @property
    def native_p_mod_raw(self) -> NDArray[np.float64]:
        return self.p_mod_raw[: self.n_native]

    @property
    def ivt_p_mod_raw(self) -> NDArray[np.float64]:
        return self.p_mod_raw[self.n_native :]

    @property
    def native_p_mod_hmm(self) -> NDArray[np.float64]:
        return self.p_mod_hmm[: self.n_native]

    @property
    def ivt_p_mod_hmm(self) -> NDArray[np.float64]:
        return self.p_mod_hmm[self.n_native :]


@dataclass
class ReadTrajectory:
    """A single read's observation path across sorted genomic positions."""

    read_name: str
    positions: list[int]
    """Sorted genomic positions where this read appears."""
    is_native: bool
    indices: list[int]
    """Index of this read within each position's read ordering
    (native-first, IVT-after)."""


@dataclass
class ContigModificationResult:
    """Full output of the hierarchical pipeline for one contig."""

    contig: str
    position_stats: dict[int, PositionStats]
    """Keyed by genomic position."""
    native_trajectories: list[ReadTrajectory]
    ivt_trajectories: list[ReadTrajectory]

    # Global IVT prior used for shrinkage
    global_mu: float
    global_sigma: float


# ---------------------------------------------------------------------------
# V1 helpers — score extraction
# ---------------------------------------------------------------------------


def _extract_ivt_distances(
    distance_matrix: NDArray[np.float64],
    n_native: int,
    n_ivt: int,
) -> NDArray[np.float64]:
    """Per-read median distance to IVT controls, log1p-transformed.

    Same semantics as ``_probability._score_distance_to_ivt`` but uses
    ``log1p`` instead of ``log`` for better numerical behaviour near zero.

    Returns shape ``(n_native + n_ivt,)``.
    """
    n_total = n_native + n_ivt
    ivt_indices = np.arange(n_native, n_total)
    scores = np.empty(n_total, dtype=np.float64)

    for i in range(n_total):
        if i >= n_native:
            # IVT read — leave-one-out
            others = ivt_indices[ivt_indices != i]
        else:
            others = ivt_indices

        if len(others) == 0:
            scores[i] = 0.0
        else:
            scores[i] = float(np.median(distance_matrix[i, others]))

    return np.log1p(scores)


# ---------------------------------------------------------------------------
# V1 helpers — robust IVT null estimation
# ---------------------------------------------------------------------------


def _classify_coverage(n_ivt: int) -> CoverageClass:
    """Classify IVT coverage at a position."""
    if n_ivt >= 20:
        return CoverageClass.HIGH
    if n_ivt >= 5:
        return CoverageClass.MEDIUM
    if n_ivt >= 1:
        return CoverageClass.LOW
    return CoverageClass.ZERO


def _fit_robust_ivt_null(
    ivt_scores: NDArray[np.float64],
) -> tuple[float, float]:
    """Fit location and scale from IVT scores using median + MAD.

    Returns (mu, sigma) where sigma = MAD * 1.4826.
    Falls back to (0, 1) if < 2 IVT reads.
    """
    if len(ivt_scores) < 2:
        return 0.0, 1.0

    mu = float(np.median(ivt_scores))
    mad = float(np.median(np.abs(ivt_scores - mu)))
    sigma = max(mad * _MAD_SCALE, _MIN_SIGMA)
    return mu, sigma


# ---------------------------------------------------------------------------
# V1 helpers — hierarchical shrinkage
# ---------------------------------------------------------------------------


def _shrink_parameters(
    positions: list[int],
    raw_params: dict[int, tuple[float, float]],
    coverages: dict[int, int],
    *,
    window: int = 15,
    kappa_high: float = 0.5,
    kappa_medium: float = 2.0,
    kappa_low: float = 5.0,
) -> dict[int, tuple[float, float]]:
    """Hierarchical shrinkage: position → local window → global prior.

    For each position *j* with raw (μ_j, σ_j) and n_ivt_j IVT reads:

        κ = coverage-dependent shrinkage strength
        μ_local = weighted median of nearby raw μ values
        σ_local = weighted median of nearby raw σ values

        μ̂_j = (n_j * μ_j + κ * μ_local) / (n_j + κ)
        σ̂_j = (n_j * σ_j + κ * σ_local) / (n_j + κ)

    For ZERO-coverage positions, μ̂ = μ_local, σ̂ = σ_local entirely.

    Parameters
    ----------
    positions : sorted list of genomic positions
    raw_params : {pos: (mu_raw, sigma_raw)}
    coverages : {pos: n_ivt}
    window : half-window size in positions (not bases)
    kappa_high, kappa_medium, kappa_low : shrinkage strengths

    Returns
    -------
    {pos: (mu_shrunk, sigma_shrunk)}
    """
    sorted_pos = sorted(positions)
    n_pos = len(sorted_pos)

    # Global prior (fallback for everything)
    all_mus = [raw_params[p][0] for p in sorted_pos if coverages[p] >= 1]
    all_sigmas = [raw_params[p][1] for p in sorted_pos if coverages[p] >= 1]

    if len(all_mus) == 0:
        global_mu, global_sigma = 0.0, 1.0
    else:
        global_mu = float(np.median(all_mus))
        global_sigma = float(np.median(all_sigmas))

    result: dict[int, tuple[float, float]] = {}

    for idx, pos in enumerate(sorted_pos):
        n_j = coverages[pos]
        mu_j, sigma_j = raw_params[pos]

        # Determine κ
        cov_cls = _classify_coverage(n_j)
        if cov_cls == CoverageClass.HIGH:
            kappa = kappa_high
        elif cov_cls == CoverageClass.MEDIUM:
            kappa = kappa_medium
        elif cov_cls == CoverageClass.LOW:
            kappa = kappa_low
        else:
            # ZERO coverage — fully rely on local/global
            kappa = float("inf")

        # Local window — gather neighbours (by position index, not base)
        lo = max(0, idx - window)
        hi = min(n_pos, idx + window + 1)
        nbr_mus = []
        nbr_sigmas = []
        for k_idx in range(lo, hi):
            nbr_pos = sorted_pos[k_idx]
            if nbr_pos != pos and coverages[nbr_pos] >= 1:
                nbr_mus.append(raw_params[nbr_pos][0])
                nbr_sigmas.append(raw_params[nbr_pos][1])

        if len(nbr_mus) > 0:
            mu_local = float(np.median(nbr_mus))
            sigma_local = float(np.median(nbr_sigmas))
        else:
            mu_local = global_mu
            sigma_local = global_sigma

        # Shrink
        if cov_cls == CoverageClass.ZERO:
            mu_shrunk = mu_local
            sigma_shrunk = sigma_local
        else:
            mu_shrunk = (n_j * mu_j + kappa * mu_local) / (n_j + kappa)
            sigma_shrunk = (n_j * sigma_j + kappa * sigma_local) / (n_j + kappa)

        sigma_shrunk = max(sigma_shrunk, _MIN_SIGMA)
        result[pos] = (mu_shrunk, sigma_shrunk)

    return result


# ---------------------------------------------------------------------------
# V2 — Anchored two-component mixture on native z-scores
# ---------------------------------------------------------------------------


def _anchored_mixture_em(
    z_native: NDArray[np.float64],
    z_ivt: NDArray[np.float64],
    z_all: NDArray[np.float64],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    pi_threshold: float = 0.05,
    separation_threshold: float = 0.8,
) -> tuple[NDArray[np.float64], float, bool]:
    """EM with null fixed to IVT distribution, alternative free.

    The null component is fitted from IVT z-scores (should be ~N(0,1) if
    shrinkage worked correctly).  The alternative is initialised shifted
    right and fitted via EM on native z-scores only.

    Parameters
    ----------
    z_native : z-scores of native reads at this position
    z_ivt : z-scores of IVT reads at this position
    z_all : z-scores of ALL reads (native + IVT, in that order)

    Returns
    -------
    (p_mod_all, pi, null_gate_active)
        p_mod_all has shape matching z_all
    """
    from baleen.eventalign._probability import (
        _normal_pdf, _normal_log_likelihood, _EPS as _PROB_EPS,
    )

    if len(z_ivt) < 2 or len(z_native) < 2:
        return np.zeros_like(z_all), 0.0, True

    # Null from IVT z-scores (should be centred near 0)
    mu0 = float(np.median(z_ivt))
    mad0 = float(np.median(np.abs(z_ivt - mu0)))
    sigma0 = max(mad0 * _MAD_SCALE, _MIN_SIGMA)

    # Init alternative — shifted right
    pi = 0.1
    mu1 = max(float(np.mean(z_native)), mu0 + 0.5 * sigma0)
    sigma1 = sigma0

    for _ in range(max_iter):
        f0 = _normal_pdf(z_native, mu0, sigma0)
        f1 = _normal_pdf(z_native, mu1, sigma1)
        denom = (1.0 - pi) * f0 + pi * f1 + _EPS
        r = (pi * f1) / denom

        pi_new = float(np.mean(r))
        r_sum = float(np.sum(r)) + _EPS
        mu1_new = float(np.sum(r * z_native)) / r_sum
        sigma1_new = math.sqrt(
            max(float(np.sum(r * (z_native - mu1_new) ** 2)) / r_sum,
                _MIN_SIGMA ** 2)
        )

        if abs(pi_new - pi) < tol and abs(mu1_new - mu1) < tol:
            pi, mu1, sigma1 = pi_new, mu1_new, sigma1_new
            break
        pi, mu1, sigma1 = pi_new, mu1_new, sigma1_new

    # Gates
    separation = abs(mu1 - mu0) / max(sigma0, _MIN_SIGMA)
    n = len(z_native)

    ll_null = _normal_log_likelihood(z_native, mu0, sigma0)
    f0_n = _normal_pdf(z_native, mu0, sigma0)
    f1_n = _normal_pdf(z_native, mu1, sigma1)
    ll_mix = float(np.sum(np.log((1 - pi) * f0_n + pi * f1_n + _EPS)))
    bic_null = -2 * ll_null + 2 * math.log(max(n, 1))
    bic_mix = -2 * ll_mix + 5 * math.log(max(n, 1))

    null_gate = (
        pi < pi_threshold
        or bic_mix >= bic_null
        or separation < separation_threshold
    )

    if null_gate:
        return np.zeros_like(z_all), pi, True

    # Posteriors using pi-weighted prior from EM
    f0_all = _normal_pdf(z_all, mu0, sigma0)
    f1_all = _normal_pdf(z_all, mu1, sigma1)
    denom_all = (1.0 - pi) * f0_all + pi * f1_all + _EPS
    probs = (pi * f1_all) / denom_all
    return np.clip(probs, 0.0, 1.0), pi, False


# ---------------------------------------------------------------------------
# V3 — Read trajectory construction
# ---------------------------------------------------------------------------


def _build_read_trajectories(
    contig_result: ContigResult,
    sorted_positions: list[int],
) -> tuple[list[ReadTrajectory], list[ReadTrajectory]]:
    """Build per-read trajectories across positions.

    Returns (native_trajectories, ivt_trajectories).
    """
    # Map: read_name → [(position, index_in_that_position, is_native)]
    native_map: dict[str, list[tuple[int, int]]] = {}
    ivt_map: dict[str, list[tuple[int, int]]] = {}

    for pos in sorted_positions:
        pr = contig_result.positions[pos]
        for idx, name in enumerate(pr.native_read_names):
            native_map.setdefault(name, []).append((pos, idx))
        for idx, name in enumerate(pr.ivt_read_names):
            # Index in the combined array = n_native + idx
            ivt_map.setdefault(name, []).append((pos, pr.n_native_reads + idx))

    native_trajs = []
    for name, entries in native_map.items():
        entries.sort(key=lambda x: x[0])
        native_trajs.append(ReadTrajectory(
            read_name=name,
            positions=[e[0] for e in entries],
            is_native=True,
            indices=[e[1] for e in entries],
        ))

    ivt_trajs = []
    for name, entries in ivt_map.items():
        entries.sort(key=lambda x: x[0])
        ivt_trajs.append(ReadTrajectory(
            read_name=name,
            positions=[e[0] for e in entries],
            is_native=False,
            indices=[e[1] for e in entries],
        ))

    return native_trajs, ivt_trajs


# ---------------------------------------------------------------------------
# V3 — Forward-backward HMM (pure NumPy)
# ---------------------------------------------------------------------------


def _gap_transition_matrix(
    gap_bases: int,
    p_stay_per_base: float = 0.98,
) -> NDArray[np.float64]:
    """Compute 2×2 transition matrix accounting for genomic gap.

    State 0 = unmodified, State 1 = modified.

    For a gap of *g* bases, the probability of remaining in the same state
    is ``p_stay_per_base ** g``, so larger gaps relax spatial correlation.

    Returns shape (2, 2) row-stochastic matrix.
    """
    p_stay = p_stay_per_base ** max(gap_bases, 1)
    p_stay = max(min(p_stay, 1.0 - _EPS), _EPS)
    p_switch = 1.0 - p_stay
    return np.array([
        [p_stay, p_switch],
        [p_switch, p_stay],
    ], dtype=np.float64)


def _forward_backward(
    emissions: NDArray[np.float64],
    positions: list[int],
    *,
    init_prob: NDArray[np.float64] | None = None,
    p_stay_per_base: float = 0.98,
) -> NDArray[np.float64]:
    """Two-state forward-backward on a read trajectory.

    Parameters
    ----------
    emissions : shape (T, 2)
        emissions[t, 0] = P(observation | unmodified)
        emissions[t, 1] = P(observation | modified)
    positions : length T, sorted genomic positions
    init_prob : shape (2,), initial state distribution (default: [0.5, 0.5])
    p_stay_per_base : per-base probability of staying in the same state

    Returns
    -------
    posteriors : shape (T,) — P(modified | all observations)
    """
    T = emissions.shape[0]
    if T == 0:
        return np.array([], dtype=np.float64)

    if init_prob is None:
        init_prob = np.array([0.5, 0.5], dtype=np.float64)

    # Forward pass
    alpha = np.zeros((T, 2), dtype=np.float64)
    scale = np.zeros(T, dtype=np.float64)

    alpha[0] = init_prob * emissions[0]
    scale[0] = alpha[0].sum()
    if scale[0] > 0:
        alpha[0] /= scale[0]
    else:
        alpha[0] = np.array([0.5, 0.5])
        scale[0] = _EPS

    for t in range(1, T):
        gap = positions[t] - positions[t - 1]
        trans = _gap_transition_matrix(gap, p_stay_per_base)
        alpha[t] = (alpha[t - 1] @ trans) * emissions[t]
        scale[t] = alpha[t].sum()
        if scale[t] > 0:
            alpha[t] /= scale[t]
        else:
            alpha[t] = np.array([0.5, 0.5])
            scale[t] = _EPS

    # Backward pass
    beta = np.zeros((T, 2), dtype=np.float64)
    beta[T - 1] = np.array([1.0, 1.0])

    for t in range(T - 2, -1, -1):
        gap = positions[t + 1] - positions[t]
        trans = _gap_transition_matrix(gap, p_stay_per_base)
        beta[t] = trans @ (emissions[t + 1] * beta[t + 1])
        bt_sum = beta[t].sum()
        if bt_sum > 0:
            beta[t] /= bt_sum
        else:
            beta[t] = np.array([1.0, 1.0])

    # Posterior
    gamma = alpha * beta
    gamma_sum = gamma.sum(axis=1, keepdims=True)
    gamma_sum = np.maximum(gamma_sum, _EPS)
    gamma /= gamma_sum

    return gamma[:, 1]  # P(state=modified)


def _run_hmm_on_trajectories(
    trajectories: list[ReadTrajectory],
    position_stats: dict[int, PositionStats],
    *,
    min_positions: int = 3,
    p_stay_per_base: float = 0.98,
    hmm_params: HMMParams | None = None,
) -> None:
    """Run forward-backward on each trajectory and write p_mod_hmm in-place.

    For each read at each position, the HMM posterior replaces the
    ``p_mod_hmm`` entry in the corresponding :class:`PositionStats`.

    Reads with fewer than *min_positions* observations are skipped (their
    p_mod_hmm remains as the V2 p_mod_raw fallback).

    When *hmm_params* is provided, its ``p_stay_per_base``, ``init_prob``,
    and ``emission_transform`` override the defaults.
    """
    # Resolve HMM parameters
    effective_p_stay = hmm_params.p_stay_per_base if hmm_params else p_stay_per_base
    effective_init = hmm_params.init_prob if hmm_params else None
    emission_transform = hmm_params.emission_transform if hmm_params else None

    for traj in trajectories:
        if len(traj.positions) < min_positions:
            continue

        T = len(traj.positions)
        emissions = np.zeros((T, 2), dtype=np.float64)

        for t_idx, (pos, read_idx) in enumerate(
            zip(traj.positions, traj.indices)
        ):
            ps = position_stats[pos]
            p_mod = ps.p_mod_raw[read_idx]

            if emission_transform is None:
                # Default: use raw p_mod directly
                p_mod_safe = max(min(p_mod, 1.0 - 1e-10), 1e-10)
                emissions[t_idx, 0] = 1.0 - p_mod_safe
                emissions[t_idx, 1] = p_mod_safe
            else:
                # Lazy import to resolve type at runtime
                from baleen.eventalign._hmm_training import (
                    EmissionCalibrator,
                    EmissionKDE,
                )

                if isinstance(emission_transform, EmissionCalibrator):
                    calibrated = emission_transform.transform(
                        np.array([p_mod])
                    )[0]
                    calibrated = max(min(calibrated, 1.0 - 1e-10), 1e-10)
                    emissions[t_idx, 0] = 1.0 - calibrated
                    emissions[t_idx, 1] = calibrated
                elif isinstance(emission_transform, EmissionKDE):
                    p_unmod, p_mod_kde = emission_transform.emission_probs(
                        np.array([p_mod])
                    )
                    emissions[t_idx, 0] = p_unmod[0]
                    emissions[t_idx, 1] = p_mod_kde[0]

        posteriors = _forward_backward(
            emissions,
            traj.positions,
            init_prob=effective_init,
            p_stay_per_base=effective_p_stay,
        )

        # Write back
        for t_idx, (pos, read_idx) in enumerate(
            zip(traj.positions, traj.indices)
        ):
            position_stats[pos].p_mod_hmm[read_idx] = posteriors[t_idx]


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def compute_sequential_modification_probabilities(
    contig_result: ContigResult,
    *,
    shrinkage_window: int = 15,
    kappa_high: float = 0.5,
    kappa_medium: float = 2.0,
    kappa_low: float = 5.0,
    mixture_max_iter: int = 100,
    mixture_pi_threshold: float = 0.05,
    mixture_separation: float = 0.8,
    hmm_min_positions: int = 3,
    hmm_p_stay_per_base: float = 0.98,
    run_hmm: bool = True,
    hmm_params: HMMParams | None = None,
) -> ContigModificationResult:
    """Run the full V1→V2→V3 hierarchical pipeline on a contig.

    Parameters
    ----------
    contig_result : ContigResult
        Output of the eventalign pipeline for one contig.
    shrinkage_window : int
        Half-window in positions (not bases) for local shrinkage.
    kappa_high, kappa_medium, kappa_low : float
        Shrinkage strengths for high/medium/low coverage positions.
    mixture_max_iter : int
        Max EM iterations for V2 mixture.
    mixture_pi_threshold : float
        Null-gate threshold on mixing proportion.
    mixture_separation : float
        Minimum effect-size separation for V2 gate.
    hmm_min_positions : int
        Minimum trajectory length for HMM.
    hmm_p_stay_per_base : float
        Per-base state persistence for HMM transitions.
    run_hmm : bool
        If False, skip V3 entirely.
    hmm_params : HMMParams | None
        Trained HMM parameters.  When provided, overrides
        ``hmm_p_stay_per_base`` and uses learned emission transforms
        and initial state probabilities.

    Returns
    -------
    ContigModificationResult
    """
    sorted_positions = sorted(contig_result.positions.keys())

    if len(sorted_positions) == 0:
        return ContigModificationResult(
            contig=contig_result.contig,
            position_stats={},
            native_trajectories=[],
            ivt_trajectories=[],
            global_mu=0.0,
            global_sigma=1.0,
        )

    # ── V1a: Extract scores and fit robust IVT null per position ──────────

    raw_params: dict[int, tuple[float, float]] = {}
    coverages: dict[int, int] = {}
    all_scores: dict[int, NDArray[np.float64]] = {}

    for pos in sorted_positions:
        pr = contig_result.positions[pos]
        scores = _extract_ivt_distances(
            pr.distance_matrix, pr.n_native_reads, pr.n_ivt_reads
        )
        all_scores[pos] = scores

        ivt_scores = scores[pr.n_native_reads:]
        mu_raw, sigma_raw = _fit_robust_ivt_null(ivt_scores)
        raw_params[pos] = (mu_raw, sigma_raw)
        coverages[pos] = pr.n_ivt_reads

    # ── V1b: Hierarchical shrinkage ──────────────────────────────────────

    shrunk_params = _shrink_parameters(
        sorted_positions,
        raw_params,
        coverages,
        window=shrinkage_window,
        kappa_high=kappa_high,
        kappa_medium=kappa_medium,
        kappa_low=kappa_low,
    )

    # Global IVT prior (for reporting)
    all_mus = [raw_params[p][0] for p in sorted_positions if coverages[p] >= 1]
    global_mu = float(np.median(all_mus)) if all_mus else 0.0
    all_sigmas = [raw_params[p][1] for p in sorted_positions if coverages[p] >= 1]
    global_sigma = float(np.median(all_sigmas)) if all_sigmas else 1.0

    # ── V1c: Z-scores and p-values ───────────────────────────────────────

    position_stats: dict[int, PositionStats] = {}

    for pos in sorted_positions:
        pr = contig_result.positions[pos]
        scores = all_scores[pos]
        mu_s, sigma_s = shrunk_params[pos]
        mu_r, sigma_r = raw_params[pos]

        z = (scores - mu_s) / max(sigma_s, _MIN_SIGMA)
        # One-sided: P(Z ≥ z) — high z means far from IVT null
        p_null_vals = 1.0 - _norm_dist.cdf(z)

        n_total = pr.n_native_reads + pr.n_ivt_reads
        position_stats[pos] = PositionStats(
            position=pos,
            reference_kmer=pr.reference_kmer,
            coverage_class=_classify_coverage(pr.n_ivt_reads),
            n_ivt=pr.n_ivt_reads,
            n_native=pr.n_native_reads,
            mu_raw=mu_r,
            sigma_raw=sigma_r,
            mu_shrunk=mu_s,
            sigma_shrunk=sigma_s,
            scores=scores,
            z_scores=z,
            p_null=np.asarray(p_null_vals, dtype=np.float64),
            p_mod_raw=np.zeros(n_total, dtype=np.float64),
            mixture_pi=0.0,
            mixture_null_gate=True,
            p_mod_hmm=np.full(n_total, np.nan, dtype=np.float64),
        )

    # ── V2: Anchored mixture ─────────────────────────────────────────────

    for pos in sorted_positions:
        ps = position_stats[pos]
        z_native = ps.z_scores[: ps.n_native]
        z_ivt = ps.z_scores[ps.n_native :]

        p_mod_all, pi, null_gate = _anchored_mixture_em(
            z_native,
            z_ivt,
            ps.z_scores,
            max_iter=mixture_max_iter,
            pi_threshold=mixture_pi_threshold,
            separation_threshold=mixture_separation,
        )
        ps.p_mod_raw[:] = p_mod_all
        ps.mixture_pi = pi
        ps.mixture_null_gate = null_gate

        # Default HMM to V2 result (will be overwritten if HMM runs)
        ps.p_mod_hmm[:] = p_mod_all

    # ── V3: HMM smoothing ────────────────────────────────────────────────

    native_trajs, ivt_trajs = _build_read_trajectories(
        contig_result, sorted_positions,
    )

    if run_hmm:
        _run_hmm_on_trajectories(
            native_trajs,
            position_stats,
            min_positions=hmm_min_positions,
            p_stay_per_base=hmm_p_stay_per_base,
            hmm_params=hmm_params,
        )
        _run_hmm_on_trajectories(
            ivt_trajs,
            position_stats,
            min_positions=hmm_min_positions,
            p_stay_per_base=hmm_p_stay_per_base,
            hmm_params=hmm_params,
        )

    return ContigModificationResult(
        contig=contig_result.contig,
        position_stats=position_stats,
        native_trajectories=native_trajs,
        ivt_trajectories=ivt_trajs,
        global_mu=global_mu,
        global_sigma=global_sigma,
    )
