"""HMM training modes for the hierarchical modification detection pipeline.

Provides three training modes for the V3 HMM:

Mode A — Unsupervised (default)
    No labeled data needed.  Returns hardcoded defaults identical to the
    current ``_hierarchical.py`` behavior.

Mode B — Semi-supervised
    Requires labeled modification positions.  Fits a Platt-scaling
    calibrator on V2 ``p_mod_raw`` values and learns ``init_prob`` from
    base-rate.  Transition stays at default.

Mode C — Fully supervised
    Requires more labeled data (≥ 50 positions, ≥ 3 contigs).  MLE-
    trained transition probability from labeled trajectories, KDE-based
    emission model, and learned ``init_prob``.

Public API
----------
HMMParams
    Frozen container for all HMM parameters (works with all 3 modes).
EmissionCalibrator
    Platt-scaling sigmoid calibrator (Mode B).
EmissionKDE
    KDE-based emission likelihood model (Mode C).
create_unsupervised_params
    Build default HMMParams (Mode A).
train_semi_supervised
    Fit Mode B parameters from labeled data.
train_supervised
    Fit Mode C parameters from labeled data.
labels_from_known_modifications
    Convert biological modification positions to training labels.
cross_validate_hmm
    Cross-validate HMM training to detect overfitting.
CVResult
    Cross-validation results container.
save_hmm_params / load_hmm_params
    JSON serialisation for cross-species parameter transfer.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as _opt
from scipy.stats import gaussian_kde as _gaussian_kde

if TYPE_CHECKING:
    from baleen.eventalign._hierarchical import (
        ContigModificationResult,
        PositionStats,
        ReadTrajectory,
    )
    from baleen.eventalign._pipeline import ContigResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

EmissionTransform = "EmissionCalibrator | EmissionKDE | None"

# ---------------------------------------------------------------------------
# Emission models
# ---------------------------------------------------------------------------


@dataclass
class EmissionCalibrator:
    """Platt-scaling calibrator for V2 → V3 emission mapping (Mode B).

    Transforms raw ``p_mod`` via sigmoid: ``σ(a·x + b)``.
    """

    a: float
    b: float

    def transform(self, p_mod_raw: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map raw P(mod) to calibrated P(mod)."""
        z = self.a * np.asarray(p_mod_raw, dtype=np.float64) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "calibrator", "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EmissionCalibrator:
        return cls(a=float(d["a"]), b=float(d["b"]))


@dataclass
class EmissionKDE:
    """KDE-based emission likelihood model (Mode C).

    Stores two pre-evaluated density curves on a fixed grid:
    ``P(p_mod_raw | unmodified)`` and ``P(p_mod_raw | modified)``.

    At inference time, :meth:`emission_probs` returns per-observation
    likelihoods via linear interpolation on the grid.
    """

    grid: NDArray[np.float64]            # shape (n_bins,)
    density_unmod: NDArray[np.float64]   # shape (n_bins,)
    density_mod: NDArray[np.float64]     # shape (n_bins,)

    def emission_probs(
        self, p_mod_raw: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(P(obs|unmod), P(obs|mod))`` via interpolation.

        Values are clamped to ``[1e-10, ∞)`` to avoid log(0) issues.
        """
        x = np.asarray(p_mod_raw, dtype=np.float64)
        p_unmod = np.interp(x, self.grid, self.density_unmod)
        p_mod = np.interp(x, self.grid, self.density_mod)
        # Floor to avoid zero-emission
        p_unmod = np.maximum(p_unmod, 1e-10)
        p_mod = np.maximum(p_mod, 1e-10)
        return p_unmod, p_mod

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "kde",
            "grid": self.grid.tolist(),
            "density_unmod": self.density_unmod.tolist(),
            "density_mod": self.density_mod.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EmissionKDE:
        return cls(
            grid=np.array(d["grid"], dtype=np.float64),
            density_unmod=np.array(d["density_unmod"], dtype=np.float64),
            density_mod=np.array(d["density_mod"], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# HMMParams container
# ---------------------------------------------------------------------------


@dataclass
class HMMParams:
    """Learned or default HMM parameters for V3.

    All fields have defaults so the dataclass can be constructed
    incrementally or via :func:`create_unsupervised_params`.
    """

    mode: Literal["unsupervised", "semi_supervised", "supervised"] = "unsupervised"
    p_stay_per_base: float = 0.98
    init_prob: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, 0.5], dtype=np.float64)
    )
    emission_transform: EmissionCalibrator | EmissionKDE | None = None
    # Metadata
    training_species: list[str] = field(default_factory=list)
    n_training_positions: int = 0
    n_training_reads: int = 0


# ---------------------------------------------------------------------------
# Mode A — unsupervised
# ---------------------------------------------------------------------------


def create_unsupervised_params() -> HMMParams:
    """Build default (unsupervised) HMM parameters.

    Equivalent to the hardcoded values in ``_hierarchical.py``.
    """
    return HMMParams(mode="unsupervised")


# ---------------------------------------------------------------------------
# Mode B — semi-supervised
# ---------------------------------------------------------------------------


def _fit_platt_scaling(
    raw_probs: NDArray[np.float64],
    true_labels: NDArray[np.float64],
) -> tuple[float, float]:
    """Fit Platt-scaling parameters via MLE of logistic regression.

    Minimises negative log-likelihood of ``σ(a·x + b)`` predicting
    *true_labels* from *raw_probs*.

    Returns ``(a, b)``.
    """

    def _neg_log_likelihood(params: NDArray[np.float64]) -> float:
        a, b = params
        z = a * raw_probs + b
        # Numerically stable log-sigmoid
        log_p = -np.logaddexp(0, -z)      # log σ(z)
        log_1_p = -np.logaddexp(0, z)     # log (1 - σ(z))
        nll = -(true_labels * log_p + (1.0 - true_labels) * log_1_p).sum()
        return float(nll)

    result = _opt.minimize(
        _neg_log_likelihood,
        x0=np.array([1.0, 0.0]),
        method="L-BFGS-B",
    )
    return float(result.x[0]), float(result.x[1])


def _learn_transition_from_labels(
    training_data: dict[str, ContigModificationResult],
    labels: dict[tuple[str, int], bool],
) -> float:
    """Learn p_stay_per_base from labeled data via MLE on state transitions.

    Counts same-state vs different-state transitions along read
    trajectories, weighted by 1/gap to normalize for genomic distance.

    Returns the learned p_stay, clamped to [0.8, 0.999].
    Falls back to 0.98 if insufficient transition data.
    """
    same_count = 0.0
    diff_count = 0.0

    for contig_name, cmr in training_data.items():
        all_trajs = list(cmr.native_trajectories) + list(cmr.ivt_trajectories)
        is_ivt_offset = len(cmr.native_trajectories)

        for traj_idx, traj in enumerate(all_trajs):
            is_ivt = traj_idx >= is_ivt_offset
            labeled_pairs: list[tuple[int, int]] = []
            for pos in traj.positions:
                key = (contig_name, pos)
                if key not in labels:
                    continue
                if is_ivt:
                    state = 0
                else:
                    state = 1 if labels[key] else 0
                labeled_pairs.append((pos, state))

            for i in range(len(labeled_pairs) - 1):
                pos_i, state_i = labeled_pairs[i]
                pos_j, state_j = labeled_pairs[i + 1]
                gap = max(pos_j - pos_i, 1)
                if state_i == state_j:
                    same_count += 1.0 / gap
                else:
                    diff_count += 1.0 / gap

    total = same_count + diff_count
    if total < 5.0:
        return 0.98

    p_stay = same_count / total
    return max(0.8, min(p_stay, 0.999))


def train_semi_supervised(
    training_data: dict[str, ContigModificationResult],
    labels: dict[tuple[str, int], bool],
    *,
    species_name: str = "",
    species_names: list[str] | None = None,
    learn_transitions: bool = True,
) -> HMMParams:
    """Train Mode B (semi-supervised) HMM parameters.

    Parameters
    ----------
    training_data
        ``{contig_name: ContigModificationResult}`` — must have been
        computed with V1→V2 (``run_hmm=False`` is fine).
    labels
        ``{(contig, pipeline_position): is_modified}`` — True means the
        position is known to carry a modification.
    species_name
        Optional single species tag stored in metadata.
    species_names
        Optional list of species names for multi-organism pooling.
        Takes precedence over ``species_name`` if provided.
    learn_transitions
        If True (default), learn ``p_stay_per_base`` from labeled
        trajectories instead of using the hardcoded 0.98 default.

    Returns
    -------
    HMMParams
        With Platt-calibrated emission transform, learned ``init_prob``,
        and optionally learned transition parameters.

    Raises
    ------
    ValueError
        If fewer than 20 labeled positions are provided, or fewer than
        10 positive / 10 negative labels.
    """
    # ── Validate label counts ────────────────────────────────────────────
    n_pos = sum(1 for v in labels.values() if v)
    n_neg = sum(1 for v in labels.values() if not v)
    if n_pos + n_neg < 20:
        raise ValueError(
            f"Semi-supervised training requires >= 20 labeled positions, "
            f"got {n_pos + n_neg}"
        )
    if n_pos < 10:
        raise ValueError(
            f"Need >= 10 positive (modified) labels, got {n_pos}"
        )
    if n_neg < 10:
        raise ValueError(
            f"Need >= 10 negative (unmodified) labels, got {n_neg}"
        )

    # ── Collect (p_mod_raw, is_modified) pairs ───────────────────────────
    raw_vals: list[float] = []
    true_vals: list[float] = []
    n_reads_total = 0

    for (contig, pos), is_mod in labels.items():
        if contig not in training_data:
            continue
        cmr = training_data[contig]
        if pos not in cmr.position_stats:
            continue

        ps = cmr.position_stats[pos]
        n_reads_total += ps.n_native + ps.n_ivt

        if is_mod:
            # Native reads at modified positions → label 1
            for i in range(ps.n_native):
                raw_vals.append(float(ps.p_mod_raw[i]))
                true_vals.append(1.0)
            # IVT reads at modified positions → label 0 (IVT never modified)
            for i in range(ps.n_native, ps.n_native + ps.n_ivt):
                raw_vals.append(float(ps.p_mod_raw[i]))
                true_vals.append(0.0)
        else:
            # All reads at unmodified positions → label 0
            for i in range(ps.n_native + ps.n_ivt):
                raw_vals.append(float(ps.p_mod_raw[i]))
                true_vals.append(0.0)

    if len(raw_vals) == 0:
        raise ValueError(
            "No reads found at labeled positions — check contig/position keys."
        )

    raw_arr = np.array(raw_vals, dtype=np.float64)
    true_arr = np.array(true_vals, dtype=np.float64)

    # ── Fit Platt scaling ────────────────────────────────────────────────
    a, b = _fit_platt_scaling(raw_arr, true_arr)
    calibrator = EmissionCalibrator(a=a, b=b)

    # ── Learn transition parameters from labeled trajectories ────────────
    if learn_transitions:
        p_stay = _learn_transition_from_labels(training_data, labels)
        logger.info("Semi-supervised learned p_stay=%.4f from labeled data", p_stay)
    else:
        p_stay = 0.98

    # ── Learned init_prob from base rate ─────────────────────────────────
    base_rate = n_pos / max(n_pos + n_neg, 1)
    init_prob = np.array([1.0 - base_rate, base_rate], dtype=np.float64)

    # ── Species metadata ─────────────────────────────────────────────────
    if species_names is not None:
        species_list = list(species_names)
    elif species_name:
        species_list = [species_name]
    else:
        species_list = []

    return HMMParams(
        mode="semi_supervised",
        p_stay_per_base=p_stay,
        init_prob=init_prob,
        emission_transform=calibrator,
        training_species=species_list,
        n_training_positions=n_pos + n_neg,
        n_training_reads=n_reads_total,
    )


# ---------------------------------------------------------------------------
# Mode C — fully supervised
# ---------------------------------------------------------------------------


def train_supervised(
    training_data: dict[str, ContigModificationResult],
    labels: dict[tuple[str, int], bool],
    *,
    species_name: str = "",
    kde_n_bins: int = 200,
    kde_bandwidth: float | None = None,
) -> HMMParams:
    """Train Mode C (fully supervised) HMM parameters.

    Parameters
    ----------
    training_data
        ``{contig_name: ContigModificationResult}`` — V1→V2 results.
    labels
        ``{(contig, pipeline_position): is_modified}``
    species_name
        Optional species tag.
    kde_n_bins
        Number of evaluation points for KDE grid.
    kde_bandwidth
        Explicit bandwidth for KDE; ``None`` = Scott's rule (default).

    Returns
    -------
    HMMParams
        With MLE transition, KDE emission model, and learned ``init_prob``.

    Raises
    ------
    ValueError
        If fewer than 50 labeled positions or fewer than 3 contigs.
    """
    n_pos = sum(1 for v in labels.values() if v)
    n_neg = sum(1 for v in labels.values() if not v)
    contig_set = {c for c, _ in labels.keys()}

    if n_pos + n_neg < 50:
        raise ValueError(
            f"Supervised training requires >= 50 labeled positions, "
            f"got {n_pos + n_neg}"
        )
    if len(contig_set) < 3:
        raise ValueError(
            f"Supervised training requires >= 3 contigs, got {len(contig_set)}"
        )

    # ── 1. Collect per-read p_mod_raw by label ───────────────────────────
    mod_vals: list[float] = []
    unmod_vals: list[float] = []
    n_reads_total = 0

    for (contig, pos), is_mod in labels.items():
        if contig not in training_data:
            continue
        cmr = training_data[contig]
        if pos not in cmr.position_stats:
            continue

        ps = cmr.position_stats[pos]
        n_reads_total += ps.n_native + ps.n_ivt

        if is_mod:
            # Native reads → modified
            for i in range(ps.n_native):
                mod_vals.append(float(ps.p_mod_raw[i]))
            # IVT reads → unmodified
            for i in range(ps.n_native, ps.n_native + ps.n_ivt):
                unmod_vals.append(float(ps.p_mod_raw[i]))
        else:
            # All reads → unmodified
            for i in range(ps.n_native + ps.n_ivt):
                unmod_vals.append(float(ps.p_mod_raw[i]))

    if len(mod_vals) < 5 or len(unmod_vals) < 5:
        raise ValueError(
            f"Need >= 5 reads in each class for KDE fitting. "
            f"Got {len(mod_vals)} modified, {len(unmod_vals)} unmodified."
        )

    # ── 2. Fit KDE emission model ────────────────────────────────────────
    mod_arr = np.array(mod_vals, dtype=np.float64)
    unmod_arr = np.array(unmod_vals, dtype=np.float64)

    bw_kwargs = {"bw_method": kde_bandwidth} if kde_bandwidth is not None else {}
    kde_mod = _gaussian_kde(mod_arr, **bw_kwargs)
    kde_unmod = _gaussian_kde(unmod_arr, **bw_kwargs)

    grid = np.linspace(0.0, 1.0, kde_n_bins)
    emission_kde = EmissionKDE(
        grid=grid,
        density_unmod=kde_unmod(grid).astype(np.float64),
        density_mod=kde_mod(grid).astype(np.float64),
    )

    # ── 3. MLE transition from labeled trajectories ──────────────────────
    same_count = 0.0
    diff_count = 0.0

    for contig_name, cmr in training_data.items():
        all_trajs = list(cmr.native_trajectories) + list(cmr.ivt_trajectories)
        is_ivt_offset = len(cmr.native_trajectories)

        for traj_idx, traj in enumerate(all_trajs):
            is_ivt = traj_idx >= is_ivt_offset
            # Build state sequence at labeled positions
            labeled_pairs: list[tuple[int, int]] = []  # (position, state)
            for pos in traj.positions:
                key = (contig_name, pos)
                if key not in labels:
                    continue
                if is_ivt:
                    state = 0  # IVT reads are always unmodified
                else:
                    state = 1 if labels[key] else 0
                labeled_pairs.append((pos, state))

            # Consecutive pairs weighted by 1/gap
            for i in range(len(labeled_pairs) - 1):
                pos_i, state_i = labeled_pairs[i]
                pos_j, state_j = labeled_pairs[i + 1]
                gap = max(pos_j - pos_i, 1)
                if state_i == state_j:
                    same_count += 1.0 / gap
                else:
                    diff_count += 1.0 / gap

    total_transitions = same_count + diff_count
    if total_transitions > 0:
        p_stay = same_count / total_transitions
    else:
        p_stay = 0.98  # fallback to default

    # Clamp to [0.8, 0.999]
    p_stay = max(0.8, min(p_stay, 0.999))

    # ── 4. Learned init_prob ─────────────────────────────────────────────
    base_rate = n_pos / max(n_pos + n_neg, 1)
    init_prob = np.array([1.0 - base_rate, base_rate], dtype=np.float64)

    species_list = [species_name] if species_name else []

    return HMMParams(
        mode="supervised",
        p_stay_per_base=p_stay,
        init_prob=init_prob,
        emission_transform=emission_kde,
        training_species=species_list,
        n_training_positions=n_pos + n_neg,
        n_training_reads=n_reads_total,
    )


# ---------------------------------------------------------------------------
# Labels helper
# ---------------------------------------------------------------------------


def labels_from_known_modifications(
    known_mods: dict[tuple[str, int], tuple[str, str]],
    contig_results: dict[str, ContigModificationResult],
    *,
    position_offset: int = 3,
    auto_negatives: bool = True,
    min_coverage: int = 5,
) -> dict[tuple[str, int], bool]:
    """Convert known biological modification sites to training labels.

    Parameters
    ----------
    known_mods
        ``{(contig, bio_position): (mod_short, mod_full), ...}`` where
        *bio_position* is the 1-based biological coordinate.
    contig_results
        ``{contig_name: ContigModificationResult}`` keyed by contig name.
    position_offset
        ``bio_position - offset = pipeline_position``.  Default 3 for
        eventalign 5-mer centre.
    auto_negatives
        If True, positions with ``n_native + n_ivt >= min_coverage`` that
        are **not** in *known_mods* become negative (unmodified) labels.
    min_coverage
        Minimum total read coverage for auto-negative positions.

    Returns
    -------
    labels
        ``{(contig, pipeline_position): is_modified}``
    """
    labels: dict[tuple[str, int], bool] = {}

    # Positive labels from known modifications
    for (contig, bio_pos), (_mod_short, _mod_full) in known_mods.items():
        pipeline_pos = bio_pos - position_offset
        if contig not in contig_results:
            continue
        cmr = contig_results[contig]
        if pipeline_pos not in cmr.position_stats:
            continue
        labels[(contig, pipeline_pos)] = True

    # Auto-negative labels
    if auto_negatives:
        for contig_name, cmr in contig_results.items():
            for pos, ps in cmr.position_stats.items():
                key = (contig_name, pos)
                if key in labels:
                    continue  # already labeled (positive)
                if ps.n_native + ps.n_ivt >= min_coverage:
                    labels[key] = False

    return labels


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Cross-validation results."""

    per_fold_auroc: list[float]
    per_fold_auprc: list[float]
    mean_auroc: float
    mean_auprc: float
    std_auroc: float
    std_auprc: float
    fold_details: list[dict[str, Any]]


def _manual_auroc(
    y_true: NDArray[np.float64],
    y_score: NDArray[np.float64],
) -> float:
    """Compute AUROC without sklearn.

    Uses the trapezoidal-rule approach on sorted scores.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    n_pos = y_true.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Sort by descending score
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = 0.0
    fp = 0.0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for i in range(n):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr = fpr
        prev_tpr = tpr

    return float(auc)


def _manual_auprc(
    y_true: NDArray[np.float64],
    y_score: NDArray[np.float64],
) -> float:
    """Compute AUPRC without sklearn.

    Uses interpolated precision-recall with trapezoidal integration.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.0

    # Sort by descending score
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = 0.0
    fp = 0.0
    auc = 0.0
    prev_recall = 0.0

    for i in range(n):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        auc += (recall - prev_recall) * precision
        prev_recall = recall

    return float(auc)


def cross_validate_hmm(
    contig_results: dict[str, ContigResult],
    labels: dict[tuple[str, int], bool],
    mode: Literal["semi_supervised", "supervised"],
    *,
    cv_strategy: Literal["leave_one_contig_out", "kfold"] = "leave_one_contig_out",
    k: int = 5,
    **hierarchical_kwargs,
) -> CVResult:
    """Cross-validate HMM training to detect overfitting.

    Parameters
    ----------
    contig_results
        Raw pipeline output per contig (``ContigResult``).
    labels
        ``{(contig, pipeline_position): is_modified}``
    mode
        Training mode to evaluate (``"semi_supervised"`` or ``"supervised"``).
    cv_strategy
        ``"leave_one_contig_out"`` (default) or ``"kfold"`` (by contig).
    k
        Number of folds for k-fold CV.
    **hierarchical_kwargs
        Forwarded to
        :func:`~baleen.eventalign._hierarchical.compute_sequential_modification_probabilities`.

    Returns
    -------
    CVResult
    """
    # Lazy import to avoid circular dependency
    from baleen.eventalign._hierarchical import (
        compute_sequential_modification_probabilities,
    )

    # ── Build folds ──────────────────────────────────────────────────────
    contigs_with_labels = sorted({c for c, _ in labels.keys()})
    # Only keep contigs that exist in contig_results
    contigs_with_labels = [c for c in contigs_with_labels if c in contig_results]

    if len(contigs_with_labels) < 2:
        raise ValueError(
            "Need labels in >= 2 contigs for cross-validation, "
            f"got {len(contigs_with_labels)}"
        )

    if cv_strategy == "leave_one_contig_out":
        folds = [([c], [x for x in contigs_with_labels if x != c])
                 for c in contigs_with_labels]
    else:
        # k-fold by contig
        n = len(contigs_with_labels)
        fold_size = max(1, n // k)
        folds = []
        for i in range(0, n, fold_size):
            test_contigs = contigs_with_labels[i : i + fold_size]
            train_contigs = [c for c in contigs_with_labels if c not in test_contigs]
            if train_contigs:
                folds.append((test_contigs, train_contigs))

    # ── Run V1+V2 on all contigs once ────────────────────────────────────
    v2_results: dict[str, ContigModificationResult] = {}
    for contig_name in contigs_with_labels:
        cr = contig_results[contig_name]
        v2_results[contig_name] = compute_sequential_modification_probabilities(
            cr, run_hmm=False, **hierarchical_kwargs
        )

    # ── Per-fold train/test ──────────────────────────────────────────────
    per_fold_auroc: list[float] = []
    per_fold_auprc: list[float] = []
    fold_details: list[dict[str, Any]] = []

    for test_contigs, train_contigs in folds:
        # Build train labels & data
        train_labels = {
            (c, p): v for (c, p), v in labels.items() if c in train_contigs
        }
        train_data = {c: v2_results[c] for c in train_contigs if c in v2_results}

        # Check minimum requirements — skip fold if insufficient
        n_train_pos = sum(1 for v in train_labels.values() if v)
        n_train_neg = sum(1 for v in train_labels.values() if not v)

        try:
            if mode == "semi_supervised":
                hmm_params = train_semi_supervised(train_data, train_labels)
            else:
                hmm_params = train_supervised(train_data, train_labels)
        except ValueError as e:
            logger.warning(
                "Skipping fold (test=%s): insufficient training data — %s",
                test_contigs,
                e,
            )
            continue

        # Run V3 with trained params on test contigs
        y_true_list: list[float] = []
        y_score_list: list[float] = []

        for test_contig in test_contigs:
            if test_contig not in contig_results:
                continue
            cr = contig_results[test_contig]
            test_result = compute_sequential_modification_probabilities(
                cr, hmm_params=hmm_params, **hierarchical_kwargs
            )

            # Collect scores at labeled test positions
            test_labels = {
                (c, p): v
                for (c, p), v in labels.items()
                if c == test_contig
            }
            for (_, pos), is_mod in test_labels.items():
                if pos not in test_result.position_stats:
                    continue
                ps = test_result.position_stats[pos]
                if is_mod:
                    # Native reads → y_true=1
                    for i in range(ps.n_native):
                        y_true_list.append(1.0)
                        y_score_list.append(float(ps.p_mod_hmm[i]))
                    # IVT reads → y_true=0
                    for i in range(ps.n_native, ps.n_native + ps.n_ivt):
                        y_true_list.append(0.0)
                        y_score_list.append(float(ps.p_mod_hmm[i]))
                else:
                    # All reads → y_true=0
                    for i in range(ps.n_native + ps.n_ivt):
                        y_true_list.append(0.0)
                        y_score_list.append(float(ps.p_mod_hmm[i]))

        if len(y_true_list) == 0:
            continue

        y_true = np.array(y_true_list, dtype=np.float64)
        y_score = np.array(y_score_list, dtype=np.float64)

        auroc = _manual_auroc(y_true, y_score)
        auprc = _manual_auprc(y_true, y_score)
        per_fold_auroc.append(auroc)
        per_fold_auprc.append(auprc)
        fold_details.append({
            "test_contigs": test_contigs,
            "train_contigs": train_contigs,
            "n_test_reads": len(y_true_list),
            "n_train_positions": n_train_pos + n_train_neg,
            "auroc": auroc,
            "auprc": auprc,
        })

    if len(per_fold_auroc) == 0:
        raise ValueError("No folds completed — insufficient data for CV.")

    return CVResult(
        per_fold_auroc=per_fold_auroc,
        per_fold_auprc=per_fold_auprc,
        mean_auroc=float(np.mean(per_fold_auroc)),
        mean_auprc=float(np.mean(per_fold_auprc)),
        std_auroc=float(np.std(per_fold_auroc)),
        std_auprc=float(np.std(per_fold_auprc)),
        fold_details=fold_details,
    )


# ---------------------------------------------------------------------------
# Save / Load for cross-species transfer
# ---------------------------------------------------------------------------


def _serialize_emission_transform(
    et: EmissionCalibrator | EmissionKDE | None,
) -> dict[str, Any]:
    if et is None:
        return {"type": "none"}
    return et.to_dict()


def _deserialize_emission_transform(
    d: dict[str, Any],
) -> EmissionCalibrator | EmissionKDE | None:
    t = d.get("type", "none")
    if t == "calibrator":
        return EmissionCalibrator.from_dict(d)
    if t == "kde":
        return EmissionKDE.from_dict(d)
    return None


def save_hmm_params(params: HMMParams, path: str | Path) -> None:
    """Serialize trained HMM parameters to JSON."""
    data = {
        "mode": params.mode,
        "p_stay_per_base": params.p_stay_per_base,
        "init_prob": params.init_prob.tolist(),
        "emission_transform": _serialize_emission_transform(
            params.emission_transform
        ),
        "training_species": params.training_species,
        "n_training_positions": params.n_training_positions,
        "n_training_reads": params.n_training_reads,
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_hmm_params(path: str | Path) -> HMMParams:
    """Load previously trained HMM parameters from JSON."""
    data = json.loads(Path(path).read_text())
    return HMMParams(
        mode=data["mode"],
        p_stay_per_base=data["p_stay_per_base"],
        init_prob=np.array(data["init_prob"], dtype=np.float64),
        emission_transform=_deserialize_emission_transform(
            data["emission_transform"]
        ),
        training_species=data.get("training_species", []),
        n_training_positions=data.get("n_training_positions", 0),
        n_training_reads=data.get("n_training_reads", 0),
    )
