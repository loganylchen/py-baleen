"""Site-level aggregation of per-read HMM modification probabilities.

Converts per-read ``p_mod_hmm`` values from the hierarchical pipeline
into transcript-level modification calls with:

- **Modification ratio** (stoichiometry) via Beta-Binomial soft counts
- **P-value** via one-sided Mann-Whitney U test (native vs IVT)
- **FDR-adjusted p-value** via Benjamini-Hochberg correction
- **95% credible interval** from Beta posterior

Output is a TSV compatible with other nanopore modification detection
tools (xPore, m6Anet, Nanocompore, ELIGOS, DRUMMER).

Public API
----------
SiteResult
    Per-site aggregated modification call.
aggregate_contig
    Aggregate one contig's per-read results into site-level calls.
aggregate_all
    Aggregate all contigs, applying FDR correction across all sites.
write_site_tsv
    Write site-level results to a TSV file.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as _beta_dist
from scipy.stats import fisher_exact as _fisher_exact
from scipy.stats import mannwhitneyu as _mannwhitneyu

if TYPE_CHECKING:
    from baleen.eventalign._hierarchical import ContigModificationResult

logger = logging.getLogger(__name__)


@dataclass
class SiteResult:
    """Per-site aggregated modification call."""

    contig: str
    position: int
    kmer: str
    mod_ratio: float
    """MAP estimate of modification stoichiometry (Beta-Binomial)."""
    ci_low: float
    """2.5th percentile of Beta posterior."""
    ci_high: float
    """97.5th percentile of Beta posterior."""
    pvalue: float
    """One-sided Mann-Whitney U p-value (native > IVT)."""
    padj: float
    """Benjamini-Hochberg FDR-adjusted p-value."""
    effect_size: float
    """median(native p_mod_hmm) - median(IVT p_mod_hmm)."""
    n_native: int
    n_ivt: int
    mean_p_mod: float
    """Mean of native p_mod_hmm values."""
    stoichiometry: float
    """Fraction of native reads with p_mod_hmm > 0.5."""


def _threshold_aggregate(
    p_mod: NDArray[np.float64],
    threshold: float = 0.99,
) -> tuple[float, float, float]:
    """Compute modification ratio as fraction of reads exceeding threshold.

    Parameters
    ----------
    p_mod
        Per-read P(modified) values (native reads only).
    threshold
        Reads with p_mod > threshold are counted as modified.

    Returns
    -------
    mod_ratio, ci_low, ci_high
        mod_ratio is the fraction of reads above threshold.
        ci_low/ci_high are the 95% credible interval from a Beta posterior
        on the binary counts.
    """
    n = len(p_mod)
    n_mod = int(np.sum(p_mod > threshold))

    mod_ratio = n_mod / n if n > 0 else 0.0

    # Beta posterior on binomial counts: Beta(1 + n_mod, 1 + n - n_mod)
    alpha = 1.0 + n_mod
    beta_param = 1.0 + (n - n_mod)
    ci_low = float(_beta_dist.ppf(0.025, alpha, beta_param))
    ci_high = float(_beta_dist.ppf(0.975, alpha, beta_param))

    return mod_ratio, ci_low, ci_high


def _fisher_pvalue(
    native_p_mod: NDArray[np.float64],
    ivt_p_mod: NDArray[np.float64],
    threshold: float = 0.99,
) -> float:
    """One-sided Fisher's exact test on binary modification calls.

    Tests whether the fraction of reads above *threshold* is higher
    in native than in IVT.

    Returns 1.0 if both groups have zero modified reads.
    """
    n_nat = len(native_p_mod)
    n_ivt = len(ivt_p_mod)
    if n_nat < 1 or n_ivt < 1:
        return 1.0

    nat_mod = int(np.sum(native_p_mod > threshold))
    ivt_mod = int(np.sum(ivt_p_mod > threshold))

    # Contingency table:
    #              modified  unmodified
    #   native   [ nat_mod,  n_nat - nat_mod ]
    #   IVT      [ ivt_mod,  n_ivt - ivt_mod ]
    table = [[nat_mod, n_nat - nat_mod],
             [ivt_mod, n_ivt - ivt_mod]]

    try:
        _, p = _fisher_exact(table, alternative="greater")
        return float(p)
    except ValueError:
        return 1.0


def _benjamini_hochberg(pvalues: NDArray[np.float64]) -> NDArray[np.float64]:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(pvalues)
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = np.arange(1, n + 1)

    padj = pvalues * n / ranked
    # Enforce monotonicity (step-up)
    padj_sorted_idx = np.argsort(pvalues)[::-1]
    padj_sorted = padj[padj_sorted_idx]
    for i in range(1, n):
        padj_sorted[i] = min(padj_sorted[i], padj_sorted[i - 1])
    padj[padj_sorted_idx] = padj_sorted

    return np.minimum(padj, 1.0)


def aggregate_contig(
    cmr: ContigModificationResult,
    *,
    score_field: str = "p_mod_hmm",
    mod_threshold: float = 0.99,
) -> list[SiteResult]:
    """Aggregate per-read results into site-level calls for one contig.

    P-values are *not* FDR-corrected here; use :func:`aggregate_all`
    for multi-contig FDR correction, or apply :func:`_benjamini_hochberg`
    manually.

    Parameters
    ----------
    cmr
        Output of ``compute_sequential_modification_probabilities``.
    score_field
        Which per-read score to aggregate.  Default ``"p_mod_hmm"``.

    Returns
    -------
    list[SiteResult]
        One entry per position, sorted by position.  ``padj`` is set
        equal to ``pvalue`` (no FDR correction applied).
    """
    results: list[SiteResult] = []

    for pos in sorted(cmr.position_stats.keys()):
        ps = cmr.position_stats[pos]
        scores = getattr(ps, score_field)

        native_scores = scores[: ps.n_native]
        ivt_scores = scores[ps.n_native :]

        # Skip positions with no valid native scores
        valid_native = native_scores[~np.isnan(native_scores)]
        valid_ivt = ivt_scores[~np.isnan(ivt_scores)]
        if len(valid_native) == 0:
            continue

        # Threshold-based aggregation (native reads only)
        mod_ratio, ci_low, ci_high = _threshold_aggregate(valid_native, mod_threshold)

        # Fisher's exact test on binary calls
        pvalue = _fisher_pvalue(valid_native, valid_ivt, mod_threshold)

        # Effect size (NaN if no IVT reads — avoids systematic upward bias)
        if len(valid_ivt) > 0:
            effect_size = float(np.median(valid_native)) - float(np.median(valid_ivt))
        else:
            effect_size = float('nan')

        # Stoichiometry: fraction of native reads confidently modified
        hmm_valid = valid_native
        stoichiometry = float(np.mean(hmm_valid > 0.5)) if len(hmm_valid) > 0 else 0.0

        results.append(
            SiteResult(
                contig=cmr.contig,
                position=pos,
                kmer=ps.reference_kmer,
                mod_ratio=mod_ratio,
                ci_low=ci_low,
                ci_high=ci_high,
                pvalue=pvalue,
                padj=pvalue,  # placeholder; corrected by aggregate_all
                effect_size=effect_size,
                n_native=ps.n_native,
                n_ivt=ps.n_ivt,
                mean_p_mod=float(np.mean(valid_native)),
                stoichiometry=stoichiometry,
            )
        )

    return results


def aggregate_all(
    results: dict[str, ContigModificationResult],
    *,
    score_field: str = "p_mod_hmm",
    mod_threshold: float = 0.99,
) -> list[SiteResult]:
    """Aggregate all contigs and apply FDR correction across all sites.

    Parameters
    ----------
    results
        ``{contig_name: ContigModificationResult}``
    score_field
        Which per-read score to aggregate.
    mod_threshold
        Per-read probability threshold for counting a read as modified.

    Returns
    -------
    list[SiteResult]
        All sites across all contigs, sorted by contig then position,
        with ``padj`` set via Benjamini-Hochberg.
    """
    all_sites: list[SiteResult] = []
    for contig in sorted(results.keys()):
        all_sites.extend(
            aggregate_contig(results[contig], score_field=score_field,
                             mod_threshold=mod_threshold)
        )

    if not all_sites:
        return all_sites

    # Apply BH FDR correction
    pvalues = np.array([s.pvalue for s in all_sites], dtype=np.float64)
    padj = _benjamini_hochberg(pvalues)
    for site, adj in zip(all_sites, padj):
        site.padj = float(adj)

    return all_sites


# -- Column order for TSV output --
_TSV_COLUMNS = [
    "contig",
    "position",
    "kmer",
    "mod_ratio",
    "ci_low",
    "ci_high",
    "pvalue",
    "padj",
    "effect_size",
    "n_native",
    "n_ivt",
    "mean_p_mod",
    "stoichiometry",
]


def write_site_tsv(
    sites: list[SiteResult],
    path: str | Path,
) -> Path:
    """Write site-level results to a TSV file.

    Parameters
    ----------
    sites
        Output of :func:`aggregate_all` or :func:`aggregate_contig`.
    path
        Output file path.

    Returns
    -------
    Path
        The written file path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(_TSV_COLUMNS)
        for site in sites:
            writer.writerow([
                site.contig,
                site.position,
                site.kmer,
                f"{site.mod_ratio:.6f}",
                f"{site.ci_low:.6f}",
                f"{site.ci_high:.6f}",
                f"{site.pvalue:.6e}",
                f"{site.padj:.6e}",
                f"{site.effect_size:.6f}",
                site.n_native,
                site.n_ivt,
                f"{site.mean_p_mod:.6f}",
                f"{site.stoichiometry:.6f}",
            ])

    logger.info("Wrote %d site results to %s", len(sites), out)
    return out
