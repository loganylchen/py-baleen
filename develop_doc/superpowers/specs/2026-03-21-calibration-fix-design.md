# Fix Modification Probability Underestimation at 100% Modified Sites

## Problem

E. coli rRNA datasets have known 100% stoichiometric modifications, but the pipeline reports modification probabilities of 20-60% instead of ~100%. This indicates systematic underestimation across the V1→V2→V3 pipeline.

## Root Causes (5 identified)

### RC1: Beta calibration ignores fitted mixing proportion
**File:** `_probability.py:317-320`
**Issue:** `_calibrate_beta()` computes posteriors as `f1/(f0+f1)` — a likelihood ratio with implicit 50/50 prior — instead of using the EM-fitted π. When π converges to 0.8+ (indicating most reads are modified), this information is discarded.
**Impact:** Posteriors are systematically lower than they should be at high-modification positions.

### RC2: kNN scores compress into narrow range
**File:** `_probability.py:426-436`
**Issue:** The exponential distance weighting `exp(-d/η)` with median-based bandwidth η causes scores to cluster in [0.3, 0.8] even for clearly modified reads. The subsequent Beta EM has poor separation on this compressed range.
**Impact:** Raw scores don't reach near 1.0 even for strongly modified reads.

### RC3: EM fails when all native reads are modified
**File:** `_probability.py:264-294` (Beta), `_hierarchical.py:479-575` (anchored mixture)
**Issue:** The two-component EM expects a mixture of null + alternative in native reads. At 100% modified positions, all native reads come from the alternative component. The EM has no null subpopulation to contrast against, leading to poor convergence where π stabilizes at moderate values.
**Impact:** Positions with clear, strong modification signal get moderate instead of high probabilities.

### RC4: HMM spatial smoothing suppresses isolated peaks
**File:** `_hierarchical.py:669-688, 819-948`
**Issue:** With `p_stay_per_base=0.98`, the 3-state HMM strongly favors state persistence. E. coli rRNA has ~36 modified positions scattered across ~4500 total positions. Each modified position is surrounded by many unmodified neighbors, so the HMM smooths the modification probability toward the majority unmodified state.
**Impact:** Even positions with high V2 scores get pulled down by the HMM.

### RC5: Soft gate partially fires, diluting probabilities
**File:** `_hierarchical.py:596-611`
**Issue:** The soft gate multiplies three sigmoid weights (π, BIC, separation). The tau temperatures (`tau_pi=0.02`, `tau_bic=5.0`, `tau_sep=0.3`) are conservative — even moderate evidence can partially trigger the gate, blending the mixture posterior with the weaker z-score fallback.
**Impact:** Additional probability dilution even when modification evidence is clear.

## Design: Targeted Calibration Fixes

### Fix 1: Pi-weighted posteriors in `_calibrate_beta()`

Replace likelihood-ratio posteriors with proper Bayesian posteriors using the fitted π:

```python
# Before (flat prior):
probs = f1_all / (f0_all + f1_all + _EPS)

# After (pi-weighted):
probs = (pi * f1_all) / ((1 - pi) * f0_all + pi * f1_all + _EPS)
```

Same fix applied to `_calibrate_normal()` in the same file.

**Risk:** When π→1.0, posteriors also approach 1.0. This is actually correct behavior — if the EM determines most reads are modified, posteriors should reflect that. The BIC gate and separation gate already protect against false positives.

### Fix 2: Rank-based kNN scoring

Replace exponential-weighted IVT affinity with a rank-based score that naturally spreads across [0, 1]:

```python
# Current: exponential weighting compresses scores
weights = np.exp(-dists[neighbor_idx] / eta)
ivt_weight = np.sum(weights[is_ivt[neighbor_idx]])
score = 1.0 - ivt_weight / total_weight

# New: rank-based scoring
# For each read, compute rank of "IVT affinity" among all reads
# Then normalize to [0, 1]
# Reads with high IVT affinity (unmodified) get low scores
# Reads with low IVT affinity (modified) get high scores
```

Specifically: for each read, compute the fraction of IVT reads among its k nearest neighbors (unweighted), then use the rank of this fraction across all reads, normalized to [0, 1]. This guarantees uniform spread.

### Fix 3: Stoichiometric modification detection

Before running EM, test whether native and IVT score distributions are completely separated. If so, bypass EM and assign high probabilities directly:

```python
def _detect_stoichiometric_modification(
    native_scores, ivt_scores, overlap_threshold=0.05
):
    """Detect positions where ALL native reads are clearly modified.

    Uses Mann-Whitney U test + distribution overlap check.
    Returns True if native and IVT are fully separated.
    """
    # Check distribution overlap
    ivt_max = np.percentile(ivt_scores, 95)
    native_min = np.percentile(native_scores, 5)

    if native_min > ivt_max:
        # No overlap — stoichiometric modification
        return True

    # Also check via rank-biserial correlation
    from scipy.stats import mannwhitneyu
    U, p = mannwhitneyu(native_scores, ivt_scores, alternative='greater')
    r = 1 - 2*U/(len(native_scores)*len(ivt_scores))

    return p < 0.001 and r > 0.8
```

When detected, assign probabilities based on distance-from-IVT quantile rather than EM posteriors.

### Fix 4: Adaptive HMM spatial penalty

Make `p_stay_per_base` adaptive based on local modification density. In regions with many modified positions close together, reduce the spatial penalty (lower p_stay) to let the HMM follow the data more closely:

```python
def _adaptive_p_stay(
    positions: list[int],
    p_mod_scores: dict[int, float],  # mean V2 score per position
    base_p_stay: float = 0.98,
    min_p_stay: float = 0.90,
) -> float:
    """Compute adaptive p_stay based on local modification evidence.

    If many nearby positions show modification evidence,
    reduce p_stay to let the HMM follow the data.
    """
    high_mod_count = sum(1 for s in p_mod_scores.values() if s > 0.3)
    mod_density = high_mod_count / max(len(positions), 1)

    # Linear interpolation: high density → lower p_stay
    return base_p_stay - (base_p_stay - min_p_stay) * min(mod_density * 5, 1.0)
```

### Fix 5: Relaxed soft gate temperatures

Increase the tau parameters to make the soft gate less aggressive:

```python
# Before:
tau_pi = 0.02    # very sharp sigmoid
tau_bic = 5.0
tau_sep = 0.3

# After:
tau_pi = 0.05    # more gradual transition
tau_bic = 10.0   # less sensitive to BIC difference
tau_sep = 0.5    # less sensitive to separation
```

This ensures the gate only significantly fires when there is strong evidence AGAINST modification, rather than partially firing at moderate evidence levels.

## Files Modified

| File | Changes |
|------|---------|
| `baleen/eventalign/_probability.py` | Fix 1 (pi-weighted posteriors), Fix 2 (rank-based kNN), Fix 3 (stoichiometric detection) |
| `baleen/eventalign/_hierarchical.py` | Fix 4 (adaptive HMM), Fix 5 (relaxed gate tau) |

## Diagnostic Notebook

Before implementing fixes, create a diagnostic notebook (`notebooks/calibration_diagnostic.ipynb`) that:

1. Loads pipeline results and runs the hierarchical pipeline
2. For each known modified position, shows:
   - Raw kNN score distribution (native vs IVT)
   - Beta EM fit quality (null vs alt components)
   - Gate weight breakdown (which gates are firing)
   - HMM input vs output (how much smoothing reduces the score)
3. Identifies which root cause is dominant for each position
4. After fixes, re-runs and shows improvement

This notebook is designed to run on the remote machine with real data and send results back for analysis.

## Testing Strategy

1. Run diagnostic notebook on real data BEFORE fixes to quantify each root cause
2. Implement fixes incrementally, running the notebook after each
3. Verify: known modified sites should show p_mod > 0.8
4. Verify: unmodified sites should NOT increase (no false positives)
5. Run existing test suite to ensure no regressions
