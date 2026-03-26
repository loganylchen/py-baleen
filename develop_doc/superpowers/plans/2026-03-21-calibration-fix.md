# Fix Modification Probability Underestimation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix systematic underestimation of modification probabilities at known 100% modified E. coli rRNA sites (currently reporting 20-60% instead of ~100%).

**Architecture:** Three targeted fixes in priority order: (1) reduce HMM over-smoothing via lower p_stay default and emission floor, (2) improve kNN score spread via rank-based scoring, (3) use pi-weighted Bayesian posteriors in calibration. Each fix is independent and testable.

**Tech Stack:** Python, NumPy, SciPy

**Spec:** `docs/superpowers/specs/2026-03-21-calibration-fix-design.md`

**Deferred from spec (intentional):**
- **Fix 3 (stoichiometric detection):** The spec proposes a `_detect_stoichiometric_modification()` function to bypass EM when native/IVT are fully separated. Deferred because the simpler fixes (HMM + kNN + pi-weighted) should be tried first. If underestimation persists after these 3 fixes, stoichiometric detection can be added as a follow-up.
- **Fix 4 (adaptive p_stay):** The spec proposes making p_stay adaptive to local modification density. Replaced with a simpler global reduction (0.98→0.92) + emission floor. If specific positions still suffer from over-smoothing after this, adaptive p_stay can be added.

---

## File Structure

| File | Role | Changes |
|------|------|---------|
| `baleen/eventalign/_hierarchical.py` | V2 soft gate + V3 HMM pipeline | Task 1: lower p_stay default, add emission floor; Task 3: relax soft gate tau |
| `baleen/eventalign/_probability.py` | kNN scoring + Beta/Normal calibration | Task 2: rank-based kNN; Task 3: pi-weighted posteriors |
| `baleen/eventalign/_hmm_training.py` | HMM parameter defaults | Task 1: update unsupervised defaults |
| `tests/test_hierarchical.py` | Tests for hierarchical pipeline | Task 1: new tests for HMM fix |
| `tests/test_probability.py` | Tests for probability algorithms | Task 2-3: new tests for kNN + calibration |

---

### Task 1: Fix HMM Over-Smoothing (RC4 — highest impact)

**Why:** Sites with calibrated kNN scores of 0.95 get crushed to 0.04 by the HMM. With `p_stay_per_base=0.98` and ~36 modified sites scattered across ~4500 positions, the HMM treats each modified position as an isolated outlier and smooths it toward unmodified.

**Files:**
- Modify: `baleen/eventalign/_hierarchical.py:967` (default parameter)
- Modify: `baleen/eventalign/_hierarchical.py:860-915` (emission construction)
- Modify: `baleen/eventalign/_hmm_training.py:154-181` (HMMParams defaults)
- Test: `tests/test_hierarchical.py`

- [ ] **Step 1: Write failing test — HMM preserves strong modification signal**

Add to `tests/test_hierarchical.py`:

```python
class TestHMMPreservesStrongSignal:
    """Verify that the HMM does not crush strong modification signals."""

    def test_isolated_modified_position_preserved(self):
        """A single modified position surrounded by unmodified should retain
        a high HMM probability (>0.5) for native reads."""
        # 20 positions, only position index 10 is modified
        cr = _make_contig_result(
            n_positions=20,
            n_native=15,
            n_ivt=10,
            modified_positions={10},
            position_start=100,
            position_step=1,
            seed=42,
        )
        result = compute_sequential_modification_probabilities(cr)
        pos = 110  # position_start + 10
        ps = result.position_stats[pos]
        mean_hmm_native = float(np.mean(ps.native_p_mod_hmm))
        # Must be > 0.5 — the data clearly shows modification
        assert mean_hmm_native > 0.5, (
            f"HMM crushed strong modification signal: mean_hmm_native={mean_hmm_native:.3f}"
        )

    def test_unmodified_positions_stay_low(self):
        """Unmodified positions should remain low even with reduced smoothing."""
        cr = _make_contig_result(
            n_positions=20,
            n_native=15,
            n_ivt=10,
            modified_positions={10},
            position_start=100,
            position_step=1,
            seed=42,
        )
        result = compute_sequential_modification_probabilities(cr)
        # Check an unmodified position far from the modified one
        unmod_pos = 100  # index 0
        ps = result.position_stats[unmod_pos]
        mean_hmm_native = float(np.mean(ps.native_p_mod_hmm))
        assert mean_hmm_native < 0.3, (
            f"Unmodified position has too-high HMM: {mean_hmm_native:.3f}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hierarchical.py::TestHMMPreservesStrongSignal -v`
Expected: FAIL — `test_isolated_modified_position_preserved` fails because HMM crushes the signal.

- [ ] **Step 3: Lower default p_stay_per_base from 0.98 to 0.92**

In `baleen/eventalign/_hierarchical.py`, change the default parameter at line 967:

```python
# Before:
    hmm_p_stay_per_base: float = 0.98,

# After:
    hmm_p_stay_per_base: float = 0.92,
```

**Rationale:** 0.98^1 = 0.98 per base, meaning the HMM has 98% inertia at each step. For a modified position surrounded by 10 unmodified on each side, the smoothing is devastating. 0.92 reduces inertia so isolated peaks survive: 0.92^5 = 0.66 (still meaningful correlation at 5 bases) vs 0.98^5 = 0.90 (too rigid).

- [ ] **Step 4: Update HMMParams default p_stay_per_base**

In `baleen/eventalign/_hmm_training.py`, change line 169:

```python
# Before:
    p_stay_per_base: float = 0.98

# After:
    p_stay_per_base: float = 0.92
```

- [ ] **Step 5: Add emission floor to prevent HMM from zeroing out strong signals**

In `baleen/eventalign/_hierarchical.py`, in `_run_hmm_on_trajectories()`, after constructing emissions for each read at each position (after the if/elif emission construction block, around line 935), add a floor to prevent the HMM from completely ignoring strong emission evidence:

Find this block (around lines 911-915):
```python
            elif emission_transform is None:
                # Default 2-state: use raw p_mod directly
                p_mod_safe = max(min(p_mod, 1.0 - 1e-10), 1e-10)
                emissions[t_idx, 0] = 1.0 - p_mod_safe
                emissions[t_idx, 1] = p_mod_safe
```

Insert BEFORE `posteriors = _forward_backward(` at line 937, after all emission construction if/elif blocks complete:

```python
        # Emission floor: prevent any state from having near-zero emission,
        # which lets transition priors completely override observation evidence.
        emissions = np.maximum(emissions, 0.01)
        # Re-normalize rows
        row_sums = emissions.sum(axis=1, keepdims=True)
        emissions /= row_sums
```

- [ ] **Step 6: Run tests to verify fix works**

Run: `pytest tests/test_hierarchical.py::TestHMMPreservesStrongSignal -v`
Expected: PASS — both tests pass.

- [ ] **Step 7: Run full test suite to check for regressions**

Run: `pytest tests/test_hierarchical.py -v`
Expected: All existing tests pass.

- [ ] **Step 8: Commit**

```bash
git add baleen/eventalign/_hierarchical.py baleen/eventalign/_hmm_training.py tests/test_hierarchical.py
git commit -m "fix: reduce HMM over-smoothing that crushes modification signals

Lower p_stay_per_base from 0.98 to 0.92 and add emission floor (0.01)
to prevent transition priors from completely overriding observation
evidence at isolated modified positions."
```

---

### Task 2: Improve kNN Score Spread (RC2 — second highest impact)

**Why:** Raw kNN scores cluster in [0.3, 0.6] even for clearly modified reads because exponential distance weighting with median-based bandwidth compresses the range. 69% of known mod sites are affected.

**Files:**
- Modify: `baleen/eventalign/_probability.py:399-438`
- Test: `tests/test_probability.py`

- [ ] **Step 1: Write failing test — kNN scores separate well for block-structured matrices**

Add to `tests/test_probability.py`:

```python
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
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `pytest tests/test_probability.py::TestKnnScoreSpread -v`
Expected: `test_modified_reads_score_high` likely fails (scores compressed).

- [ ] **Step 3: Replace exponential weighting with rank-based kNN scoring**

Replace the function `_score_knn_ivt_purity` in `baleen/eventalign/_probability.py` (lines 399-438):

```python
def _score_knn_ivt_purity(
    distance_matrix: NDArray[np.float64],
    n_native: int,
    n_ivt: int,
    k: Optional[int] = None,
) -> NDArray[np.float64]:
    """Compute per-read kNN IVT-purity scores.

    Score = 1 - (IVT fraction among k nearest neighbors).
    Higher score = fewer IVT neighbors = more likely modified.
    Returns values in [0, 1].

    Uses unweighted neighbor counting for better score spread,
    then applies rank-based normalization to ensure scores span [0, 1].
    """
    n_total = n_native + n_ivt
    if k is None:
        k = int(_clip(round(math.sqrt(n_total)), 3, 15))
    k = min(k, n_total - 1)

    is_ivt = np.zeros(n_total, dtype=bool)
    is_ivt[n_native:] = True

    raw_scores = np.empty(n_total, dtype=np.float64)

    for i in range(n_total):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf  # exclude self
        neighbor_idx = np.argpartition(dists, k)[:k]

        # Unweighted IVT fraction among k neighbors
        ivt_count = int(np.sum(is_ivt[neighbor_idx]))
        raw_scores[i] = 1.0 - ivt_count / k

    # Rank-based normalization: map to [0, 1] using ranks
    # This ensures good spread regardless of the raw score distribution
    ranks = np.argsort(np.argsort(raw_scores)).astype(np.float64)
    scores = ranks / max(n_total - 1, 1)

    # Blend: 70% rank-based + 30% raw to preserve some absolute signal
    scores = 0.7 * scores + 0.3 * raw_scores

    return scores
```

Key changes:
- **Unweighted** neighbor counting instead of exponential weighting (simpler, more robust)
- **`np.argpartition`** instead of `np.argsort` (O(n) vs O(n log n))
- **Rank-based normalization** ensures scores span [0, 1] with good spread
- **k upper bound increased** from 10 to 15 for better statistics
- **70/30 blend** of rank-based + raw preserves absolute signal while improving spread

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_probability.py::TestKnnScoreSpread -v`
Expected: PASS

- [ ] **Step 5: Run full probability test suite**

Run: `pytest tests/test_probability.py -v`
Expected: All tests pass. Some tests that check exact kNN output values may need threshold adjustments — update if needed.

- [ ] **Step 6: Commit**

```bash
git add baleen/eventalign/_probability.py tests/test_probability.py
git commit -m "fix: improve kNN score spread with rank-based normalization

Replace exponential distance weighting with unweighted neighbor counting
plus rank-based normalization. Ensures kNN scores span [0, 1] with good
separation for clearly modified reads. Uses argpartition for O(n) perf."
```

---

### Task 3: Pi-Weighted Posteriors + Relaxed Soft Gate (RC1 + RC5)

**Why:** Beta calibration uses flat 50/50 prior for posteriors, ignoring the fitted mixing proportion π. When π=0.8 (most reads modified), posteriors should be higher. Also, soft gate tau parameters are too aggressive — partially closing the gate even when evidence is moderate.

**Files:**
- Modify: `baleen/eventalign/_probability.py:317-321` (Beta posteriors)
- Modify: `baleen/eventalign/_probability.py:226-229` (Normal posteriors)
- Modify: `baleen/eventalign/_hierarchical.py:488-489` (soft gate tau defaults)
- Test: `tests/test_probability.py`

- [ ] **Step 1: Write failing test — pi-weighted posteriors are higher when pi is high**

Add to `tests/test_probability.py`:

```python
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
```

- [ ] **Step 2: Run test to check current behavior**

Run: `pytest tests/test_probability.py::TestPiWeightedPosteriors -v`
Expected: `test_high_pi_gives_high_posteriors` may fail.

- [ ] **Step 3: Use pi-weighted posteriors in `_calibrate_beta()`**

In `baleen/eventalign/_probability.py`, replace lines 317-321:

```python
    # Before (flat 50/50 prior):
    f0_all = _beta_pdf(scores_all, a0, b0)
    f1_all = _beta_pdf(scores_all, a1, b1)
    denom_all = f0_all + f1_all + _EPS
    probs = f1_all / denom_all
    probs = np.clip(probs, 0.0, 1.0)

    # After (pi-weighted Bayesian posterior):
    f0_all = _beta_pdf(scores_all, a0, b0)
    f1_all = _beta_pdf(scores_all, a1, b1)
    denom_all = (1.0 - pi) * f0_all + pi * f1_all + _EPS
    probs = (pi * f1_all) / denom_all
    probs = np.clip(probs, 0.0, 1.0)
```

- [ ] **Step 4: Use pi-weighted posteriors in `_calibrate_normal()`**

In `baleen/eventalign/_probability.py`, replace lines 226-230:

```python
    # Before (flat 50/50 prior):
    f0_all = _normal_pdf(scores_all, mu0, sigma0)
    f1_all = _normal_pdf(scores_all, mu1, sigma1)
    denom_all = f0_all + f1_all + _EPS
    probs = f1_all / denom_all
    probs = np.clip(probs, 0.0, 1.0)

    # After (pi-weighted Bayesian posterior):
    f0_all = _normal_pdf(scores_all, mu0, sigma0)
    f1_all = _normal_pdf(scores_all, mu1, sigma1)
    denom_all = (1.0 - pi) * f0_all + pi * f1_all + _EPS
    probs = (pi * f1_all) / denom_all
    probs = np.clip(probs, 0.0, 1.0)
```

Also remove or update the comment at lines 223-225 that says "This avoids the problem where pi→1..." — this was the original rationale for flat prior, but pi-weighted is now correct because the BIC gate already protects against false positives.

- [ ] **Step 4b: Use pi-weighted posteriors in `mds_gmm()` inline computation**

In `baleen/eventalign/_probability.py`, replace lines 656-659:

```python
    # Before (flat 50/50 prior):
    f0_all = _mvn_pdf(coords, mu0, sigma0)
    f1_all = _mvn_pdf(coords, mu1, sigma1)
    denom_all = f0_all + f1_all + _EPS
    probs = f1_all / denom_all

    # After (pi-weighted Bayesian posterior):
    f0_all = _mvn_pdf(coords, mu0, sigma0)
    f1_all = _mvn_pdf(coords, mu1, sigma1)
    denom_all = (1.0 - pi) * f0_all + pi * f1_all + _EPS
    probs = (pi * f1_all) / denom_all
```

- [ ] **Step 5: Relax soft gate tau parameters**

In `baleen/eventalign/_hierarchical.py`, update the default parameters in `_anchored_mixture_em()` at lines 488-490:

```python
    # Before:
    tau_pi: float = 0.02,
    tau_bic: float = 5.0,
    tau_sep: float = 0.3,

    # After:
    tau_pi: float = 0.05,
    tau_bic: float = 10.0,
    tau_sep: float = 0.5,
```

This makes the soft gate less aggressive — it will only substantially fire when there is strong evidence AGAINST modification, rather than partially firing at moderate evidence levels.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_probability.py::TestPiWeightedPosteriors -v`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add baleen/eventalign/_probability.py baleen/eventalign/_hierarchical.py tests/test_probability.py
git commit -m "fix: use pi-weighted posteriors and relax soft gate

Replace flat 50/50 prior with pi-weighted Bayesian posteriors in both
_calibrate_beta() and _calibrate_normal(). When EM determines most reads
are modified (high pi), posteriors now correctly reflect this.

Also relax soft gate tau parameters to prevent premature gating at
positions with moderate modification evidence."
```

---

### Task 4: Update Diagnostic Notebook + Verify on Real Data

**Why:** After implementing fixes, the diagnostic notebook should be updated to compare before/after, and the user needs to run it on their data server to verify improvements.

**Files:**
- Modify: `notebooks/calibration_diagnostic.ipynb`

- [ ] **Step 1: Add comparison section to notebook**

Add a new cell at the end of `notebooks/calibration_diagnostic.ipynb` that summarizes the expected improvement:

```python
print("\n" + "=" * 70)
print("POST-FIX VERIFICATION")
print("=" * 70)
print("\nIf fixes are applied correctly, you should see:")
print("  1. HMM P(mod) for known mod sites: > 0.5 (was ~0.02-0.46)")
print("  2. Raw kNN native mean for mod sites: > 0.6 (was ~0.3-0.6)")
print("  3. Cal kNN native mean for mod sites: > 0.7 (was ~0.4-0.9)")
print("  4. Gate weight: > 0.9 at most mod sites")
print("\nRe-run this notebook after applying fixes to verify.")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/calibration_diagnostic.ipynb
git commit -m "docs: update diagnostic notebook with post-fix verification"
```

---

### Task 5: Run Full Test Suite and Integration Verification

**Files:**
- All test files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 2: Verify no import errors**

Run: `python -c "from baleen.eventalign._hierarchical import compute_sequential_modification_probabilities; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify pipeline can still run end-to-end (dry run)**

Run: `python -c "
from baleen.eventalign._hierarchical import compute_sequential_modification_probabilities
from baleen.eventalign._pipeline import ContigResult, PositionResult
import numpy as np

# Quick smoke test with synthetic data
rng = np.random.RandomState(42)
n_nat, n_ivt = 10, 10
n = n_nat + n_ivt
dm = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        if i < n_nat and j < n_nat:
            d = 1.0
        elif i >= n_nat and j >= n_nat:
            d = 1.0
        else:
            d = 5.0
        d += rng.normal(0, 0.1)
        dm[i,j] = dm[j,i] = max(d, 0)

pr = PositionResult(
    position=100,
    reference_kmer='AACGT',
    n_native_reads=n_nat,
    n_ivt_reads=n_ivt,
    native_read_names=[f'n{i}' for i in range(n_nat)],
    ivt_read_names=[f'i{i}' for i in range(n_ivt)],
    distance_matrix=dm,
)
cr = ContigResult(contig='test', native_depth=10.0, ivt_depth=10.0, positions={100: pr})
result = compute_sequential_modification_probabilities(cr)
ps = result.position_stats[100]
print(f'HMM native mean: {np.nanmean(ps.native_p_mod_hmm):.3f}')
print(f'HMM IVT mean: {np.nanmean(ps.ivt_p_mod_hmm):.3f}')
assert np.nanmean(ps.native_p_mod_hmm) > 0.5, 'Native HMM too low'
assert np.nanmean(ps.ivt_p_mod_hmm) < 0.3, 'IVT HMM too high'
print('Smoke test PASSED')
"
```
Expected: `Smoke test PASSED`

- [ ] **Step 4: Final commit if any adjustments were needed**

```bash
git add -u
git commit -m "fix: adjust test thresholds after calibration improvements"
```
