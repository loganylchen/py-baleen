# Baleen Pre-Publication Fix Tracker

Prioritized from TODO_improvements.md review (2026-03-25).
Ordered by impact on modification calling accuracy.

---

## Tier 1 — Directly affects modification calling accuracy

### Fix 1: Forward-backward backward pass normalization (TODO #3)
**File:** `baleen/eventalign/_hierarchical.py` (`_forward_backward`)
**Issue:** Backward pass normalizes `beta[t]` by its own sum instead of forward
scale factors. Non-standard; introduces subtle posterior bias especially for
short trajectories and extreme emissions.
**Fix:** Use forward scale factors for backward normalization (standard scaled F-B).
**Status:** ✅ Fixed (2026-03-25)

### Fix 2: `_anchored_mixture_em` posterior missing mixing weights (TODO #1)
**File:** `baleen/eventalign/_hierarchical.py` (`_anchored_mixture_em`)
**Issue:** `raw_posterior = f1 / (f0 + f1)` omits pi weights. When pi is small
(e.g. 0.05), P(mod) is systematically inflated → higher false positive rate.
**Fix:** Use `(pi * f1) / ((1-pi) * f0 + pi * f1)`.
**Status:** ✅ Fixed (2026-03-25)

### Fix 3: BIC log-likelihood _EPS guard fragile (TODO #5)
**Files:** `baleen/eventalign/_probability.py` (`_calibrate_normal`, `_calibrate_beta`),
`baleen/eventalign/_hierarchical.py` (`_anchored_mixture_em`)
**Issue:** `log((1-pi)*f0 + pi*f1 + 1e-300)` — when PDFs underflow to 0,
log(1e-300) ≈ -690 distorts BIC, causing null gate to fire on truly modified
positions (false negatives).
**Fix:** Added `_normal_logpdf`, `_beta_logpdf`, `_mixture_log_likelihood` helpers.
All three BIC computation sites now use log-sum-exp.
**Status:** ✅ Fixed (2026-03-25)

---

## Tier 2 — Affects results under specific conditions

### Fix 4: EM convergence checks incomplete (TODO #7)
**Files:** `baleen/eventalign/_probability.py` (`_calibrate_normal`, `_calibrate_beta`)
**Issue:** `_calibrate_normal` doesn't check `sigma1`; `_calibrate_beta` only
checks `pi`, not `a1`/`b1`. Parameters may still drift at loop exit.
**Fix:** Check convergence of all updated parameters (pi, mu1, sigma1 / pi, a1, b1).
**Status:** ✅ Fixed (2026-03-25)

### Fix 5: Beta method-of-moments `common` overflow (TODO #2)
**File:** `baleen/eventalign/_probability.py` (`_fit_beta`, `_calibrate_beta`)
**Issue:** No upper bound on `common`; can reach 2.5e9 for small variance,
collapsing Beta to Dirac delta.
**Fix:** `common = max(min(common, 1000.0), 2.0)` in both `_fit_beta` and
`_calibrate_beta` EM loop.
**Status:** ✅ Fixed (2026-03-25)

### Fix 6: `_mvn_pdf` uses `det` instead of `slogdet` (TODO #6)
**File:** `baleen/eventalign/_probability.py` (`_mvn_pdf`)
**Issue:** `np.linalg.det` can underflow/overflow for near-singular covariances.
**Fix:** Use `np.linalg.slogdet` and compute norm factor in log-space:
`log_norm = -0.5 * (d * log(2π) + logdet)`, then `exp(log_norm + exponent)`.
**Status:** ✅ Fixed (2026-03-25)

---

## Tier 3 — Reproducibility and user experience

### Fix 7: `_contig_pooled_mixture_em` return type (TODO #4)
**File:** `baleen/eventalign/_hierarchical.py`
**Status:** ⬚ TODO

### Fix 8: CI has no test step (TODO #15)
**File:** `.github/workflows/docker.yml`
**Status:** ⬚ TODO

### Fix 9: Dependencies have no version constraints (TODO #18)
**Files:** `pyproject.toml`, `setup.py`
**Status:** ⬚ TODO

### Fix 10: No end-to-end integration test (TODO #26)
**Status:** ⬚ TODO
