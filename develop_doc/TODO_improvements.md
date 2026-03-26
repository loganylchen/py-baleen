# Baleen Improvement Tracker

Code review conducted 2026-03-24. Organized by priority.

---

## P0: Algorithm Correctness Bugs

### 1. `_anchored_mixture_em` posterior missing mixing weights
**File:** `baleen/eventalign/_hierarchical.py:603-606`

```python
denom_all = f0_all + f1_all + _EPS
raw_posterior = f1_all / denom_all
```

Should be `(pi * f1) / ((1-pi)*f0 + pi*f1)`. Without weights, when `pi` is small (e.g. 0.05), the posterior for the alternative component is artificially inflated. `_calibrate_normal` in `_probability.py:229-230` does this correctly — inconsistency between the two code paths.

### 2. Beta method-of-moments `common` can overflow
**File:** `baleen/eventalign/_probability.py:131-136`

`v` clamped to `1e-10` means `common` can reach `2.5e9` for `m=0.5`. Beta parameters become extreme (Dirac delta), EM collapses. Fix: add `common = min(common, 1000)`.

### 3. Forward-backward backward pass normalization is non-standard
**File:** `baleen/eventalign/_hierarchical.py:801-810`

`beta[t]` normalized by its own sum instead of forward scale factors. Mathematically different from scaled forward-backward. Can introduce subtle posterior bias. When `bt_sum=0`, setting `beta[t] = ones` produces degenerate posteriors. Final gamma renormalization (line 813-815) partially corrects but doesn't fully compensate.

### 4. `_contig_pooled_mixture_em` return type annotation is wrong
**File:** `baleen/eventalign/_hierarchical.py:403-431`

Declares `-> tuple[float, float]` but returns `(None, None)` when `len(all_z_native) < 3`. Should be `tuple[float | None, float | None]`.

---

## P1: Numerical Stability

### 5. `_EPS` guard in log-likelihood is fragile
**File:** `baleen/eventalign/_probability.py:208,302`

`np.log((1-pi)*f0 + pi*f1 + _EPS)` — when PDFs underflow to exact 0 (30+ sigma), `_EPS=1e-300` produces `log(1e-300) = -690`, distorting BIC comparisons. More robust: compute in log-space with log-sum-exp.

### 6. `_mvn_pdf` uses `np.linalg.det` instead of `slogdet`
**File:** `baleen/eventalign/_probability.py:554-564`

For degenerate covariances, `det` can be `1e-30`, making norm factor enormous and forcing posteriors to 0/1 regardless of evidence. Fix: use `np.linalg.slogdet` and work in log-space.

### 7. EM convergence checks are incomplete
**File:** `baleen/eventalign/_probability.py:198,292`

- `_calibrate_normal`: doesn't check `sigma1` convergence
- `_calibrate_beta`: only checks `pi`, not `a1`/`b1`

Parameters can still be drifting when loop exits.

### 8. `_anchored_mixture_em` with `use_global=True` overestimates `pi`
**File:** `baleen/eventalign/_hierarchical.py:562-570`

Freezes `mu1`/`sigma1` while updating `pi` — valid constrained EM but tends to overestimate `pi` when global alternative is poorly aligned with local data.

### 9. 3-state HMM Flank initial probability inflates P(Modified) for short trajectories
**File:** `baleen/eventalign/_hierarchical.py:817`, `_hmm_training.py:171`

`init_prob = [0.7, 0.2, 0.1]` — the 20% Flank weight gets redistributed to Modified in the posterior. For trajectories < 5 positions this can dominate.

---

## P1: Performance Bottlenecks

### 10. Python loops in `_score_distance_to_ivt` and `_extract_ivt_distances`
**Files:** `baleen/eventalign/_probability.py:347-357`, `baleen/eventalign/_hierarchical.py:207-215`

Per-read median distances to IVT via `for i in range(n_total)`. Vectorizable: `np.median(D[:, n_native:], axis=1)`.

### 11. kNN copies distance row per read
**File:** `baleen/eventalign/_probability.py:428`

`.copy()` per iteration creates O(n) allocation per read. Use mask or filter argpartition result instead.

### 12. Pure Python DTW open-boundary fallback
**File:** `baleen/_cuda_dtw/__init__.py:136-160`

Nested Python for-loop: 200x200 signal = 40k iterations per pair. No warning emitted when falling back to this path.

### 13. `_classical_mds` allocates unnecessary O(n²) matrices
**File:** `baleen/eventalign/_probability.py:499-502`

`H = np.eye(n) - np.ones((n,n))/n` materializes full centering matrix. Replace with: `B = -0.5 * (D_sq - D_sq.mean(axis=1, keepdims=True) - D_sq.mean(axis=0, keepdims=True) + D_sq.mean())`.

### 14. `dtw_multi_position_pairwise` pads to global max length
**File:** `baleen/_cuda_dtw/__init__.py:532-543`

One long signal forces all positions to pad to its length. Per-position padding would reduce memory for heterogeneous signal lengths.

---

## P1: CI/CD and Packaging

### 15. CI has no test step
**File:** `.github/workflows/docker.yml`

Builds and pushes Docker images without ever running pytest. Add `pip install . && pytest` job gating Docker builds.

### 16. GPU Docker image is not actually GPU-enabled
**File:** `.github/workflows/docker.yml`

`build-gpu` runs on `ubuntu-latest` (no nvcc), setup.py silently falls back to CPU-only. The image labeled GPU has no CUDA.

### 17. `latest` Docker tag never applied
**File:** `.github/workflows/docker.yml`

Trigger condition is `refs/heads/main` but default branch is `dev`. Change to `refs/heads/dev`.

### 18. Dependencies have no version constraints
**Files:** `pyproject.toml`, `setup.py`

All deps are bare names. Suggested minimums: `numpy>=1.24`, `scipy>=1.9`, `tslearn>=0.6`, `pysam>=0.21`, `tqdm>=4.60`, `pandas>=1.5`.

### 19. No pytest configuration
**File:** `pyproject.toml`

Add `[tool.pytest.ini_options]` with `testpaths = ["tests"]` to avoid full filesystem scan.

### 20. Stale `_cuda_dtw/Makefile`
**File:** `baleen/_cuda_dtw/Makefile`

References nonexistent `dtw_c_api.cpp` (actual file is `dtw_api.cpp`). Not used in build. Delete it.

### 21. `setup.py` uses `std=c++11` for CUDA
**File:** `setup.py:189`

CUDA 12.x supports c++17. Current flag limits host-side code.

---

## P2: Test Coverage Gaps

### 22. `_contig_pooled_mixture_em` — zero test coverage
**File:** `baleen/eventalign/_hierarchical.py:403`

Every other private function in `_hierarchical.py` has tests except this one.

### 23. CLI flag paths untested
**File:** `baleen/cli.py`

- `--no-hmm` branch (lines 232-253)
- `--hmm-params` file loading
- `--no-read-bam` flag
- `--no-cuda` resolving to `use_cuda=False`
- `_cmd_aggregate` with `--ref`/BAM writing path

### 24. `dtw_pairwise` variable-length input validation untested
**File:** `baleen/_cuda_dtw/__init__.py:353-358`

`ValueError` for heterogeneous lengths in list input — no test for `dtw_pairwise([[1,2], [1,2,3]])`.

### 25. `FilterReason.NO_MAPPED_READS_*` untested
**File:** `baleen/eventalign/_bam.py:335-357`

Reachable when `min_mapq` filtering removes all reads after index stats check.

### 26. No end-to-end integration test
DTW → HMM → aggregation → TSV — no single test chains all four stages.

### 27. No multi-contig integration test
`TestRunPipeline` only uses single-contig BAMs.

### 28. No test for `run_pipeline` with `threads > 1`
Multi-threaded `ProcessPoolExecutor` branch never exercised in tests.

### 29. No test for `run_pipeline` with `padding > 0`
Padded signal extraction path never exercised through full pipeline.

### 30. `_parse_int`, `_parse_float`, `_parse_samples` — no unit tests
**File:** `baleen/eventalign/_signal.py:46-61`

Only exercised indirectly through `parse_eventalign`.

### 31. Duplicate test helpers across files
`_make_block_distance_matrix` and `_make_contig_result` copy-pasted across `test_hierarchical.py`, `test_hmm_training.py`, `test_aggregation.py`. Should be in shared `conftest.py`.

---

## P2: Code Quality

### 32. Dead `_DtwDistanceFn` Protocol
**File:** `baleen/eventalign/_pipeline.py:36-44`

Defined, used only in a `cast()`, never appears in any function signature.

### 33. `_compute_pairwise_batch` misleadingly named
**File:** `baleen/eventalign/_pipeline.py:127-140`

Name implies batch/cdist but implementation is a plain loop. `_cuda_dtw._dtw_pairwise_cpu` does use `cdist_dtw`.

### 34. `EmissionTransform` is a string, not a type alias
**File:** `baleen/eventalign/_hmm_training.py:71`

`EmissionTransform = "EmissionCalibrator | EmissionKDE | None"` — provides no type-checking value.

### 35. `load_results` has no error handling
**File:** `baleen/eventalign/_pipeline.py:183-187`

No guard for missing files, corrupt pickles, or old-schema files missing `"metadata"` key.

### 36. Dead branch in `_cmd_aggregate`
**File:** `baleen/cli.py:301-304`

`isinstance(loaded, tuple)` else branch is unreachable — `load_results` always returns a tuple.

### 37. `PathLike` redefined in 4 modules
**Files:** `_pipeline.py:33`, `_bam.py:79`, `_f5c.py:19`, `_read_bam.py:37`

Same `Union[str, Path]` alias. Should be in shared module or just use `str | Path`.

### 38. `_EPS` and `_MIN_SIGMA` duplicated
**Files:** `_probability.py:40-41`, `_hierarchical.py:54-55`

Identical constants defined independently. Should be shared.

### 39. `write_read_bam` not in public API
**Files:** `baleen/eventalign/__init__.py`, `baleen/cli.py:37`

CLI imports from private `_read_bam` module directly. Either export or document as CLI-only.

### 40. `position=-1` magic sentinel
**File:** `baleen/eventalign/_probability.py:387,479,597,674`

Used in `ModificationProbabilities` then mutated post-construction. `Optional[int]` with `None` would be cleaner.

### 41. Closure created inside loop
**File:** `baleen/eventalign/_bam.py:250-251`

`_coverage_read_callback` captures loop-invariant values. Define once outside with `functools.partial`.

### 42. Dead `pass` block in `dtw_pairwise`
**File:** `baleen/_cuda_dtw/__init__.py:373-378`

`if sequences.shape[0] > 1: pass` with a comment saying it does nothing. Remove.

### 43. Deferred imports in `_hierarchical.py` are unnecessary
**File:** `baleen/eventalign/_hierarchical.py:419,524`

Imports from `_probability.py` inside function bodies to avoid circular imports, but no runtime circular dependency exists.

### 44. Triple duplication of pairwise distance loop skeleton
**Files:** `_cuda_dtw/__init__.py:197-208`, `_pipeline.py:143-167`, `_cuda_dtw/__init__.py:465-476`

Same `for i in range(n): for j in range(i+1, n)` pattern in three places.

### 45. `validate_bam` called redundantly
**Files:** `baleen/eventalign/_bam.py:219,424`, `baleen/eventalign/_pipeline.py:492-493`

Each BAM validated at least twice: once in pipeline setup, once per contig split. `n_contigs * 2` redundant pysam opens.

### 46. Parallel mode loses contig name on failure
**File:** `baleen/eventalign/_pipeline.py:597-601`

`future.result()` re-raises with no indication of which contig failed, even though `futures[future]` has the name.

### 47. `score_field: str` not type-validated
**File:** `baleen/eventalign/_aggregation.py:151-154`

`getattr(ps, score_field)` with no validation. Should use `Literal["p_mod_hmm", "p_mod_knn", "p_mod_raw"]`.

### 48. Pickle format undocumented
**File:** `baleen/eventalign/_pipeline.py:170-180`

`save_results`/`load_results` use pickle with no schema versioning or format documentation. Cross-version compatibility risk.

---

## P3: Documentation / Packaging Polish

### 49. No README.md at project root
### 50. No `license` field in pyproject.toml or setup.py
### 51. `author` field only in setup.py, no email
### 52. Dockerfile.cpu has defensive workaround that can be removed now that entry_points work
### 53. `egg-info/requires.txt` is stale (missing `tqdm` and `pandas`)
### 54. No Docker layer caching in CI
### 55. CI triggers on every push to every branch (`branches: ["**"]`)
