# Baleen Development Roadmap

**Consolidated:** 2026-04-01
**Source:** TODO_fixes.md (all done), TODO_improvements.md, FUTURE_IMPROVEMENTS.md, BENCHMARK_TODO.md, superpowers plans

---

## Completed

The following items have been finished and no longer need tracking:

- **Pre-publication fixes (10/10)**: Forward-backward normalization, posterior mixing weights, BIC log-sum-exp, EM convergence, Beta overflow, slogdet, return type, CI test job, dependency versions, integration tests
- **GPU batch DTW**: `dtw_multi_position_pairwise` with CUDA streams (16 concurrent positions)
- **Position-level parallelism**: Round-robin across CUDA streams
- **FDR correction**: Benjamini-Hochberg in `_aggregation.py`
- **Progress bars**: Stage-aware tqdm
- **Read-level BAM output**: `write_read_bam()`, `load_read_results()`, `--no-read-bam` CLI flag
- **Benchmark scripts**: `run_benchmark.sh` (multi-threshold × multi-scoring), `evaluate_benchmark.py` (site + transcript level, 6 score columns)
- **Docker `latest` tag**: Fixed to trigger on `dev` branch
- **HMM over-smoothing fix**: `p_stay_per_base` 0.98→0.92, emission floor 0.01 (底层函数签名默认值待统一)
- **kNN score spread fix**: 无权重邻居计数 + rank-based normalization + 70/30 blend
- **Pi-weighted posteriors**: `_calibrate_normal`, `_calibrate_beta`, `mds_gmm` 全部使用贝叶斯后验
- **Soft gate tau relaxed**: tau_pi 0.02→0.05, tau_bic 5→10, tau_sep 0.3→0.5

---

## P1: Algorithm & Numerical Issues (Not Yet Fixed)

### 4. `use_global=True` overestimates pi

`_hierarchical.py:562-570` — Freezes mu1/sigma1 while updating pi in constrained EM, tends to overestimate when global alternative poorly fits local data.

### 5. 3-state HMM Flank initial probability

`_hierarchical.py:817`, `_hmm_training.py:171` — `init_prob=[0.7, 0.2, 0.1]`: 20% Flank weight gets redistributed to Modified in posterior, dominates for trajectories <5 positions.

---

## P1: Performance Bottlenecks

### 6. Python loops in distance scoring

`_probability.py:347-357`, `_hierarchical.py:207-215` — Per-read median to IVT via Python loop. **Vectorizable:** `np.median(D[:, n_native:], axis=1)`.

### 7. kNN per-read copy

`_probability.py:428` — `.copy()` per iteration = O(n) allocation per read. Use mask/filter on `argpartition` result instead.

### 8. Pure Python DTW fallback (no warning)

`_cuda_dtw/__init__.py:136-160` — Nested Python for-loop: 200×200 = 40k iterations per pair. No warning when this path is used.

### 9. MDS O(n²) centering matrix

`_probability.py:499-502` — `H = eye(n) - ones(n,n)/n` materializes full matrix. **Fix:** `B = -0.5*(D² - row_mean - col_mean + grand_mean)`.

### 10. Global max padding in multi-position DTW

`_cuda_dtw/__init__.py:532-543` — One long signal forces all positions to pad to its length. Per-position padding would reduce memory.

---

## P1: CI/CD & Packaging

### 11. Stale Makefile

`_cuda_dtw/Makefile` — References nonexistent `dtw_c_api.cpp` (actual: `dtw_api.cpp`). Not used in build. Delete.

### 13. C++ standard outdated

`setup.py:189` — Uses `std=c++11`; CUDA 12.x supports c++17.

---

## P2: Test Coverage Gaps

| # | Gap | Location |
|---|-----|----------|
| 14 | `_contig_pooled_mixture_em` — zero tests | `_hierarchical.py:403` |
| 15 | CLI flags untested: `--no-hmm`, `--hmm-params`, `--no-cuda`, `_cmd_aggregate` w/ BAM | `cli.py` |
| 16 | Variable-length DTW input validation | `_cuda_dtw/__init__.py:353-358` |
| 17 | `FilterReason.NO_MAPPED_READS_*` | `_bam.py:335-357` |
| 18 | Multi-contig integration test | single-contig only in `TestRunPipeline` |
| 19 | `run_pipeline` with `threads > 1` | ProcessPoolExecutor branch never exercised |
| 20 | `run_pipeline` with `padding > 0` | padded signal path untested |
| 21 | `_parse_int`, `_parse_float`, `_parse_samples` | `_signal.py:46-61`, only indirect |
| 22 | Duplicate test helpers | `_make_block_distance_matrix` etc. in 3 files → consolidate to `conftest.py` |

---

## P2: Code Quality

| # | Issue | Location |
|---|-------|----------|
| 23 | Dead `_DtwDistanceFn` Protocol | `_pipeline.py:36-44` |
| 24 | `_compute_pairwise_batch` misleading name (plain loop, not batch) | `_pipeline.py:127-140` |
| 25 | `EmissionTransform` is a string literal, not type alias | `_hmm_training.py:71` |
| 26 | `load_results` no error handling (corrupt pickle, missing keys) | `_pipeline.py:183-187` |
| 27 | Dead `isinstance(loaded, tuple)` branch in `_cmd_aggregate` | `cli.py:301-304` |
| 28 | `PathLike` redefined in 4 modules | `_pipeline.py`, `_bam.py`, `_f5c.py`, `_read_bam.py` |
| 29 | `_EPS` and `_MIN_SIGMA` duplicated | `_probability.py:40-41`, `_hierarchical.py:54-55` |
| 30 | `write_read_bam` not in `__all__` (CLI imports from private module) | `__init__.py` |
| 31 | `position=-1` magic sentinel → use `Optional[int]` | `_probability.py:387,479,597,674` |
| 32 | Closure inside loop → `functools.partial` | `_bam.py:250-251` |
| 33 | Dead `pass` block | `_cuda_dtw/__init__.py:373-378` |
| 34 | Unnecessary deferred imports | `_hierarchical.py:419,524` |
| 35 | Triple duplication of pairwise loop skeleton | `_cuda_dtw`, `_pipeline.py` |
| 36 | `validate_bam` called redundantly (2× per contig) | `_bam.py`, `_pipeline.py` |
| 37 | Parallel mode loses contig name on failure | `_pipeline.py:597-601` |
| 38 | `score_field: str` not type-validated → `Literal[...]` | `_aggregation.py:151-154` |
| 39 | Pickle format undocumented, no schema versioning | `_pipeline.py:170-180` |

---

## P3: Packaging & Polish

| # | Issue |
|---|-------|
| 40 | No `license` field in pyproject.toml |
| 41 | Stale `egg-info/requires.txt` (missing tqdm, pandas) |
| 42 | No Docker layer caching in CI |
| 43 | CI triggers on every push to every branch (`branches: ["**"]`) |
| 44 | Dockerfile.cpu defensive workaround (can be removed now) |

---

## Future: Detection Performance

### Signal Feature Engineering
- K-mer context expansion (7-mer / 9-mer)
- Per-read signal normalization (Z-score before DTW)

### kNN Enhancements
- Adaptive k selection (silhouette / elbow) — **deprioritized** (significant compute overhead)
- ~~Local outlier factor (LOF) anomaly score~~ — done: `_score_lof()` + blended into `knn_ivt_purity` (30% LOF weight)
- ~~IVT-native ratio weighting~~ — done: `ratio_correction` in `_score_knn_ivt_purity`, adjusts expected IVT fraction per read

### HMM Enhancements
- Learned transitions from labeled data
- 5-state HMM (strong/weak modification)
- Hierarchical HMM (site + read level)
- Duration modeling (HSMM)

---

## Future: Computational Efficiency

### DTW
- Pruned DTW (Sakoe-Chiba band) — 2-5x faster
- FastDTW approximation — 10-100x faster
- Sparse distance matrix (only within threshold)

### Memory
- Float32 distance matrices — 50% reduction (1 line change)
- Streaming DTW results — O(1) memory per contig
- HDF5 chunked storage — 5-10x storage reduction
- Lazy signal loading

### Parallelization
- CPU-GPU overlap (parse next batch while GPU computes current)
- f5c parallelism (concurrent subprocesses — often the actual wall-clock bottleneck)
- Distributed processing (Ray/Dask)
- ~~GPU kNN~~ / ~~GPU HMM~~: deprioritized (<1% of pipeline time)

---

## Future: Statistical Methodology

### Null Distribution
- Position-specific nulls per k-mer type
- Empirical null from shuffling
- Bayesian hierarchical null
- Mixture null for IVT subpopulations

### Calibration
- Platt scaling with CV
- Isotonic regression
- Temperature scaling
- Ensemble calibrators

### Multiple Testing
- Storey's q-value (adaptive FDR with pi0)
- Independent hypothesis weighting (weight by coverage)
- Spatial FDR (nearby position correlation)

---

## Future: New Capabilities

### Modification Type Classification
- Multi-class HMM (m6A, psi, m5C, etc.)
- One-vs-rest classifiers
- Transfer learning via MODOMICS

### Quantification
- Stoichiometry estimation
- Beta-binomial modeling (overdispersion)
- Read-level mixture deconvolution

### Robustness
- Batch effect correction (run-to-run)
- Negative control positions for calibration
- Bootstrap confidence intervals
- Outlier read removal

---

## Future: User Experience

### CLI
- Checkpointing / resume interrupted runs
- Dry-run mode
- YAML config file support

### Output
- UCSC track hub output
- Interactive HTML report
- BED output with scores

### Validation
- Synthetic spike-in validation
- Cross-sample reproducibility metrics
