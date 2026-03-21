# Baleen Future Improvements Roadmap

**Generated:** 2026-03-21
**Status:** Ideas for future development

---

## Quick Wins (High Impact, Low Effort)

| # | Category | Improvement | Description | Estimated Effort |
|---|----------|-------------|-------------|-----------------|
| 1 | Memory | Float32 distance matrices | Use float32 instead of float64 for distance matrices → 50% memory reduction | 1 line |
| 2 | UX | Progress bars | Add tqdm progress bars to contig/position loops | 30 min |
| 3 | Speed | Pre-compute IVT-IVT distances | IVT pairwise distances don't change per position → O(n_ivt²) saved per position | 1-2 hours |
| 4 | Speed | Sakoe-Chiba band DTW | Restrict warping path to diagonal band → 2-5x speedup, ~1% accuracy loss | 2-3 hours |
| 5 | Stats | FDR correction | Add Benjamini-Hochberg q-values to output | 1 hour |
| 6 | Storage | HDF5 output | Alternative to pickle with compression → 5-10x storage reduction | 4 hours |
| 7 | UX | Checkpointing | Resume interrupted runs | 4-6 hours |

---

## Category 1: Detection Performance (Accuracy)

### 1.1 Signal Feature Engineering
- [ ] **Multi-feature DTW**: Add stdv, duration, dwell time as additional dimensions beyond mean signal
- [ ] **Derivative DTW**: Use signal slope (dI/dt) instead of raw current to reduce baseline drift
- [ ] **K-mer context expansion**: Try 7-mer or 9-mer instead of 5-mer for longer-range context
- [ ] **Per-read signal normalization**: Z-score normalize each read's signal before DTW

### 1.2 kNN Improvements
- [ ] **Adaptive k selection**: Use silhouette score or elbow method instead of `k = sqrt(n_total)`
- [ ] **Rank-based weighting**: Alternative to exp(-d/η) weighting
- [ ] **Local outlier factor (LOF)**: Replace kNN purity with LOF anomaly score
- [ ] **IVT-native ratio weighting**: Weight kNN score by overall IVT:native ratio

### 1.3 HMM Enhancements
- [ ] **Learned transitions**: Learn P(stay) from labeled data instead of fixed 0.98
- [ ] **5-state HMM**: Add "strong" and "weak" modification states
- [ ] **Hierarchical HMM**: Two-level model (site + read level)
- [ ] **Duration modeling (HSMM)**: Explicit state duration instead of geometric

---

## Category 2: Computational Efficiency

### 2.1 DTW Optimizations
- [ ] **Pruned DTW (Sakoe-Chiba band)**: 2-5x faster
- [ ] **FastDTW approximation**: 10-100x faster with multi-resolution
- [ ] **Sparse distance matrix**: Only compute distances within threshold
- [ ] **Batch GPU DTW**: Process all positions together for better GPU utilization

### 2.2 Memory Optimizations
- [ ] **Streaming DTW results**: Write to disk immediately → O(1) memory per contig
- [ ] **Float32 storage**: 50% memory reduction
- [ ] **HDF5 chunked storage**: 5-10x storage reduction with compression
- [ ] **Lazy signal loading**: Load on-demand for low-memory systems

### 2.3 Parallelization
- [ ] **GPU kNN**: Move kNN computation to CUDA → 10-50x faster
- [ ] **GPU HMM forward-backward**: 5-20x faster
- [ ] **Position-level parallelism**: Process multiple positions in parallel
- [ ] **Distributed processing**: Ray/Dask for multi-node scaling

---

## Category 3: Statistical Methodology

### 3.1 Null Distribution
- [ ] **Position-specific nulls**: Fit separate null per k-mer type
- [ ] **Empirical null from shuffling**: Permute IVT labels for non-parametric null
- [ ] **Bayesian hierarchical null**: Full Bayesian treatment with priors
- [ ] **Mixture null**: Account for IVT subpopulations

### 3.2 Calibration
- [ ] **Platt scaling with CV**: Cross-validate to avoid overfitting
- [ ] **Isotonic regression**: Non-parametric calibration
- [ ] **Temperature scaling**: Simple post-hoc calibration
- [ ] **Ensemble calibrators**: Combine Beta + Platt + Isotonic

### 3.3 Multiple Testing
- [ ] **Benjamini-Hochberg FDR**: Control false discovery rate
- [ ] **Storey's q-value**: Adaptive FDR with π₀ estimation
- [ ] **Independent hypothesis weighting**: Weight by coverage
- [ ] **Spatial FDR**: Account for nearby position correlation

---

## Category 4: New Capabilities

### 4.1 Modification Type Classification
- [ ] **Multi-class HMM**: States for m6A, Ψ, m5C, etc.
- [ ] **One-vs-rest classifiers**: Separate model per modification type
- [ ] **Transfer learning**: Use MODOMICS database

### 4.2 Quantification
- [ ] **Stoichiometry estimation**: Estimate % modified at each site
- [ ] **Beta-binomial modeling**: Account for overdispersion
- [ ] **Read-level mixture deconvolution**: Separate mod/unmod distributions

### 4.3 Robustness
- [ ] **Batch effect correction**: Run-to-run variability
- [ ] **Negative control positions**: Known-unmodified sites for calibration
- [ ] **Bootstrap confidence intervals**: Resample reads for CI
- [ ] **Outlier read removal**: Filter aberrant reads before DTW

---

## Category 5: User Experience

### 5.1 CLI
- [ ] Progress bars (tqdm)
- [ ] Checkpointing / resume
- [ ] Dry-run mode
- [ ] YAML config file support

### 5.2 Output
- [ ] UCSC track hub output
- [ ] Per-read BAM tags with P(mod)
- [ ] Interactive HTML report
- [ ] BED output with scores

### 5.3 Testing
- [ ] Synthetic spike-in validation
- [ ] Cross-sample reproducibility metrics
- [ ] Benchmark suite with known modifications

---

## Priority Matrix

```
                    HIGH IMPACT
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    │  GPU kNN/HMM       │  Modification      │
    │  Sakoe-Chiba DTW   │  classification    │
    │  Float32 storage   │  Stoichiometry     │
    │                    │                    │
LOW │────────────────────┼────────────────────│ HIGH
    │                    │                    │ EFFORT
    │  Progress bars     │  Hierarchical HMM  │
    │  FDR correction    │  Full Bayesian     │
    │  HDF5 output       │  Distributed       │
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
                    LOW IMPACT
```

---

## Notes

- Items marked with ⭐ are recommended starting points
- Effort estimates assume familiarity with the codebase
- Impact estimates based on expected accuracy/speed improvements
