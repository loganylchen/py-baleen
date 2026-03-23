# Baleen Future Improvements Roadmap

**Generated:** 2026-03-21
**Updated:** 2026-03-22
**Status:** Ideas for future development

---

## Quick Wins (High Impact, Low Effort)

| # | Category | Improvement | Description | Estimated Effort |
|---|----------|-------------|-------------|-----------------|
| 1 | Memory | Float32 distance matrices | Use float32 instead of float64 for distance matrices → 50% memory reduction | 1 line |
| 2 | UX | Progress bars | Stage-aware tqdm bars for pipeline, per-contig DTW, and mod-calling | Done |
| 3 | Speed | Pre-compute IVT-IVT distances | IVT pairwise distances don't change per position → O(n_ivt²) saved per position | 1-2 hours |
| 4 | Speed | Sakoe-Chiba band DTW | Restrict warping path to diagonal band → 2-5x speedup, ~1% accuracy loss | 2-3 hours |
| 5 | Stats | FDR correction | Add Benjamini-Hochberg q-values to output (already in `_aggregation.py`) | Done |
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
- [x] **Batch GPU DTW**: Process all positions together for better GPU utilization — *Implemented via `dtw_multi_position_pairwise` with CUDA streams (16 concurrent positions by default). Reduces cudaMalloc calls from 5000/contig to ~37, enables 75-100% GPU occupancy on H100/3090.*

### 2.2 Memory Optimizations
- [ ] **Streaming DTW results**: Write to disk immediately → O(1) memory per contig
- [ ] **Float32 storage**: 50% memory reduction
- [ ] **HDF5 chunked storage**: 5-10x storage reduction with compression
- [ ] **Lazy signal loading**: Load on-demand for low-memory systems

### 2.3 Parallelization
- [ ] ~~**GPU kNN**~~: *Low value — kNN operates on the already-computed N×N distance matrix using `np.argpartition` (O(N) partial sort). For N=100 reads, total kNN time is ~10-50ms per contig (<1% of pipeline time). GPU kernel launch overhead would negate any speedup.*
- [ ] ~~**GPU HMM forward-backward**~~: *Low value — HMM uses 2-3 states with tiny matrices (2×2 or 3×3). Per-read forward-backward is microseconds. GPU launch overhead dominates for such small problems.*
- [x] **Position-level parallelism**: Process multiple positions in parallel — *Implemented via CUDA streams in `opendba_dtw_multi_position_pairwise`. Positions are assigned round-robin across streams for concurrent GPU execution.*
- [ ] **CPU-GPU overlap**: Parse/extract signals for next batch while GPU computes current batch — *Would hide CPU-side signal extraction latency behind GPU DTW computation.*
- [ ] **f5c parallelism**: Run f5c eventalign for multiple contigs concurrently — *f5c is often the actual wall-clock bottleneck, not DTW. Multiple concurrent f5c subprocesses could saturate I/O.*
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
- [x] **Benjamini-Hochberg FDR**: Control false discovery rate — *Implemented in `_aggregation.py`*
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
- [x] Progress bars (tqdm) — *Implemented: stage-aware bars for pipeline contigs, per-contig DTW, and V1/V2/kNN mod-calling stages*
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
    │  Sakoe-Chiba DTW   │  Modification      │
    │  Float32 storage   │  classification    │
    │  f5c parallelism   │  Stoichiometry     │
    │  CPU-GPU overlap   │                    │
    │                    │                    │
LOW │────────────────────┼────────────────────│ HIGH
    │                    │                    │ EFFORT
    │  Progress bars     │  Hierarchical HMM  │
    │  HDF5 output       │  Full Bayesian     │
    │                    │  Distributed       │
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
                    LOW IMPACT

  DONE: Batch GPU DTW, Position-level parallelism, FDR correction,
        Progress bars
  DEPRIORITIZED: GPU kNN, GPU HMM (both <1% of pipeline time)
```

---

## Notes

- Items marked with ⭐ are recommended starting points
- Effort estimates assume familiarity with the codebase
- Impact estimates based on expected accuracy/speed improvements
