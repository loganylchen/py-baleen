# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Install package (CUDA auto-detected if nvcc available)
pip install .

# Install CPU-only (skip CUDA compilation)
BALEEN_NO_CUDA=1 pip install .

# Run all tests
pytest

# Run specific test file
pytest tests/test_dtw.py

# Run single test
pytest tests/test_dtw.py::test_dtw_distance_basic -v
```

## Architecture Overview

Baleen is a CUDA-accelerated DTW (Dynamic Time Warping) and nanopore signal analysis pipeline for detecting RNA modifications by comparing native and IVT (in vitro transcribed) nanopore signals.

### Package Structure

```
baleen/
├── __init__.py              # Re-exports public API from eventalign
├── _cuda_dtw/               # CUDA DTW implementation with CPU fallback
│   └── __init__.py          # Python wrapper (dtw_distance, dtw_pairwise, etc.)
└── eventalign/              # Main analysis pipeline
    ├── __init__.py          # Public API exports
    ├── _pipeline.py         # run_pipeline(), save/load_results()
    ├── _bam.py              # BAM parsing, contig stats, filtering
    ├── _f5c.py              # f5c eventalign CLI wrapper
    ├── _signal.py           # Signal extraction and grouping by position
    ├── _probability.py      # Modification probability algorithms
    ├── _hierarchical.py     # Hierarchical Bayesian + HMM pipeline (V1→V2→V3)
    └── _hmm_training.py     # HMM training modes (unsupervised/semi-supervised/supervised)
```

### Data Flow

1. **Input**: Native + IVT BAM/FASTQ/BLOW5 files + reference FASTA
2. **Event alignment**: f5c eventalign produces per-read signal tables per position
3. **Signal grouping**: Group signals by genomic position, find common positions
4. **DTW computation**: Pairwise DTW distance matrices per position (CUDA or tslearn fallback)
5. **Modification calling**: Three-stage hierarchical pipeline:
   - V1: Empirical-Bayes null scoring with hierarchical shrinkage
   - V2: Anchored two-component mixture EM
   - V3: HMM forward-backward smoothing along read trajectories

### Key Data Classes

- `PositionResult`: Per-position DTW distance matrix + metadata
- `ContigResult`: All position results for one contig
- `PositionStats`: Per-position V1→V2→V3 outputs (z-scores, p-values, probabilities)
- `ContigModificationResult`: Full hierarchical pipeline output for one contig
- `HMMParams`: Learned or default HMM parameters for V3

### DTW Backend Selection

The `_cuda_dtw` module auto-selects backend at import time:
- CUDA (GPU) if `_cuda_dtw` C extension compiled successfully
- CPU (tslearn) fallback otherwise

Use `use_cuda=True/False` to force backend, or `None` for auto-select.

### Modification Probability Algorithms

Three algorithms in `_probability.py`, all sharing EM calibration:
1. `distance_to_ivt`: Median DTW distance to IVT controls
2. `knn_ivt_purity`: k-NN IVT affinity score
3. `mds_gmm`: MDS embedding + Gaussian mixture

### HMM Training Modes

Three modes in `_hmm_training.py`:
- **Unsupervised** (default): Hardcoded defaults, no labeled data needed
- **Semi-supervised**: Platt-scaling calibrator from labeled positions
- **Supervised**: MLE transitions + KDE emissions from labeled trajectories

## External Dependencies

- **f5c**: External CLI tool for nanopore event alignment. Must be on PATH.
- **pysam**: BAM file parsing
- **tslearn**: CPU DTW fallback
- **scipy**: Statistical functions, optimization
