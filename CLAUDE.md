# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Install package (CUDA auto-detected if nvcc available)
pip install .

# Install CPU-only (skip CUDA compilation)
BALEEN_NO_CUDA=1 pip install .

# Target specific GPU archs (comma-separated compute capabilities without dot)
BALEEN_CUDA_ARCHS=86,90 pip install .
# Or auto-detect installed GPU
BALEEN_CUDA_ARCHS=native pip install .

# Run all tests
pytest

# Run specific test file
pytest tests/test_dtw.py

# Run single test
pytest tests/test_dtw.py::test_dtw_distance_basic -v

# Benchmark (requires testdata/ with mixing stoichiometries)
python benchmarks/bench.py run --threads 2 --repeat 5
python benchmarks/bench.py compare  # tabulate recent runs
```

## CLI Usage

```bash
# Full pipeline: DTW + HMM + site-level aggregation
baleen run \
    --native-bam native.bam --native-fastq native.fq.gz --native-blow5 native.blow5 \
    --ivt-bam ivt.bam --ivt-fastq ivt.fq.gz --ivt-blow5 ivt.blow5 \
    --ref ref.fa -o results/

# Site-level aggregation only (from saved results)
baleen aggregate -i results/pipeline_results.pkl -o results/sites.tsv
```

## Commit Style

Conventional commits: `feat:`, `fix:`, `perf:`, `build:`, `bench:`, `ci:`, `refactor:`, `test:`, `docs:`.

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

## CUDA Kernel Architecture

- **FP32 only** — `DTWDistance<float>` template, always float. FP16 would break Pascal consumer GPUs (1/64 FP32 throughput).
- **Wavefront parallelism**: one thread per row of cost matrix, diagonal sweep. `blockDim.x = 1024` (max threads per block). Three rolling diagonals in shared memory (~12 KB).
- **One block per pair** for pairwise mode; grid.x = num_comparisons. Outer loop over reference sequences is serial.
- **Cost function**: squared Euclidean distance, `sqrt` only at the end. Path matrix = nullptr for pairwise (no memory waste).
- **No Sakoe-Chiba band** — a soft-band variant was tried and reverted because setting out-of-band cells to INF without reducing thread count/diagonals is pure overhead. A real band optimization requires skipping diagonals and sizing `blockDim.x` to `min(1024, 2*band_width+1)`.
- Source files: `dtw.hpp` (kernel), `dtw_api.cpp` (Python-C bridge), `multithreading.cpp` (CPU thread pool).

## External Dependencies

- **f5c**: External CLI tool for nanopore event alignment. Must be on PATH.
- **pysam**: BAM file parsing
- **tslearn**: CPU DTW fallback
- **scipy**: Statistical functions, optimization
- **numba** (optional): JIT-compiled HMM forward-backward kernel (`@njit(cache=True)`), kicks in when installed


# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
