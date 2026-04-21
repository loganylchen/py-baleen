# Baleen Workflow Description for Publication-Quality Figure

## Overview

Baleen is a computational pipeline for detecting RNA modifications from Oxford Nanopore direct RNA sequencing (DRS) data. It compares **native RNA** (containing modifications) against **in vitro transcribed (IVT) RNA** (modification-free control) by quantifying signal shape differences using Dynamic Time Warping (DTW), a Beta-calibrated kNN IVT-purity score, and a gap-aware per-read HMM.

---

## Visual Layout Recommendation

**Overall structure:** Left-to-right horizontal flow with three main panels:
- **Panel A: Data Preparation** (input files → signal extraction)
- **Panel B: Distance Computation** (DTW pairwise matrix)
- **Panel C: Statistical Inference** (kNN + Beta EM → HMM)

Use a **color scheme** of blues/greens for native data, oranges/yellows for IVT data, and purple for statistical outputs.

---

## Panel A: Data Preparation & Signal Extraction

### A1. Input Data
**Visual:** Two parallel tracks stacked vertically

```
┌─────────────────────────────────────────────────────────────┐
│  NATIVE SAMPLE                    IVT SAMPLE (Control)      │
│  ┌─────────────────────┐         ┌─────────────────────┐   │
│  │ BAM (alignments)    │         │ BAM (alignments)    │   │
│  │ FASTQ (reads)       │         │ FASTQ (reads)       │   │
│  │ BLOW5 (raw signal)  │         │ BLOW5 (raw signal)  │   │
│  └─────────────────────┘         └─────────────────────┘   │
│           │                               │                 │
│           ▼                               ▼                 │
│  [Native reads with           [IVT reads without            │
│   modifications]               modifications]               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Reference FASTA │
                    │ (genome/transcriptome)     │
                    └─────────────────┘
```

**Caption:** Native RNA (top, blue) contains endogenous modifications (m6A, Ψ, etc.) that alter ionic current patterns. IVT RNA (bottom, orange) serves as modification-free control.

### A2. Event Alignment
**Visual:** f5c eventalign process

```
Native BAM + FASTQ + BLOW5 + Reference
              │
              ▼
    ┌─────────────────────┐
    │   f5c eventalign    │
    │  (per-read signal   │
    │   alignment to ref) │
    └─────────────────────┘
              │
              ▼
┌───────────────────────────────────────────┐
│ Eventalign TSV (per-contig)               │
│ ┌─────────────────────────────────────┐   │
│ │ read_name  pos  kmer  signal_mean   │   │
│ │ read_001   100   GAC   [45.2,46.1,..]│   │
│ │ read_001   101   ACT   [44.8,45.3,..]│   │
│ │ read_002   100   GAC   [45.0,45.8,..]│   │
│ └─────────────────────────────────────┘   │
└───────────────────────────────────────────┘
```

**Key point:** Each row represents one read's signal at one genomic position.

### A3. Signal Grouping by Position
**Visual:** Signal arrays grouped by genomic coordinate

```
Position 100 (kmer: GAC)
┌─────────────────────────────────────────────────┐
│ Native reads:                                   │
│   read_001: ━━━━━━━━━━  (normalized signal)    │
│   read_002: ━━━━━━━━━━                          │
│   read_003: ━━━━━━━━━━                          │
│                                                 │
│ IVT reads:                                      │
│   read_101: ──────────  (different shape)      │
│   read_102: ──────────                          │
│   read_103: ──────────                          │
└─────────────────────────────────────────────────┘
```

**Key insight:** Modified positions show signal shape divergence between native and IVT reads.

---

## Panel B: Dynamic Time Warping Distance Computation

### B1. Pairwise DTW Matrix
**Visual:** Symmetric distance matrix heatmap

```
         Native         IVT
      ┌─────────────────────────────┐
      │  N1   N2   N3   I1   I2   I3 │
      ├─────────────────────────────┤
N1    │  0   1.2  1.5  4.8  5.1  4.9 │ ◄─ native-native: small
N2    │ 1.2   0   1.3  4.6  5.0  4.7 │
N3    │ 1.5  1.3   0   4.9  5.2  4.8 │
      ├─────────────────────────────┤
I1    │ 4.8  4.6  4.9   0   1.1  0.9 │ ◄─ IVT-IVT: small
I2    │ 5.1  5.0  5.2  1.1   0   1.0 │
I3    │ 4.9  4.7  4.8  0.9  1.0   0 │
      └─────────────────────────────┘
            ▲
            └── native-IVT: large (modification signal)
```

**Color coding:**
- Dark blue: small distances (similar signals)
- Yellow/white: large distances (dissimilar signals)

**Caption:** DTW computes optimal alignment between signal pairs, capturing shape differences. The (n_native + n_ivt)² matrix encodes all pairwise similarities.

### B2. DTW Algorithm Detail (optional subpanel)
**Visual:** Two signals being warped

```
Signal A (native):  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁
                    ╲ ╱╲ ╱╲ ╱╲ ╱╲ ╱
Signal B (IVT):     ▁▃▄▅▆▇█▇▆▅▄▃▁

DTW finds optimal non-linear alignment
that minimizes cumulative distance.
```

---

## Panel C: Statistical Inference (kNN + Beta EM → HMM)

### C1. kNN IVT-Purity Scoring
**Visual:** k-nearest neighbor classification in DTW space

```
┌─────────────────────────────────────────────────────────────┐
│  For each read i, find k nearest neighbors in DTW space:   │
│                                                             │
│         Native (●)         IVT (○)                         │
│                                                             │
│              ○  ○                                          │
│           ●     ○     ○                                    │
│        ●  ●  i ────► ○  ○    ← k=6 neighbors              │
│           ●     ○     ○                                    │
│              ○  ○                                          │
│                                                             │
│  kNN score[i] = 1 - (weighted IVT fraction among k NN)     │
│                                                             │
│  High score = few IVT neighbors = likely modified          │
│  Low score = many IVT neighbors = likely unmodified        │
└─────────────────────────────────────────────────────────────┘
```

**Beta Calibration:**
```
Raw kNN scores ∈ [0,1] → calibrated via Beta null + Beta alternative EM

IVT reads: fit Beta(a₀, b₀) as null distribution
Native reads: EM fits Beta(a₁, b₁) as alternative

P(mod | score) = f_alt(score) / [f_null(score) + f_alt(score)]
```

**Caption:** kNN IVT-purity scoring quantifies how isolated a read is from IVT controls in DTW distance space. Modified reads cluster together, away from IVT neighbors.

### C2. HMM Spatial Smoothing
**Visual:** Hidden Markov Model along read trajectory

```
Read trajectory (genomic positions along a single read):
┌──────────────────────────────────────────────────────────┐
│   pos: 100    101    102    103    104    105    106    │
│                                                          │
│   ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐            │
│   │ U │───▶│ U │───▶│ F │───▶│ M │───▶│ F │───▶ ...    │
│   └───┘    └───┘    └───┘    └───┘    └───┘            │
│                                                          │
│   States: U=Unmodified, F=Flank, M=Modified             │
│                                                          │
│   Transitions: P(stay) = 0.98^(gap_in_bases)            │
│   Emissions: P(p_mod_knn | state)                       │
└──────────────────────────────────────────────────────────┘

3-State HMM Topology:
┌─────────────────────────────────────────┐
│                                         │
│    ┌─────────────────────────────┐     │
│    │      Unmodified (U)         │     │
│    │   Beta(2, 8) — mean ≈ 0.2   │     │
│    └──────────┬──────────────────┘     │
│               │                         │
│               ▼                         │
│    ┌─────────────────────────────┐     │
│    │        Flank (F)            │     │
│    │   Beta(3, 3) — mean = 0.5   │     │
│    └──────────┬──────────────────┘     │
│               │                         │
│               ▼                         │
│    ┌─────────────────────────────┐     │
│    │      Modified (M)           │     │
│    │   Beta(8, 2) — mean ≈ 0.8   │     │
│    └─────────────────────────────┘     │
│                                         │
└─────────────────────────────────────────┘
```

**Forward-Backward Algorithm:**
```
α_t(s) = P(x_1...x_t, q_t = s)  [forward]
β_t(s) = P(x_t+1...x_T | q_t = s)  [backward]

P(mod | trajectory) = Σ_s∈{F,M} γ_t(s)
where γ_t(s) ∝ α_t(s) × β_t(s)
```

**Caption:** The HMM runs along each read's genomic trajectory, smoothing modification probabilities and capturing the ±2-base signal halo around true modification sites.

---

## Panel D: Output & Aggregation

### D1. Per-Position Summary
**Visual:** Site-level aggregation

```
┌─────────────────────────────────────────────┐
│ Position │ Kmer │ P(mod)_kNN │ P(mod)_HMM   │
│──────────┼──────┼────────────┼──────────────│
│   142    │ GGACU│   0.92     │   0.95       │
│   143    │ GACUA│   0.08     │   0.05       │
│   144    │ ACUAG│   0.88     │   0.93       │
└─────────────────────────────────────────────┘
```

---

## Key Algorithmic Features to Highlight

1. **kNN IVT-purity scoring:** Quantifies neighborhood composition in DTW space — modified reads cluster together, away from IVT neighbors. Calibrated via Beta EM. Serves as the HMM emission source.

2. **Parallelization:** Contigs processed in parallel using `ProcessPoolExecutor` with spawn context for CUDA safety

3. **Open-boundary DTW:** Allows signals to be compared with flexible start/end points, accommodating variable read lengths

4. **CUDA acceleration:** GPU-accelerated DTW computation with automatic CPU fallback

5. **Gap-aware transitions:** HMM transition probability P(stay) = 0.98^gap naturally handles uneven genomic spacing

6. **3-state topology:** Explicit Flank state absorbs signal halo around modification sites

---

## Suggested Figure Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  A. DATA PREPARATION              B. DTW COMPUTATION               │
│  ┌─────────────────────┐         ┌─────────────────────┐          │
│  │                     │         │                     │          │
│  │   [Input panel]     │────────▶│   [DTW matrix]      │          │
│  │                     │         │                     │          │
│  └─────────────────────┘         └─────────────────────┘          │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  C. STATISTICAL INFERENCE                                          │
│  ┌──────────────────────┐     ┌────────────┐                       │
│  │ kNN + Beta EM        │────▶│   HMM      │                       │
│  │ (p_mod_knn)          │     │ (p_mod_hmm)│                       │
│  └──────────────────────┘     └────────────┘                       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  D. OUTPUT                                                          │
│  ┌─────────────────────────────────────────────────────┐           │
│  │  Site-level modification probabilities (TSV)        │           │
│  └─────────────────────────────────────────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Summary Box (for figure legend)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Key Equations                                                      │
│                                                                     │
│  DTW: D(A,B) = min_{warping} Σ |A_i - B_j|                        │
│                                                                     │
│  kNN: score[i] = 1 - Σ w_j·I[IVT](j) / Σ w_j  (kNN purity)         │
│       P(mod|score) calibrated via Beta EM  →  p_mod_knn             │
│                                                                     │
│  HMM: γ_t(s) ∝ α_t(s)·β_t(s)  via forward-backward                  │
│       P(stay|gap) = p_stay^gap  (gap-aware transitions)             │
│       Emissions from p_mod_knn                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Color Palette Suggestion

| Element | Color | Hex |
|---------|-------|-----|
| Native data | Blue | #3498db |
| IVT data | Orange | #e67e22 |
| Unmodified state | Light blue | #85c1e9 |
| Modified state | Red/Pink | #e74c3c |
| Flank state | Purple | #9b59b6 |
| Statistical inference | Green | #27ae60 |
| Background | White/Light gray | #f8f9fa |

---

## Figure Legend Text (Draft)

**Figure X: Baleen workflow for RNA modification detection from nanopore DRS data.**

**(A)** Data preparation. Native RNA (containing modifications) and IVT RNA (modification-free control) are aligned to a reference using f5c eventalign, producing per-read signal tables for each genomic position.

**(B)** Pairwise DTW distance computation. For each position, a symmetric distance matrix captures signal shape differences between all read pairs. Native-IVT distances are elevated at modified positions.

**(C)** Statistical inference. kNN IVT-purity scoring quantifies neighborhood composition in DTW space and is calibrated via Beta EM into `p_mod_knn`. A 3-state HMM with gap-aware transitions then smooths these probabilities along each read's genomic trajectory via forward–backward, yielding the final per-read, per-position modification probabilities.

**(D)** Output. Site-level modification probabilities are aggregated across reads via per-site thresholding and Fisher combination into `mod_ratio` and `pvalue`.
