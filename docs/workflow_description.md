# Baleen Workflow Description for Publication-Quality Figure

## Overview

Baleen is a computational pipeline for detecting RNA modifications from Oxford Nanopore direct RNA sequencing (DRS) data. It compares **native RNA** (containing modifications) against **in vitro transcribed (IVT) RNA** (modification-free control) by quantifying signal shape differences using Dynamic Time Warping (DTW) and a hierarchical Bayesian-HMM framework.

---

## Visual Layout Recommendation

**Overall structure:** Left-to-right horizontal flow with three main panels:
- **Panel A: Data Preparation** (input files → signal extraction)
- **Panel B: Distance Computation** (DTW pairwise matrix)
- **Panel C: Statistical Inference** (V1 → V2 → V3 hierarchical pipeline)

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

## Panel C: Hierarchical Statistical Inference (V1 → V2 → V3)

### C1. V1: Empirical-Bayes Null Scoring
**Visual:** Three-step process

```
Step 1: Extract IVT Distances
┌─────────────────────────────────────────────┐
│ For each read i:                            │
│   score[i] = median(DTW(i, IVT_controls))   │
│                                             │
│ Native reads → high scores (distant from IVT)│
│ IVT reads → low scores (similar to IVT)     │
└─────────────────────────────────────────────┘
                    │
                    ▼
Step 2: Fit Robust Null (IVT only)
┌─────────────────────────────────────────────┐
│   μ_IVT = median(IVT_scores)                │
│   σ_IVT = MAD(IVT_scores) × 1.482           │
│                                             │
│   [Gaussian null distribution]              │
│         ▲                                   │
│        ╱ ╲                                  │
│       ╱   ╲                                 │
│      ╱     ╲                                │
│  ───┴───────┴───                            │
│     μ_IVT                                   │
└─────────────────────────────────────────────┘
                    │
                    ▼
Step 3: Hierarchical Shrinkage
┌─────────────────────────────────────────────┐
│ Position j with n_j IVT reads:              │
│                                             │
│ μ̂_j = (n_j × μ_j + κ × μ_local) / (n_j + κ) │
│                                             │
│ κ depends on IVT coverage:                  │
│   HIGH (≥20): κ=0.5  (trust position)       │
│   MEDIUM (5-19): κ=2.0                      │
│   LOW (1-4): κ=5.0 (shrink to neighbors)    │
│   ZERO: use local/global only               │
│                                             │
│ Output: z-score = (score - μ̂) / σ̂          │
└─────────────────────────────────────────────┘
```

**Caption:** V1 establishes a robust null distribution from IVT controls, with hierarchical shrinkage to borrow strength across positions.

### C2. V2a: kNN IVT-Purity Scoring (Default Emission Source)
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

### C2b. V2b: Anchored Two-Component Mixture EM (Alternative Scoring)
**Visual:** Mixture model fitting on z-scores

```
┌─────────────────────────────────────────────────────────┐
│            Native z-scores histogram                   │
│                                                         │
│     ▓▓▓▓▓▓▓                                              │
│    ▓▓▓▓▓▓▓▓▓▓     ░░░░░░░                               │
│   ▓▓▓▓▓▓▓▓▓▓▓▓   ░░░░░░░░░░                             │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ░░░░░░░░░░░░░                            │
│ ─┴──────────────┴────────────────┴───                    │
│  0   null (unmod)   alt (modified)                      │
│                                                         │
│ Null: N(μ_IVT, σ_IVT)  — fixed from IVT                │
│ Alt:  N(μ_alt, σ_alt) — learned via EM                 │
│                                                         │
│ P(mod | z) = π × f_alt(z) / [(1-π)×f_null(z) + π×f_alt]│
└─────────────────────────────────────────────────────────┘
```

**Soft Gating Mechanism:**
```
gate_weight = σ(π) × σ(BIC_mix - BIC_null) × σ(separation)

Final: p_mod_raw = gate_weight × mixture_posterior + (1-gate_weight) × z_fallback
```

**Note:** The pipeline computes BOTH `p_mod_raw` (mixture) and `p_mod_knn` (kNN). By default, `p_mod_knn` is used as the HMM emission source.

### C3. V3: HMM Spatial Smoothing
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
│   Emissions: P(p_mod_knn | state)  ← DEFAULT            │
└──────────────────────────────────────────────────────────┘

Emission Source Selection:
┌─────────────────────────────────────────────────────────┐
│  DEFAULT: emission_source = "p_mod_knn"                │
│           Uses kNN IVT-purity scores as HMM emissions  │
│                                                         │
│  ALTERNATIVE: emission_source = "p_mod_raw"            │
│               Uses V2 mixture posteriors as emissions  │
└─────────────────────────────────────────────────────────┘

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

**Caption:** V3 applies a 3-state HMM along each read's genomic trajectory, smoothing modification probabilities and capturing the ±2-base signal halo around true modification sites.

---

## Panel D: Output & Aggregation

### D1. Per-Position Summary
**Visual:** Site-level aggregation

```
┌─────────────────────────────────────────────────────────┐
│ Position │ Kmer │ P(mod)_raw │ P(mod)_kNN │ P(mod)_HMM │
│──────────┼──────┼────────────┼────────────┼────────────│
│   142    │ GGACU│   0.87     │   0.92     │   0.95     │
│   143    │ GACUA│   0.12     │   0.08     │   0.05     │
│   144    │ ACUAG│   0.91     │   0.88     │   0.93     │
└─────────────────────────────────────────────────────────┘
```

### D2. Three Training Modes
**Visual:** Decision tree

```
                    Labeled data?
                   ╱            ╲
                 No              Yes
                 │                │
                 ▼                ▼
         ┌──────────────┐  ┌──────────────────┐
         │  UNSUPERVISED│  │ Positions < 50?  │
         │  (defaults)  │  │  or < 3 contigs? │
         └──────────────┘  │   ╱        ╲     │
                           │  Yes        No   │
                           │   │          │   │
                           ▼   ▼          ▼   ▼
                    ┌─────────────┐ ┌─────────────┐
                    │SEMI-SUPERVISED│ │ SUPERVISED │
                    │ (Platt scale)│ │(MLE+KDE)   │
                    └─────────────┘ └─────────────┘
```

---

## Key Algorithmic Features to Highlight

1. **kNN IVT-purity scoring (default):** Quantifies neighborhood composition in DTW space — modified reads cluster together, away from IVT neighbors. Calibrated via Beta EM. Used as the DEFAULT emission source for HMM.

2. **Parallelization:** Contigs processed in parallel using `ProcessPoolExecutor` with spawn context for CUDA safety

3. **Open-boundary DTW:** Allows signals to be compared with flexible start/end points, accommodating variable read lengths

4. **CUDA acceleration:** GPU-accelerated DTW computation with automatic CPU fallback

5. **Hierarchical shrinkage:** Low-coverage positions borrow strength from neighboring positions

6. **Soft gating:** Continuous blending instead of hard binary decisions reduces boundary artifacts (for mixture-based scoring)

7. **Gap-aware transitions:** HMM transition probability P(stay) = 0.98^gap naturally handles uneven genomic spacing

8. **3-state topology:** Explicit Flank state absorbs signal halo around modification sites

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
│  C. HIERARCHICAL INFERENCE (V1 → V2 → V3)                          │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐                 │
│  │    V1      │──▶│    V2      │──▶│    V3      │                 │
│  │ (EB null)  │   │ (Mixture)  │   │   (HMM)    │                 │
│  └────────────┘   └────────────┘   └────────────┘                 │
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
│  V1: z = (score - μ̂) / σ̂   where μ̂, σ̂ are shrunk null parameters  │
│      μ̂_j = (n_j × μ_j + κ × μ_local) / (n_j + κ)  (shrinkage)      │
│                                                                     │
│  V2a (kNN): score[i] = 1 - Σ w_j·I[IVT](j) / Σ w_j  (kNN purity)   │
│             P(mod|score) calibrated via Beta EM (DEFAULT)           │
│                                                                     │
│  V2b (Mixture): P(mod|z) = π·f_alt(z) / [(1-π)·f_null + π·f_alt]   │
│                 with soft gating: gate = σ(π)·σ(ΔBIC)·σ(sep)        │
│                                                                     │
│  V3: γ_t(s) ∝ α_t(s)·β_t(s)  via forward-backward                  │
│      P(stay|gap) = p_stay^gap  (gap-aware transitions)              │
│      Emissions from p_mod_knn (default) or p_mod_raw                │
│                                                                     │
│  DTW: D(A,B) = min_{warping} Σ |A_i - B_j|                        │
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

**(C)** Hierarchical statistical inference. V1 (Empirical-Bayes): Robust null distribution from IVT controls with hierarchical shrinkage. V2a (kNN IVT-purity, default): Quantifies neighborhood composition in DTW space, calibrated via Beta EM. V2b (Mixture EM, alternative): Two-component mixture with soft gating produces raw P(mod). V3 (HMM): 3-state forward-backward smoothing along read trajectories, using kNN scores as default emission source, yields final modification probabilities.

**(D)** Output. Site-level modification probabilities are aggregated across reads, with optional semi-supervised or supervised HMM training using labeled modification sites.
