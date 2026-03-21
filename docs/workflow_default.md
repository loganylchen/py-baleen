# Baleen Default Pipeline Workflow

## One-Command Summary

```bash
baleen run \
  --native-bam native.bam \
  --native-fastq native.fastq \
  --native-blow5 native.blow5 \
  --ivt-bam ivt.bam \
  --ivt-fastq ivt.fastq \
  --ivt-blow5 ivt.blow5 \
  --ref reference.fasta \
  --output-dir results/
```

This runs the entire V1 → V2 → V3 pipeline with all defaults.

---

## Default Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   INPUT                    PROCESSING              OUTPUT               │
│   ─────                    ──────────              ──────               │
│                                                                         │
│   Native BAM/FASTQ/BLOW5 ─┐                                          │
│                           ├─► f5c eventalign ─► DTW matrix ─► V1 ─► V2 ─► V3 ─► p_mod
│   IVT BAM/FASTQ/BLOW5 ────┘                                          │
│   Reference FASTA                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Preparation & DTW

### 1.1 Input Files
```
┌─────────────────────────────────────────────────────┐
│  NATIVE (with modifications)     IVT (control)      │
│  ┌─────────────────────┐    ┌─────────────────────┐│
│  │ BAM   (alignments)  │    │ BAM   (alignments)  ││
│  │ FASTQ (sequences)   │    │ FASTQ (sequences)   ││
│  │ BLOW5 (ion signal)  │    │ BLOW5 (ion signal)  ││
│  └─────────────────────┘    └─────────────────────┘│
│              │                      │               │
│              └──────────┬───────────┘               │
│                         ▼                           │
│               ┌─────────────────┐                  │
│               │ Reference FASTA │                  │
│               └─────────────────┘                  │
└─────────────────────────────────────────────────────┘
```

### 1.2 f5c Eventalign
```
BAM + FASTQ + BLOW5 + Reference
            │
            ▼
    ┌─────────────────┐
    │  f5c eventalign │
    └─────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│ Per-read signal table for each position   │
│                                           │
│ read_001 @ pos=100: [45.2, 46.1, 44.8...]│
│ read_002 @ pos=100: [44.9, 45.3, 46.0...]│
│ read_101 @ pos=100: [42.1, 41.8, 42.5...]\
│ (IVT control)                             │
└───────────────────────────────────────────┘
```

### 1.3 Pairwise DTW Distance Matrix
```
For each genomic position, compute DTW distances between ALL read pairs:

              Native reads      IVT reads
              N1   N2   N3      I1   I2   I3
         ┌────────────────────────────────────
    N1   │  0   1.2  1.5      4.8  5.1  4.9   ◄── native-native: SMALL
    N2   │ 1.2   0   1.3      4.6  5.0  4.7       (similar signals)
    N3   │ 1.5  1.3   0       4.9  5.2  4.8
         ├────────────────────────────────────
    I1   │ 4.8  4.6  4.9       0   1.1  0.9   ◄── IVT-IVT: SMALL
    I2   │ 5.1  5.0  5.2      1.1   0   1.0       (similar signals)
    I3   │ 4.9  4.7  4.8      0.9  1.0   0
         └────────────────────────────────────
                    ▲
                    └── native-IVT: LARGE at modified positions
                        (different signal shapes)
```

**Key insight:** Modified reads look different from IVT reads → large DTW distances.

---

## Stage 2 (V1): Empirical-Bayes Null Scoring

### 2.1 Extract Per-Read Scores
```
For each read i at position p:

  score[i] = log1p( median DTW distance to all IVT reads )

  Native reads ─► HIGH scores (far from IVT)
  IVT reads    ─► LOW scores  (close to IVT)
```

### 2.2 Fit Robust Null from IVT
```
Using only IVT reads:

  μ_IVT = median(IVT_scores)
  σ_IVT = MAD(IVT_scores) × 1.4826

  This defines the NULL distribution: N(μ_IVT, σ_IVT²)
```

### 2.3 Hierarchical Shrinkage
```
Position with low IVT coverage borrows strength from neighbors:

  ┌─────────────────────────────────────────────────────────┐
  │  Position j with n_j IVT reads:                        │
  │                                                        │
  │  μ̂_j = (n_j × μ_j + κ × μ_local) / (n_j + κ)          │
  │  σ̂_j = (n_j × σ_j + κ × σ_local) / (n_j + κ)          │
  │                                                        │
  │  κ (shrinkage strength) depends on IVT coverage:       │
  │    n_IVT ≥ 20  → κ = 0.5   (trust this position)       │
  │    5 ≤ n_IVT < 20 → κ = 2.0                            │
  │    1 ≤ n_IVT < 5  → κ = 5.0   (borrow more)            │
  │    n_IVT = 0      → use local/global entirely          │
  └─────────────────────────────────────────────────────────┘

Output: z-score = (score - μ̂) / σ̂
```

---

## Stage 2 (V2): kNN IVT-Purity Scoring

### 2.4 k-Nearest Neighbor Classification
```
For each read i, find k nearest neighbors in DTW distance space:

         Native (●)         IVT (○)

              ○  ○
           ●     ○     ○
        ●  ●  i ────► ○  ○    ← k=6 nearest neighbors
           ●     ○     ○
              ○  ○


kNN_score[i] = 1 - (weighted IVT fraction among k neighbors)

  ┌─────────────────────────────────────────────────────────┐
  │  High kNN score = few IVT neighbors = likely MODIFIED  │
  │  Low kNN score  = many IVT neighbors = likely UNMOD    │
  └─────────────────────────────────────────────────────────┘
```

### 2.5 Beta Calibration via EM
```
Raw kNN scores ∈ [0,1] → calibrate to proper probabilities

  IVT reads:  fit Beta(a₀, b₀) as NULL distribution
  Native reads: EM fits Beta(a₁, b₁) as ALTERNATIVE

  P(mod | kNN_score) = f_alt(score) / [f_null(score) + f_alt(score)]

Output: p_mod_knn ∈ [0, 1] for each read
```

---

## Stage 3 (V3): HMM Spatial Smoothing

### 3.1 Build Read Trajectories
```
For each read, extract its path through genomic positions:

Read "read_001" trajectory:
  pos: 100 → 101 → 102 → 103 → 104 → 105
  p_mod_knn: [0.12, 0.08, 0.45, 0.92, 0.88, 0.15]
```

### 3.2 Three-State HMM
```
┌──────────────────────────────────────────────────────────┐
│   pos: 100    101    102    103    104    105           │
│                                                          │
│   ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐            │
│   │ U │───▶│ U │───▶│ F │───▶│ M │───▶│ F │───▶ ...    │
│   └───┘    └───┘    └───┘    └───┘    └───┘            │
│                                                          │
│   States: U=Unmodified, F=Flank, M=Modified             │
└──────────────────────────────────────────────────────────┘

State Definitions:
  ┌─────────────────────────────────────────────┐
  │  Unmodified (U): Beta(2, 8) — mean ≈ 0.2   │
  │  Flank (F):      Beta(3, 3) — mean = 0.5   │
  │  Modified (M):   Beta(8, 2) — mean ≈ 0.8   │
  └─────────────────────────────────────────────┘

Transitions (gap-aware):
  P(stay in same state) = 0.98^(gap_in_bases)

  Example: gap = 5 bases → P(stay) = 0.98^5 = 0.90
```

### 3.3 Forward-Backward Algorithm
```
Input: p_mod_knn values along trajectory
Output: Smoothed p_mod_hmm

  α_t(s) = P(observations_1..t, state_t = s)     [forward]
  β_t(s) = P(observations_t+1..T | state_t = s)  [backward]

  Posterior: γ_t(s) ∝ α_t(s) × β_t(s)

  Final: p_mod_hmm[t] = γ_t(Modified) + γ_t(Flank)
```

**Why HMM?**
- Removes isolated noise spikes
- Captures ±2-base signal halo around true modifications
- Enforces spatial continuity along the read

---

## Output

### Per-Read, Per-Position Probabilities
```
┌────────────────────────────────────────────────────────────────┐
│ read_name   │ position │ kmer  │ p_mod_knn │ p_mod_hmm (final)│
│─────────────┼──────────┼───────┼───────────┼──────────────────│
│ read_001    │   142    │ GGACU │   0.92    │      0.95        │
│ read_001    │   143    │ GACUA │   0.08    │      0.05        │
│ read_001    │   144    │ ACUAG │   0.88    │      0.93        │
│ read_002    │   142    │ GGACU │   0.87    │      0.91        │
│ ...         │   ...    │ ...   │   ...     │      ...         │
└────────────────────────────────────────────────────────────────┘
```

### Site-Level Aggregation
```
┌─────────────────────────────────────────────────────────────┐
│ position │ kmer  │ n_native │ n_ivt │ mean_p_mod │ stderr  │
│──────────┼───────┼──────────┼───────┼────────────┼─────────│
│   142    │ GGACU │    25    │  30   │    0.94    │  0.02   │
│   144    │ ACUAG │    23    │  28   │    0.91    │  0.03   │
│   200    │ CUGGA │    30    │  35   │    0.08    │  0.01   │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary: Default Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   INPUT                                                                 │
│   Native + IVT BAM/FASTQ/BLOW5 + Reference FASTA                       │
│                                                                         │
│                         │                                               │
│                         ▼                                               │
│                                                                         │
│   DTW MATRIX                                                            │
│   Pairwise signal shape distances (n_native + n_ivt)² matrix           │
│                                                                         │
│                         │                                               │
│                         ▼                                               │
│                                                                         │
│   V1: EMPIRICAL-BAYES NULL                                              │
│   • score = log(median DTW to IVT)                                     │
│   • Fit null from IVT: μ, σ via median + MAD                           │
│   • Hierarchical shrinkage: borrow from neighbors if low coverage      │
│   • Output: z-scores                                                   │
│                                                                         │
│                         │                                               │
│                         ▼                                               │
│                                                                         │
│   V2: kNN IVT-PURITY (DEFAULT)                                         │
│   • kNN_score = 1 - (IVT fraction among k nearest neighbors)          │
│   • Calibrate via Beta EM: P(mod | kNN_score)                          │
│   • Output: p_mod_knn                                                  │
│                                                                         │
│                         │                                               │
│                         ▼                                               │
│                                                                         │
│   V3: HMM SMOOTHING                                                     │
│   • 3-state: Unmodified → Flank → Modified                             │
│   • Emissions: p_mod_knn (default)                                     │
│   • Gap-aware transitions: P(stay) = 0.98^gap                         │
│   • Forward-backward → posterior                                       │
│   • Output: p_mod_hmm (FINAL)                                          │
│                                                                         │
│                         │                                               │
│                         ▼                                               │
│                                                                         │
│   OUTPUT                                                                │
│   Per-read, per-position modification probabilities                    │
│   Aggregated site-level statistics                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Equations (Default Pipeline)

| Stage | Equation |
|-------|----------|
| **V1 Score** | `score[i] = log1p(median(DTW(i, IVT_controls)))` |
| **V1 Null** | `μ = median(IVT_scores), σ = MAD(IVT_scores) × 1.4826` |
| **V1 Shrinkage** | `μ̂ = (n × μ_local + κ × μ_window) / (n + κ)` |
| **V2 kNN** | `kNN[i] = 1 - Σ wⱼ·I[IVT](j) / Σ wⱼ` (k neighbors) |
| **V2 Calibrate** | `P(mod\|kNN) = f_Beta_alt(kNN) / [f_Beta_null + f_Beta_alt]` |
| **V3 Transition** | `P(stay\|gap) = 0.98^gap` |
| **V3 Emission** | `P(obs\|state) = Beta(p_mod_knn; a_state, b_state)` |
| **V3 Posterior** | `γ_t(s) ∝ α_t(s) × β_t(s)` (forward-backward) |

---

## Figure Legend (for publication)

**Figure X: Baleen pipeline for RNA modification detection from nanopore direct RNA sequencing.**

**(A)** Native RNA (containing modifications) and IVT RNA (modification-free control) are aligned to a reference using f5c eventalign, producing per-read signal tables for each genomic position.

**(B)** Pairwise DTW distance matrices are computed for each position. Modified reads show elevated distances to IVT controls.

**(C)** V1: Empirical-Bayes null scoring with hierarchical shrinkage establishes robust baseline statistics. V2: kNN IVT-purity scoring quantifies neighborhood composition in DTW space, calibrated via Beta EM. V3: A 3-state HMM with gap-aware transitions smooths probabilities along read trajectories.

**(D)** Final output: per-read modification probabilities aggregated to site-level statistics.
