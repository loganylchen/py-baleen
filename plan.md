# AUPRC Improvement Plan — Remaining Phases

## Current Status

**Phase 1 (complete, on branches):**
- `phase-1a` — Position-level evaluation added to `internal_benchmarking.ipynb`
- `phase-1b` — Pi-weighted posterior (builds on 1a)
- `phase-1c` — Soft null gate (builds on 1b)
- `phase-1d` — IVT-to-IVT knn bandwidth (builds on 1c)

Each branch builds on the previous. User should test each by re-running `internal_benchmarking.ipynb` and comparing AUROC/AUPRC.

---

## Phase 2: Multi-organism supervised training

**Requires:** User's yeast rRNA and human cell line datasets (native + IVT + labels).

### 2a. Create multi-organism training infrastructure
- **New notebook:** `multi_organism_training.ipynb`
- Process yeast rRNA and human cell line datasets through the same pipeline (f5c → signals → DTW → probabilities)
- Define known modification sites for yeast and human (user provides these)
- Generate labeled `(position, is_modified)` pairs across all three organisms

### 2b. Train supervised HMM with cross-organism validation
- Use existing `train_hmm_supervised()` and `train_hmm_semi_supervised()` in `_hmm_training.py`
- **Training:** yeast rRNA + human cell line
- **Testing:** E. coli rRNA (held-out)
- Or: leave-one-organism-out cross-validation
- Key parameters to learn:
  - `p_stay_per_base` (currently hardcoded 0.98)
  - `init_prob` (currently 0.5/0.5)
  - KDE emission model (learned P(score | modified) and P(score | unmodified))

### 2c. Calibrate emission model across organisms
- Train `EmissionCalibrator` (Platt scaling) and `EmissionKDE` on pooled multi-organism data
- This teaches the HMM what "modified" and "unmodified" scores look like across different sequence contexts

**Files:** `_hmm_training.py`, new `notebooks/multi_organism_training.ipynb`

---

## Phase 3: Ensemble and advanced methods

### 3a. Score combination / ensemble
- Use knn scores as input to hierarchical V1→V2→V3 pipeline (instead of distance_to_ivt)
- Or: train logistic regression combining `knn_score + hier_V2_raw + distance_to_ivt` using labeled data from Phase 2

### 3b. Position-level aggregation before HMM
- First aggregate per-position (fraction of reads with P(mod) > 0.5, or mean native read probability)
- Then run HMM on position-level scores along the genome
- Reduces noise from individual reads

**Files:** `_probability.py`, `_hierarchical.py`, possibly new `_ensemble.py`

---

## Next Steps

1. User tests Phase 1 branches on E. coli data, reports AUROC/AUPRC changes
2. User provides yeast + human datasets and known modification sites
3. Implement Phase 2a-c with those datasets
4. Implement Phase 3a-b after training data is available
