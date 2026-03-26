# Baleen Algorithm — Main Figure Description

> **Purpose:** This document provides a comprehensive description of the Baleen
> algorithm workflow for generating a publication-quality main figure suitable
> for a Nature-style paper. Feed this entire document to an AI figure generator
> (e.g., BioRender AI, Gemini, GPT-4o with DALL-E) to produce the diagram.

---

## Figure Prompt for AI Generation

Create a multi-panel scientific workflow diagram in Nature publication style for an algorithm called **Baleen** — a hierarchical Bayesian framework for detecting RNA modifications from nanopore direct RNA sequencing. The figure should use a clean, minimal design with a consistent left-to-right or top-to-bottom flow. Use a muted scientific color palette (blues, teals, warm oranges, soft grays). Each stage should be clearly labeled and visually distinct. The overall layout should span a full page width (~180 mm) with 4–5 horizontal panels or a vertical cascade of stages.

---

## Panel A — Input & Signal Alignment

**Title:** "Nanopore signal alignment"

Show two parallel input streams converging:

1. **Native RNA** (colored warm orange): A nanopore device icon producing a raw ionic current trace (squiggly line), with a BAM file icon and a BLOW5 signal file icon.
2. **IVT control RNA** (colored cool blue): Same nanopore device producing a similar but distinct current trace, with matching BAM and BLOW5 icons. Label this "in vitro transcribed (unmodified control)."
3. **Reference genome** (gray): A double-helix or linear sequence bar labeled "Reference FASTA."

These three inputs feed into a box labeled **"f5c eventalign"** — an alignment engine that maps raw signal events to genomic 5-mer positions. Show small arrows from each event to a position on the reference.

**Output:** A schematic table showing columns: `position`, `kmer (5-mer)`, `read_name`, `signal samples`. Show that this produces per-read signal arrays grouped by genomic position. Emphasize that only **common positions** (present in both native and IVT) proceed downstream.

---

## Panel B — Pairwise DTW Distance Matrix

**Title:** "Per-position DTW distance computation"

At a single genomic position, show:

1. A set of **signal traces** stacked vertically — top group colored orange (native reads, ~15–30), bottom group colored blue (IVT reads, ~10–20). Each trace is a short time series of current values.
2. Between two traces, show a **DTW alignment path** — a warping grid with the optimal path drawn diagonally, illustrating that Dynamic Time Warping handles variable-length signals.
3. The output: a symmetric **distance matrix** (heatmap) of size (n_native + n_ivt) × (n_native + n_ivt). Use a block structure to visually show:
   - **Top-left block** (native × native): low distance if unmodified, variable if modified
   - **Bottom-right block** (IVT × IVT): consistently low distance (self-similar controls)
   - **Off-diagonal blocks** (native × IVT): HIGH distance at modified positions (the modification signal), LOW distance at unmodified positions
4. Label: "GPU-accelerated (CUDA) or CPU fallback." Show a small GPU chip icon.

**Key visual:** Show two contrasting distance matrices side by side — one for an **unmodified position** (uniform low distances) and one for a **modified position** (clear block structure with high native-vs-IVT distances).

---

## Panel C — Three-Stage Hierarchical Pipeline (V1 → V2 → V3)

**Title:** "Hierarchical empirical-Bayes modification scoring"

This is the central and largest panel. Show three connected stages flowing left to right (or top to bottom), each with increasing sophistication.

### Stage V1 — Empirical-Bayes Null Scoring

**Subtitle:** "Robust IVT null with hierarchical shrinkage"

1. **Score extraction:** From each distance matrix, extract a per-read score:
   - Show formula: `score = log₁p(median distance to IVT) + corrections`
   - The corrections include: asymmetric IVT baseline subtraction, native cohesion bonus
   - Visually: arrows from the distance matrix to a single score vector

2. **Robust null estimation:** From IVT scores at each position:
   - `μ = median(IVT scores)`, `σ = MAD × 1.4826`
   - Show a small normal distribution fitted to the IVT score histogram

3. **Hierarchical shrinkage** (the key innovation): Show three nested levels:
   - **Position level** (innermost): raw (μ, σ) from this position's IVT reads
   - **Local window** (middle ring): weighted median of ±15 neighboring positions
   - **Global prior** (outer ring): contig-wide median of all positions
   - Formula: `μ_shrunk = (n·μ_raw + κ·μ_local) / (n + κ)`
   - Show κ varying by coverage: HIGH (κ=0.5, trust data), MEDIUM (κ=2), LOW (κ=5), ZERO (κ=∞, use prior)
   - Visually: a genomic coordinate axis with positions shown as dots, arrows showing borrowing of strength from neighbors

4. **Z-scores:** `z = (score − μ_shrunk) / σ_shrunk` → one-sided p-values

### Stage V2 — Anchored Two-Component Mixture with Soft Gating

**Subtitle:** "Per-position EM with continuous confidence gating"

1. **Contig-pooled EM:** First, pool native z-scores across all positions to estimate a global alternative component N(μ₁, σ₁). Show a histogram of pooled z-scores with two overlaid Gaussians (null in blue, alternative in orange).

2. **Per-position anchored EM:** At each position:
   - **Null component** (fixed, blue): N(μ₀, σ₀) from this position's IVT z-scores
   - **Alternative component** (free, orange): N(μ₁, σ₁) — fits to native reads shifted rightward
   - Show the EM iteration: E-step assigns soft responsibilities, M-step updates alternative
   - Low-coverage positions borrow global (μ₁, σ₁) instead of fitting locally

3. **Soft gating** (novel feature): Show three sigmoid curves multiplied together:
   - **π-gate:** Is the mixing proportion large enough? `σ((π − 0.05) / τ)`
   - **BIC-gate:** Does the mixture model beat the null model? `σ((BIC_null − BIC_mix) / τ)`
   - **Separation-gate:** Is the effect size meaningful? `σ((|μ₁−μ₀|/σ₀ − 0.8) / τ)`
   - Product → `gate_weight ∈ [0, 1]`
   - Final probability = `gate_weight × mixture_posterior + (1 − gate_weight) × z_score_fallback`
   - Show a gradient bar from 0 (fully gated, use z-score fallback) to 1 (fully open, trust mixture)

4. **kNN IVT-purity scoring** (parallel complementary path):
   - For each read, find k nearest neighbors in the distance matrix
   - Compute weighted IVT fraction among neighbors: `kNN_score = 1 − IVT_affinity`
   - Calibrate via two-component **Beta mixture** EM → `p_mod_knn ∈ [0, 1]`
   - Show: scatter of reads in distance space, k-NN circles around a read, color-coded by IVT (blue) vs native (orange)

**Output:** Two parallel per-read probability tracks: `p_mod_raw` (from mixture) and `p_mod_knn` (from kNN).

### Stage V3 — HMM Spatial Smoothing Along Read Trajectories

**Subtitle:** "3-state forward-backward with gap-aware transitions"

This is the most visually rich sub-panel.

1. **Read trajectory construction:** Show a genomic coordinate axis (horizontal) with multiple positions marked. A single nanopore read spans multiple positions (a long RNA molecule passes through the pore). Draw a colored path connecting the positions where this read appears — this is its "trajectory." Show 3–4 overlapping read trajectories of different lengths.

2. **3-State HMM topology** (the key new contribution): Draw a state-transition diagram with three circles:
   - **State U (Unmodified)** — blue circle, left
   - **State F (Flank)** — gray/teal circle, center
   - **State M (Modified)** — orange/red circle, right
   - Self-loops on each state (p_stay, dominant)
   - U → F and F → U arrows (allowed)
   - F → M and M → F arrows (allowed)
   - **U → M and M → U: crossed out / dashed / red X** (FORBIDDEN — must pass through Flank)
   - Label the asymmetric Flank split: F→U = 40%, F→M = 60%
   - Below the diagram, show: `p_stay = 0.98^gap` with a decay curve — larger genomic gaps weaken the state persistence

3. **Biological rationale inset:** A small schematic showing the 5-mer sequencing window. A modification at position X perturbs the signal at positions X−2 through X+2 (the "halo effect"). The Flank state captures these perturbed-but-not-modified flanking positions. Show 5 positions on a line: [U, F, M, F, U] with the modification star at center.

4. **Emission distributions:** Show three overlaid Beta PDFs on a [0, 1] axis:
   - **U → Beta(2, 8):** peaked near 0.2 (low kNN scores)
   - **F → Beta(3, 3):** symmetric, peaked at 0.5 (moderate scores)
   - **M → Beta(8, 2):** peaked near 0.8 (high kNN scores)

5. **Forward-backward inference:** Show a trellis/lattice diagram:
   - Horizontal axis: 5–7 genomic positions along a read trajectory
   - Vertical axis: 3 states (U, F, M)
   - At each position, show emission likelihood as circle size
   - Forward messages (α) flow left-to-right
   - Backward messages (β) flow right-to-left
   - Final posterior γ = α ⊙ β (normalized) gives P(Modified) at each position
   - Color-code the posterior: blue (low) → red (high P(modified))

6. **Output:** The posterior `γ[:, 2]` (Modified state only) is written back as `p_mod_hmm` for each read at each position. Show a genomic track where each position has a colored bar indicating P(modified), with the Flank positions correctly showing LOW p_mod_hmm despite elevated raw scores.

---

## Panel D — Output & Comparison

**Title:** "Per-position modification probability"

Show the final output as a **genome browser-style track**:

1. **Top track:** Reference sequence with 5-mer annotations
2. **Middle track:** Per-position P(modified) as a bar chart or heatmap strip, colored blue (unmodified) through white to red (modified). Show ~20 positions with 2–3 true modification sites clearly standing out.
3. **Bottom track:** Individual read-level heatmap (rows = reads, columns = positions) showing that the HMM smooths noisy per-read calls into coherent modification signals.

Include a small **comparison inset** showing:
- **Without Flank state (2-state):** False positive halo at ±2 positions around true modification
- **With Flank state (3-state):** Clean peak at the true modification site, flanking positions correctly absorbed

---

## Panel E (optional) — Training Modes

**Title:** "HMM parameter learning"

Show three modes as a progression:

1. **Mode A — Unsupervised (default):** No labels needed. Beta emission priors are hardcoded. Arrow from "kNN scores" directly to "3-state HMM."
2. **Mode B — Semi-supervised:** Labeled modification sites → Platt-scaling calibrator (sigmoid: `σ(ax + b)`) learns to map raw kNN scores to calibrated emissions. Show the sigmoid curve.
3. **Mode C — Supervised:** More labels → KDE emission densities learned from data. Show two smooth density curves (unmodified vs modified) replacing the parametric Beta distributions.

Below, show **leave-one-contig-out cross-validation** as a simple fold diagram: train on N−1 contigs, test on the held-out contig, repeat.

---

## Style Notes for the AI Generator

- **Color palette:** Native/Modified = warm orange (#ED7D31), IVT/Unmodified = cool blue (#5B9BD5), Flank = teal/gray (#70AD47 or #808080), background = white, text = dark gray
- **Font:** Sans-serif (Helvetica/Arial), 7–9 pt for labels, 10–12 pt for panel titles
- **Arrows:** Thin (0.5 pt), dark gray, with small arrowheads
- **Mathematical notation:** Use proper subscripts and Greek letters (μ, σ, κ, γ, α, β, π, Φ)
- **Scale:** Full page width (180 mm), 200–250 mm height
- **Resolution:** 300 dpi minimum, vector preferred (SVG/PDF)
- **No 3D effects, no gradients on boxes** — flat, clean, publication-ready
- **Panel labels:** Bold uppercase (A), (B), (C), (D) in top-left corner of each panel
- **Nature style:** Dense but readable, every element serves a purpose, no decorative elements
