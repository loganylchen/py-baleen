# Baleen Workflow Figure Description

> For AI-based figure generation (NanoBanana / similar tools).
> Target: Nature Biotechnology quality, single-panel or multi-panel workflow figure.

---

## Figure Title

**Baleen: Hierarchical Bayesian framework for RNA modification detection from nanopore signal comparison**

---

## Overall Layout

A **left-to-right horizontal workflow** with 4 major blocks, connected by arrows. Each block contains vertically stacked sub-steps. Use a clean, muted color palette (blues, greens, warm accents for highlights). Avoid clutter — emphasize the hierarchical architecture.

**Four major blocks:**

1. **Input & Signal Extraction** (left, cool gray/blue)
2. **DTW Distance Computation** (center-left, teal/cyan)
3. **Hierarchical Modification Calling** (center-right, the core — use 3 vertically stacked sub-panels for V1→V2→V3, warm gradient from light to deep)
4. **Output** (right, green/teal)

---

## Block 1: Input & Signal Extraction

**Color:** Light steel blue / cool gray

### Sub-step 1a: Inputs (show as icons/file shapes)
- **Native nanopore reads**: BAM + FASTQ + BLOW5 (labeled "Native")
- **IVT control reads**: BAM + FASTQ + BLOW5 (labeled "IVT control")
- **Reference transcriptome**: FASTA
- Show native and IVT as two parallel input streams converging

### Sub-step 1b: Event Alignment
- Label: **"f5c eventalign"**
- Small text: "per-read ionic current signals aligned to reference k-mers"
- Arrow from inputs → event alignment
- Show a schematic of a nanopore read being aligned to reference positions, with raw signal segments mapped to k-mer positions

### Sub-step 1c: Signal Grouping
- Label: **"Signal grouping by position"**
- Show signals from multiple reads stacked at a single genomic position
- Visual: small waveform snippets grouped into a column per position
- Note: "Flanking positions concatenated for context (padding = 1)"

### Sub-step 1d: Contig Filtering
- Small annotation: "Depth filter (min. 15 reads in both native & IVT)"
- Show a funnel or filter icon

---

## Block 2: DTW Distance Computation

**Color:** Teal / cyan

### Visual
- Show two signal traces being compared with a warping path between them
- Below: a symmetric **distance matrix** heatmap (native reads on one axis, IVT on the other), with clear block structure:
  - Within-IVT block (low distance, cool color)
  - Within-native block
  - Between native-IVT block (high distance if modified, warm color)

### Labels
- **"Pairwise DTW"**
- **HIGHLIGHT badge:** "CUDA-accelerated batch kernel" — show a GPU icon
- Small text: "All positions in a single GPU call via multi-position kernel with 16 concurrent CUDA streams"
- Note: "CPU fallback (tslearn) when GPU unavailable"

### Key feature to highlight
- The **batch multi-position GPU kernel** is a key technical innovation — one kernel launch processes all positions in a contig simultaneously, not one-at-a-time

---

## Block 3: Hierarchical Modification Calling (V1 → V2 → V3)

**Color:** Warm gradient — light gold (V1) → amber (V2) → deep orange/coral (V3)

This is the **core innovation** and should occupy the most visual space. Show three vertically stacked sub-panels with arrows flowing downward (V1 → V2 → V3), each refining the previous stage's output.

### Sub-panel V1: Empirical-Bayes Null Estimation

**Label:** "V1: Robust IVT Null + Hierarchical Shrinkage"

**Visual concept:**
- Left side: Show a bell curve (Normal distribution) fitted to IVT scores = the "null" distribution
- Right side: Show a **three-level shrinkage diagram**:
  - Bottom level: individual position estimates (noisy dots)
  - Middle level: local window smoothed (±15 positions, smoother line)
  - Top level: global prior (single horizontal line)
  - Arrows showing "borrowing strength" between levels
- Show how shrinkage strength adapts to coverage: thick arrows for low-coverage positions (strong pull toward prior), thin arrows for high-coverage (trust local data)

**Key annotations:**
- "Robust median + MAD (outlier-resistant)"
- "Coverage-adaptive shrinkage: HIGH (κ=0.5) → LOW (κ=5.0)"
- Output arrow labeled: "per-read z-scores & p-values"

**HIGHLIGHT:** The 3-level hierarchical shrinkage (position → local window → global) that adapts to coverage — this enables **low-coverage position recovery**

### Sub-panel V2: Anchored Mixture Model + Soft Gating

**Label:** "V2: Anchored Two-Component EM + Soft Gating"

**Visual concept:**
- Show two overlapping distributions:
  - **Null component** (blue, anchored to IVT) — labeled "fixed to IVT baseline"
  - **Alternative component** (red/orange, fitted to native) — labeled "modification signal"
- Show the EM iteration refining the alternative component
- **Soft gating diagram**: Three sigmoid curves side by side (π gate, BIC gate, separation gate), each outputting a continuous weight ∈ [0,1], multiplied together → final gate weight
- Contrast with a crossed-out binary gate (traditional hard thresholding)

**Key annotations:**
- "Null anchored to IVT (prevents mixture collapse)"
- "Global pooling bootstraps low-coverage positions"
- "Soft gates replace hard binary decisions"
- Output arrow labeled: "per-read P(modified) + confidence weight"

**HIGHLIGHT:** The **soft gating** mechanism (3 continuous sigmoid gates replacing binary decisions) — this is a key methodological contribution. Also highlight the **anchored null** preventing collapse.

### Sub-panel V3: HMM Spatial Smoothing

**Label:** "V3: Gap-Aware HMM (Forward-Backward)"

**Visual concept:**
- Show a **horizontal chain** of states along genomic positions, with:
  - Circles for states (Unmodified / Modified, or 3-state with Flank)
  - Arrows between states representing transitions
  - Transition arrow thickness varies by genomic gap distance (thicker = more positions between → more transition probability)
- Below the chain: show raw V2 probabilities (noisy) vs. smoothed V3 output (clean)
- A small inset showing the exponential decay: "p_stay = 0.98^gap"

**Key annotations:**
- "Gap-aware transitions: transition probability scales with genomic distance"
- "Forward-backward algorithm with numerical scaling"
- "3 training modes: unsupervised / semi-supervised / fully supervised"
- Output arrow labeled: "spatially smoothed P(modified) per read"

**HIGHLIGHT:** The **gap-aware transition model** — transition probabilities decay exponentially with genomic distance between analyzed positions. This is biologically motivated: nearby positions are more likely to share modification status.

---

## Block 4: Output

**Color:** Forest green / teal

### Sub-step 4a: Site-Level Aggregation
- Label: **"Beta-Binomial aggregation + Mann-Whitney U test"**
- Show a genomic track with colored dots (red = modified, blue = unmodified)
- Annotation: "BH-corrected p-values (FDR control)"
- Output: **site_results.tsv** (file icon)
  - Columns shown: position, mod_ratio, p_adj, effect_size, 95% CI

### Sub-step 4b: Read-Level mod-BAM
- Label: **"Standard mod-BAM (MM/ML tags)"**
- Show a BAM record icon with MM:Z and ML:B:C tags highlighted
- Annotation: "Compatible with modkit, modbamtools, IGV"
- **HIGHLIGHT badge:** "Industry-standard format" — show logos/icons for modkit, IGV
- Output: **read_results.bam** (file icon)

---

## Elements to Highlight (Stars / Badges / Callouts)

Use small star or badge icons to call out these innovations:

1. **CUDA batch DTW kernel** — "Single GPU call for all positions per contig"
2. **3-level hierarchical shrinkage** — "Borrows strength across positions and coverage levels"
3. **Soft gating** — "Continuous confidence weighting replaces binary thresholds"
4. **Gap-aware HMM** — "Biologically motivated spatial smoothing"
5. **Standard mod-BAM output** — "Native compatibility with downstream ecosystem"

---

## Color Palette Suggestion

| Element | Color | Hex |
|---------|-------|-----|
| Native reads | Warm coral | #E8734A |
| IVT control | Steel blue | #5B8DB8 |
| V1 stage | Light gold | #F5D76E |
| V2 stage | Amber | #F0A830 |
| V3 stage | Deep coral | #E85D4A |
| Output/positive | Forest green | #4CAF7D |
| Background | Off-white | #FAFAFA |
| Arrows/connections | Dark gray | #4A4A4A |
| Highlight badges | Deep teal | #2C8C99 |

---

## Optional: Bottom Annotation Strip

A thin horizontal strip below the main figure showing the **data dimensionality reduction** at each stage:

```
Input signals     →  Distance matrices  →  Z-scores (V1)  →  P(mod) (V2)  →  Smoothed P(mod) (V3)  →  Site calls
[N×L waveforms]      [N×N per position]    [N per pos]        [N per pos]      [N per pos]              [1 per pos]
```

This emphasizes how the pipeline progressively distills raw signal into interpretable modification calls.

---

## Figure Legend (Draft)

**Figure 1. Overview of the Baleen modification detection pipeline.** Native and IVT control nanopore reads are event-aligned to a reference transcriptome using f5c, and ionic current signals are grouped by genomic position. CUDA-accelerated pairwise DTW distances are computed in a single batched GPU kernel across all positions per contig. A three-stage hierarchical framework then calls modifications: (V1) Empirical-Bayes null estimation with coverage-adaptive three-level shrinkage produces per-read z-scores; (V2) an anchored two-component EM mixture with continuous soft gating yields raw modification probabilities; (V3) a gap-aware Hidden Markov Model applies forward-backward spatial smoothing along read trajectories, where transition probabilities decay exponentially with genomic distance. Site-level results are aggregated via Beta-Binomial estimation with Benjamini-Hochberg FDR correction. Per-read probabilities are output in standard mod-BAM format (MM/ML tags), enabling direct visualization in IGV and analysis with modkit.
