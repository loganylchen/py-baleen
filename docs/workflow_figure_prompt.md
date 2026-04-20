# Baleen Workflow Figure — AI Image-Gen Prompts

Curated prompt templates for generating a **Nature-journal quality workflow
figure** of the Baleen pipeline. Paste one of the blocks below directly into
Nano Banana Pro / Gemini 3 Pro / Flux / DALL·E / Imagen.

The prompts below are engineered to surface Baleen's algorithmic
innovations (not just a generic "DTW + HMM" cartoon). Each visual motif is
chosen to encode a specific methodological claim of the paper.

---

## Algorithmic highlights — what the figure must convey

Before writing a prompt, understand what distinguishes Baleen from a
"plain" score-and-threshold modification caller. The visuals must signal
these six innovations:

1. **Batched multi-position CUDA DTW.** All genomic positions of a contig
   are processed in **one kernel launch** — wavefront parallelism, one
   CUDA thread per cost-matrix row, three rolling diagonals held in
   ~12 KB of shared memory, one thread block per read pair, 16 concurrent
   CUDA streams, FP32 squared-Euclidean cost, single `sqrt` at the end.
   Scale: `O(10⁴)` pairwise alignments concurrently on a single GPU.
2. **Coverage-adaptive three-level James-Stein shrinkage (V1).** The
   per-position IVT null distribution is a *weighted composition* of
   (i) the position-specific MLE, (ii) a local k-mer-context window
   prior, and (iii) the transcriptome-wide global prior. Mixing weights
   are **functions of the position's coverage** — high coverage trusts
   the position; low coverage falls back to context / global.
3. **Null-anchored two-component mixture EM with continuous soft gating
   (V2).** The null component is **frozen** at the V1-calibrated Beta
   (or Normal) parameters; only the alternative component and the
   mixture weight `π` are updated by EM. This resolves the label-switching
   / identifiability problem of free 2-component mixtures. A **continuous
   sigmoid on the BIC gap**, not a binary reject/accept, gates the final
   posterior — removing the brittle threshold behaviour of classical
   mixture callers. A λ-regularised shrinkage pulls alternative
   parameters toward the transcriptome-wide alternative prior.
4. **Gap-aware per-read forward–backward HMM (V3).** Unlike site-level
   HMMs that smooth across sites in an abstract graph, V3 runs a
   **2-state HMM along each individual read's trajectory**, with
   transitions whose probabilities depend on the **genomic distance
   between the read's consecutive called sites**. Emissions are V2
   posteriors. Forward–backward yields per-read marginal posteriors
   at every called site — bit-exact, `numba`-JIT, no `fastmath`.
5. **Per-read output → single-molecule combinatorics.** The pipeline
   does not collapse to per-site stoichiometry; it emits
   `p(mod | read, position)` as standard mod-BAM (`MM:Z` / `ML:B:C`).
   Two or more sites on the **same molecule** can be phased, yielding
   co-deposition / mutual-exclusion / independence contrasts that bulk
   methods cannot observe.
6. **Streaming architecture.** DTW → HMM → Beta-Binomial aggregation is
   **fused per contig**; distance matrices are discarded after inference.
   Memory footprint is independent of transcriptome size.

Every prompt that follows exposes these six themes through one specific
visual hook (wavefront diagonal, shrinkage funnel, anchored pin, gap-aware
chain, per-read phasing grid, streaming conveyor). When a first draft
loses the sophistication, re-inject these hooks one at a time.

---

## Design anchors

- **Palette**: *teal* (`#2F8F9D`) = **native** RNA and "modified" calls;
  *coral* (`#E7734A`) = **IVT** control and "co-deposition" accent;
  *warm red* (`#B33A3A`) = "mutual exclusion" accent; *slate*
  (`#2F3E46`) = skeletons/axes; *light slate* (`#C6CCD1`) = unmodified
  calls; *cream* (`#F5F1EA`) = one optional depth panel; pure white
  (`#FFFFFF`) background. No rainbow gradients, no drop shadows.
- **Typography**: clean sans-serif (Helvetica Neue / Inter / Source Sans),
  black `#111`, weights 400/600, title 600 only. Lowercase axis labels,
  title-case sub-panel titles, italic for technical sub-captions.
- **Line weight**: 1.0–1.5 pt skeletons, 0.75 pt axis ticks.
- **Math callouts**: small monospace formulas (9-pt) are welcome where
  they encode a claim (e.g. `π·f₁(x) / [(1−π)·f₀(x) + π·f₁(x)]` in the
  V2 panel) — use them sparingly, one per sub-panel at most.
- **Iconography**: flat vector, thin stroke, 2-color fills, geometric.
  Pictograms over photoreal.
- **Composition**: panel letters (a, b, c, …) bottom-left in 600-weight;
  ~15 % gutters; minimal chevron connectors.
- **Resolution targets**: `3000 × 1200 px` (2.5 : 1) landscape five-panel,
  `3600 × 1200 px` (3 : 1) landscape six-panel, `1400 × 2000 px` (7 : 10)
  portrait.

---

## Prompt A — Expanded landscape workflow (canonical, six panels)

This is the **recommended manuscript figure 1**. It includes the
combinatorial readout as panel *f*, widening the aspect ratio to 3 : 1.

> **Prompt:**
> A publication-quality scientific workflow figure for a computational
> biology pipeline called Baleen, rendered as a single horizontal panel
> at a 3 : 1 aspect ratio, in a flat vector illustration style
> reminiscent of *Nature Methods* figure 1. Pure white background, no
> drop shadows, no skeuomorphic textures. Six stages arranged
> left-to-right, each separated by a thin slate-gray chevron arrow
> (1 pt), each stage labelled with a lowercase panel letter
> (a, b, c, d, e, f) in 600-weight sans-serif bottom-left.
>
> **Stage a — Single-molecule input.** Two stacked nanopore cross-sections
> (simplified cylindrical pore with protein glyph), a single-stranded
> RNA molecule threading through each. Upper pore labelled "native RNA"
> in teal (#2F8F9D); lower pore labelled "IVT control" in coral
> (#E7734A). To the right of each pore, a short noisy ionic-current
> trace in the matching color, with mild step-like structure and a
> horizontal dashed mean line. Below: a thin slate horizontal bar
> representing the reference transcriptome, with gene-track tick marks
> and a small "ref" label. A tiny 9-pt caption underneath:
> *"direct-RNA nanopore sequencing — paired native vs IVT control"*.
>
> **Stage b — Event alignment (f5c).** A rectangular matrix where each
> row is a read and each column is a reference k-mer (four-letter cells
> A / C / G / U labelled at the top). Inside each cell, a miniature
> waveform glyph representing the ionic-current segment aligned to that
> k-mer. Two such matrices side-by-side, teal-tinted (native) and
> coral-tinted (IVT). A small uppercase label "f5c eventalign" sits
> above. A 9-pt italic caption underneath:
> *"per-read signal segmented and aligned to reference k-mers"*.
>
> **Stage c — Batched CUDA pairwise DTW.** At the top, three overlapping
> isometric square cost matrices (to imply many positions processed
> together), each rendered with a **bright diagonal wavefront band**
> (a teal → white gradient running along the anti-diagonal, signaling
> the rolling three-diagonal shared-memory sweep). Below the matrices,
> a compact stack of three symmetric distance matrices in a single-hue
> blue ramp (#E8F0F4 → #1F4E5F), diagonal zero visible. To the right of
> the matrices, a small GPU-chip pictogram with **16 horizontal stream
> lanes** emanating from its right side (each lane a thin coral-to-teal
> gradient line), denoting concurrent CUDA streams. A tiny italic
> 9-pt caption below: *"one thread block per read pair, 16 CUDA streams,
> entire contig in one kernel launch"*.
>
> **Stage d — Hierarchical modification calling (V1 · V2 · V3).** A
> rounded cream (#F5F1EA) panel spanning ~30 % of the figure width,
> divided into three horizontally stacked sub-blocks of equal height,
> each with its own mini-title in 10-pt 600-weight:
>
> (i) **V1 · empirical-Bayes null (three-level shrinkage).** A
> funnel-shaped diagram: at the top, a wide band of light coral small
> dots representing many IVT reads at one position; the band narrows
> downward through three shrinkage stages labelled in 8-pt slate,
> left-to-right or top-to-bottom: *"position"* → *"local k-mer window"*
> → *"global transcriptome"*, with **a small coverage gauge** (circular
> dial, 0 %–100 %) sitting beside the funnel whose needle position
> determines how wide each stage is — high coverage narrows to the
> position estimate quickly, low coverage remains wide until global.
> Output: a single teal Beta density curve at the bottom of the funnel,
> labelled *"Beta(α₀, β₀)"* in 9-pt italic.
>
> (ii) **V2 · null-anchored mixture EM.** Two overlaid probability
> density curves: a **pinned coral null** (Beta(α₀, β₀) from V1, with a
> small pin / padlock glyph above it denoting the *"anchored, fixed"*
> status) and a **teal alternative** (Beta(α₁, β₁), being learned, with
> a small curved arrow showing parameter motion). A thin continuous
> sigmoid gate curve runs horizontally along the baseline between the
> two densities, with its inflection marked by a small vertical tick
> (NOT a step function — a smooth S-curve). A miniature
> 9-pt monospace formula callout floats to the right of the densities:
> `γ = σ(ΔBIC) · π · f₁ / [(1−π) f₀ + π f₁]`. Beneath: an inset EM
> convergence trajectory — a tiny line plot of `log-likelihood vs
> iteration` going up and plateauing at ~10–30 iterations, labelled in
> 8-pt slate. A small λ-shrinkage glyph (a tiny tether arrow from
> Beta(α₁, β₁) pointing toward a faint "global prior" ghost curve)
> signals the regularisation toward transcriptome-wide alternative
> parameters.
>
> (iii) **V3 · gap-aware per-read forward–backward HMM.** A horizontal
> graphical-model strip: five circular hidden-state nodes in a row,
> each split diagonally into a teal "mod" half and a light-slate
> "unmod" half (signalling a 2-state HMM). Adjacent nodes connected by
> **arrows whose length AND stroke weight vary with the genomic
> distance label printed beneath the arrow** (e.g. "Δ = 1", "Δ = 3",
> "Δ = 7", "Δ = 2", "Δ = 12") — making the "gap-aware" nature
> visually explicit. Each node has a short downward emission arrow to
> a small observed posterior circle (with a tiny histogram glyph
> inside). **Two curved sweeping arrows arch above the chain**, one
> from left to right labelled "α (forward)" in coral, one from right
> to left labelled "β (backward)" in teal, marking the two passes of
> the forward–backward algorithm. A tiny 9-pt italic caption beneath:
> *"per-read trajectory · bit-exact numba JIT · transitions depend on
> genomic gap"*.
>
> **Stage e — Per-site statistical readout.** Top half: a miniature TSV
> table with columns `contig | pos | mod_ratio | ci_low | ci_high |
> padj`, three body rows with realistic-looking values (e.g.
> `ENST…1234 | 742 | 0.68 | 0.61 | 0.74 | 3.2e-08`). Bottom half: a
> small volcano plot inset, `-log₁₀(padj)` on y-axis vs `mod_ratio
> difference` on x-axis, with scattered dots, a dashed significance
> threshold (horizontal, labelled "BH FDR 0.05"), and a handful of
> teal highlighted points above the threshold. Title in 10-pt 600-weight
> "site_results.tsv — Beta-Binomial posterior + BH FDR". A 9-pt italic
> caption: *"Mann-Whitney U native vs IVT, HMM-posterior-weighted
> counts, 95 % credible intervals"*.
>
> **Stage f — Single-molecule combinatorial readout.** A narrow
> vertical stack on the far right, three stacked elements:
> (top) a compact stack of six horizontal per-read tracks (each a thin
> slate line) across three candidate sites i / j / k marked by vertical
> dashed guides; at each site a filled circle — teal for modified,
> light slate for unmodified — with a pattern that clearly shows two
> reads carrying both i and j (co-deposited), two reads carrying only
> one (exclusive), and two reads carrying neither or all three;
> (middle) a 2 × 2 contingency grid for sites (i, j), single-hue teal
> ramp shading, with a 9-pt monospace annotation below:
> `log-odds = +2.31, Fisher p = 4.8e-06`;
> (bottom) a mini diverging colorbar swatch labelled "log-odds ratio"
> running from warm red (#B33A3A, "exclusive") through white at zero
> to coral (#E7734A, "co-deposited"). Title in 10-pt 600-weight
> "combinatorial phasing (per-read)". Tiny italic 9-pt caption:
> *"single-molecule resolution — bulk stoichiometry cannot resolve
> these configurations"*.
>
> **Global typography & finish.** All type in Helvetica Neue or Inter,
> black #111. Stage titles 12-pt 600-weight, sub-titles 10-pt
> 600-weight, body labels 9-pt 400-weight, italic technical captions
> 9-pt italic slate. Formula callouts in monospace 9-pt. Line weight
> 1.0–1.5 pt for skeletons, 0.75 pt for axis ticks. **No drop shadows,
> no rainbow gradients** — only the named single-hue / diverging
> ramps. Arrows are thin slate-gray chevrons. Generous 12–15 % gutters
> between stages. Overall aesthetic: clean, dense but uncluttered,
> editorial, publication-ready — signalling algorithmic sophistication
> through specific mathematical motifs (wavefront diagonal, shrinkage
> funnel, anchored pin, gap-aware chain, forward/backward arcs,
> phasing grid) rather than decorative flourish.

---

## Prompt A-compact — Five-panel landscape (pipeline only, no combinatorial)

Use this for a cleaner five-panel pipeline-only figure when the
combinatorial analysis lives in a separate figure 2.

> **Prompt:**
> A *Nature Methods*-style scientific workflow figure, 2.5 : 1
> landscape, pure white background, flat vector style, five stages
> labelled a–e in 600-weight lowercase sans-serif.
>
> (a) **Input** — two stacked nanopore cross-sections threading RNA
> (upper teal "native", lower coral "IVT"), each with a short noisy
> ionic-current trace; slate reference bar underneath.
>
> (b) **Event alignment (f5c)** — two read × k-mer matrices
> (teal-tinted native, coral-tinted IVT) filled with miniature waveform
> glyphs, labelled "f5c eventalign".
>
> (c) **Batched CUDA pairwise DTW** — three overlapping isometric cost
> matrices with a bright teal diagonal wavefront band, below them a
> stack of blue-ramp symmetric distance matrices; to the right a GPU
> chip pictogram with 16 stream lanes; 9-pt italic caption
> *"one block per read pair · 16 CUDA streams · single-launch"*.
>
> (d) **Hierarchical calling — V1 · V2 · V3** in a rounded cream panel:
>  **V1** shrinkage funnel from IVT reads through three levels
>  (position → local k-mer window → global) with a small coverage dial
>  controlling the funnel width, terminating in a teal `Beta(α₀, β₀)`
>  density;
>  **V2** overlaid coral-pinned (padlock glyph) null density plus a
>  teal learnable alternative density, a smooth horizontal sigmoid
>  gate along the baseline, a small EM-convergence inset, a λ-shrinkage
>  tether arrow toward a ghost global-prior curve, monospace callout
>  `γ = σ(ΔBIC) · π·f₁ / [(1−π)f₀ + π·f₁]`;
>  **V3** five-node two-state HMM chain with emission arrows, variable
>  arrow lengths/widths labelled "Δ = 1, 3, 7, 2, 12" (gap-aware
>  transitions), and two arcing arrows above the chain — coral
>  "α forward" and teal "β backward" — for forward–backward.
>
> (e) **Output** — miniature TSV (`contig | pos | mod_ratio | ci_low |
> ci_high | padj`) with a volcano-plot inset (dashed BH-FDR threshold,
> teal significant points), and a colored per-base mod-BAM read strip
> (teal = modified, light slate = unmodified) labelled "MM / ML tags".
>
> Palette: teal #2F8F9D, coral #E7734A, slate #2F3E46, light slate
> #C6CCD1, cream #F5F1EA, white. Helvetica / Inter 9–12 pt, 1 pt
> lines, no drop shadows, no rainbow gradients. Publication-ready,
> editorial, dense but uncluttered — algorithmic sophistication
> encoded through specific visual motifs (wavefront diagonal,
> shrinkage funnel, anchored pin, gap-aware chain, forward/backward
> arcs), not decorative flourish.

---

## Prompt B — Short version (rapid iteration, Flux-schnell scale)

> Flat vector *Nature Methods*-style scientific workflow, 2.5 : 1
> landscape, white background, five stages labelled a–e.
> (a) Native + IVT nanopores threading RNA with ionic-current traces
> (teal / coral). (b) Two read × k-mer waveform matrices,
> "f5c eventalign". (c) Three overlapping cost matrices with a teal
> diagonal wavefront band + blue-ramp symmetric distance heatmaps +
> GPU chip with 16 stream lanes, "batched CUDA DTW".
> (d) Cream panel with three sub-blocks: V1 coverage-controlled
> shrinkage funnel (position→local→global) ending in a teal
> `Beta(α₀,β₀)`; V2 pinned-coral null + learnable-teal alternative
> with a smooth sigmoid soft-gate + EM convergence inset +
> λ-shrinkage tether; V3 five-node 2-state HMM chain with
> genomic-gap-labelled variable arrows (Δ=1,3,7,2,12) and two arcing
> forward/backward passes. (e) Miniature TSV with volcano-plot inset
> + mod-BAM per-base strip. Palette teal #2F8F9D / coral #E7734A /
> warm red #B33A3A / slate / cream / white. Helvetica / Inter, 1 pt
> lines, no shadows, publication-ready, algorithmic motifs
> (wavefront, funnel, anchor, gap-aware chain, F/B arcs) preserved.

---

## Prompt C — Vertical layout (portrait, supplementary / slide)

> *Nature Methods*-style flat-vector workflow, 7 : 10 portrait, white
> background, five stages stacked top-to-bottom in horizontal bands
> connected by downward chevrons, labels a–e.
> **(a) Input** — paired teal-native / coral-IVT pores with
> ionic-current traces and a reference bar.
> **(b) f5c eventalign** — two read × k-mer waveform matrices.
> **(c) Batched CUDA DTW** — overlapping cost matrices with diagonal
> wavefront band, blue-ramp distance matrices, GPU chip with 16
> stream lanes; caption *"entire contig in one kernel launch"*.
> **(d) V1 · V2 · V3 in a cream panel** — V1 coverage-adaptive
> three-level shrinkage funnel (position → local → global) with a
> coverage dial; V2 pinned-coral null + learnable-teal alternative
> with a continuous sigmoid soft-gate, EM convergence inset, and a
> λ-shrinkage tether; V3 five-node 2-state HMM chain with
> genomic-gap-labelled variable-width transitions and arcing
> forward/backward passes.
> **(e) Output** — miniature TSV with volcano-plot inset plus a
> colored mod-BAM read strip (MM / ML tags).
> Palette teal / coral / warm red / slate / cream / white.
> Helvetica / Inter, 1 pt lines, no shadows, no gradients except the
> named single-hue ramps. Publication-ready.

---

## Prompt D — Ultra-compact one-sentence (prompt-budget-constrained)

> Clean *Nature Methods*-style flat-vector workflow, 2.5:1 on white,
> teal/coral/slate palette, five panels a–e: (a) native+IVT nanopores
> with ionic-current traces; (b) f5c eventalign matrices; (c) batched
> CUDA DTW with wavefront-diagonal cost matrices, blue-ramp distance
> heatmaps, and a GPU chip with 16 stream lanes; (d) cream panel with
> V1 coverage-adaptive three-level shrinkage funnel, V2 pinned-null
> + learnable-alt densities with a smooth sigmoid soft-gate and EM
> convergence inset, V3 five-node 2-state HMM chain with gap-labelled
> variable transitions and forward/backward arcs; (e) site TSV with
> volcano-plot inset plus a per-base mod-BAM strip. Helvetica, 1 pt
> lines, no shadows, no gradients except named ramps, algorithmic
> motifs preserved, publication-ready.

---

## Stage-by-stage annotation reference

If a tool lets you caption panels separately, use these **exact**
wordings — they match the paper in preparation and the CLI vocabulary.

| Panel | Title | Algorithmic subtitle |
|-------|-------|----------------------|
| **a** | Input | Paired native direct-RNA reads + in-vitro-transcribed control + reference transcriptome |
| **b** | Event alignment | Per-read ionic-current signals segmented and aligned to reference k-mers (`f5c eventalign`) |
| **c** | Batched CUDA pairwise DTW | Wavefront parallelism, one thread block per read pair, 16 concurrent CUDA streams, entire contig in one kernel launch |
| **d(i)** | V1 · empirical-Bayes null | Coverage-adaptive three-level James-Stein shrinkage (position → local k-mer window → global) |
| **d(ii)** | V2 · anchored mixture EM | Null-frozen two-component mixture with continuous soft-gating (`σ(ΔBIC)`) and λ-regularised alternative prior |
| **d(iii)** | V3 · gap-aware forward–backward | Per-read 2-state HMM whose transition probabilities depend on genomic gap between consecutive called sites |
| **e** | Per-site statistical readout | Beta-Binomial MAP + 95 % credible intervals · Mann-Whitney U (native vs IVT) · BH-adjusted FDR |
| **f** | Combinatorial phasing | Single-molecule mod-BAM output (`MM:Z` / `ML:B:C`) enables co-deposition / mutual-exclusion contrasts over arbitrary site sets |

---

## Editing tips — re-injecting algorithmic motifs when drafts look generic

| Draft problem | Prompt fix |
|---------------|------------|
| DTW looks like a generic heatmap | *"cost matrices must show a bright anti-diagonal wavefront band (teal → white gradient), signalling rolling-diagonal shared-memory sweep"* |
| DTW stage doesn't read as parallel | *"add 16 horizontal stream lanes emanating from a GPU-chip pictogram, each lane a thin coral-to-teal gradient, denoting concurrent CUDA streams"* |
| V1 looks like a single arrow | *"replace with a three-level funnel narrowing through 'position → local k-mer window → global', with a circular coverage dial whose needle controls funnel width"* |
| V2 looks like just two Gaussians | *"add a padlock or pin glyph above the null curve labelled 'anchored', a curved parameter-motion arrow on the alternative, a smooth S-shaped sigmoid gate along the baseline (NOT a step), a 9-pt monospace formula callout `γ = σ(ΔBIC)·π·f₁/[(1−π)f₀+π·f₁]`, an EM convergence inset (log-likelihood vs iteration plateauing), and a λ-shrinkage tether to a ghost global-prior curve"* |
| V3 looks like a vanilla HMM chain | *"vary the arrow lengths and stroke weights between adjacent nodes according to printed genomic-gap labels (Δ=1, 3, 7, 2, 12), split each node diagonally into teal 'mod' and light-slate 'unmod' halves, and add two arcing passes above the chain — coral 'α forward' left-to-right and teal 'β backward' right-to-left"* |
| Too slick / marketing-y | *"remove decorative flourish, desaturate palette, restrict to the named single-hue / diverging ramps, re-emphasise specific mathematical motifs"* |
| Too busy | *"drop the formula callouts, keep only the visual motifs; widen gutters to 18 %"* |
| Wrong typography | *"all type in Helvetica Neue 400/600, black #111, lowercase axis labels, title-case sub-panel titles, 9-pt italic for technical captions, 9-pt monospace for formula callouts only"* |

---

## Recommended negative prompts (SD / Flux)

```
photorealism, 3D render, drop shadows, rainbow gradient, neon glow,
cartoon mascots, stock-photo scientists, textured paper, grunge,
hand-drawn sketch, watermark, JPEG artifacts, cluttered composition,
decorative swirls, marketing infographic, isometric office scene,
generic heatmap without wavefront, vanilla HMM without gap-awareness
```

---

## Single-read combinatorial analysis — standalone figure

If the combinatorial readout deserves its own figure (rather than being
folded in as panel *f* of Prompt A), use Prompt E below.

### Scientific framing

Because Baleen emits per-read modification probabilities in standard
mod-BAM (`MM:Z` / `ML:B:C`), modifications at two or more sites can be
phased on the **same molecule**:

| Observation | Interpretation |
|-------------|----------------|
| Positions *i* and *j* mod together more often than product of marginals | **Co-deposition** — shared writer, structural co-dependency, or coupled regulation |
| Positions *i* and *j* mod together less often than product of marginals | **Mutual exclusion** — writer/eraser competition, allele- or isoform-specific patterns |
| Joint rate ≈ product of marginals | **Independence** — bulk per-site stoichiometry is sufficient |

These readouts are computed directly from the per-read output (e.g.
`modkit extract`, or `load_read_results()` + Fisher / log-odds /
phi-coefficient statistics over any pair or set of sites).

### Prompt E — Standalone combinatorial figure (four sub-panels)

> **Prompt:**
> A publication-quality scientific figure titled *"Single-molecule
> combinatorial modification analysis"*, single horizontal panel at
> 2.5 : 1, pure white background, flat vector illustration in *Nature
> Methods* style. Four sub-panels labelled a, b, c, d (lowercase
> 600-weight, bottom-left), connected only by visual flow (no arrows).
> Palette: teal #2F8F9D for modified calls, light slate #C6CCD1 for
> unmodified calls, coral #E7734A for co-deposition accent, warm red
> #B33A3A for mutual-exclusion accent, neutral slate #2F3E46 for
> skeletons, white background.
>
> **(a) Per-read evidence.** A stack of ~12 horizontal read tracks,
> each a thin slate line. Three candidate modification sites at
> columns i, j, k are marked by vertical dashed guides. At each site,
> a filled circle — teal if modified, light slate if unmodified. The
> joint pattern across reads is deliberately diverse: some reads
> carry (i ∧ j), some only i, some only j, some neither, a few carry
> all three. Thin 8-pt gray labels *"read 1 … read 12"*. Title in
> 10-pt 600-weight *"per-read modification calls (mod-BAM, MM/ML
> tags)"*. A 9-pt italic caption below the stack:
> *"each row = one molecule; columns = candidate sites"*.
>
> **(b) 2 × 2 contingency.** A compact 2 × 2 grid for sites (i, j):
> rows `i mod | i unmod`, columns `j mod | j unmod`. Cell shading
> uses a single-hue teal ramp (#E8F0F4 → #2F8F9D), darker = higher
> count. Counts printed inside cells in 9-pt monospace. Below the
> grid, annotation in 9-pt italic:
> *"log-odds = +2.31, Fisher p = 4.8e-06"*. Tiny title
> *"pairwise joint distribution"*.
>
> **(c) Pairwise co-deposition / exclusion map.** A square symmetric
> heatmap over ~10 candidate sites, axes labelled by site index.
> Diverging palette: warm red (#B33A3A) for negative log-odds
> (mutual exclusion), white at zero, coral (#E7734A) for positive
> log-odds (co-deposition). Diagonal masked in light gray. A thin
> horizontal colorbar along the right edge labelled
> *"log-odds ratio (reads)"* with ticks at −3, 0, +3. Title
> *"pairwise co-deposition map"*. A 9-pt italic caption
> *"BH-FDR-filtered pairs; Fisher's exact per cell"*.
>
> **(d) Stoichiometric breakdown vs bulk confusion.** Four vertical
> bars giving the fraction of reads in each joint class for sites
> (i, j): *"both"* (teal), *"only i"* (teal 60 %), *"only j"*
> (teal 30 %), *"neither"* (light slate). Above the bars, three tiny
> molecule schematics with distinct dot patterns, labelled
> *"co-deposited"*, *"exclusive"*, *"independent"*. To the right, a
> small ghost "bulk average" bar with a question mark, and a 9-pt
> italic caption
> *"bulk stoichiometry cannot distinguish these three molecular
> configurations — they all yield the same per-site modification
> ratios"*.
>
> Typography throughout: Helvetica Neue or Inter, black #111.
> Sub-panel titles 10-pt 600-weight, body 9-pt, monospace for counts
> and statistics, italic for technical captions. Line weight 1 pt
> skeletons, 0.75 pt axis ticks. **No drop shadows, no rainbow
> gradients** — only the named single-hue teal ramp and diverging
> red-white-coral ramp. Generous 15 % gutters. Overall aesthetic:
> editorial, precise, minimal, publication-ready — the figure must
> read as a *single-molecule advantage* claim, not a generic
> correlation plot.

### Prompt E-short

> Flat vector *Nature Methods*-style figure, 2.5 : 1 landscape, white
> background, four sub-panels a–d for single-molecule combinatorial
> modification analysis from per-read mod-BAM output. (a) Stack of
> ~12 read tracks with teal/light-slate mod dots across three sites
> i, j, k (vertical dashed guides); title *"per-read modification
> calls"*. (b) Compact 2 × 2 contingency grid, teal-ramp shading,
> 9-pt italic annotation `log-odds = +2.31, Fisher p = 4.8e-06`.
> (c) Square symmetric pairwise heatmap over ~10 sites, diverging
> warm-red (#B33A3A) ↔ white ↔ coral (#E7734A) palette, colorbar
> *"log-odds ratio"*, title *"pairwise co-deposition map"*. (d) Four
> vertical bars for "both / only i / only j / neither" classes, with
> three molecule schematics labelled *"co-deposited / exclusive /
> independent"* and a ghost bulk-average bar annotated
> *"bulk stoichiometry cannot distinguish these configurations"*.
> Palette: teal #2F8F9D, warm red #B33A3A, coral #E7734A, slate
> #2F3E46, light slate #C6CCD1, white. Helvetica / Inter, 1 pt
> lines, no shadows, no gradients outside the named ramps,
> publication-ready.

---

## Reuse

- **Prompt A** — canonical manuscript figure 1, six panels, 3 : 1
  landscape, pipeline + combinatorial readout combined.
- **Prompt A-compact** — five-panel pipeline-only variant when the
  combinatorial story lives in figure 2.
- **Prompt B** — rapid iteration for short-prompt-window models.
- **Prompt C** — portrait layout for poster / supplementary / slides.
- **Prompt D** — one-sentence ultra-compact for tight prompt budgets.
- **Prompt E / E-short** — standalone combinatorial figure showcasing
  the single-molecule advantage.

All prompts are engineered to expose Baleen's six algorithmic signatures
(wavefront-diagonal CUDA DTW, three-level coverage-adaptive shrinkage,
null-anchored mixture EM with soft gating, gap-aware per-read HMM,
single-molecule combinatorics, streaming architecture). If a draft loses
any of these signals, consult the **Editing tips** table to re-inject
the missing motif with a single targeted sentence.
