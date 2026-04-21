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
these seven innovations (items 3 and 3b are both shown in panel C as
adjacent sub-blocks; default runtime uses the 3b / kNN branch as the
HMM emission source):

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
   parameters toward the transcriptome-wide alternative prior. **Note on
   the default runtime path:** V2 mixture posteriors (`p_mod_raw`) are
   computed every call but are **not** the HMM emission source in the
   default unsupervised pipeline. The default emission source is the
   kNN IVT-purity score (see bullet 3b). `p_mod_raw` is consumed only
   when `emission_source` is switched to `"p_mod_raw"` or when the HMM
   is trained in semi-supervised / supervised mode (`_hmm_training.py`
   reads it as the calibration signal).
3b. **kNN IVT-purity + Beta-EM calibration (default HMM emission
   source).** For each read, a k-weighted IVT-purity score is computed
   from its position in DTW space (fraction of k nearest neighbours
   that are IVT controls, distance-weighted). The raw score is
   calibrated by a Beta(a₀, b₀) / Beta(a₁, b₁) two-component EM whose
   null is anchored on the IVT subset. This produces `p_mod_knn`, which
   is what the V3 HMM actually reads as emissions in the default
   pipeline.
4. **Gap-aware per-read forward–backward HMM (V3).** Unlike site-level
   HMMs that smooth across sites in an abstract graph, V3 runs a
   **2-state HMM along each individual read's trajectory**, with
   transitions whose probabilities depend on the **genomic distance
   between the read's consecutive called sites**. Emissions are
   `p_mod_knn` by default (V2 mixture posteriors `p_mod_raw` are an
   alternative emission source selected by training mode or by an
   explicit `--emission-source` override). Forward–backward yields
   per-read marginal posteriors at every called site — bit-exact,
   `numba`-JIT, no `fastmath`.
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
visual hook (**DTW warping correspondence + wavefront accent** for the
merged signal/DTW panel, shrinkage funnel, anchored pin, gap-aware
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
- **Composition**: every panel is visually demarcated two ways at once —
  (i) thin slate vertical rules (0.5 pt, `#2F3E46`) separate adjacent
  panels, and (ii) a **prominent bold uppercase letter label**
  (A, B, C, …) sits in the **top-left corner** of each panel, rendered
  in 18-pt **700-weight** Helvetica black on an optional light-slate
  (`#E4E7EA`) 24 × 24 px rounded square. The letter must be large
  enough to read at thumbnail size (~ 1/20 of figure height). Minimal
  thin slate chevron connectors (1 pt) bridge adjacent panels at mid-height.
  ~15 % gutters between panels.
- **Resolution targets**: `3000 × 1200 px` (2.5 : 1) landscape five-panel,
  `3600 × 1200 px` (3 : 1) landscape six-panel, `1400 × 2000 px` (7 : 10)
  portrait.

---

## Prompt A — Expanded landscape workflow (canonical, five panels)

This is the **recommended manuscript figure 1**. Former panels B
("signal extraction") and C ("CUDA DTW") are merged into a single
panel B so the figure tells a cleaner story: **signals in → pairwise
distance out**. The combinatorial readout remains as panel *E*, at a
2.75 : 1 aspect ratio.

> **Prompt:**
> A publication-quality scientific workflow figure for a computational
> biology pipeline called Baleen, rendered as a single horizontal panel
> at a 2.75 : 1 aspect ratio, in a flat vector illustration style
> reminiscent of *Nature Methods* figure 1. Pure white background, no
> drop shadows, no skeuomorphic textures. **Five panels arranged
> left-to-right, each panel visually demarcated by a thin 0.5-pt slate
> (#2F3E46) vertical rule separating it from its neighbour, plus a
> prominent bold uppercase letter label — A, B, C, D, E — in the
> top-left corner of each panel rendered at 18-pt 700-weight black
> Helvetica, seated inside an optional light-slate (#E4E7EA) 24 × 24 px
> rounded square so the panel identity is unambiguous even at thumbnail
> size.** A thin slate chevron connector (1 pt) sits at mid-height
> between adjacent panels indicating data flow.
>
> **Relative panel sizing (important).** Visual hierarchy:
> **C (largest, algorithmic centrepiece) > D (tall stacked output) >
> B (merged signals + DTW, wider than A but narrower than C) > A
> (compact input) > E (narrow side column)**. Panel C dominates. Panel
> B is the only place DTW and f5c appear — keep its internal density
> moderate (one clear alignment motif, not a gallery of matrices).
>
> **Panel A — Single-molecule input.** Bold uppercase "A" label in the
> top-left corner (18-pt 700-weight, on light-slate rounded square).
> Two stacked nanopore cross-sections
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
> **Panel B — Signal comparison via batched CUDA DTW** *(merged
> preprocessing + algorithm; one unified story)*. Bold uppercase "B"
> label in the top-left corner (18-pt 700-weight, on light-slate
> rounded square). Render as **three horizontally adjacent sub-regions
> within the same panel**, flowing left-to-right, with a subtle thin
> chevron glyph between adjacent sub-regions to indicate transformation
> (NOT separate bordered boxes):
>
>  *(left sub-region) Per-position signal pairs.* A compact vertical
>  stack of three tiny sub-cells, one per reference position, each
>  containing two overlaid ionic-current traces — a **teal "native"**
>  line above and a **coral "IVT"** line below — segmented by faint
>  vertical ticks into k-mer windows. Tiny 7-pt grey 3-character
>  attribution *"via f5c eventalign"* underneath, unobtrusive.
>
>  *(middle sub-region — the hero motif) DTW warping.* A single
>  zoomed-in position callout (thin hairline leader line coming in
>  from the left sub-region). Inside: two ionic-current traces plotted
>  back-to-back, teal native on top and coral IVT below (mirrored
>  about a horizontal axis). **Approximately 8–12 thin slate curved
>  correspondence lines** span the gap between them, connecting
>  matching sample points — the classical DTW warping correspondence.
>  A small 9-pt italic label beneath: *"dynamic time warping
>  alignment → distance"*. This motif is the **visual centre of mass**
>  of the panel: it literally shows what DTW does.
>
>  *(right sub-region) Batched CUDA execution.* A single compact
>  symmetric distance matrix (blue single-hue ramp #E8F0F4 → #1F4E5F,
>  diagonal zero visible) labelled *"reads × reads"* in 8-pt slate
>  underneath. Overlaid on one corner of this matrix, a small teal →
>  white diagonal wavefront accent stripe (signalling the rolling
>  three-diagonal GPU sweep). To the right, a minimal GPU-chip
>  pictogram with four thin parallel stream lanes emanating from its
>  side. A tiny 9-pt italic caption spanning beneath the whole
>  right sub-region: *"all positions of the contig in a single
>  kernel launch · 16 CUDA streams"*.
>
> The three sub-regions together read as: *signals extracted → warped
> comparison → batched distance matrix*. One coherent algorithmic
> statement, one panel.
>
> **Panel C — Hierarchical modification calling (V1 · V2 · V3)** *(the
> algorithmic centrepiece — largest panel)*. Bold uppercase "C" label
> in the top-left corner (18-pt 700-weight, on light-slate rounded
> square). A rounded cream (#F5F1EA) sub-panel spanning ~32–35 % of
> the figure width — the **dominant panel of the figure**, clearly
> wider than panel B,
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
> **Panel D — Per-site statistical readout (reference-anchored stack).**
> Bold uppercase "D" label in the top-left corner (18-pt 700-weight, on
> light-slate rounded square). Render as a **vertically stacked
> five-layer "genome-browser" track** — all layers share a common
> horizontal reference axis (a thin slate bar at the bottom with a
> `5' → 3'` arrowhead, tick marks, and three highlighted candidate
> positions marked by faint vertical guide lines spanning all layers).
> Layers from top to bottom:
>
>  *(i) posterior-density ridgeline.* For each of ~12 reference
>  positions along the axis, a tiny **teal violin / ridge** encoding
>  the full Beta-Binomial posterior over the per-site modification
>  rate. Shape width scales with posterior concentration (narrow =
>  high coverage, wide = low coverage). The three candidate positions
>  are filled solid teal; the others are outlined only. This is
>  **distinct from a bar chart** — the violins convey posterior
>  uncertainty, not a point estimate.
>
>  *(ii) 95 % credible-interval forest.* Directly below each violin,
>  a horizontal whisker (IQR bar + 95 % CI ticks) centred on the
>  posterior MAP estimate, rendered in slate, with a short vertical
>  tick at the MAP. Colour of the MAP tick matches significance:
>  teal if the site passes BH-FDR, light slate otherwise.
>
>  *(iii) Manhattan-style significance lollipops.* `-log₁₀(p_adj)`
>  plotted as thin vertical stems rising from the reference axis
>  (stem height = evidence); stems capped with a small filled
>  circle. A dashed horizontal line at `-log₁₀(0.05)` with a 9-pt
>  italic "BH FDR 0.05" label. Stems above the threshold are teal,
>  below are light slate. Three stems correspond to the candidate
>  positions, visibly taller.
>
>  *(iv) Mann-Whitney native-vs-IVT mini-inset.* One callout balloon
>  anchored to the tallest lollipop, containing two overlaid density
>  curves — a teal "native" and a coral "IVT" distribution of per-read
>  `p(mod)` — with the area of non-overlap lightly hatched and a
>  tiny 9-pt monospace callout inside: `U = 1.8e4, p = 4.8e−09`.
>  This encodes the Mann-Whitney U test that feeds each lollipop.
>
>  *(v) per-read mod-BAM strip.* A compact stack of ~8 horizontal
>  read tracks underneath the reference axis, each a thin slate line
>  with teal dots at called modified positions and faint light-slate
>  dots at unmodified positions. A small tag annotation on the right:
>  `MM:Z / ML:B:C` in 9-pt monospace. Caption in 9-pt italic:
>  *"per-read calls retained — bulk-invisible information preserved
>  for downstream phasing (Panel E)"*.
>
> Panel title in 10-pt 600-weight: *"per-site readout · Beta-Binomial
> posterior · BH-FDR-adjusted"*. Top-right corner carries a tiny
> 9-pt monospace glyph `site_results.tsv` to anchor the output
> filename without dominating the panel. The whole panel reads as a
> **reference-coordinate-aligned stack** rather than a generic
> table + volcano — every row is pinned to the same genomic axis,
> emphasising that the output is spatially resolved along the
> transcript.
>
> **Panel E — Single-molecule combinatorial readout.** Bold uppercase
> "E" label in the top-left corner (18-pt 700-weight, on light-slate
> rounded square). A narrow vertical stack on the far right, three
> stacked elements:
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
> through specific mathematical motifs (DTW warping correspondence +
> wavefront accent, shrinkage funnel, anchored pin, gap-aware chain,
> forward/backward arcs, phasing grid) rather than decorative flourish.

---

## Prompt A-compact — Four-panel landscape (pipeline only, no combinatorial)

Use this for a cleaner four-panel pipeline-only figure when the
combinatorial analysis lives in a separate figure 2.

> **Prompt:**
> A *Nature Methods*-style scientific workflow figure, 2.25 : 1
> landscape, pure white background, flat vector style, **four panels
> each marked by a bold uppercase letter A–D in the top-left corner
> (18-pt 700-weight black, inside an optional 24 × 24 px light-slate
> rounded square), separated by thin 0.5-pt slate vertical rules**.
>
> **Relative sizing:** C (V1·V2·V3, centrepiece) > D (output stack) >
> B (merged signals+DTW) > A (input). Panel C dominates.
>
> **(A) Input** — two stacked nanopore cross-sections threading RNA
> (upper teal "native", lower coral "IVT"), each with a short noisy
> ionic-current trace; slate reference bar underneath.
>
> **(B) Signal comparison via batched CUDA DTW** *(merged preprocessing
> + algorithm)* — three sub-regions flowing left to right inside one
> panel: (left) tiny stacked pairs of teal-native / coral-IVT
> ionic-current traces per reference position, with a 7-pt grey
> *"via f5c eventalign"* attribution; (middle, hero motif) a zoomed-in
> single position showing two traces plotted back-to-back (mirrored)
> connected by ~10 thin slate curved correspondence lines — the
> classical DTW warping correspondence, with a 9-pt italic caption
> *"dynamic time warping alignment → distance"*; (right) one compact
> blue-ramp symmetric distance matrix labelled *"reads × reads"*, a
> small diagonal teal → white wavefront accent stripe on one corner,
> a minimal GPU chip with four thin parallel stream lanes, 9-pt italic
> caption *"all positions of the contig in one kernel launch · 16
> CUDA streams"*. The warping correspondence is the visual centre of
> mass; cost matrices and GPU chip are accents.
>
> **(C) Hierarchical calling — V1 · V2 · V3** (bold uppercase "C" label
> in the top-left corner, 18-pt 700-weight on light-slate rounded
> square) in a rounded cream panel:
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
> **(D) Output — reference-anchored stack** (bold uppercase "D" label
> in the top-left corner, 18-pt 700-weight on light-slate rounded
> square). A vertically stacked multi-layer genome-browser track
> pinned to a shared horizontal reference axis (`5' → 3'` arrow,
> tick marks, three candidate positions highlighted by faint vertical
> guides spanning all layers): (i) a **teal Beta-Binomial posterior
> ridgeline** — one tiny violin per reference position encoding the
> full per-site modification-rate posterior (width = uncertainty, not
> a bar chart); (ii) a **95 % credible-interval forest** of horizontal
> whiskers centred at each MAP, tick coloured teal if BH-FDR
> significant else light slate; (iii) **Manhattan-style
> `-log₁₀(p_adj)` lollipops** with a dashed "BH FDR 0.05" threshold,
> teal above / slate below; (iv) a **Mann-Whitney mini-inset** balloon
> anchored to the tallest lollipop with overlaid teal-native / coral-IVT
> per-read `p(mod)` densities and a `U, p` monospace callout; (v) a
> **per-read mod-BAM strip** of ~8 read tracks with teal "modified"
> dots and an `MM:Z / ML:B:C` tag annotation. A tiny `site_results.tsv`
> glyph in the top-right anchors the output filename.
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

> Flat vector *Nature Methods*-style scientific workflow, 2.25 : 1
> landscape, white background, **four panels marked by bold uppercase
> letter labels A–D (18-pt 700-weight) in each top-left corner,
> separated by thin 0.5-pt slate vertical rules**.
> **(A)** Native + IVT nanopores threading RNA with ionic-current
> traces (teal / coral). **(B, merged signals + DTW)** Three flowing
> sub-regions in one panel: per-position paired teal-native /
> coral-IVT ionic-current traces with a 7-pt grey *"via f5c eventalign"*
> attribution → a zoomed-in DTW warping motif (two traces mirrored,
> connected by ~10 thin slate correspondence curves — the hero motif) →
> one blue-ramp symmetric distance matrix labelled "reads × reads"
> with a small diagonal wavefront accent and a minimal GPU chip with
> a few stream lanes. **(C, centrepiece)** Cream panel with three
> sub-blocks: V1 coverage-controlled
> shrinkage funnel (position→local→global) ending in a teal
> `Beta(α₀,β₀)`; V2 pinned-coral null + learnable-teal alternative
> with a smooth sigmoid soft-gate + EM convergence inset +
> λ-shrinkage tether; V3 five-node 2-state HMM chain with
> genomic-gap-labelled variable arrows (Δ=1,3,7,2,12) and two arcing
> forward/backward passes. **(D)** Reference-anchored stack on one
> shared 5′→3′ axis — teal Beta-Binomial posterior ridgeline of
> per-position violins, 95 %-CI forest whiskers, Manhattan-style
> `-log₁₀(p_adj)` lollipops with dashed BH-FDR line, a Mann-Whitney
> balloon inset (teal-native / coral-IVT overlaid densities, `U, p`
> callout), and a per-read mod-BAM strip (`MM:Z / ML:B:C`). Palette
> teal #2F8F9D / coral #E7734A / warm red #B33A3A / slate / cream /
> white. Helvetica / Inter, 1 pt lines, no shadows, publication-ready,
> algorithmic motifs (DTW warping correspondence, shrinkage funnel,
> anchor pin, gap-aware chain, F/B arcs, reference-anchored posterior
> stack) preserved. Panel sizing C > D > B > A; C is the algorithmic
> centrepiece.

---

## Prompt C — Vertical layout (portrait, supplementary / slide)

> *Nature Methods*-style flat-vector workflow, 7 : 10 portrait, white
> background, **four panels stacked top-to-bottom in horizontal bands
> connected by downward chevrons, each panel bearing a bold uppercase
> letter label A–D in its top-left corner (18-pt 700-weight) inside
> an optional 24 × 24 px light-slate rounded square, separated by
> thin 0.5-pt slate horizontal rules**. Sizing C > D > B > A.
> **(A) Input** — paired teal-native / coral-IVT pores with
> ionic-current traces and a reference bar.
> **(B) Signal comparison via batched CUDA DTW** — three flowing
> sub-regions within one panel: per-position paired teal-native /
> coral-IVT ionic-current traces with a 7-pt grey *"via f5c eventalign"*
> attribution → a zoomed-in DTW warping motif (two mirrored traces
> connected by ~10 thin slate correspondence curves, the hero motif)
> → one compact blue-ramp symmetric distance matrix with a small
> diagonal wavefront accent and a minimal GPU chip with a few stream
> lanes; caption *"dynamic time warping · single kernel launch · 16
> CUDA streams"*.
> **(C) V1 · V2 · V3 in a cream panel (centrepiece)** — V1
> coverage-adaptive three-level shrinkage funnel (position → local →
> global) with a coverage dial; V2 pinned-coral null + learnable-teal
> alternative with a continuous sigmoid soft-gate, EM convergence
> inset, and a λ-shrinkage tether; V3 five-node 2-state HMM chain
> with genomic-gap-labelled variable-width transitions and arcing
> forward/backward passes.
> **(D) Reference-anchored output stack** — layers pinned to one
> shared horizontal `5′→3′` reference axis: teal Beta-Binomial
> posterior ridgeline (per-position violins), 95 %-CI forest
> whiskers, Manhattan `-log₁₀(p_adj)` lollipops with dashed BH-FDR
> line, a Mann-Whitney balloon (overlaid teal-native / coral-IVT
> `p(mod)` densities, `U, p` callout), and a per-read mod-BAM
> strip (MM:Z / ML:B:C tags).
> Palette teal / coral / warm red / slate / cream / white.
> Helvetica / Inter, 1 pt lines, no shadows, no gradients except the
> named single-hue ramps. Publication-ready.

---

## Prompt D — Ultra-compact one-sentence (prompt-budget-constrained)

> Clean *Nature Methods*-style flat-vector workflow, 2.25:1 on white,
> teal/coral/slate palette, **four panels each marked by a bold
> uppercase letter A–D (18-pt 700-weight) in the top-left corner,
> separated by thin slate vertical rules** (sizing C > D > B > A;
> C dominates as algorithmic centrepiece): (A) native+IVT nanopores
> with ionic-current traces; (B, merged signals + DTW) three flowing
> sub-regions in one panel — per-position paired teal-native /
> coral-IVT ionic-current traces with a tiny grey "via f5c eventalign"
> attribution, a zoomed-in DTW warping motif (two mirrored traces
> connected by ~10 thin slate correspondence curves — hero motif),
> and one compact blue-ramp symmetric distance matrix with a diagonal
> wavefront accent plus a minimal GPU chip with a few stream lanes;
> (C, centrepiece) cream panel with V1 coverage-adaptive three-level
> shrinkage funnel, V2 pinned-null + learnable-alt densities with a
> smooth sigmoid soft-gate and EM convergence inset, V3 five-node
> 2-state HMM chain with gap-labelled variable transitions and
> forward/backward arcs; (D) reference-anchored output stack on a
> shared 5′→3′ axis — teal Beta-Binomial posterior violins, 95 %-CI
> forest whiskers, Manhattan `-log₁₀(p_adj)` lollipops with dashed
> BH-FDR line, Mann-Whitney balloon (native/IVT `p(mod)` densities,
> `U,p` callout), and per-read mod-BAM strip. Helvetica, 1 pt lines,
> no shadows, no gradients except named ramps, algorithmic motifs
> preserved, publication-ready.

---

## Stage-by-stage annotation reference

If a tool lets you caption panels separately, use these **exact**
wordings — they match the paper in preparation and the CLI vocabulary.

| Panel | Title | Algorithmic subtitle |
|-------|-------|----------------------|
| **A** | Input | Paired native direct-RNA reads + in-vitro-transcribed control + reference transcriptome |
| **B** | Signal comparison via batched CUDA DTW | Per-position paired ionic-current traces (extracted via `f5c eventalign`) → DTW warping correspondence → blue-ramp symmetric distance matrix; entire contig in one kernel launch, 16 concurrent CUDA streams |
| **C(i)** | V1 · empirical-Bayes null | Coverage-adaptive three-level James-Stein shrinkage (position → local k-mer window → global); outputs `z_scores` used by semi-supervised training |
| **C(ii)** | V2 · anchored mixture EM | Null-frozen two-component mixture with continuous soft-gating (`σ(ΔBIC)`) and λ-regularised alternative prior; outputs `p_mod_raw` — alternative HMM emission source, also the training signal for semi-/supervised HMM |
| **C(ii-default)** | kNN IVT-purity + Beta EM | k-weighted IVT-purity score in DTW space, Beta-EM calibrated; outputs `p_mod_knn` — **the default HMM emission source** |
| **C(iii)** | V3 · gap-aware forward–backward | Per-read 2-state HMM whose transition probabilities depend on genomic gap between consecutive called sites; emissions = `p_mod_knn` (default) or `p_mod_raw` (training / override) |
| **D** | Per-site reference-anchored stack | Beta-Binomial posterior ridgeline + 95 % CI forest + Manhattan `-log₁₀(p_adj)` lollipops (BH-FDR) + Mann-Whitney native-vs-IVT inset + per-read mod-BAM strip, all pinned to one `5' → 3'` reference axis |
| **E** | Combinatorial phasing | Single-molecule mod-BAM output (`MM:Z` / `ML:B:C`) enables co-deposition / mutual-exclusion contrasts over arbitrary site sets |

---

## Editing tips — re-injecting algorithmic motifs when drafts look generic

| Draft problem | Prompt fix |
|---------------|------------|
| Panel B reads as "just another heatmap" | *"reorganise Panel B into three flowing sub-regions: (i) per-position paired teal-native / coral-IVT ionic-current traces with a 7-pt grey 'via f5c eventalign' attribution, (ii) a zoomed-in DTW warping callout — two mirrored traces connected by ~10 thin slate correspondence curves (the hero motif), (iii) one compact blue-ramp reads × reads distance matrix with a diagonal wavefront accent and a minimal GPU chip with a few stream lanes"* |
| DTW correspondence motif is missing | *"add two mirrored ionic-current traces (teal native above, coral IVT below) connected by ~10 thin slate curved lines showing sample-to-sample correspondence — this is the literal visual of dynamic time warping and should be the centre of mass of Panel B"* |
| DTW stage doesn't read as batched / parallel | *"add a small diagonal teal → white wavefront accent on one corner of the distance matrix and a minimal GPU-chip pictogram with four thin parallel stream lanes — accents only, not the focus"* |
| V1 looks like a single arrow | *"replace with a three-level funnel narrowing through 'position → local k-mer window → global', with a circular coverage dial whose needle controls funnel width"* |
| V2 looks like just two Gaussians | *"add a padlock or pin glyph above the null curve labelled 'anchored', a curved parameter-motion arrow on the alternative, a smooth S-shaped sigmoid gate along the baseline (NOT a step), a 9-pt monospace formula callout `γ = σ(ΔBIC)·π·f₁/[(1−π)f₀+π·f₁]`, an EM convergence inset (log-likelihood vs iteration plateauing), and a λ-shrinkage tether to a ghost global-prior curve"* |
| V3 looks like a vanilla HMM chain | *"vary the arrow lengths and stroke weights between adjacent nodes according to printed genomic-gap labels (Δ=1, 3, 7, 2, 12), split each node diagonally into teal 'mod' and light-slate 'unmod' halves, and add two arcing passes above the chain — coral 'α forward' left-to-right and teal 'β backward' right-to-left"* |
| Panel D reads as a generic table + volcano | *"replace with a reference-anchored vertically stacked genome-browser track on a shared 5'→3' axis: a Beta-Binomial posterior ridgeline (per-position teal violins, width = uncertainty), a 95 %-CI forest of horizontal whiskers, Manhattan-style -log₁₀(p_adj) lollipops with a dashed BH-FDR line, a Mann-Whitney inset balloon with overlaid teal-native / coral-IVT p(mod) densities, and a per-read mod-BAM strip underneath — every layer pinned to the same reference coordinates"* |
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
folded in as panel *E* of Prompt A), use Prompt E below.

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

- **Prompt A** — canonical manuscript figure 1, five panels, 2.75 : 1
  landscape, pipeline + combinatorial readout combined (former panels
  B and C merged into a single DTW-warping panel B).
- **Prompt A-compact** — four-panel pipeline-only variant when the
  combinatorial story lives in figure 2.
- **Prompt B** — rapid iteration for short-prompt-window models.
- **Prompt C** — portrait layout for poster / supplementary / slides.
- **Prompt D** — one-sentence ultra-compact for tight prompt budgets.
- **Prompt E / E-short** — standalone combinatorial figure showcasing
  the single-molecule advantage.

All prompts are engineered to expose Baleen's seven algorithmic
signatures (**DTW warping correspondence + wavefront-accent CUDA DTW**,
three-level coverage-adaptive shrinkage, null-anchored mixture EM with
soft gating, kNN IVT-purity + Beta-EM calibration as the default
emission source, gap-aware per-read HMM, single-molecule
combinatorics, streaming architecture). If a draft loses any of these
signals, consult the **Editing tips** table to re-inject the missing
motif with a single targeted sentence.
