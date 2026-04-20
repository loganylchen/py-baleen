# Baleen Workflow Figure — AI Image-Gen Prompts

Curated prompt templates for generating a **Nature-journal quality workflow
figure** of the Baleen pipeline. Paste one of the blocks below directly into
Nano Banana Pro / Gemini 3 Pro / Flux / DALL·E / Imagen.

---

## Design anchors

Before the prompts, the style that defines what "Nature-quality" looks like
in this document:

- **Palette**: two accent colors — *teal* (`#2F8F9D`) for **native** RNA,
  *coral* (`#E7734A`) for **IVT** control. Neutral slate `#2F3E46` for
  skeletons and axes. Background pure white `#FFFFFF`, with one optional
  cream panel `#F5F1EA` for depth. No rainbow gradients, no drop shadows.
- **Typography**: clean sans-serif (Helvetica Neue / Inter / Source Sans),
  black `#111`, weights 400/600, title 600 only. Axis labels in lowercase.
- **Line weight**: 1.0–1.5 pt for skeletons, 0.75 pt for axis ticks.
- **Iconography**: flat vector, thin-stroke, 2-color fills, geometric.
  Pictograms over photoreal. No skeuomorphic textures.
- **Layout**: single panel, left-to-right flow, five stages connected by
  minimal arrows (`▸` or thin chevrons). Panel letters `a`–`e` bottom-left
  of each stage in 600-weight.
- **White space**: generous, ~15 % gutter between stages. No crowding.
- **Resolution**: target `3000×1200 px` (2.5 : 1) for landscape, or
  `1400×2000 px` (7 : 10) for vertical.

---

## Prompt A — Primary landscape workflow (recommended)

Copy this for a single horizontal figure suitable for a Nature Methods /
Nature Biotechnology main-text figure 1.

> **Prompt:**
> A publication-quality scientific workflow figure for a computational
> biology pipeline, rendered as a single horizontal panel at a 2.5:1 aspect
> ratio, in a flat vector illustration style reminiscent of *Nature Methods*
> figure 1. Pure white background. Five stages arranged left-to-right,
> connected by thin slate-gray chevron arrows, each stage labelled with a
> lowercase panel letter (a, b, c, d, e) in 600-weight sans-serif at
> bottom-left.
>
> **Stage a — Input.** Two stacked schematic nanopore pores, each with a
> single-stranded RNA molecule threading through. Upper pore is labelled
> "native RNA" in teal (#2F8F9D); lower pore is labelled "IVT control" in
> coral (#E7734A). Beside each, a short ionic-current trace (noisy
> horizontal signal waveform) in the matching color. Below: a thin gray
> horizontal bar representing the reference transcriptome, with tick marks.
>
> **Stage b — Event alignment.** A rectangular matrix of small signal
> segments, each segment a miniature waveform glyph aligned to a k-mer
> (A/C/G/U letters). Two such matrices side-by-side, one teal-tinted, one
> coral-tinted, with a tiny label "f5c eventalign" in uppercase 500-weight
> on top.
>
> **Stage c — Pairwise DTW distance matrices.** A small stack of three
> square heatmaps (overlapping, isometric), each heatmap a symmetric
> distance matrix with diagonal-zero pattern, using a single-hue blue
> ramp (#E8F0F4 → #1F4E5F). Below the stack, a small GPU chip icon with
> parallel streaming arrows, labelled "CUDA DTW, one block per pair" in
> 500-weight italic.
>
> **Stage d — Hierarchical modification calling.** Three vertically stacked
> sub-blocks inside a rounded cream (#F5F1EA) panel:
>   (i) a thin horizontal track showing three brackets at increasing scale
>       — "position → local window → global" — with a downward shrinkage
>       arrow, labelled "V1 · empirical-Bayes null";
>   (ii) two overlaid bell curves, coral and teal, with a thin vertical
>       soft-gate sigmoid between them, labelled
>       "V2 · anchored mixture EM";
>   (iii) a small graphical model: five circular hidden-state nodes in a
>       row connected by thin horizontal arrows, with short emission arrows
>       pointing downward to observed probabilities; variable-width gaps
>       between nodes to hint "gap-aware". Labelled
>       "V3 · gap-aware HMM forward–backward".
>
> **Stage e — Output.** Two outputs side-by-side:
>   (i) a miniature TSV table with header row `contig | pos | mod_ratio |
>       padj`, three body rows, and a tiny volcano-plot inset showing
>       dots crossing a dashed significance threshold, labelled
>       "site_results.tsv";
>   (ii) a short BAM-like read strip with per-base colored circles (pale
>       gray for unmodified, teal for modified), labelled
>       "read_results.bam  (MM / ML tags)".
>
> Typography throughout: Helvetica Neue or Inter. Body 10 pt equivalent,
> stage titles 12 pt 600-weight. No drop shadows, no gradients except the
> single-hue blue heatmaps. 1.0–1.5 pt line weights. Panel letters a–e in
> 600-weight. Overall aesthetic: clean, minimal, publication-ready,
> conveying scientific precision.

---

## Prompt B — Short version (for quick iteration)

Use when a model has a shorter prompt window (Nano Banana quick drafts,
Flux-schnell).

> Flat vector scientific workflow figure, 2.5:1 landscape, pure white
> background, Nature Methods style. Five stages left-to-right with
> chevron arrows and lowercase panel letters a–e.
> **(a)** Two nanopore pores threading RNA — upper teal "native", lower
> coral "IVT" — each with a noisy ionic-current trace.
> **(b)** Two small matrices of waveform-over-k-mer glyphs labelled
> "f5c eventalign".
> **(c)** Stack of three blue-ramp symmetric heatmaps + GPU chip glyph,
> labelled "CUDA pairwise DTW".
> **(d)** Rounded cream panel with three stacked sub-blocks: a hierarchical
> shrinkage bracket (V1 empirical-Bayes), two overlaid coral/teal Gaussians
> with a soft-gate sigmoid (V2 anchored mixture EM), and a 5-node HMM
> chain with emissions and variable gaps (V3 gap-aware forward–backward).
> **(e)** A mini TSV table with a tiny volcano-plot inset
> ("site_results.tsv"), plus a colored per-base BAM strip
> ("read_results.bam, MM/ML tags").
> Palette: teal #2F8F9D, coral #E7734A, slate #2F3E46, cream #F5F1EA,
> white background. Helvetica/Inter typography. No drop shadows, no
> gradients. 1 pt line weight. Publication-ready, clean, minimal.

---

## Prompt C — Vertical layout (portrait, for supplementary figures)

Use for a 7:10 portrait figure or slide insert.

> Flat vector scientific workflow, 7:10 portrait, pure white background,
> *Nature Methods* style. Five stages stacked top-to-bottom, each in a
> horizontal band ~200 px tall, separated by thin downward chevron arrows.
> Lowercase panel letters a–e bottom-left of each band.
> **(a) Input** — paired pore-through-RNA icons (teal native / coral IVT)
> with matched ionic-current traces and a reference bar underneath.
> **(b) Event alignment** — two eventalign matrices (waveform over k-mer),
> teal- and coral-tinted, labelled "f5c".
> **(c) DTW** — three overlapping symmetric heatmaps plus GPU-chip glyph,
> "CUDA pairwise DTW, one block per pair".
> **(d) Three-stage hierarchical calling** — cream rounded panel
> containing: shrinkage brackets (position/local/global) for V1; overlaid
> coral/teal bell curves with a soft-gate sigmoid for V2; five-node HMM
> chain with emission arrows and variable-width gaps for V3.
> **(e) Output** — mini TSV with volcano-plot inset
> ("site_results.tsv") beside a colored per-base BAM read strip
> ("read_results.bam"). Palette: teal #2F8F9D, coral #E7734A, slate,
> cream, white. Helvetica/Inter, 1 pt lines, no shadows, no gradients.

---

## Prompt D — One-sentence ultra-compact (for models with very tight prompt budgets)

> Clean Nature-Methods-style flat-vector workflow diagram, 2.5:1
> landscape on white, teal/coral/slate palette, five panels a–e:
> (a) native+IVT nanopore pores with ionic-current traces; (b) f5c
> eventalign matrices; (c) stack of blue-ramp DTW distance heatmaps
> with a GPU chip; (d) cream panel containing three stacked blocks —
> V1 hierarchical shrinkage brackets, V2 overlaid Gaussians with
> soft-gate sigmoid, V3 five-node HMM chain with emissions and gaps;
> (e) site_results TSV with volcano-plot inset plus a per-base mod-BAM
> strip. Helvetica, 1 pt lines, no shadows, publication-ready.

---

## Stage-by-stage annotation reference

If a tool lets you caption panels separately, use these exact wordings —
they match the paper in preparation and the CLI vocabulary.

| Panel | Title | Subtitle |
|-------|-------|----------|
| **a** | Input | Native direct-RNA reads + IVT control + reference transcriptome |
| **b** | Event alignment | Per-read ionic-current signals grouped by position (`f5c eventalign`) |
| **c** | Pairwise DTW | CUDA-accelerated distance matrices, one block per read pair |
| **d** | Hierarchical modification calling | V1 empirical-Bayes null · V2 anchored mixture EM · V3 gap-aware HMM |
| **e** | Output | Per-site TSV with BH-adjusted p-values + standard mod-BAM (MM/ML tags) |

---

## Editing tips when the first draft isn't right

| Problem | Fix in prompt |
|---------|---------------|
| Colors too saturated | Add "desaturated, editorial palette" |
| Looks like a slide, not a paper figure | Add "single-column figure, *Nature Methods* aesthetic, no background shapes, no drop shadows" |
| Cartoonish icons | Replace "icons" with "geometric pictograms, thin stroke, flat fills" |
| Typography wrong | Add "all type in Helvetica Neue 400/600, black #111, lowercase axis labels" |
| Too busy | Remove one sub-panel or increase gutter: "generous 15 % white-space gutters between stages" |
| Arrows ugly | Replace "arrows" with "thin slate-gray chevron connectors, 1 pt" |
| HMM chain wrong | Specify: "five circular hidden-state nodes in a horizontal row, thin arrows between adjacent nodes, downward emission arrows to small observed-probability circles, variable-width horizontal spacing between nodes to denote gap-aware transitions" |

---

## Recommended negative prompts (for Stable Diffusion / Flux)

```
photorealism, 3D render, drop shadows, rainbow gradient, neon glow,
cartoon mascots, stock-photo scientists, textured paper, grunge,
hand-drawn sketch, watermark, JPEG artifacts, cluttered composition
```

---

## Reuse

Copy any block above verbatim. Treat **Prompt A** as canonical for the
manuscript figure 1; **Prompt B** for rapid iteration; **Prompt C** for
poster/supplementary; **Prompt D** for prompt-budget-constrained tools.
