# Roadmap note: multi-scale DTW + complementary level features

Status: **design note only — not implemented.** Captured from the v1.0.1
benchmark analysis (July 2026) for consideration in a future baleen version.
Supporting data & figures live in `/SSD/logan/projects/baleen-paper`
(`figures/demo/`, `data/demo/`, `scripts/demo_figure.py`).

## TL;DR

Baleen's per-position DTW comparison is a **shape/dwell** detector; it is
comparatively weak on modifications whose only signature is a **sharp local
current-level shift** (e.g. m6A, m2A, ho5C). A conventional per-read mean
current test (à la xpore) is the opposite. The two are **complementary**, and
the loss on level-shift sites is partly an artefact of the `--padding 1`
window. A future version should compute DTW at **multiple padding scales in one
pass** and let the hierarchical model (V1/V2/V3) fuse them (plus, optionally, an
explicit level feature) with *adaptive* weights rather than the current single
fixed scale.

## Evidence (E. coli 23S, 26 truth sites, depth 500, 100% native)

Per-site *separation* AUROC (native vs IVT), scored as median DTW distance to
IVT reads for DTW, |mean − IVT median| for the level test:

| signal | mean AUROC (26 sites) | notes |
|---|---|---|
| mean current (level) | 0.742 | conventional / xpore-like |
| DTW `padding=1` (default; shape + context) | 0.754 | current baleen |
| DTW `padding=0` (center k-mer only) | 0.716 | |
| fuse pad0+pad1 (z-sum) | 0.759 | DTW-only; lifts the floor |
| fuse pad0+pad1+level (z-sum) | 0.776 | best fixed rule |
| oracle (best modality per site) | 0.802 | ceiling for adaptive fusion |

Complementarity is site-class-dependent:

- **DTW wins (13/26)** — shape/dwell/context modifications: m2G, Cm, m5U, most
  Psi. `figures/demo/dtw_vs_meancurrent_scatter.png` (points above the diagonal).
  Extreme: m2G `ecoli23S:1835`, DTW-distance AUROC 0.865 while mean current is
  0.51 (native 93.0 vs IVT 92.7 pA — indistinguishable).
- **mean wins (8/26)** — sharp level-shift modifications: m6A, m2A, ho5C, Gm.
  Extreme: m6A `ecoli23S:2030`, mean 0.85 vs DTW `pad1` 0.56.

## Mechanism: the `padding=1` window dilutes local level shifts

Baleen does **not** normalise the signal before DTW (raw pA `--scale-events`
samples; only resampling to a fixed length for the CUDA kernel), so amplitude
*is* in the DTW distance in principle. The weakness on level-shift sites comes
from the **window**: with `--padding 1` the DTW signal spans the center k-mer ±
1 flank, so a single-position current step is averaged with ~2 unmodified flanks
(~1/3 dilution).

Verified by re-running with `--padding 0` (`demo_dtw_p0/`):

- Level-shift sites improve: m6A 2030 `pad1 0.556 → pad0 0.678` (+0.12), D 2449
  +0.11, m5C 1962 +0.09; mean Δ over level-dominant sites **+0.04**.
- Shape sites regress (they need the context): m1G 745 `0.960 → 0.829` (−0.13),
  m3Psi 1915 −0.10; mean Δ over shape-dominant sites **−0.09**.

So `padding` is a knob trading local-level sensitivity against shape/context
sensitivity. Even de-diluted, DTW at m6A (0.68) < mean (0.85): a holistic
shape-distance is inherently weaker than a targeted level test for a *pure*
level step — hence the level feature is genuinely additive.

## Proposal for the next version

1. **Multi-scale DTW in a single pass.** Event alignment (f5c) + per-position
   signal grouping is the expensive, shared step and should run **once**. Then
   compute the pairwise DTW distance matrix at **two windowings** from the same
   grouped signals — `pad0` (center k-mer only) and `pad1` (center ± flank);
   `pad0`'s signal is a subset of `pad1`'s. Cost ≈ **one extra DTW pass** (DTW on
   the GPU is cheap relative to eventalign), *not* a second full pipeline run.
   This is what the July-2026 analysis emulated with two separate `--padding`
   runs — a redundant hack that a real implementation collapses into one pass.

2. **Fuse in the hierarchical model, not by naive sum.** A fixed equal-weight
   z-sum already beats any single scale on average (0.759 DTW-only, 0.776 with
   level) but stays below the oracle (0.802) because it **dilutes sites where one
   scale dominates** (Cm 2498 `pad1 0.955 → sum 0.914`; m7G 2069 0.723 → 0.583).
   Feed the pad0 and pad1 distance-derived features (and optionally the explicit
   level feature) as **separate channels** into V1/V2/V3 so the model can weight
   them per k-mer / modification context, approaching the oracle.

3. **Optional explicit level feature.** For pure level-shift modifications,
   pad0+pad1 fusion alone recovers only part of the gap (m6A 2030 → ~0.65 vs
   0.73 with the level feature). A cheap per-read `|mean_current − IVT_median|`
   channel closes it. Keep it optional/toggle so the model stays model-free by
   default.

## Where this touches the code (for whoever implements it)

- Signal windowing / DTW dispatch: `baleen/eventalign/_pipeline.py`
  (`_process_contig*`, the per-position signal extraction feeding
  `_cuda_dtw.dtw_*`). Produce two distance matrices per `PositionResult` instead
  of one.
- `PositionResult` (`_pipeline.py`): carry `distance_matrix_pad0` and
  `distance_matrix_pad1` (or a list of scales).
- Scoring: `baleen/eventalign/_hierarchical.py` (V1 null / V2 mixture / V3 HMM)
  consumes the per-scale features; add the level channel from the already-parsed
  `event_level_mean` in `_signal.py`.
- CLI: expose `--dtw-scales` (default `1`, opt-in `0,1`) rather than a single
  `--padding`.

## Reproduce the analysis

All numbers/figures: `/SSD/logan/projects/baleen-paper`
(`scripts/demo_figure.py` for the illustrations; the padding and fusion tables
were produced from the `--keep-intermediate` DTW distance matrices of
`ecoliRNA500_10` native_0/control_0 at `--padding 1` and `--padding 0`).
