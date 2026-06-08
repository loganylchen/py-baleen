# Changelog

Baleen is pre-1.0 (Development Status: Alpha). This page summarises notable
changes by theme; see the
[full commit history](https://github.com/loganylchen/py-baleen/commits/dev) for
detail.

## Unreleased (dev)

### Features

- **Read-ID intersection** — every stage is gated on
  `reads(BAM) ∩ reads(FASTQ) ∩ reads(BLOW5)` per condition, so `f5c` silently
  dropping reads absent from the signal file no longer biases depth statistics,
  the `--min-depth` filter, or subsampling. Disable with
  `--no-read-intersection`.
- **`--resume`** — interrupted runs reuse per-contig slices already on disk,
  guarded by a `.run_params.json` input/parameter fingerprint.
- **`--depth-mode`** — choose how `--min-depth` is interpreted; the default is
  now `read_count` (total mapped reads on the contig) rather than
  `mean_coverage` (**breaking** default change).

### Performance

- **Streaming per-contig flush** — DTW → HMM → aggregation are fused per contig
  and written to disk immediately, bounding peak memory regardless of
  transcriptome size.
- **cuDTW++ warp-shuffle kernel** replaces the previous wavefront DTW kernel.
- **Numba-JIT EM loops** in the HMM calibration path (`_calibrate_beta`,
  `_anchored_mixture_em`), roughly a 19× per-call speedup on calibration.
- **`emission_source` gating** — the default `p_mod_knn` path skips the V1/V2
  computation entirely.

### Fixes

- Chunked `merge_contig_bams` to survive thousands of per-contig slices.
- Closed path-traversal gaps in per-contig filename handling.
- Per-position buffer stride fix in the multi-position CUDA DTW kernel.

### Build

- GPU Docker build now **fails loudly** if the `_cuda_dtw` extension silently
  falls back to CPU.

### Reverted

- The Sakoe-Chiba band DTW constraint was reverted — the soft-band
  implementation added overhead without reducing thread/diagonal count and was
  measured slower.

## API note

`run_pipeline_streaming(...)` returns a **2-tuple** `(output_paths, metadata)`,
where `output_paths` is a dict with keys `site_tsv`, `read_bam`,
`per_contig_dir`, `n_total_sites`, and `n_significant`. (Earlier internal
revisions returned a 3-tuple; that shape is no longer used.)
