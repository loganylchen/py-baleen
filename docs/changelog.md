# Changelog

This page summarises notable changes by theme; see the
[full commit history](https://github.com/loganylchen/py-baleen/commits/main) for
detail.

## v1.0.1 — read-ID intersection fix

Patch release fixing a silent-empty-output bug in the default pipeline.

### Fixes

- **Read-ID intersection no longer trusts f5c's `.index.readdb`.** The FASTQ
  side of `reads(BAM) ∩ reads(FASTQ) ∩ reads(BLOW5)` now comes solely from the
  FASTQ headers. Previously a leftover f5c single-BLOW5 readdb (`*<TAB>blow5`,
  whose first column is a wildcard, not a read id) collapsed the FASTQ side to
  `{"*"}` (count = 1), silently emptying the intersection and producing **no
  output** under the documented default flags. The empty-intersection warning
  now prints each source's count and an example id so an id-format mismatch is
  diagnosable from the log alone.
- **GPU image now fails the build if `pyslow5` is missing.** The builder's
  `pip install … | tee` masked dependency-install failures, so the published
  v1.0.0 GPU image shipped without `pyslow5` and crashed the read-ID
  intersection at runtime. An explicit import check now fails the build loudly,
  matching the existing CUDA-extension check.

## v1.0.0 — first stable release

First public release. Baleen detects RNA modifications by comparing native vs
IVT nanopore signals with CUDA-accelerated DTW and a three-stage hierarchical
Bayesian / HMM pipeline. Event alignment is performed by **f5c** (the GPU
Docker image uses f5c's CUDA build so eventalign also runs on the GPU).

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
