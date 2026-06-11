# Performance & Scaling

DTW dominates the wall-clock time of a Baleen run; everything after it (HMM,
aggregation) is comparatively cheap. This page covers the DTW backend, memory
behaviour, and the knobs that control throughput.

## DTW backend

The `baleen._dtw` module is a thin shim over the
[krill](https://loganylchen.github.io/krill-dist/) engine. The backend is
selected by **which krill wheel is installed** plus device presence — not by a
compile-time flag:

- **GPU** when krill's `cu122` wheel is installed and a CUDA device is present.
- **CPU** otherwise (krill's plain wheel, or no device).

Check which one is active:

```python
from baleen._dtw import backend, is_available
print("DTW backend:", backend())     # "gpu" or "cpu"
print("GPU available:", is_available())
```

Force a backend per run:

| Flag | Effect |
|------|--------|
| `--cuda 0` / `--cuda 0,1` / `--cuda all` | Use the listed GPU device(s). |
| `--no-cuda` | Force the CPU backend. |
| `--gpu-memory-limit BYTES` | Cap the GPU memory budget for concurrent DTW workers. |

### CUDA kernel characteristics

- **FP32 only.** The kernel is templated on `float`; FP16 is deliberately not
  used because it cripples Pascal consumer GPUs (1/64 FP32 throughput).
- **Wavefront parallelism** — one thread per row of the cost matrix sweeping the
  anti-diagonal, `blockDim.x = 1024`, three rolling diagonals in shared memory.
- **One block per pair** in pairwise mode; the grid spans all comparisons.
- **No Sakoe-Chiba band.** A soft-band variant was tried and reverted: marking
  out-of-band cells as infinite without shrinking the thread count or diagonal
  count is pure overhead on a latency-bound kernel.

## Streaming architecture & memory

DTW → HMM → aggregation are **fused per contig**: each worker computes a
contig, writes its `site_results` rows and mod-BAM slice to `per_contig/`, then
frees the in-memory result before returning a lightweight summary. The main
process merges the slices at the end.

The practical consequence: **peak memory stays bounded regardless of
transcriptome size** — you do not accumulate every contig's per-position
read-name lists in RAM. This is what lets Baleen process thousands of contigs
without OOM.

## Throughput knobs

| Flag | Effect on performance |
|------|-----------------------|
| `--threads N` | Parallel contig workers (`ProcessPoolExecutor`). More workers = more concurrency. |
| `--subsample-n N` | Caps reads per condition per contig (default 300). Fewer reads → fewer DTW pairs → faster, at some statistical cost. |
| `--no-subsample` | Disables the cap — slower, more memory, on deep data. |
| `--min-depth` / `--depth-mode` | Skip shallow contigs entirely. |
| `--target` | Restrict to specific contigs. |

!!! tip "`--threads` controls contig parallelism"
    krill eventalign runs in-process, not as a separate multithreaded
    subprocess, so there is no per-call thread budget to balance. `--threads`
    simply sets how many contig workers run in parallel.

## Resuming long runs

`--resume` reuses per-contig slices already under `<output_dir>/per_contig/`,
skipping their workers entirely. A `.run_params.json` fingerprint of the inputs
and run-affecting parameters guards correctness: if anything drifted, the resume
aborts and lists the mismatches rather than silently mixing incompatible
results. See the [CLI Reference](cli.md#baleen-run).
