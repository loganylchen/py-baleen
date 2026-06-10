# Performance & Scaling

DTW dominates the wall-clock time of a Baleen run; everything after it (HMM,
aggregation) is comparatively cheap. This page covers the DTW backend, memory
behaviour, and the knobs that control throughput.

## DTW backend

The `_cuda_dtw` module selects a backend **at import time**:

- **CUDA (GPU)** if the `_cuda_dtw` C extension compiled with CUDA support.
- **CPU (`tslearn`)** fallback otherwise.

Check which one is active:

```python
from baleen._cuda_dtw import backend, is_available
print("DTW backend:", backend())     # "cuda" or "cpu"
print("CUDA available:", is_available())
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
| `--threads N` | Parallel contig workers (`ProcessPoolExecutor`). More workers = more concurrency, but each f5c call then gets fewer CPU threads. |
| `--f5c-threads N` | CPU threads per `f5c eventalign` call. Default auto = `total_cores / threads`. |
| `--subsample-n N` | Caps reads per condition per contig (default 300). Fewer reads → fewer DTW pairs → faster, at some statistical cost. |
| `--no-subsample` | Disables the cap — slower, more memory, on deep data. |
| `--min-depth` / `--depth-mode` | Skip shallow contigs entirely. |
| `--target` | Restrict to specific contigs. |

!!! tip "Balancing `--threads` and `--f5c-threads`"
    `f5c` is itself multithreaded. If you set `--threads 16` on a 16-core
    machine, the auto rule gives each f5c call only 1 thread. For
    f5c-bound workloads, fewer pipeline workers with more f5c threads each can
    be faster — profile both.

## Resuming long runs

`--resume` reuses per-contig slices already under `<output_dir>/per_contig/`,
skipping their workers entirely. A `.run_params.json` fingerprint of the inputs
and run-affecting parameters guards correctness: if anything drifted, the resume
aborts and lists the mismatches rather than silently mixing incompatible
results. See the [CLI Reference](cli.md#baleen-run).
