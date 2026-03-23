# GPU Utilization Optimization: Multi-Position Batch DTW

**Date**: 2026-03-21
**Status**: Approved
**Target hardware**: NVIDIA H100 (80GB, 132 SMs) and RTX 3090 (24GB, 82 SMs)

## Problem

The baleen pipeline processes DTW pairwise distances one genomic position at a time. Each position triggers a full `cudaMalloc → memcpy → kernel → memcpy → cudaFree` cycle. With typical workloads (100+ reads, 1000+ positions per contig), this results in:

- **<10% GPU utilization**: Each position generates only ~99 thread blocks (for 100 reads). H100 needs 1000+ blocks for full occupancy.
- **Thousands of unnecessary malloc/free cycles**: 1000 positions = 5000 `cudaMalloc` + 5000 `cudaFree` calls per contig.
- **No concurrency across positions**: The GPU finishes one position's computation and sits idle while Python prepares the next.
- **Repeated `cudaGetDeviceProperties` queries**: Called on every DTW invocation.

## Solution: CUDA Streams + Memory Pool

Batch all positions for a contig into a single GPU call using CUDA streams for concurrent execution.

### Architecture

```
BEFORE:
  for each position:
    cudaMalloc (5 buffers)
    cudaMemcpy H→D
    DTW kernel (99 blocks)
    cudaDeviceSynchronize
    cudaMemcpy D→H
    cudaFree (5 buffers)

AFTER:
  cudaMalloc (all buffers once)
  cudaMemcpy H→D (all data once)
  for each position p:
    stream = streams[p % num_streams]
    cudaMemsetAsync(cost_bufs, stream)
    DTW kernels on stream (99 blocks × wavefront chunks)
  cudaDeviceSynchronize (once)
  cudaMemcpy D→H (all results once)
  cudaFree (once)
```

### New C/CUDA Function

File: `baleen/_cuda_dtw/dtw_api.cpp`

```c
int opendba_dtw_multi_position_pairwise(
    const float *all_sequences,        // All positions' padded signals, concatenated
    const size_t *all_seq_lengths,     // All actual sequence lengths, concatenated
    const size_t *position_seq_counts, // Number of sequences per position
    size_t num_positions,
    size_t global_max_length,          // Max signal length across all positions
    int use_open_start,
    int use_open_end,
    float *out_distances,              // All upper-triangle distances, concatenated
    int num_cuda_streams               // Number of CUDA streams (default 16)
);
```

#### Memory Layout

All positions use `global_max_length` as stride for simplicity:

```
d_sequences: [pos0: n_0 × G] [pos1: n_1 × G] ... [posP: n_P × G]
  where G = global_max_length

d_seq_lengths: [pos0: n_0 lengths] [pos1: n_1 lengths] ... [posP: n_P lengths]

d_distances: [pos0: n_0*(n_0-1)/2] [pos1: n_1*(n_1-1)/2] ... [posP: n_P*(n_P-1)/2]
```

#### Per-Stream Cost Buffers

Each stream needs its own pair of cost buffers (ping-pong pattern):

```
d_cost_a[s]: max_n_per_stream × global_max_length × sizeof(float)
d_cost_b[s]: max_n_per_stream × global_max_length × sizeof(float)

where max_n_per_stream = max(position_seq_counts) - 1
```

**Critical: Cost buffer stride alignment.** The DTW kernel indexes cost buffers as `dtwCostSoFar[first_seq_length * blockIdx.x]`, where `first_seq_length` comes from `gpu_sequence_lengths[first_seq_index]`. Since different positions may have different actual signal lengths, the kernel uses the *actual* length as the stride, not `global_max_length`. This means:

- Cost buffers must be allocated with `global_max_length` stride (for worst case), but the kernel will index with the *actual* position's max length.
- This is safe because `actual_length <= global_max_length`, so the kernel never reads past the allocated buffer. Consecutive blocks access `[0, len)`, `[len, 2*len)`, etc. — no overlap since blocks are independent within a reference sequence.
- The wavefront loop bound `i < first_seq_length + blockDim.x` correctly uses the actual length, so the kernel doesn't over-process padded zeros.

Total per-stream cost memory for 100 reads × 100 max_length: ~40KB × 2 = 80KB.
For 16 streams: 1.3MB. For 32 streams: 2.6MB. Negligible on both H100 and 3090.

#### Execution Flow

```c
// 1. Cache device properties (once per session)
ensure_device_props();

// 2. Compute offsets
size_t seq_offsets[num_positions];    // Where each position's sequences start
size_t dist_offsets[num_positions];   // Where each position's distances start

// 3. Allocate all GPU memory
cudaMalloc(&d_sequences, total_sequences * global_max_length * sizeof(float));
cudaMalloc(&d_seq_lengths, total_sequences * sizeof(size_t));
cudaMalloc(&d_distances, total_pairs * sizeof(float));
for (int s = 0; s < num_streams; s++) {
    cudaMalloc(&d_cost_a[s], max_n * global_max_length * sizeof(float));
    cudaMalloc(&d_cost_b[s], max_n * global_max_length * sizeof(float));
}

// 4. Upload all data (single bulk transfer)
cudaMemcpy(d_sequences, all_sequences, ..., H2D);
cudaMemcpy(d_seq_lengths, all_seq_lengths, ..., H2D);

// 5. Create streams
cudaStream_t streams[num_streams];
for (int s = 0; s < num_streams; s++) cudaStreamCreate(&streams[s]);

// 6. Process positions across streams
for (size_t p = 0; p < num_positions; p++) {
    int s = p % num_streams;
    size_t n = position_seq_counts[p];
    size_t seq_offset = seq_offsets[p];

    // Skip positions with fewer than 2 sequences (no pairs to compute)
    if (n < 2) continue;

    // DTW for this position (reuse existing kernel)
    for (size_t i = 0; i < n - 1; i++) {
        size_t num_comparisons = n - i - 1;

        // CRITICAL: Reset cost buffers BEFORE EACH reference sequence iteration.
        // The kernel leaves non-zero values from the previous i's computation.
        // This matches the existing opendba_dtw_pairwise_batch behavior (line 248-250).
        cudaMemsetAsync(d_cost_a[s], 0, num_comparisons * global_max_length * sizeof(float), streams[s]);
        cudaMemsetAsync(d_cost_b[s], 0, num_comparisons * global_max_length * sizeof(float), streams[s]);

        float *cur = d_cost_a[s], *nxt = d_cost_b[s];

        for (size_t offset = 0; offset < global_max_length; offset += max_threads) {
            DTWDistance<float><<<num_comparisons, thread_block, shared_mem, streams[s]>>>(
                nullptr, global_max_length,
                nullptr, global_max_length,
                i, offset,
                &d_sequences[seq_offset * global_max_length],
                global_max_length, n,
                &d_seq_lengths[seq_offset],
                cur, nxt,
                nullptr, 0,  // no path matrix
                &d_distances[dist_offsets[p]],
                use_open_start, use_open_end
            );
            float *tmp = cur; cur = nxt; nxt = tmp;
        }
    }
}

// 7. Sync all streams, copy results back, free everything
cudaDeviceSynchronize();
cudaMemcpy(out_distances, d_distances, total_pairs * sizeof(float), D2H);
// ... cleanup ...
```

### Python Wrapper

File: `baleen/_cuda_dtw/__init__.py`

```python
def dtw_multi_position_pairwise(
    position_signals: list[list[np.ndarray]],
    use_open_start: bool = False,
    use_open_end: bool = False,
    use_cuda: Optional[bool] = None,
    num_streams: int = 16,
) -> list[np.ndarray]:
    """
    Batch-compute pairwise DTW distances for multiple positions in one GPU call.

    Parameters
    ----------
    position_signals : list of list of np.ndarray
        position_signals[p][r] is the 1D float32 signal for position p, read r.
    num_streams : int
        Number of CUDA streams for concurrent position processing.

    Returns
    -------
    list of np.ndarray
        Distance matrices, one per position. Each is (n_p, n_p) float64.
    """
```

Implementation:
1. Compute `global_max_length = max(len(s) for pos in position_signals for s in pos)`
2. Pad all signals to `global_max_length`, concatenate into `(total_seqs, global_max_length)` float32
3. Build `lengths` (int64) and `position_seq_counts` (int64) arrays
4. Call C function
5. Split flat upper-triangle result into per-position symmetric matrices
6. Handle edge case: positions with n < 2 sequences produce empty (0×0) or (1×1) distance matrices
7. CPU fallback: sequential loop over positions using existing `_dtw_pairwise_cpu`

### Pipeline Integration

File: `baleen/eventalign/_pipeline.py`

Replace the per-position loop in `_process_contig` with a two-phase approach:

```python
# Phase 1: Collect all signals (CPU)
all_position_data = []  # [(pos, kmer, native_names, ivt_names, signals)]
for pos in common_positions:
    native_names, native_signals = extract(native_by_pos, pos)
    ivt_names, ivt_signals = extract(ivt_by_pos, pos)
    if native_signals and ivt_signals:
        all_position_data.append((pos, kmer, native_names, ivt_names,
                                  native_signals + ivt_signals))

# Phase 2: Batch GPU DTW
all_signals_lists = [d[4] for d in all_position_data]
all_matrices = _cuda_dtw.dtw_multi_position_pairwise(
    all_signals_lists,
    use_open_start=use_open_start,
    use_open_end=use_open_end,
    use_cuda=use_cuda,
    num_streams=num_cuda_streams,
)

# Phase 3: Package results
for (pos, kmer, nat_names, ivt_names, _), matrix in zip(all_position_data, all_matrices):
    position_results[pos] = PositionResult(
        position=pos, reference_kmer=kmer,
        n_native_reads=len(nat_names), n_ivt_reads=len(ivt_names),
        native_read_names=nat_names, ivt_read_names=ivt_names,
        distance_matrix=matrix,
    )
```

### New Pipeline Parameter

```python
def run_pipeline(
    ...,
    num_cuda_streams: int = 16,  # NEW: number of CUDA streams for batch DTW
) -> ...:
```

### Device Properties Caching

```c
// Module-level cached properties
static int g_max_threads = 0;
static bool g_props_cached = false;

static int ensure_device_props() {
    if (!g_props_cached) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        g_max_threads = prop.maxThreadsPerBlock;
        g_props_cached = true;
    }
    return 0;
}
```

Reset in `opendba_dtw_cleanup()` by setting `g_props_cached = false`.

### Backward Compatibility

- Existing `dtw_pairwise_varlen` function unchanged (used by tests and standalone calls)
- New `dtw_multi_position_pairwise` is additive
- `run_pipeline` default behavior unchanged (just faster)
- CPU fallback path preserved for non-CUDA environments

### Expected Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| `cudaMalloc` calls / contig (1000 pos) | 5,000 | 5 + 2×num_streams |
| `cudaMemcpy` H→D / contig | 2,000 | 2 |
| Max concurrent GPU thread blocks | ~99 | ~1,584 (16 positions × 99) |
| GPU occupancy (H100, 132 SMs) | ~8% | ~75-100% |
| GPU occupancy (3090, 82 SMs) | ~12% | ~75-100% |
| `cudaGetDeviceProperties` calls | 1 per position | 1 per session |

### Testing Strategy

1. **Correctness**: Compare batch results against per-position results for known test data
2. **GPU utilization**: Use `nvidia-smi` or `nsys` to measure actual SM occupancy before/after
3. **Regression**: Existing `test_dtw.py` tests must pass unchanged
4. **New tests**: `test_multi_position_batch.py` with various position counts and signal sizes

### Files to Modify

1. `baleen/_cuda_dtw/dtw_api.cpp` — New `opendba_dtw_multi_position_pairwise` + Python binding
2. `baleen/_cuda_dtw/dtw_api.h` — New function declaration
3. `baleen/_cuda_dtw/__init__.py` — New `dtw_multi_position_pairwise` wrapper
4. `baleen/eventalign/_pipeline.py` — Restructured `_process_contig`, new `num_cuda_streams` param
5. `baleen/__init__.py` — Export new function if needed
6. `tests/test_dtw.py` — New batch correctness tests

### Risks and Mitigations

1. **GPU memory overflow on very large batches**: If total_sequences × global_max_length exceeds GPU memory, fall back to chunked processing (sub-batches of positions). Monitor with `cudaMemGetInfo`.
2. **Stream overhead for small workloads**: For <10 positions, stream creation overhead may exceed benefit. Add a threshold: if num_positions < min_batch_threshold (default 4), use the existing single-position path.
3. **Cost buffer contention**: Positions on the same stream share cost buffers sequentially. This is correct (CUDA stream semantics guarantee ordering) but limits parallelism to num_streams positions at a time. 16-32 streams is sufficient for both H100 and 3090.
