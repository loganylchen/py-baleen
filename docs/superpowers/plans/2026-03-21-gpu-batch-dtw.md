# GPU Batch DTW Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batch all per-position DTW computations into a single GPU call using CUDA streams to achieve 75-100% GPU utilization on H100 and 3090.

**Architecture:** New C/CUDA function `opendba_dtw_multi_position_pairwise` processes all positions for a contig in one call. Positions are assigned round-robin to CUDA streams for concurrent execution. GPU memory is allocated once and reused. The existing `DTWDistance` kernel is reused unchanged.

**Tech Stack:** CUDA C++, Python C API, NumPy

**Spec:** `docs/superpowers/specs/2026-03-21-gpu-utilization-optimization-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `baleen/_cuda_dtw/dtw_api.h` | Modify | Add `opendba_dtw_multi_position_pairwise` declaration |
| `baleen/_cuda_dtw/dtw_api.cpp` | Modify | Add device props cache, new C function, new Python binding |
| `baleen/_cuda_dtw/__init__.py` | Modify | Add `dtw_multi_position_pairwise` Python wrapper |
| `baleen/eventalign/_pipeline.py` | Modify | Restructure `_process_contig` to batch DTW, add `num_cuda_streams` param |
| `tests/test_dtw.py` | Modify | Add multi-position batch correctness tests |

---

### Task 1: Device Properties Caching in C

**Files:**
- Modify: `baleen/_cuda_dtw/dtw_api.cpp:1-21` (top of file, after includes)

- [ ] **Step 1: Add cached device properties near the top of dtw_api.cpp**

Insert after the `CUDA_CHECK` macro definition (line 20), before `opendba_dtw_cuda`:

```cpp
// ============================================================================
// Cached Device Properties
// ============================================================================

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

- [ ] **Step 2: Update `opendba_dtw_cleanup` to reset cache**

Change the existing `opendba_dtw_cleanup` function (currently at line 130-133):

```cpp
void opendba_dtw_cleanup()
{
    g_props_cached = false;
    g_max_threads = 0;
    cudaDeviceReset();
}
```

- [ ] **Step 3: Replace `cudaGetDeviceProperties` calls in existing functions with cached version**

In `opendba_dtw_cuda` (around line 65-67), replace:
```cpp
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int max_threads = deviceProp.maxThreadsPerBlock;
```
with:
```cpp
    ensure_device_props();
    int max_threads = g_max_threads;
```

In `opendba_dtw_pairwise_batch` (around line 178-180), make the same replacement.

In `opendba_dtw_pairwise_varlen` (around line 390-392), make the same replacement.

- [ ] **Step 4: Rebuild and verify existing tests still pass**

Run: `pip install -e . && pytest tests/test_dtw.py -v`
Expected: All existing tests PASS. No behavioral change.

- [ ] **Step 5: Commit**

```bash
git add baleen/_cuda_dtw/dtw_api.cpp
git commit -m "refactor: cache cudaGetDeviceProperties across DTW calls"
```

---

### Task 2: New C Function — `opendba_dtw_multi_position_pairwise`

**Files:**
- Modify: `baleen/_cuda_dtw/dtw_api.h`
- Modify: `baleen/_cuda_dtw/dtw_api.cpp` (insert before Python C API section, around line 470)

- [ ] **Step 1: Add declaration to dtw_api.h**

Add before the closing `#ifdef __cplusplus` / `#endif`:

```c
    /**
     * @brief Compute pairwise DTW for multiple positions in one batched GPU call.
     *
     * All positions' padded signals are concatenated into a single flat array.
     * CUDA streams enable concurrent processing of different positions.
     *
     * @param all_sequences        Concatenated padded signals: sum(n_i) * global_max_length floats
     * @param all_seq_lengths      Actual lengths for all sequences: sum(n_i) values
     * @param position_seq_counts  Number of sequences per position: num_positions values
     * @param num_positions        Number of genomic positions
     * @param global_max_length    Max signal length across all positions (padding width)
     * @param use_open_start       Open start boundary
     * @param use_open_end         Open end boundary
     * @param out_distances        Output: concatenated full distance matrices (sum(n_i^2) floats)
     * @param num_cuda_streams     Number of CUDA streams for concurrency
     * @return 0=success, non-zero=error
     */
    int opendba_dtw_multi_position_pairwise(
        const float *all_sequences,
        const size_t *all_seq_lengths,
        const size_t *position_seq_counts,
        size_t num_positions,
        size_t global_max_length,
        int use_open_start,
        int use_open_end,
        float *out_distances,
        int num_cuda_streams);
```

- [ ] **Step 2: Implement the C function in dtw_api.cpp**

Insert before the `// Python C API Bindings` section:

```cpp
// ============================================================================
// Multi-Position Batched Pairwise DTW (CUDA Streams)
// ============================================================================

int opendba_dtw_multi_position_pairwise(
    const float *all_sequences,
    const size_t *all_seq_lengths,
    const size_t *position_seq_counts,
    size_t num_positions,
    size_t global_max_length,
    int use_open_start,
    int use_open_end,
    float *out_distances,
    int num_cuda_streams)
{
    if (!all_sequences || !all_seq_lengths || !position_seq_counts ||
        !out_distances || num_positions == 0 || global_max_length == 0)
    {
        fprintf(stderr, "Invalid input parameters for multi-position batch DTW\n");
        return -1;
    }

    ensure_device_props();
    int max_threads = g_max_threads;

    // Compute offsets and sizes
    size_t total_sequences = 0;
    size_t total_out_floats = 0;
    size_t max_n = 0;
    for (size_t p = 0; p < num_positions; p++)
    {
        size_t n = position_seq_counts[p];
        total_sequences += n;
        total_out_floats += n * n;  // full matrix per position
        if (n > max_n) max_n = n;
    }

    // Compute per-position offsets
    size_t *seq_offsets = new size_t[num_positions];
    size_t *dist_offsets = new size_t[num_positions];  // into upper-triangle buffer
    size_t *out_offsets = new size_t[num_positions];   // into output full-matrix buffer
    {
        size_t seq_off = 0, dist_off = 0, out_off = 0;
        for (size_t p = 0; p < num_positions; p++)
        {
            seq_offsets[p] = seq_off;
            dist_offsets[p] = dist_off;
            out_offsets[p] = out_off;
            size_t n = position_seq_counts[p];
            seq_off += n;
            dist_off += (n * (n - 1)) / 2;
            out_off += n * n;
        }
    }

    size_t total_pairs = 0;
    for (size_t p = 0; p < num_positions; p++)
    {
        size_t n = position_seq_counts[p];
        total_pairs += (n * (n - 1)) / 2;
    }

    // Allocate GPU memory — all at once
    float *d_sequences = nullptr;
    size_t *d_seq_lengths = nullptr;
    float *d_distances = nullptr;

    CUDA_CHECK(cudaMalloc(&d_sequences, total_sequences * global_max_length * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_seq_lengths, total_sequences * sizeof(size_t)));
    if (total_pairs > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_distances, total_pairs * sizeof(float)));
    }

    // Upload all data in one transfer
    CUDA_CHECK(cudaMemcpy(d_sequences, all_sequences,
                          total_sequences * global_max_length * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lengths, all_seq_lengths,
                          total_sequences * sizeof(size_t),
                          cudaMemcpyHostToDevice));

    // Create streams
    if (num_cuda_streams < 1) num_cuda_streams = 1;
    if (num_cuda_streams > 256) num_cuda_streams = 256;

    cudaStream_t *streams = new cudaStream_t[num_cuda_streams];
    for (int s = 0; s < num_cuda_streams; s++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    // Allocate per-stream cost buffers
    size_t max_comparisons = (max_n > 0) ? max_n - 1 : 0;
    size_t cost_buf_size = max_comparisons * global_max_length * sizeof(float);

    float **d_cost_a = new float*[num_cuda_streams];
    float **d_cost_b = new float*[num_cuda_streams];
    for (int s = 0; s < num_cuda_streams; s++)
    {
        d_cost_a[s] = nullptr;
        d_cost_b[s] = nullptr;
        if (cost_buf_size > 0)
        {
            CUDA_CHECK(cudaMalloc(&d_cost_a[s], cost_buf_size));
            CUDA_CHECK(cudaMalloc(&d_cost_b[s], cost_buf_size));
        }
    }

    // Kernel launch config
    dim3 thread_block(max_threads, 1, 1);
    size_t shared_mem = thread_block.x * 3 * sizeof(float);

    // Process all positions across streams
    for (size_t p = 0; p < num_positions; p++)
    {
        size_t n = position_seq_counts[p];
        if (n < 2) continue;

        int s = p % num_cuda_streams;
        size_t seq_offset = seq_offsets[p];

        for (size_t i = 0; i < n - 1; i++)
        {
            size_t num_comparisons = n - i - 1;

            // Reset cost buffers before each reference sequence
            cudaMemsetAsync(d_cost_a[s], 0,
                            num_comparisons * global_max_length * sizeof(float),
                            streams[s]);
            cudaMemsetAsync(d_cost_b[s], 0,
                            num_comparisons * global_max_length * sizeof(float),
                            streams[s]);

            float *d_current_cost = d_cost_a[s];
            float *d_next_cost = d_cost_b[s];

            for (size_t offset = 0; offset < global_max_length; offset += max_threads)
            {
                DTWDistance<float><<<num_comparisons, thread_block, shared_mem, streams[s]>>>(
                    nullptr, global_max_length,
                    nullptr, global_max_length,
                    i, offset,
                    &d_sequences[seq_offset * global_max_length],
                    global_max_length, n,
                    &d_seq_lengths[seq_offset],
                    d_current_cost,
                    d_next_cost,
                    (unsigned char *)nullptr, 0,  // no path matrix
                    &d_distances[dist_offsets[p]],
                    use_open_start,
                    use_open_end);
                CUDA_CHECK(cudaGetLastError());

                float *tmp = d_current_cost;
                d_current_cost = d_next_cost;
                d_next_cost = tmp;
            }
        }
    }

    // Sync all streams
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back and convert upper-triangle to full matrices
    if (total_pairs > 0)
    {
        float *h_upper_triangle = new float[total_pairs];
        CUDA_CHECK(cudaMemcpy(h_upper_triangle, d_distances,
                              total_pairs * sizeof(float),
                              cudaMemcpyDeviceToHost));

        for (size_t p = 0; p < num_positions; p++)
        {
            size_t n = position_seq_counts[p];
            size_t out_off = out_offsets[p];
            size_t pair_off = dist_offsets[p];

            // Fill full symmetric matrix
            size_t pair_idx = 0;
            for (size_t i = 0; i < n; i++)
            {
                out_distances[out_off + i * n + i] = 0.0f;  // diagonal
                for (size_t j = i + 1; j < n; j++)
                {
                    float dist = h_upper_triangle[pair_off + pair_idx];
                    out_distances[out_off + i * n + j] = dist;
                    out_distances[out_off + j * n + i] = dist;
                    pair_idx++;
                }
            }
        }

        delete[] h_upper_triangle;
    }
    else
    {
        // Handle positions with n <= 1 (fill diagonal zeros)
        for (size_t p = 0; p < num_positions; p++)
        {
            size_t n = position_seq_counts[p];
            size_t out_off = out_offsets[p];
            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = 0; j < n; j++)
                {
                    out_distances[out_off + i * n + j] = (i == j) ? 0.0f : 0.0f;
                }
            }
        }
    }

    // Cleanup
    for (int s = 0; s < num_cuda_streams; s++)
    {
        if (d_cost_a[s]) cudaFree(d_cost_a[s]);
        if (d_cost_b[s]) cudaFree(d_cost_b[s]);
        cudaStreamDestroy(streams[s]);
    }
    delete[] d_cost_a;
    delete[] d_cost_b;
    delete[] streams;

    cudaFree(d_sequences);
    cudaFree(d_seq_lengths);
    if (d_distances) cudaFree(d_distances);

    delete[] seq_offsets;
    delete[] dist_offsets;
    delete[] out_offsets;

    return 0;
}
```

- [ ] **Step 3: Verify compilation**

Run: `pip install -e .`
Expected: Build succeeds with no errors.

- [ ] **Step 4: Commit**

```bash
git add baleen/_cuda_dtw/dtw_api.h baleen/_cuda_dtw/dtw_api.cpp
git commit -m "feat: add opendba_dtw_multi_position_pairwise C function"
```

---

### Task 3: Python C API Binding for Multi-Position Function

**Files:**
- Modify: `baleen/_cuda_dtw/dtw_api.cpp` (Python C API section, after `py_dtw_pairwise_varlen`)

- [ ] **Step 1: Add the Python binding function**

Insert after the `py_dtw_pairwise_varlen` function (around line 730), before `py_dtw_cleanup`:

```cpp
static PyObject *py_dtw_multi_position_pairwise(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *sequences_array;
    PyArrayObject *lengths_array;
    PyArrayObject *counts_array;
    int use_open_start = 0;
    int use_open_end = 0;
    int num_cuda_streams = 16;

    static char *kwlist[] = {
        (char *)"sequences", (char *)"lengths", (char *)"counts",
        (char *)"use_open_start", (char *)"use_open_end",
        (char *)"num_cuda_streams", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!|iii", kwlist,
                                     &PyArray_Type, &sequences_array,
                                     &PyArray_Type, &lengths_array,
                                     &PyArray_Type, &counts_array,
                                     &use_open_start, &use_open_end,
                                     &num_cuda_streams))
    {
        return NULL;
    }

    // Validate sequences array: 2D float32
    if (PyArray_NDIM(sequences_array) != 2)
    {
        PyErr_SetString(PyExc_ValueError,
                        "sequences must be a 2D array (total_sequences, global_max_length)");
        return NULL;
    }
    if (PyArray_TYPE(sequences_array) != NPY_FLOAT32)
    {
        PyErr_SetString(PyExc_TypeError, "sequences must be float32 dtype");
        return NULL;
    }

    // Validate lengths array: 1D int64
    if (PyArray_NDIM(lengths_array) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "lengths must be a 1D array");
        return NULL;
    }

    // Validate counts array: 1D int64
    if (PyArray_NDIM(counts_array) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "counts must be a 1D array");
        return NULL;
    }

    npy_intp *seq_dims = PyArray_DIMS(sequences_array);
    size_t total_sequences = (size_t)seq_dims[0];
    size_t global_max_length = (size_t)seq_dims[1];
    size_t num_positions = (size_t)PyArray_DIM(counts_array, 0);

    if ((size_t)PyArray_DIM(lengths_array, 0) != total_sequences)
    {
        PyErr_SetString(PyExc_ValueError,
                        "lengths array size must match total number of sequences");
        return NULL;
    }

    // Convert lengths to size_t array
    size_t *h_lengths = new size_t[total_sequences];
    for (size_t i = 0; i < total_sequences; i++)
    {
        long long val;
        if (PyArray_TYPE(lengths_array) == NPY_INT64)
            val = *((long long *)PyArray_GETPTR1(lengths_array, i));
        else if (PyArray_TYPE(lengths_array) == NPY_INT32)
            val = *((int *)PyArray_GETPTR1(lengths_array, i));
        else
        {
            delete[] h_lengths;
            PyErr_SetString(PyExc_TypeError, "lengths must be int32 or int64 dtype");
            return NULL;
        }
        if (val <= 0 || (size_t)val > global_max_length)
        {
            delete[] h_lengths;
            PyErr_Format(PyExc_ValueError,
                         "length[%zu]=%lld out of range (1..%zu)", i, val, global_max_length);
            return NULL;
        }
        h_lengths[i] = (size_t)val;
    }

    // Convert counts to size_t array
    size_t *h_counts = new size_t[num_positions];
    size_t check_total = 0;
    for (size_t p = 0; p < num_positions; p++)
    {
        long long val;
        if (PyArray_TYPE(counts_array) == NPY_INT64)
            val = *((long long *)PyArray_GETPTR1(counts_array, p));
        else if (PyArray_TYPE(counts_array) == NPY_INT32)
            val = *((int *)PyArray_GETPTR1(counts_array, p));
        else
        {
            delete[] h_lengths;
            delete[] h_counts;
            PyErr_SetString(PyExc_TypeError, "counts must be int32 or int64 dtype");
            return NULL;
        }
        if (val < 0)
        {
            delete[] h_lengths;
            delete[] h_counts;
            PyErr_Format(PyExc_ValueError, "counts[%zu]=%lld must be non-negative", p, val);
            return NULL;
        }
        h_counts[p] = (size_t)val;
        check_total += (size_t)val;
    }

    if (check_total != total_sequences)
    {
        delete[] h_lengths;
        delete[] h_counts;
        PyErr_Format(PyExc_ValueError,
                     "sum(counts)=%zu != total sequences=%zu", check_total, total_sequences);
        return NULL;
    }

    // Compute total output size
    size_t total_out_floats = 0;
    for (size_t p = 0; p < num_positions; p++)
        total_out_floats += h_counts[p] * h_counts[p];

    // Allocate output as flat 1D array
    npy_intp out_dim = (npy_intp)total_out_floats;
    PyArrayObject *out_array = (PyArrayObject *)PyArray_ZEROS(1, &out_dim, NPY_FLOAT32, 0);
    if (out_array == NULL)
    {
        delete[] h_lengths;
        delete[] h_counts;
        return NULL;
    }

    float *sequences_data = (float *)PyArray_DATA(sequences_array);
    float *out_data = (float *)PyArray_DATA(out_array);

    int result = opendba_dtw_multi_position_pairwise(
        sequences_data, h_lengths, h_counts,
        num_positions, global_max_length,
        use_open_start, use_open_end,
        out_data, num_cuda_streams);

    delete[] h_lengths;
    delete[] h_counts;

    if (result != 0)
    {
        Py_DECREF(out_array);
        PyErr_SetString(PyExc_RuntimeError, "CUDA multi-position batch DTW failed");
        return NULL;
    }

    return (PyObject *)out_array;
}
```

- [ ] **Step 2: Register the new function in `DtwMethods` array**

Add this entry before the `cleanup` entry in the `DtwMethods` array:

```cpp
    {"dtw_multi_position_pairwise", (PyCFunction)py_dtw_multi_position_pairwise,
     METH_VARARGS | METH_KEYWORDS,
     "Compute pairwise DTW distances for multiple positions in one batched GPU call.\n\n"
     "Parameters\n"
     "----------\n"
     "sequences : np.ndarray\n"
     "    2D padded array (total_sequences, global_max_length) in float32\n"
     "lengths : np.ndarray\n"
     "    1D array of actual sequence lengths (int64)\n"
     "counts : np.ndarray\n"
     "    1D array of sequence counts per position (int64)\n"
     "use_open_start : int, optional\n"
     "    Enable open start boundary (default: 0)\n"
     "use_open_end : int, optional\n"
     "    Enable open end boundary (default: 0)\n"
     "num_cuda_streams : int, optional\n"
     "    Number of CUDA streams (default: 16)\n\n"
     "Returns\n"
     "-------\n"
     "np.ndarray\n"
     "    Flat 1D array of concatenated distance matrices (float32)\n"},
```

- [ ] **Step 3: Rebuild and verify**

Run: `pip install -e .`
Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add baleen/_cuda_dtw/dtw_api.cpp
git commit -m "feat: add Python C binding for multi-position batch DTW"
```

---

### Task 4: Python Wrapper — `dtw_multi_position_pairwise`

**Files:**
- Modify: `baleen/_cuda_dtw/__init__.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dtw.py`:

```python
class TestMultiPositionBatchCPU:
    """dtw_multi_position_pairwise must produce same results as per-position calls."""

    def test_batch_matches_individual(self):
        """Batch result must match individual dtw_pairwise_varlen calls."""
        from baleen._cuda_dtw import dtw_multi_position_pairwise, dtw_pairwise_varlen

        rng = np.random.default_rng(42)
        # 3 positions, each with different numbers of variable-length signals
        position_signals = [
            [rng.standard_normal(rng.integers(5, 20)).astype(np.float32) for _ in range(4)],
            [rng.standard_normal(rng.integers(5, 20)).astype(np.float32) for _ in range(6)],
            [rng.standard_normal(rng.integers(5, 20)).astype(np.float32) for _ in range(3)],
        ]

        batch_results = dtw_multi_position_pairwise(position_signals, use_cuda=False)

        assert len(batch_results) == 3
        for p, signals in enumerate(position_signals):
            expected = dtw_pairwise_varlen(signals, use_cuda=False)
            np.testing.assert_allclose(
                batch_results[p], expected, rtol=1e-5,
                err_msg=f"Position {p} mismatch",
            )

    def test_single_position(self):
        from baleen._cuda_dtw import dtw_multi_position_pairwise, dtw_pairwise_varlen

        signals = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0], dtype=np.float32),
        ]
        batch_results = dtw_multi_position_pairwise([signals], use_cuda=False)
        expected = dtw_pairwise_varlen(signals, use_cuda=False)
        np.testing.assert_allclose(batch_results[0], expected, rtol=1e-5)

    def test_position_with_one_signal(self):
        """Position with n=1 should produce a 1x1 zero matrix."""
        from baleen._cuda_dtw import dtw_multi_position_pairwise

        position_signals = [
            [np.array([1.0, 2.0], dtype=np.float32)],  # n=1
            [np.array([1.0, 2.0], dtype=np.float32),
             np.array([3.0, 4.0], dtype=np.float32)],  # n=2
        ]
        results = dtw_multi_position_pairwise(position_signals, use_cuda=False)
        assert results[0].shape == (1, 1)
        assert results[0][0, 0] == 0.0
        assert results[1].shape == (2, 2)

    def test_many_positions(self):
        from baleen._cuda_dtw import dtw_multi_position_pairwise

        rng = np.random.default_rng(99)
        position_signals = [
            [rng.standard_normal(rng.integers(3, 15)).astype(np.float32)
             for _ in range(rng.integers(2, 8))]
            for _ in range(20)
        ]
        results = dtw_multi_position_pairwise(position_signals, use_cuda=False)
        assert len(results) == 20
        for p, (matrix, signals) in enumerate(zip(results, position_signals)):
            n = len(signals)
            assert matrix.shape == (n, n)
            np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-6)
            assert np.allclose(matrix, matrix.T)

    def test_empty_list_raises(self):
        from baleen._cuda_dtw import dtw_multi_position_pairwise

        with pytest.raises(ValueError, match="at least 1"):
            dtw_multi_position_pairwise([], use_cuda=False)

    def test_use_cuda_true_raises_without_gpu(self):
        from baleen._cuda_dtw import CUDA_AVAILABLE, dtw_multi_position_pairwise

        if not CUDA_AVAILABLE:
            signals = [[np.array([1.0, 2.0], dtype=np.float32),
                         np.array([3.0, 4.0], dtype=np.float32)]]
            with pytest.raises(RuntimeError, match="CUDA"):
                dtw_multi_position_pairwise(signals, use_cuda=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dtw.py::TestMultiPositionBatchCPU -v`
Expected: FAIL — `ImportError: cannot import name 'dtw_multi_position_pairwise'`

- [ ] **Step 3: Implement `dtw_multi_position_pairwise` in `__init__.py`**

Add to `baleen/_cuda_dtw/__init__.py`, after the existing imports at the top (around line 27), add a conditional import:

```python
try:
    from ._cuda_dtw import dtw_multi_position_pairwise as _dtw_multi_position_cuda
except (ImportError, AttributeError):
    _dtw_multi_position_cuda = None  # Not available (older build or no CUDA)
```

Then add the function before the `cleanup` function (around line 470):

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
    use_open_start : bool
        Enable open start boundary condition.
    use_open_end : bool
        Enable open end boundary condition.
    use_cuda : bool or None
        Backend selection (None=auto, True=force CUDA, False=force CPU).
    num_streams : int
        Number of CUDA streams for concurrent processing (default 16).

    Returns
    -------
    list of np.ndarray
        Distance matrices, one per position. Each is (n_p, n_p) float64.
    """
    if len(position_signals) < 1:
        raise ValueError("Need at least 1 position, got 0")

    # Prep all signals
    prepped: list[list[np.ndarray]] = []
    counts: list[int] = []
    for pos_sigs in position_signals:
        ps = [np.ascontiguousarray(np.asarray(s, dtype=np.float32)) for s in pos_sigs]
        if any(len(s) == 0 for s in ps):
            raise ValueError("All signals must be non-empty")
        prepped.append(ps)
        counts.append(len(ps))

    want_cuda = use_cuda is True or (use_cuda is None and CUDA_AVAILABLE)

    if want_cuda:
        if not CUDA_AVAILABLE or _dtw_multi_position_cuda is None:
            raise RuntimeError(
                "CUDA backend requested but not available. "
                "Install with CUDA support or use use_cuda=False."
            )

        # Compute global max length
        global_max_len = max(
            len(s) for pos_sigs in prepped for s in pos_sigs
        )
        total_seqs = sum(counts)

        # Build padded array and lengths
        padded = np.zeros((total_seqs, global_max_len), dtype=np.float32)
        lengths = np.empty(total_seqs, dtype=np.int64)
        idx = 0
        for pos_sigs in prepped:
            for s in pos_sigs:
                padded[idx, :len(s)] = s
                lengths[idx] = len(s)
                idx += 1

        counts_arr = np.array(counts, dtype=np.int64)

        flat_result = _dtw_multi_position_cuda(
            padded, lengths, counts_arr,
            use_open_start=int(use_open_start),
            use_open_end=int(use_open_end),
            num_cuda_streams=num_streams,
        )

        # Split flat result into per-position matrices
        result_list: list[np.ndarray] = []
        offset = 0
        for n in counts:
            mat = np.asarray(flat_result[offset:offset + n * n], dtype=np.float64).reshape(n, n)
            result_list.append(mat)
            offset += n * n
        return result_list

    # CPU fallback: process each position individually
    result_list = []
    for pos_sigs in prepped:
        n = len(pos_sigs)
        if n < 2:
            result_list.append(np.zeros((n, n), dtype=np.float64))
            continue
        mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = _dtw_distance_cpu(
                    pos_sigs[i], pos_sigs[j],
                    use_open_start=use_open_start,
                    use_open_end=use_open_end,
                )
                mat[i, j] = d
                mat[j, i] = d
        result_list.append(mat)
    return result_list
```

Update the `__all__` list at the bottom of the file to include the new function:

```python
__all__ = [
    "dtw_distance",
    "dtw_pairwise",
    "dtw_pairwise_varlen",
    "dtw_multi_position_pairwise",
    "cleanup",
    "is_available",
    "backend",
    "CUDA_AVAILABLE",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dtw.py::TestMultiPositionBatchCPU -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Run full test suite for regression**

Run: `pytest tests/test_dtw.py -v`
Expected: All tests PASS (no regressions).

- [ ] **Step 6: Commit**

```bash
git add baleen/_cuda_dtw/__init__.py tests/test_dtw.py
git commit -m "feat: add dtw_multi_position_pairwise Python wrapper with CPU fallback"
```

---

### Task 5: Pipeline Integration — Batch DTW in `_process_contig`

**Files:**
- Modify: `baleen/eventalign/_pipeline.py`

- [ ] **Step 1: Add the `num_cuda_streams` parameter to `_process_contig`**

Add `num_cuda_streams: int` as the last parameter in `_process_contig`'s signature (after `cleanup_temp: bool`):

```python
def _process_contig(
    ...
    cleanup_temp: bool,
    num_cuda_streams: int,
) -> tuple[str, ContigResult]:
```

- [ ] **Step 2: Replace the per-position DTW loop with batched approach**

Replace the block from `position_results: dict[int, PositionResult] = {}` through the DTW timing (lines 300-349) with:

```python
    position_results: dict[int, PositionResult] = {}
    n_skipped = 0
    dtw_t0 = time.perf_counter()

    # Phase 1: Collect all signals (CPU)
    position_data: list[tuple[int, str, list[str], list[str], list[NDArray[np.float32]]]] = []
    for pos in common_positions:
        native_pos = native_by_pos[pos]
        ivt_pos = ivt_by_pos[pos]

        if padding > 0:
            native_read_names, native_signals = _signal.extract_signals_for_dtw_padded(
                native_by_pos, pos, padding,
            )
            ivt_read_names, ivt_signals = _signal.extract_signals_for_dtw_padded(
                ivt_by_pos, pos, padding,
            )
        else:
            native_read_names, native_signals = _signal.extract_signals_for_dtw(native_pos)
            ivt_read_names, ivt_signals = _signal.extract_signals_for_dtw(ivt_pos)

        if not native_signals or not ivt_signals:
            logger.debug(
                "    Skipping pos=%d: empty signals (native=%d, ivt=%d)",
                pos, len(native_signals), len(ivt_signals),
            )
            n_skipped += 1
            continue

        all_signals = native_signals + ivt_signals
        kmer = native_pos.reference_kmer
        logger.debug(
            "    [Position %d/%d] pos=%d kmer=%s  %d signals (%d native + %d ivt)",
            len(position_data) + 1, len(common_positions), pos,
            kmer,
            len(all_signals), len(native_signals), len(ivt_signals),
        )
        position_data.append((
            pos, kmer,
            native_read_names, ivt_read_names, all_signals,
        ))

    # Phase 2: Batch DTW (single GPU call for all positions)
    if position_data:
        all_signal_lists = [d[4] for d in position_data]
        all_matrices = _cuda_dtw.dtw_multi_position_pairwise(
            all_signal_lists,
            use_open_start=use_open_start,
            use_open_end=use_open_end,
            use_cuda=use_cuda,
            num_streams=num_cuda_streams,
        )

        # Phase 3: Package results
        for (pos, kmer, nat_names, ivt_names, _sigs), matrix in zip(position_data, all_matrices):
            position_results[pos] = PositionResult(
                position=pos,
                reference_kmer=kmer,
                n_native_reads=len(nat_names),
                n_ivt_reads=len(ivt_names),
                native_read_names=nat_names,
                ivt_read_names=ivt_names,
                distance_matrix=matrix,
            )

    dtw_elapsed = _fmt_elapsed(time.perf_counter() - dtw_t0)
```

- [ ] **Step 3: Add `num_cuda_streams` to `run_pipeline` signature**

Add the parameter after `threads`:

```python
def run_pipeline(
    ...
    threads: int = 1,
    num_cuda_streams: int = 16,
) -> tuple[dict[str, ContigResult], PipelineMetadata]:
```

Add a log line in the initial logging block (around line 418):

```python
    logger.info("  open_start=%s  open_end=%s  min_mapq=%d  primary_only=%s  cuda_streams=%d",
                use_open_start, use_open_end, min_mapq, primary_only, num_cuda_streams)
```

- [ ] **Step 4: Pass `num_cuda_streams` through to `_process_contig` calls**

In both the parallel (`executor.submit`) and sequential code paths, add `num_cuda_streams=num_cuda_streams` to the `_process_contig` call arguments. Add it after the `cleanup_temp=cleanup_temp` line in both places.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add baleen/eventalign/_pipeline.py
git commit -m "feat: batch all positions into single GPU call via dtw_multi_position_pairwise"
```

---

### Task 6: GPU Build Verification and Integration Test

**Files:**
- Modify: `tests/test_dtw.py` (add GPU-conditional tests)

- [ ] **Step 1: Add GPU-conditional batch test**

Add to `tests/test_dtw.py`:

```python
@pytest.mark.skipif(
    not __import__("baleen._cuda_dtw", fromlist=["CUDA_AVAILABLE"]).CUDA_AVAILABLE,
    reason="CUDA not available",
)
class TestMultiPositionBatchGPU:
    """GPU batch DTW must produce same results as CPU."""

    def test_gpu_matches_cpu(self):
        from baleen._cuda_dtw import dtw_multi_position_pairwise

        rng = np.random.default_rng(42)
        position_signals = [
            [rng.standard_normal(rng.integers(5, 20)).astype(np.float32) for _ in range(4)],
            [rng.standard_normal(rng.integers(5, 20)).astype(np.float32) for _ in range(6)],
            [rng.standard_normal(rng.integers(5, 20)).astype(np.float32) for _ in range(3)],
        ]

        cpu_results = dtw_multi_position_pairwise(position_signals, use_cuda=False)
        gpu_results = dtw_multi_position_pairwise(position_signals, use_cuda=True)

        for p in range(3):
            np.testing.assert_allclose(
                gpu_results[p], cpu_results[p], rtol=1e-4,
                err_msg=f"Position {p}: GPU != CPU",
            )

    def test_gpu_many_positions(self):
        """Stress test: 50 positions to verify stream concurrency works."""
        from baleen._cuda_dtw import dtw_multi_position_pairwise

        rng = np.random.default_rng(123)
        position_signals = [
            [rng.standard_normal(rng.integers(5, 30)).astype(np.float32)
             for _ in range(rng.integers(3, 15))]
            for _ in range(50)
        ]

        gpu_results = dtw_multi_position_pairwise(position_signals, use_cuda=True, num_streams=16)
        cpu_results = dtw_multi_position_pairwise(position_signals, use_cuda=False)

        for p in range(50):
            np.testing.assert_allclose(
                gpu_results[p], cpu_results[p], rtol=1e-4,
                err_msg=f"Position {p}: GPU != CPU",
            )

    def test_gpu_different_stream_counts(self):
        """Verify correctness with different numbers of streams."""
        from baleen._cuda_dtw import dtw_multi_position_pairwise

        rng = np.random.default_rng(77)
        position_signals = [
            [rng.standard_normal(rng.integers(5, 15)).astype(np.float32) for _ in range(5)]
            for _ in range(10)
        ]

        cpu_results = dtw_multi_position_pairwise(position_signals, use_cuda=False)
        for nstreams in [1, 4, 8, 16, 32]:
            gpu_results = dtw_multi_position_pairwise(
                position_signals, use_cuda=True, num_streams=nstreams,
            )
            for p in range(10):
                np.testing.assert_allclose(
                    gpu_results[p], cpu_results[p], rtol=1e-4,
                    err_msg=f"Position {p}, streams={nstreams}: GPU != CPU",
                )
```

- [ ] **Step 2: Run CPU tests**

Run: `pytest tests/test_dtw.py -v -k "not GPU"`
Expected: All PASS.

- [ ] **Step 3: Rebuild CUDA extension and run GPU tests (on GPU server)**

Run: `pip install -e . && pytest tests/test_dtw.py -v`
Expected: All tests PASS including GPU tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_dtw.py
git commit -m "test: add GPU correctness tests for multi-position batch DTW"
```

---

## Execution Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Cache device properties | `dtw_api.cpp` |
| 2 | New C function | `dtw_api.h`, `dtw_api.cpp` |
| 3 | Python C binding | `dtw_api.cpp` |
| 4 | Python wrapper + CPU tests | `__init__.py`, `test_dtw.py` |
| 5 | Pipeline integration | `_pipeline.py` |
| 6 | GPU integration tests | `test_dtw.py` |

Tasks 1-3 must be sequential (each builds on prior). Tasks 4 and 5 can run in parallel after Task 3. Task 6 runs after all others.
