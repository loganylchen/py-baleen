// cudtw_wrapper.cu — cuDTW++ warp-shuffle kernel wrapper for baleen
//
// Query is passed through global memory (no __constant__ serialisation),
// enabling CUDA-stream concurrency across pairwise queries.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// --------------------------------------------------------------------------
// cuDTW++ kernel includes
// --------------------------------------------------------------------------
typedef float    value_t;
typedef uint64_t index_t;

#include "cudtw/include/DTW.hpp"

// --------------------------------------------------------------------------
// Error-checking macro
// --------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            return -1;                                                         \
        }                                                                      \
    } while (0)

// --------------------------------------------------------------------------
// Length bucketing
// --------------------------------------------------------------------------
static int select_bucket(int len) {
    if (len <= 127)  return 127;
    if (len <= 255)  return 255;
    if (len <= 511)  return 511;
    if (len <= 1023) return 1023;
    if (len <= 2047) return 2047;
    return -1;  // too long — caller must resample
}

// --------------------------------------------------------------------------
// Single-pair DTW  (kept for API compat; not perf-critical)
// --------------------------------------------------------------------------
int opendba_dtw_cuda(
    const float *seq1, size_t len1,
    const float *seq2, size_t len2,
    float *out_distance)
{
    if (!seq1 || !seq2 || !out_distance || len1 == 0 || len2 == 0) {
        fprintf(stderr, "cudtw_wrapper: invalid input\n");
        return -1;
    }

    size_t max_len = (len1 > len2) ? len1 : len2;
    int bucket = select_bucket((int)max_len);
    if (bucket < 0) {
        fprintf(stderr, "cudtw_wrapper: sequence too long (%zu), max 2047\n", max_len);
        return -1;
    }

    float *h_query   = (float *)calloc(bucket, sizeof(float));
    float *h_subject = (float *)calloc(bucket, sizeof(float));
    if (!h_query || !h_subject) { free(h_query); free(h_subject); return -1; }
    memcpy(h_query,   seq1, len1 * sizeof(float));
    memcpy(h_subject, seq2, len2 * sizeof(float));

    float *d_query = nullptr, *d_subject = nullptr, *d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_query,   bucket * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_subject, bucket * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist,    sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_query,   h_query,   bucket * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_subject, h_subject, bucket * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_query); free(h_subject);

    bool ok = FullDTW::dist<value_t, index_t>(
        d_query, d_subject, d_dist, (index_t)1, (index_t)bucket);
    if (!ok) {
        cudaFree(d_query); cudaFree(d_subject); cudaFree(d_dist);
        fprintf(stderr, "cudtw_wrapper: unsupported bucket %d\n", bucket);
        return -1;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float sq_cost;
    CUDA_CHECK(cudaMemcpy(&sq_cost, d_dist, sizeof(float), cudaMemcpyDeviceToHost));
    *out_distance = sqrtf(sq_cost);

    cudaFree(d_query);
    cudaFree(d_subject);
    cudaFree(d_dist);
    return 0;
}

// --------------------------------------------------------------------------
// Cleanup
// --------------------------------------------------------------------------
void opendba_dtw_cleanup() {
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess) {
        for (int i = 0; i < count; i++) {
            cudaSetDevice(i);
            cudaDeviceReset();
        }
    }
}

// --------------------------------------------------------------------------
// Internal: stream-concurrent pairwise DTW over one padded subject buffer.
//
// d_subjects: [N * bucket] device buffer (padded to bucket width).
// d_out:      [N * N] device output buffer (float32, squared cost).
// Each row i is launched on streams[i % num_streams]:
//   Query = d_subjects + i*bucket, Subject = d_subjects, Dist = d_out + i*N.
// Caller is responsible for the final cudaDeviceSynchronize + sqrtf.
// --------------------------------------------------------------------------
static int launch_pairwise_rows(
    const float *d_subjects, float *d_out,
    size_t N, int bucket,
    cudaStream_t *streams, int num_streams)
{
    for (size_t i = 0; i < N; i++) {
        cudaStream_t s = streams[i % (size_t)num_streams];
        bool ok = FullDTW::dist<value_t, index_t>(
            d_subjects + i * (size_t)bucket,
            const_cast<float *>(d_subjects),
            d_out + i * N,
            (index_t)N, (index_t)bucket, s);
        if (!ok) return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
// Host helper: take a [N*N] float32 squared-cost matrix → sqrtf + zero diag.
// --------------------------------------------------------------------------
static void finalize_matrix(float *m, size_t N) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            m[i * N + j] = (i == j) ? 0.0f : sqrtf(m[i * N + j]);
        }
    }
}

// --------------------------------------------------------------------------
// Batch pairwise DTW  (equal-length sequences, single position)
// --------------------------------------------------------------------------
int opendba_dtw_pairwise_batch(
    const float *sequences,
    size_t num_sequences,
    size_t seq_length,
    float *out_distances)
{
    if (!sequences || !out_distances || num_sequences < 2 || seq_length == 0) {
        fprintf(stderr, "cudtw_wrapper: invalid pairwise input\n");
        return -1;
    }

    int bucket = select_bucket((int)seq_length);
    if (bucket < 0) {
        fprintf(stderr, "cudtw_wrapper: seq_length %zu > 2047\n", seq_length);
        return -1;
    }

    const size_t N = num_sequences;

    // Pad all sequences to bucket on host (single contiguous buffer)
    float *h_padded = (float *)calloc(N * bucket, sizeof(float));
    if (!h_padded) return -1;
    for (size_t i = 0; i < N; i++)
        memcpy(h_padded + i * bucket, sequences + i * seq_length,
               seq_length * sizeof(float));

    // Allocate device memory: subjects (doubles as query source) + output
    float *d_subjects = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_subjects, N * bucket * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,      N * N      * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_subjects, h_padded, N * bucket * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_padded);

    // Stream pool — sized to min(N, 16) for single-position batch
    int num_streams = (N < 16) ? (int)N : 16;
    std::vector<cudaStream_t> streams(num_streams);
    for (int s = 0; s < num_streams; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    int rc = launch_pairwise_rows(d_subjects, d_out, N, bucket,
                                  streams.data(), num_streams);
    if (rc == 0) rc = (cudaGetLastError() == cudaSuccess) ? 0 : -1;
    if (rc == 0) rc = (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;

    for (int s = 0; s < num_streams; s++) cudaStreamDestroy(streams[s]);

    if (rc != 0) {
        cudaFree(d_subjects); cudaFree(d_out);
        return -1;
    }

    // One D→H copy, then sqrtf + diagonal zeroing on host
    CUDA_CHECK(cudaMemcpy(out_distances, d_out, N * N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    finalize_matrix(out_distances, N);

    cudaFree(d_subjects);
    cudaFree(d_out);
    return 0;
}

// --------------------------------------------------------------------------
// Variable-length pairwise DTW (single position)
// --------------------------------------------------------------------------
int opendba_dtw_pairwise_varlen(
    const float *sequences,
    const size_t *seq_lengths,
    size_t num_sequences,
    size_t max_length,
    float *out_distances)
{
    if (!sequences || !seq_lengths || !out_distances ||
        num_sequences < 2 || max_length == 0) {
        fprintf(stderr, "cudtw_wrapper: invalid varlen input\n");
        return -1;
    }

    size_t actual_max = 0;
    for (size_t i = 0; i < num_sequences; i++)
        if (seq_lengths[i] > actual_max) actual_max = seq_lengths[i];

    int bucket = select_bucket((int)actual_max);
    if (bucket < 0) {
        fprintf(stderr, "cudtw_wrapper: max_length %zu > 2047\n", actual_max);
        return -1;
    }

    const size_t N = num_sequences;

    // Repack to bucket width
    float *h_padded = (float *)calloc(N * bucket, sizeof(float));
    if (!h_padded) return -1;
    for (size_t i = 0; i < N; i++)
        memcpy(h_padded + i * bucket, sequences + i * max_length,
               seq_lengths[i] * sizeof(float));

    float *d_subjects = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_subjects, N * bucket * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,      N * N      * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_subjects, h_padded, N * bucket * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_padded);

    int num_streams = (N < 16) ? (int)N : 16;
    std::vector<cudaStream_t> streams(num_streams);
    for (int s = 0; s < num_streams; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    int rc = launch_pairwise_rows(d_subjects, d_out, N, bucket,
                                  streams.data(), num_streams);
    if (rc == 0) rc = (cudaGetLastError() == cudaSuccess) ? 0 : -1;
    if (rc == 0) rc = (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;

    for (int s = 0; s < num_streams; s++) cudaStreamDestroy(streams[s]);

    if (rc != 0) {
        cudaFree(d_subjects); cudaFree(d_out);
        return -1;
    }

    CUDA_CHECK(cudaMemcpy(out_distances, d_out, N * N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    finalize_matrix(out_distances, N);

    cudaFree(d_subjects);
    cudaFree(d_out);
    return 0;
}

// --------------------------------------------------------------------------
// Multi-position batched pairwise DTW — whole chunk shares one d_subjects /
// one d_out, one H→D, one D→H, one device-sync.  Kernels across positions
// and within a position fan out over a stream pool for real concurrency.
// --------------------------------------------------------------------------
int opendba_dtw_multi_position_pairwise(
    const float *all_sequences,
    const size_t *all_seq_lengths,
    const size_t *position_seq_counts,
    size_t num_positions,
    size_t global_max_length,
    float *out_distances,
    int num_cuda_streams,
    int device_id)
{
    if (!all_sequences || !all_seq_lengths || !position_seq_counts ||
        !out_distances || num_positions == 0 || global_max_length == 0) {
        fprintf(stderr, "cudtw_wrapper: invalid multi-position input\n");
        return -1;
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    // Pre-scan: per-position bucket + sequence offset + output offset.
    // Each position packs its sequences into d_subjects with stride =
    // its own bucket (NOT a global max).  The cuDTW++ kernel uses
    // num_features as both the subject stride and the walk length, so
    // buffer stride and kernel num_features MUST match per position —
    // otherwise every row beyond 0 reads from padding → garbage.
    std::vector<int>    buckets(num_positions, 0);
    std::vector<size_t> seq_index_offsets(num_positions, 0);  // sequence-count prefix sum
    std::vector<size_t> pos_float_offsets(num_positions, 0);  // float offset into d_subjects
    std::vector<size_t> out_offsets(num_positions, 0);

    size_t total_seqs   = 0;
    size_t total_out    = 0;
    size_t total_floats = 0;

    for (size_t p = 0; p < num_positions; p++) {
        size_t n = position_seq_counts[p];
        seq_index_offsets[p] = total_seqs;
        pos_float_offsets[p] = total_floats;
        out_offsets[p]       = total_out;

        size_t pos_max = 0;
        for (size_t i = 0; i < n; i++) {
            size_t l = all_seq_lengths[total_seqs + i];
            if (l > pos_max) pos_max = l;
        }
        int b = (n < 2) ? 0 : select_bucket((int)pos_max);
        if (b < 0) {
            fprintf(stderr, "cudtw_wrapper: position %zu max_len %zu > 2047\n",
                    p, pos_max);
            return -1;
        }
        buckets[p]    = b;
        total_floats += n * (size_t)b;  // 0 for n<2 positions
        total_seqs   += n;
        total_out    += n * n;
    }

    // Degenerate: every position had n<2 — just zero the output and return.
    if (total_floats == 0) {
        memset(out_distances, 0, total_out * sizeof(float));
        return 0;
    }

    // Pack sequences per-position with per-position stride.  Skipped entirely
    // for n<2 positions (they contribute 0 floats).
    float *h_padded = (float *)calloc(total_floats, sizeof(float));
    if (!h_padded) return -1;

    for (size_t p = 0; p < num_positions; p++) {
        size_t n = position_seq_counts[p];
        if (n < 2) continue;
        size_t b        = (size_t)buckets[p];
        size_t seq_off  = seq_index_offsets[p];
        size_t pos_base = pos_float_offsets[p];
        for (size_t i = 0; i < n; i++) {
            size_t slen = all_seq_lengths[seq_off + i];
            const float *src = all_sequences + (seq_off + i) * global_max_length;
            memcpy(h_padded + pos_base + i * b, src, slen * sizeof(float));
        }
    }

    float *d_subjects = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_subjects, total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,      total_out    * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_subjects, h_padded, total_floats * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_padded);

    // Zero the output buffer so positions with n<2 leave a valid zero block.
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, total_out * sizeof(float), 0));

    int ns = num_cuda_streams > 0 ? num_cuda_streams : 16;
    std::vector<cudaStream_t> streams(ns);
    for (int s = 0; s < ns; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    // Dispatch: every (position, row) pair hits a stream round-robin.
    // Buffer stride for position p is buckets[p] — same as the kernel's
    // num_features, so Query/Subject indexing is consistent.
    int launch_err = 0;
    size_t launch_counter = 0;
    for (size_t p = 0; p < num_positions && launch_err == 0; p++) {
        size_t n = position_seq_counts[p];
        if (n < 2) continue;
        size_t b        = (size_t)buckets[p];
        size_t pos_base = pos_float_offsets[p];
        size_t out_off  = out_offsets[p];
        for (size_t i = 0; i < n; i++) {
            cudaStream_t s = streams[launch_counter % (size_t)ns];
            launch_counter++;
            bool ok = FullDTW::dist<value_t, index_t>(
                d_subjects + pos_base + i * b,   // Query = row i (stride b)
                d_subjects + pos_base,           // Subject base (stride b)
                d_out      + out_off + i * n,
                (index_t)n, (index_t)b, s);
            if (!ok) { launch_err = 1; break; }
        }
    }

    int rc = launch_err;
    if (rc == 0 && cudaGetLastError()      != cudaSuccess) rc = -1;
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = -1;

    for (int s = 0; s < ns; s++) cudaStreamDestroy(streams[s]);

    if (rc != 0) {
        cudaFree(d_subjects); cudaFree(d_out);
        return -1;
    }

    CUDA_CHECK(cudaMemcpy(out_distances, d_out, total_out * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Finalize each position block: sqrtf + diagonal zero
    for (size_t p = 0; p < num_positions; p++) {
        size_t n = position_seq_counts[p];
        if (n < 2) continue;
        finalize_matrix(out_distances + out_offsets[p], n);
    }

    cudaFree(d_subjects);
    cudaFree(d_out);
    return 0;
}

// ==========================================================================
// Python C API Bindings
// ==========================================================================

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// --- dtw_distance ---
// Accepts use_open_start / use_open_end as ignored kwargs for backward
// compat with older Python callers.  cuDTW++ has no open-boundary mode.
static PyObject *py_dtw_cuda(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyArrayObject *seq1_array = NULL, *seq2_array = NULL;
    int use_open_start = 0, use_open_end = 0;
    static char *kwlist[] = {(char*)"seq1", (char*)"seq2",
                             (char*)"use_open_start", (char*)"use_open_end", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|ii", kwlist,
                                     &PyArray_Type, &seq1_array,
                                     &PyArray_Type, &seq2_array,
                                     &use_open_start, &use_open_end))
        return NULL;
    (void)use_open_start; (void)use_open_end;

    if (PyArray_NDIM(seq1_array) != 1 || PyArray_NDIM(seq2_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be 1-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(seq1_array) != NPY_FLOAT32 ||
        PyArray_TYPE(seq2_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be float32");
        return NULL;
    }

    npy_intp len1 = PyArray_DIM(seq1_array, 0);
    npy_intp len2 = PyArray_DIM(seq2_array, 0);
    if (len1 == 0 || len2 == 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays cannot be empty");
        return NULL;
    }

    float *s1 = (float *)PyArray_DATA(seq1_array);
    float *s2 = (float *)PyArray_DATA(seq2_array);
    float distance = 0.0f;

    int rc = opendba_dtw_cuda(s1, (size_t)len1, s2, (size_t)len2, &distance);
    if (rc != 0) {
        PyErr_SetString(PyExc_RuntimeError, "CUDA DTW computation failed");
        return NULL;
    }
    return PyFloat_FromDouble((double)distance);
}

// --- dtw_pairwise ---
static PyObject *py_dtw_pairwise(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyArrayObject *seq_array;
    int use_open_start = 0, use_open_end = 0;
    static char *kwlist[] = {(char*)"sequences",
                             (char*)"use_open_start", (char*)"use_open_end", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|ii", kwlist,
                                     &PyArray_Type, &seq_array,
                                     &use_open_start, &use_open_end))
        return NULL;
    (void)use_open_start; (void)use_open_end;

    if (PyArray_NDIM(seq_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "sequences must be 2D");
        return NULL;
    }
    if (PyArray_TYPE(seq_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "sequences must be float32");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(seq_array);
    size_t N = (size_t)dims[0], L = (size_t)dims[1];
    if (N < 2) {
        PyErr_SetString(PyExc_ValueError, "Need at least 2 sequences");
        return NULL;
    }

    npy_intp out_dims[2] = {(npy_intp)N, (npy_intp)N};
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(2, out_dims, NPY_FLOAT32, 0);
    if (!out) return NULL;

    int rc = opendba_dtw_pairwise_batch(
        (float *)PyArray_DATA(seq_array), N, L,
        (float *)PyArray_DATA(out));
    if (rc != 0) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "CUDA pairwise DTW failed");
        return NULL;
    }
    return (PyObject *)out;
}

// --- dtw_pairwise_varlen ---
static PyObject *py_dtw_pairwise_varlen(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyArrayObject *seq_array, *len_array;
    int use_open_start = 0, use_open_end = 0;
    static char *kwlist[] = {(char*)"sequences", (char*)"lengths",
                             (char*)"use_open_start", (char*)"use_open_end", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|ii", kwlist,
                                     &PyArray_Type, &seq_array,
                                     &PyArray_Type, &len_array,
                                     &use_open_start, &use_open_end))
        return NULL;
    (void)use_open_start; (void)use_open_end;

    if (PyArray_NDIM(seq_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "sequences must be 2D");
        return NULL;
    }
    if (PyArray_TYPE(seq_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "sequences must be float32");
        return NULL;
    }
    if (PyArray_NDIM(len_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "lengths must be 1D");
        return NULL;
    }

    npy_intp *sdims = PyArray_DIMS(seq_array);
    size_t N = (size_t)sdims[0], max_len = (size_t)sdims[1];
    if ((size_t)PyArray_DIM(len_array, 0) != N) {
        PyErr_SetString(PyExc_ValueError, "lengths size != num_sequences");
        return NULL;
    }
    if (N < 2) {
        PyErr_SetString(PyExc_ValueError, "Need at least 2 sequences");
        return NULL;
    }

    size_t *h_lengths = new size_t[N];
    for (size_t i = 0; i < N; i++) {
        long long val;
        if (PyArray_TYPE(len_array) == NPY_INT64)
            val = *((long long *)PyArray_GETPTR1(len_array, i));
        else if (PyArray_TYPE(len_array) == NPY_INT32)
            val = *((int *)PyArray_GETPTR1(len_array, i));
        else {
            delete[] h_lengths;
            PyErr_SetString(PyExc_TypeError, "lengths must be int32 or int64");
            return NULL;
        }
        if (val <= 0 || (size_t)val > max_len) {
            delete[] h_lengths;
            PyErr_Format(PyExc_ValueError,
                         "length[%zu]=%lld out of range", i, val);
            return NULL;
        }
        h_lengths[i] = (size_t)val;
    }

    npy_intp out_dims[2] = {(npy_intp)N, (npy_intp)N};
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(2, out_dims, NPY_FLOAT32, 0);
    if (!out) { delete[] h_lengths; return NULL; }

    int rc = opendba_dtw_pairwise_varlen(
        (float *)PyArray_DATA(seq_array), h_lengths, N, max_len,
        (float *)PyArray_DATA(out));
    delete[] h_lengths;

    if (rc != 0) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "CUDA varlen DTW failed");
        return NULL;
    }
    return (PyObject *)out;
}

// --- dtw_multi_position_pairwise ---
static PyObject *py_dtw_multi_position_pairwise(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *seq_array, *len_array, *cnt_array;
    int use_open_start = 0, use_open_end = 0;
    int num_cuda_streams = 16, device_id = 0;
    static char *kwlist[] = {
        (char*)"sequences", (char*)"lengths", (char*)"counts",
        (char*)"use_open_start", (char*)"use_open_end",
        (char*)"num_cuda_streams", (char*)"device_id", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!|iiii", kwlist,
                                     &PyArray_Type, &seq_array,
                                     &PyArray_Type, &len_array,
                                     &PyArray_Type, &cnt_array,
                                     &use_open_start, &use_open_end,
                                     &num_cuda_streams, &device_id))
        return NULL;
    (void)use_open_start; (void)use_open_end;

    if (PyArray_NDIM(seq_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "sequences must be 2D"); return NULL;
    }
    if (PyArray_TYPE(seq_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "sequences must be float32"); return NULL;
    }
    if (PyArray_NDIM(len_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "lengths must be 1D"); return NULL;
    }
    if (PyArray_NDIM(cnt_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "counts must be 1D"); return NULL;
    }

    npy_intp *sdims = PyArray_DIMS(seq_array);
    size_t total_seqs = (size_t)sdims[0];
    size_t gml = (size_t)sdims[1];
    size_t num_pos = (size_t)PyArray_DIM(cnt_array, 0);

    if ((size_t)PyArray_DIM(len_array, 0) != total_seqs) {
        PyErr_SetString(PyExc_ValueError, "lengths size != total sequences");
        return NULL;
    }

    size_t *h_lengths = new size_t[total_seqs];
    for (size_t i = 0; i < total_seqs; i++) {
        long long val;
        if (PyArray_TYPE(len_array) == NPY_INT64)
            val = *((long long *)PyArray_GETPTR1(len_array, i));
        else if (PyArray_TYPE(len_array) == NPY_INT32)
            val = *((int *)PyArray_GETPTR1(len_array, i));
        else {
            delete[] h_lengths;
            PyErr_SetString(PyExc_TypeError, "lengths must be int32 or int64");
            return NULL;
        }
        if (val <= 0 || (size_t)val > gml) {
            delete[] h_lengths;
            PyErr_Format(PyExc_ValueError, "length[%zu]=%lld out of range", i, val);
            return NULL;
        }
        h_lengths[i] = (size_t)val;
    }

    size_t *h_counts = new size_t[num_pos];
    size_t check_total = 0;
    for (size_t p = 0; p < num_pos; p++) {
        long long val;
        if (PyArray_TYPE(cnt_array) == NPY_INT64)
            val = *((long long *)PyArray_GETPTR1(cnt_array, p));
        else if (PyArray_TYPE(cnt_array) == NPY_INT32)
            val = *((int *)PyArray_GETPTR1(cnt_array, p));
        else {
            delete[] h_lengths; delete[] h_counts;
            PyErr_SetString(PyExc_TypeError, "counts must be int32 or int64");
            return NULL;
        }
        h_counts[p] = (size_t)val;
        check_total += (size_t)val;
    }
    if (check_total != total_seqs) {
        delete[] h_lengths; delete[] h_counts;
        PyErr_Format(PyExc_ValueError,
                     "sum(counts)=%zu != total_sequences=%zu",
                     check_total, total_seqs);
        return NULL;
    }

    size_t total_out = 0;
    for (size_t p = 0; p < num_pos; p++)
        total_out += h_counts[p] * h_counts[p];

    npy_intp out_dim = (npy_intp)total_out;
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(1, &out_dim, NPY_FLOAT32, 0);
    if (!out) { delete[] h_lengths; delete[] h_counts; return NULL; }

    int rc = opendba_dtw_multi_position_pairwise(
        (float *)PyArray_DATA(seq_array), h_lengths, h_counts,
        num_pos, gml,
        (float *)PyArray_DATA(out), num_cuda_streams, device_id);

    delete[] h_lengths;
    delete[] h_counts;

    if (rc != 0) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "CUDA multi-position DTW failed");
        return NULL;
    }
    return (PyObject *)out;
}

// --- cleanup ---
static PyObject *py_dtw_cleanup(PyObject *self, PyObject *args) {
    opendba_dtw_cleanup();
    Py_RETURN_NONE;
}

// ==========================================================================
// Module table
// ==========================================================================
static PyMethodDef DtwMethods[] = {
    {"dtw_distance", (PyCFunction)py_dtw_cuda, METH_VARARGS | METH_KEYWORDS,
     "Compute DTW distance between two sequences (cuDTW++ warp-shuffle)."},
    {"dtw_pairwise", (PyCFunction)py_dtw_pairwise, METH_VARARGS | METH_KEYWORDS,
     "Compute pairwise DTW distances (cuDTW++ warp-shuffle)."},
    {"dtw_pairwise_varlen", (PyCFunction)py_dtw_pairwise_varlen,
     METH_VARARGS | METH_KEYWORDS,
     "Compute pairwise DTW distances for variable-length sequences."},
    {"dtw_multi_position_pairwise", (PyCFunction)py_dtw_multi_position_pairwise,
     METH_VARARGS | METH_KEYWORDS,
     "Compute pairwise DTW for multiple positions (cuDTW++ warp-shuffle)."},
    {"cleanup", py_dtw_cleanup, METH_NOARGS,
     "Reset CUDA device and free resources."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef dtwmodule = {
    PyModuleDef_HEAD_INIT,
    "_cuda_dtw",
    "cuDTW++ warp-shuffle DTW for baleen",
    -1,
    DtwMethods
};

PyMODINIT_FUNC PyInit__cuda_dtw(void) {
    import_array();
    if (PyErr_Occurred()) return NULL;

    PyObject *m = PyModule_Create(&dtwmodule);
    if (!m) return NULL;

    PyModule_AddIntConstant(m, "__version_major__", 1);
    PyModule_AddIntConstant(m, "__version_minor__", 0);
    PyModule_AddStringConstant(m, "__version__", "1.0.0-cudtw");

    return m;
}
