// cudtw_wrapper.cu — cuDTW++ warp-shuffle kernel wrapper for baleen
//
// Replaces dtw_api.cpp when BALEEN_USE_CUDTW != 0 (default).
// Same Python binding signatures so __init__.py works unchanged.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>

// --------------------------------------------------------------------------
// cuDTW++ constant-memory query buffer + kernel includes
// --------------------------------------------------------------------------
typedef float    value_t;
typedef uint64_t index_t;

constexpr index_t max_features = (1UL << 16) / sizeof(value_t);  // 16384
__constant__ value_t cQuery[max_features];

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
    int /*use_open_start*/,   // ignored — cuDTW++ has no open boundary
    int /*use_open_end*/,
    float *out_distance)
{
    if (!seq1 || !seq2 || !out_distance || len1 == 0 || len2 == 0) {
        fprintf(stderr, "cudtw_wrapper: invalid input\n");
        return -1;
    }

    // Use the longer length to pick bucket (pad both to same bucket)
    size_t max_len = (len1 > len2) ? len1 : len2;
    int bucket = select_bucket((int)max_len);
    if (bucket < 0) {
        fprintf(stderr, "cudtw_wrapper: sequence too long (%zu), max 2047\n", max_len);
        return -1;
    }

    // Pad seq2 (subject) on host
    float *h_subject = (float *)calloc(bucket, sizeof(float));
    if (!h_subject) return -1;
    memcpy(h_subject, seq2, len2 * sizeof(float));

    // Upload query (seq1) to constant memory
    float *h_query = (float *)calloc(bucket, sizeof(float));
    if (!h_query) { free(h_subject); return -1; }
    memcpy(h_query, seq1, len1 * sizeof(float));
    CUDA_CHECK(cudaMemcpyToSymbol(cQuery, h_query, bucket * sizeof(float)));
    free(h_query);

    // Allocate device memory
    float *d_subject = nullptr, *d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_subject, bucket * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_subject, h_subject, bucket * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_subject);

    // Launch — 1 entry
    bool ok = FullDTW::dist<value_t, index_t>(
        d_subject, d_dist, (index_t)1, (index_t)bucket);
    if (!ok) {
        cudaFree(d_subject); cudaFree(d_dist);
        fprintf(stderr, "cudtw_wrapper: unsupported bucket %d\n", bucket);
        return -1;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float sq_cost;
    CUDA_CHECK(cudaMemcpy(&sq_cost, d_dist, sizeof(float), cudaMemcpyDeviceToHost));
    *out_distance = sqrtf(sq_cost);

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
// Batch pairwise DTW  (equal-length sequences)
// --------------------------------------------------------------------------
int opendba_dtw_pairwise_batch(
    const float *sequences,
    size_t num_sequences,
    size_t seq_length,
    int /*use_open_start*/,
    int /*use_open_end*/,
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

    size_t N = num_sequences;

    // Pad all sequences to bucket on host
    float *h_padded = (float *)calloc(N * bucket, sizeof(float));
    if (!h_padded) return -1;
    for (size_t i = 0; i < N; i++)
        memcpy(h_padded + i * bucket, sequences + i * seq_length,
               seq_length * sizeof(float));

    // Upload all subjects to device once
    float *d_subjects = nullptr, *d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_subjects, N * bucket * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_subjects, h_padded, N * bucket * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *h_row = new float[N];

    // For each query i, upload to cQuery → kernel → read row
    for (size_t i = 0; i < N; i++) {
        CUDA_CHECK(cudaMemcpyToSymbol(cQuery, h_padded + i * bucket,
                                      bucket * sizeof(float)));

        bool ok = FullDTW::dist<value_t, index_t>(
            d_subjects, d_dist, (index_t)N, (index_t)bucket);
        if (!ok) {
            free(h_padded); delete[] h_row;
            cudaFree(d_subjects); cudaFree(d_dist);
            return -1;
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_row, d_dist, N * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Fill symmetric matrix row — sqrt the squared cost
        for (size_t j = 0; j < N; j++) {
            float d = (i == j) ? 0.0f : sqrtf(h_row[j]);
            out_distances[i * N + j] = d;
        }
    }

    free(h_padded);
    delete[] h_row;
    cudaFree(d_subjects);
    cudaFree(d_dist);
    return 0;
}

// --------------------------------------------------------------------------
// Variable-length pairwise DTW
// --------------------------------------------------------------------------
int opendba_dtw_pairwise_varlen(
    const float *sequences,
    const size_t *seq_lengths,
    size_t num_sequences,
    size_t max_length,
    int use_open_start,
    int use_open_end,
    float *out_distances)
{
    // Find actual max and bucket
    size_t actual_max = 0;
    for (size_t i = 0; i < num_sequences; i++)
        if (seq_lengths[i] > actual_max) actual_max = seq_lengths[i];

    int bucket = select_bucket((int)actual_max);
    if (bucket < 0) {
        fprintf(stderr, "cudtw_wrapper: max_length %zu > 2047\n", actual_max);
        return -1;
    }

    size_t N = num_sequences;

    // Re-pad to bucket width
    float *h_padded = (float *)calloc(N * bucket, sizeof(float));
    if (!h_padded) return -1;
    for (size_t i = 0; i < N; i++)
        memcpy(h_padded + i * bucket, sequences + i * max_length,
               seq_lengths[i] * sizeof(float));

    // Upload subjects
    float *d_subjects = nullptr, *d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_subjects, N * bucket * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_subjects, h_padded, N * bucket * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *h_row = new float[N];

    for (size_t i = 0; i < N; i++) {
        CUDA_CHECK(cudaMemcpyToSymbol(cQuery, h_padded + i * bucket,
                                      bucket * sizeof(float)));

        bool ok = FullDTW::dist<value_t, index_t>(
            d_subjects, d_dist, (index_t)N, (index_t)bucket);
        if (!ok) {
            free(h_padded); delete[] h_row;
            cudaFree(d_subjects); cudaFree(d_dist);
            return -1;
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_row, d_dist, N * sizeof(float),
                              cudaMemcpyDeviceToHost));

        for (size_t j = 0; j < N; j++) {
            float d = (i == j) ? 0.0f : sqrtf(h_row[j]);
            out_distances[i * N + j] = d;
        }
    }

    free(h_padded);
    delete[] h_row;
    cudaFree(d_subjects);
    cudaFree(d_dist);
    return 0;
}

// --------------------------------------------------------------------------
// Multi-position batched pairwise DTW
// --------------------------------------------------------------------------
int opendba_dtw_multi_position_pairwise(
    const float *all_sequences,
    const size_t *all_seq_lengths,
    const size_t *position_seq_counts,
    size_t num_positions,
    size_t global_max_length,
    int /*use_open_start*/,
    int /*use_open_end*/,
    float *out_distances,
    int /*num_cuda_streams*/,   // ignored — cQuery serialises anyway
    int device_id)
{
    if (!all_sequences || !all_seq_lengths || !position_seq_counts ||
        !out_distances || num_positions == 0 || global_max_length == 0) {
        fprintf(stderr, "cudtw_wrapper: invalid multi-position input\n");
        return -1;
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    // Find per-position actual max lengths and buckets
    size_t seq_offset = 0;
    size_t out_offset = 0;

    for (size_t p = 0; p < num_positions; p++) {
        size_t n = position_seq_counts[p];

        if (n < 2) {
            // Fill diagonal zeros
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < n; j++)
                    out_distances[out_offset + i * n + j] = 0.0f;
            out_offset += n * n;
            seq_offset += n;
            continue;
        }

        // Find max length in this position
        size_t pos_max = 0;
        for (size_t i = 0; i < n; i++) {
            size_t l = all_seq_lengths[seq_offset + i];
            if (l > pos_max) pos_max = l;
        }

        int bucket = select_bucket((int)pos_max);
        if (bucket < 0) {
            fprintf(stderr, "cudtw_wrapper: position %zu max_len %zu > 2047\n",
                    p, pos_max);
            return -1;
        }

        // Pad this position's sequences to bucket
        float *h_padded = (float *)calloc(n * bucket, sizeof(float));
        if (!h_padded) return -1;
        for (size_t i = 0; i < n; i++) {
            size_t slen = all_seq_lengths[seq_offset + i];
            const float *src = all_sequences + (seq_offset + i) * global_max_length;
            memcpy(h_padded + i * bucket, src, slen * sizeof(float));
        }

        // Upload subjects
        float *d_subjects = nullptr, *d_dist = nullptr;
        CUDA_CHECK(cudaMalloc(&d_subjects, n * bucket * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dist, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_subjects, h_padded, n * bucket * sizeof(float),
                              cudaMemcpyHostToDevice));

        float *h_row = new float[n];

        for (size_t i = 0; i < n; i++) {
            CUDA_CHECK(cudaMemcpyToSymbol(cQuery, h_padded + i * bucket,
                                          bucket * sizeof(float)));

            bool ok = FullDTW::dist<value_t, index_t>(
                d_subjects, d_dist, (index_t)n, (index_t)bucket);
            if (!ok) {
                free(h_padded); delete[] h_row;
                cudaFree(d_subjects); cudaFree(d_dist);
                return -1;
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_row, d_dist, n * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            for (size_t j = 0; j < n; j++) {
                float d = (i == j) ? 0.0f : sqrtf(h_row[j]);
                out_distances[out_offset + i * n + j] = d;
            }
        }

        free(h_padded);
        delete[] h_row;
        cudaFree(d_subjects);
        cudaFree(d_dist);

        out_offset += n * n;
        seq_offset += n;
    }

    return 0;
}

// ==========================================================================
// Python C API Bindings  (same signatures as dtw_api.cpp)
// ==========================================================================

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// --- dtw_distance ---
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

    int rc = opendba_dtw_cuda(s1, (size_t)len1, s2, (size_t)len2,
                              use_open_start, use_open_end, &distance);
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
        use_open_start, use_open_end,
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
        use_open_start, use_open_end,
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

    // Convert lengths
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

    // Convert counts
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

    // Compute output size
    size_t total_out = 0;
    for (size_t p = 0; p < num_pos; p++)
        total_out += h_counts[p] * h_counts[p];

    npy_intp out_dim = (npy_intp)total_out;
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(1, &out_dim, NPY_FLOAT32, 0);
    if (!out) { delete[] h_lengths; delete[] h_counts; return NULL; }

    int rc = opendba_dtw_multi_position_pairwise(
        (float *)PyArray_DATA(seq_array), h_lengths, h_counts,
        num_pos, gml, use_open_start, use_open_end,
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

    PyModule_AddIntConstant(m, "__version_major__", 0);
    PyModule_AddIntConstant(m, "__version_minor__", 2);
    PyModule_AddStringConstant(m, "__version__", "0.2.0-cudtw");

    return m;
}
