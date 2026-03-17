#include "dtw_api.h"
#include "dtw.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <chrono>

// Define CUDA_CHECK macro for error checking
#define CUDA_CHECK(call)                                                          \
    do                                                                            \
    {                                                                             \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess)                                                   \
        {                                                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl;                    \
            return -1;                                                            \
        }                                                                         \
    } while (0)

// 复用 OpenDBA 的核函数启动逻辑，仅适配单对序列计算
int opendba_dtw_cuda(
    const float *seq1, size_t len1,
    const float *seq2, size_t len2,
    int use_open_start,
    int use_open_end,
    float *out_distance)
{
    // 1. 输入校验
    if (!seq1 || !seq2 || !out_distance || len1 == 0 || len2 == 0)
    {
        fprintf(stderr, "Invalid input parameters\n");
        return -1;
    }

    // 2. 设备内存分配（复用 OpenDBA 的内存对齐逻辑）
    float *d_seq1, *d_seq2, *d_dtw_cost, *d_new_dtw_cost;
    unsigned char *d_path_matrix;
    float *d_pairwise_dist;
    size_t path_mem_pitch;

    // 分配序列内存
    CUDA_CHECK(cudaMalloc(&d_seq1, len1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_seq2, len2 * sizeof(float)));
    // 分配 DTW 计算所需临时内存
    // CRITICAL: d_dtw_cost and d_new_dtw_cost must be sized by len1 (first_seq_length)
    // because the kernel indexes them up to first_seq_length-1
    CUDA_CHECK(cudaMalloc(&d_dtw_cost, len1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_dtw_cost, len1 * sizeof(float)));
    CUDA_CHECK(cudaMallocPitch(&d_path_matrix, &path_mem_pitch, len2 * sizeof(unsigned char), len1));
    CUDA_CHECK(cudaMalloc(&d_pairwise_dist, sizeof(float)));

    // 3. 主机→设备数据拷贝
    CUDA_CHECK(cudaMemcpy(d_seq1, seq1, len1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq2, seq2, len2 * sizeof(float), cudaMemcpyHostToDevice));
    // 初始化临时内存（must match allocation sizes above)
    CUDA_CHECK(cudaMemset(d_dtw_cost, 0, len1 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_new_dtw_cost, 0, len1 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_path_matrix, 0, path_mem_pitch * len1));
    CUDA_CHECK(cudaMemset(d_pairwise_dist, 0, sizeof(float)));

    // 4. 启动 OpenDBA 原版 DTW 核函数（参数严格对齐）
    // Get device properties to determine thread count
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int max_threads = deviceProp.maxThreadsPerBlock;

    dim3 thread_block(max_threads, 1, 1);
    size_t shared_mem = thread_block.x * 3 * sizeof(float); // 复用 OpenDBA 的共享内存计算

    // CRITICAL: The wavefront algorithm processes the second sequence in chunks of blockDim.x
    // We must call the kernel multiple times, advancing offset_within_second_seq each time
    float *d_current_cost = d_dtw_cost;
    float *d_next_cost = d_new_dtw_cost;

    for (size_t offset = 0; offset < len2; offset += max_threads)
    {
        DTWDistance<float><<<1, thread_block, shared_mem>>>(
            d_seq1, len1,
            d_seq2, len2,
            0, offset,                    // Advance offset through second sequence
            (const float *)nullptr, 0, 0, // 多序列相关参数置空/0
            (const size_t *)nullptr,
            d_current_cost,
            d_next_cost,
            d_path_matrix,
            path_mem_pitch,
            d_pairwise_dist,
            use_open_start,
            use_open_end);
        CUDA_CHECK(cudaGetLastError()); // 检查核函数启动错误

        // Swap buffers for next iteration
        float *temp = d_current_cost;
        d_current_cost = d_next_cost;
        d_next_cost = temp;
    }

    CUDA_CHECK(cudaDeviceSynchronize()); // 等待所有核函数执行完成

    // 5. 设备→主机拷贝结果
    // After the loop, d_current_cost contains the final result
    // (kernel writes to d_next_cost, then we swap, so result is in d_current_cost after swap)
    float final_cost;
    CUDA_CHECK(cudaMemcpy(&final_cost, &d_current_cost[len1 - 1], sizeof(float), cudaMemcpyDeviceToHost));

    // Apply same normalization/distance calculation as the kernel would
    // Note: The kernel computes squared differences, so we take sqrt here
    if ((use_open_end && !use_open_start) || (!use_open_end && use_open_start))
    {
        *out_distance = sqrtf(final_cost) / len1; // Normalized by sequence length
    }
    else
    {
        *out_distance = sqrtf(final_cost); // Raw distance
    }

    // 6. 释放设备内存
    CUDA_CHECK(cudaFree(d_seq1));
    CUDA_CHECK(cudaFree(d_seq2));
    CUDA_CHECK(cudaFree(d_dtw_cost));
    CUDA_CHECK(cudaFree(d_new_dtw_cost));
    CUDA_CHECK(cudaFree(d_path_matrix));
    CUDA_CHECK(cudaFree(d_pairwise_dist));

    return 0;
}

void opendba_dtw_cleanup()
{
    cudaDeviceReset();
}

// ============================================================================
// Batch Pairwise DTW
// ============================================================================

int opendba_dtw_pairwise_batch(
    const float *sequences,
    size_t num_sequences,
    size_t seq_length,
    int use_open_start,
    int use_open_end,
    float *out_distances)
{
    if (!sequences || !out_distances || num_sequences < 2 || seq_length == 0)
    {
        fprintf(stderr, "Invalid input parameters for batch DTW\n");
        return -1;
    }

    // Allocate GPU memory for all sequences
    float *d_sequences;
    size_t *d_seq_lengths;
    float *d_distances;

    size_t total_seq_size = num_sequences * seq_length * sizeof(float);
    size_t num_pairs = (num_sequences * (num_sequences - 1)) / 2; // Upper triangle

    CUDA_CHECK(cudaMalloc(&d_sequences, total_seq_size));
    CUDA_CHECK(cudaMalloc(&d_seq_lengths, num_sequences * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_distances, num_pairs * sizeof(float)));

    // Copy sequences to GPU
    CUDA_CHECK(cudaMemcpy(d_sequences, sequences, total_seq_size, cudaMemcpyHostToDevice));

    // Initialize sequence lengths (all same)
    size_t *h_seq_lengths = new size_t[num_sequences];
    for (size_t i = 0; i < num_sequences; i++)
    {
        h_seq_lengths[i] = seq_length;
    }
    CUDA_CHECK(cudaMemcpy(d_seq_lengths, h_seq_lengths, num_sequences * sizeof(size_t), cudaMemcpyHostToDevice));
    delete[] h_seq_lengths;

    // Allocate temporary buffers for DTW computation
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int max_threads = deviceProp.maxThreadsPerBlock;

    // Query available GPU memory
    size_t free_memory, total_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));

    // Maximum pairs we'll compute in parallel (when processing first sequence)
    size_t max_pairs_parallel = num_sequences - 1;

    // Memory required for cost buffers: seq_length * max_pairs_parallel * 2 * sizeof(float)
    // Example: 1000 len × 100 pairs × 8 bytes = 800 KB (reasonable)
    // With 20GB GPU, you could do: 10000 len × 1000 pairs × 8 bytes = 80 MB (still tiny!)

    size_t cost_buffer_size = seq_length * max_pairs_parallel * sizeof(float);
    size_t total_temp_memory = cost_buffer_size * 2; // Two buffers

    if (getenv("DTW_DEBUG"))
    {
        fprintf(stderr, "=== DTW Batch Pairwise Memory Usage ===\n");
        fprintf(stderr, "GPU: %.2f GB total, %.2f GB free\n",
                total_memory / 1024.0 / 1024.0 / 1024.0,
                free_memory / 1024.0 / 1024.0 / 1024.0);
        fprintf(stderr, "Input sequences: %.2f MB (%zu × %zu)\n",
                total_seq_size / 1024.0 / 1024.0, num_sequences, seq_length);
        fprintf(stderr, "Cost buffers: %.2f MB\n", total_temp_memory / 1024.0 / 1024.0);
        fprintf(stderr, "Output distances: %.2f KB (%zu pairs)\n",
                num_pairs * sizeof(float) / 1024.0, num_pairs);
        fprintf(stderr, "Total GPU memory used: %.2f MB (%.1f%% of free memory)\n",
                (total_seq_size + total_temp_memory + num_pairs * sizeof(float)) / 1024.0 / 1024.0,
                100.0 * (total_seq_size + total_temp_memory) / free_memory);
        fprintf(stderr, "\nNote: Low memory usage is by design for stability.\n");
        fprintf(stderr, "      With your GPU, you could process:\n");
        fprintf(stderr, "      - 1000 sequences × 10000 length (~800 MB)\n");
        fprintf(stderr, "      - 5000 sequences × 2000 length (~800 MB)\n");
        fprintf(stderr, "=======================================\n");
    }

    float *d_dtw_cost, *d_new_dtw_cost;

    // Allocate cost buffers: seq_length per parallel pair
    CUDA_CHECK(cudaMalloc(&d_dtw_cost, seq_length * max_pairs_parallel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_dtw_cost, seq_length * max_pairs_parallel * sizeof(float)));

    // Don't allocate path matrix for pairwise - we only need distances, not alignments
    // This saves HUGE amounts of memory (would be seq_length² per pair!)
    unsigned char *d_path_matrix = nullptr;
    size_t path_mem_pitch = 0;

    // Compute pairwise distances
    dim3 thread_block(max_threads, 1, 1);
    size_t shared_mem = thread_block.x * 3 * sizeof(float);

    if (getenv("DTW_DEBUG"))
    {
        fprintf(stderr, "\n=== Starting DTW Pairwise Computation ===\n");
        fprintf(stderr, "Total pairs to compute: %zu\n", num_pairs);
        fprintf(stderr, "Wavefront chunks per sequence: %zu\n", (seq_length + max_threads - 1) / max_threads);
        fprintf(stderr, "=========================================\n\n");
    }

    size_t total_pairs_completed = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Process each reference sequence
    for (size_t i = 0; i < num_sequences - 1; i++)
    {
        size_t num_comparisons = num_sequences - i - 1;

        // Initialize buffers for all comparisons with this reference sequence
        CUDA_CHECK(cudaMemset(d_dtw_cost, 0, seq_length * num_comparisons * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_new_dtw_cost, 0, seq_length * num_comparisons * sizeof(float)));

        float *d_current_cost = d_dtw_cost;
        float *d_next_cost = d_new_dtw_cost;

        // Process sequence in wavefront chunks
        size_t num_chunks = (seq_length + max_threads - 1) / max_threads;
        for (size_t chunk_idx = 0, offset = 0; offset < seq_length; chunk_idx++, offset += max_threads)
        {
            // Detailed progress for very long sequences
            if (getenv("DTW_DEBUG") && seq_length > 10000 && chunk_idx % 100 == 0)
            {
                fprintf(stderr, "  [Seq %zu] Chunk %zu/%zu (offset %zu/%zu)\n",
                        i, chunk_idx, num_chunks, offset, seq_length);
                fflush(stderr);
            }

            // Launch kernel with num_comparisons blocks to compute all pairs with sequence i
            DTWDistance<float><<<num_comparisons, thread_block, shared_mem>>>(
                nullptr, seq_length, // Don't pass individual seqs, use batch arrays
                nullptr, seq_length,
                i, offset, // Reference sequence index and offset
                d_sequences, seq_length, num_sequences,
                d_seq_lengths,
                d_current_cost,
                d_next_cost,
                d_path_matrix,
                path_mem_pitch,
                d_distances,
                use_open_start,
                use_open_end);
            CUDA_CHECK(cudaGetLastError());

            // Swap buffers
            float *temp = d_current_cost;
            d_current_cost = d_next_cost;
            d_next_cost = temp;
        }

        // Update completed pairs count
        total_pairs_completed += num_comparisons;

        // Synchronize and report progress AFTER GPU work is done
        if (i % 10 == 0 || getenv("DTW_DEBUG"))
        {
            CUDA_CHECK(cudaDeviceSynchronize()); // Wait for GPU to finish

            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

            if (elapsed > 0) // Avoid division by zero
            {
                float progress = 100.0 * total_pairs_completed / num_pairs;
                float pairs_per_sec = total_pairs_completed / (elapsed / 1000.0);
                size_t remaining_pairs = num_pairs - total_pairs_completed;
                float eta_sec = remaining_pairs / pairs_per_sec;

                fprintf(stderr, "[Progress] Ref seq %3zu/%zu | Completed: %6zu/%zu pairs (%.1f%%) | "
                                "Speed: %.1f pairs/sec | ETA: %.1f sec\n",
                        i + 1, num_sequences - 1, total_pairs_completed, num_pairs, progress,
                        pairs_per_sec, eta_sec);
                fflush(stderr);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    if (getenv("DTW_DEBUG") || true) // Always show completion
    {
        fprintf(stderr, "\n[Complete] Computed %zu pairs in %.2f seconds (%.1f pairs/sec)\n",
                num_pairs, total_elapsed / 1000.0, num_pairs / (total_elapsed / 1000.0));
        fflush(stderr);
    }

    // Copy results back (upper triangle format)
    float *h_upper_triangle = new float[num_pairs];
    CUDA_CHECK(cudaMemcpy(h_upper_triangle, d_distances, num_pairs * sizeof(float), cudaMemcpyDeviceToHost));

    // Convert upper triangle to full symmetric matrix
    size_t pair_idx = 0;
    for (size_t i = 0; i < num_sequences; i++)
    {
        out_distances[i * num_sequences + i] = 0.0f; // Diagonal
        for (size_t j = i + 1; j < num_sequences; j++)
        {
            float dist = h_upper_triangle[pair_idx++];
            out_distances[i * num_sequences + j] = dist;
            out_distances[j * num_sequences + i] = dist; // Symmetric
        }
    }
    delete[] h_upper_triangle;

    // Cleanup
    CUDA_CHECK(cudaFree(d_sequences));
    CUDA_CHECK(cudaFree(d_seq_lengths));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_dtw_cost));
    CUDA_CHECK(cudaFree(d_new_dtw_cost));
    // d_path_matrix is nullptr, don't free it

    return 0;
}

// ============================================================================
// Variable-length Batch Pairwise DTW
// ============================================================================

int opendba_dtw_pairwise_varlen(
    const float *sequences,
    const size_t *seq_lengths,
    size_t num_sequences,
    size_t max_length,
    int use_open_start,
    int use_open_end,
    float *out_distances)
{
    if (!sequences || !seq_lengths || !out_distances || num_sequences < 2 || max_length == 0)
    {
        fprintf(stderr, "Invalid input parameters for varlen batch DTW\n");
        return -1;
    }

    float *d_sequences;
    size_t *d_seq_lengths;
    float *d_distances;

    size_t total_seq_size = num_sequences * max_length * sizeof(float);
    size_t num_pairs = (num_sequences * (num_sequences - 1)) / 2;

    CUDA_CHECK(cudaMalloc(&d_sequences, total_seq_size));
    CUDA_CHECK(cudaMalloc(&d_seq_lengths, num_sequences * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_distances, num_pairs * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_sequences, sequences, total_seq_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lengths, seq_lengths, num_sequences * sizeof(size_t), cudaMemcpyHostToDevice));

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int max_threads = deviceProp.maxThreadsPerBlock;

    size_t max_pairs_parallel = num_sequences - 1;

    float *d_dtw_cost, *d_new_dtw_cost;

    // Cost buffers use max_length for stride so all blocks index correctly
    // (the kernel indexes as dtwCostSoFar[first_seq_length * blockIdx.x],
    //  but first_seq_length varies per reference — we use max_length to be safe)
    CUDA_CHECK(cudaMalloc(&d_dtw_cost, max_length * max_pairs_parallel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_dtw_cost, max_length * max_pairs_parallel * sizeof(float)));

    unsigned char *d_path_matrix = nullptr;
    size_t path_mem_pitch = 0;

    dim3 thread_block(max_threads, 1, 1);
    size_t shared_mem = thread_block.x * 3 * sizeof(float);

    for (size_t i = 0; i < num_sequences - 1; i++)
    {
        size_t num_comparisons = num_sequences - i - 1;

        CUDA_CHECK(cudaMemset(d_dtw_cost, 0, max_length * num_comparisons * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_new_dtw_cost, 0, max_length * num_comparisons * sizeof(float)));

        float *d_current_cost = d_dtw_cost;
        float *d_next_cost = d_new_dtw_cost;

        // Iterate over wavefront chunks up to max_length;
        // shorter second sequences exit early inside the kernel
        for (size_t offset = 0; offset < max_length; offset += max_threads)
        {
            DTWDistance<float><<<num_comparisons, thread_block, shared_mem>>>(
                nullptr, max_length,
                nullptr, max_length,
                i, offset,
                d_sequences, max_length, num_sequences,
                d_seq_lengths,
                d_current_cost,
                d_next_cost,
                d_path_matrix,
                path_mem_pitch,
                d_distances,
                use_open_start,
                use_open_end);
            CUDA_CHECK(cudaGetLastError());

            float *temp = d_current_cost;
            d_current_cost = d_next_cost;
            d_next_cost = temp;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_upper_triangle = new float[num_pairs];
    CUDA_CHECK(cudaMemcpy(h_upper_triangle, d_distances, num_pairs * sizeof(float), cudaMemcpyDeviceToHost));

    size_t pair_idx = 0;
    for (size_t i = 0; i < num_sequences; i++)
    {
        out_distances[i * num_sequences + i] = 0.0f;
        for (size_t j = i + 1; j < num_sequences; j++)
        {
            float dist = h_upper_triangle[pair_idx++];
            out_distances[i * num_sequences + j] = dist;
            out_distances[j * num_sequences + i] = dist;
        }
    }
    delete[] h_upper_triangle;

    CUDA_CHECK(cudaFree(d_sequences));
    CUDA_CHECK(cudaFree(d_seq_lengths));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_dtw_cost));
    CUDA_CHECK(cudaFree(d_new_dtw_cost));

    return 0;
}

// ============================================================================
// Python C API Bindings
// ============================================================================

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/**
 * Python wrapper for opendba_dtw_cuda
 *
 * Args:
 *     seq1: numpy array of floats (1D)
 *     seq2: numpy array of floats (1D)
 *     use_open_start: boolean (default False)
 *     use_open_end: boolean (default False)
 *
 * Returns:
 *     float: DTW distance
 */
static PyObject *py_dtw_cuda(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *seq1_array = NULL, *seq2_array = NULL;
    int use_open_start = 0;
    int use_open_end = 0;

    static char *kwlist[] = {(char *)"seq1", (char *)"seq2",
                             (char *)"use_open_start", (char *)"use_open_end", NULL};

    // Parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|ii", kwlist,
                                     &PyArray_Type, &seq1_array,
                                     &PyArray_Type, &seq2_array,
                                     &use_open_start, &use_open_end))
    {
        return NULL;
    }

    // Validate input arrays
    if (PyArray_NDIM(seq1_array) != 1 || PyArray_NDIM(seq2_array) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be 1-dimensional");
        return NULL;
    }

    if (PyArray_TYPE(seq1_array) != NPY_FLOAT32 || PyArray_TYPE(seq2_array) != NPY_FLOAT32)
    {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be of type float32");
        return NULL;
    }

    // Get array dimensions and data
    npy_intp len1 = PyArray_DIM(seq1_array, 0);
    npy_intp len2 = PyArray_DIM(seq2_array, 0);

    if (len1 == 0 || len2 == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Input arrays cannot be empty");
        return NULL;
    }

    // Get pointers to array data
    float *seq1_data = (float *)PyArray_DATA(seq1_array);
    float *seq2_data = (float *)PyArray_DATA(seq2_array);

    // Allocate output
    float distance = 0.0f;

    // Call CUDA function
    int result = opendba_dtw_cuda(
        seq1_data, (size_t)len1,
        seq2_data, (size_t)len2,
        use_open_start,
        use_open_end,
        &distance);

    if (result != 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "CUDA DTW computation failed");
        return NULL;
    }

    // Return the distance as a Python float
    return PyFloat_FromDouble((double)distance);
}

/**
 * Python wrapper for batch pairwise DTW
 */
static PyObject *py_dtw_pairwise(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *sequences_array;
    int use_open_start = 0;
    int use_open_end = 0;

    static char *kwlist[] = {(char *)"sequences", (char *)"use_open_start", (char *)"use_open_end", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|ii", kwlist,
                                     &PyArray_Type, &sequences_array,
                                     &use_open_start, &use_open_end))
    {
        return NULL;
    }

    // Validate input array
    if (PyArray_NDIM(sequences_array) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "sequences must be a 2D array (num_sequences, seq_length)");
        return NULL;
    }

    if (PyArray_TYPE(sequences_array) != NPY_FLOAT32)
    {
        PyErr_SetString(PyExc_TypeError, "sequences must be float32 dtype");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(sequences_array);
    size_t num_sequences = (size_t)dims[0];
    size_t seq_length = (size_t)dims[1];

    if (num_sequences < 2)
    {
        PyErr_SetString(PyExc_ValueError, "Need at least 2 sequences");
        return NULL;
    }

    float *sequences_data = (float *)PyArray_DATA(sequences_array);

    // Allocate output distance matrix
    npy_intp out_dims[2] = {(npy_intp)num_sequences, (npy_intp)num_sequences};
    PyArrayObject *distance_matrix = (PyArrayObject *)PyArray_ZEROS(2, out_dims, NPY_FLOAT32, 0);
    if (distance_matrix == NULL)
    {
        return NULL;
    }

    float *distances_data = (float *)PyArray_DATA(distance_matrix);

    // Call CUDA function
    int result = opendba_dtw_pairwise_batch(
        sequences_data, num_sequences, seq_length,
        use_open_start, use_open_end,
        distances_data);

    if (result != 0)
    {
        Py_DECREF(distance_matrix);
        PyErr_SetString(PyExc_RuntimeError, "CUDA batch DTW computation failed");
        return NULL;
    }

    return (PyObject *)distance_matrix;
}

static PyObject *py_dtw_pairwise_varlen(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *sequences_array;
    PyArrayObject *lengths_array;
    int use_open_start = 0;
    int use_open_end = 0;

    static char *kwlist[] = {(char *)"sequences", (char *)"lengths",
                             (char *)"use_open_start", (char *)"use_open_end", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|ii", kwlist,
                                     &PyArray_Type, &sequences_array,
                                     &PyArray_Type, &lengths_array,
                                     &use_open_start, &use_open_end))
    {
        return NULL;
    }

    if (PyArray_NDIM(sequences_array) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "sequences must be a 2D array (num_sequences, max_length)");
        return NULL;
    }

    if (PyArray_TYPE(sequences_array) != NPY_FLOAT32)
    {
        PyErr_SetString(PyExc_TypeError, "sequences must be float32 dtype");
        return NULL;
    }

    if (PyArray_NDIM(lengths_array) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "lengths must be a 1D array");
        return NULL;
    }

    npy_intp *seq_dims = PyArray_DIMS(sequences_array);
    size_t num_sequences = (size_t)seq_dims[0];
    size_t max_length = (size_t)seq_dims[1];

    if ((size_t)PyArray_DIM(lengths_array, 0) != num_sequences)
    {
        PyErr_SetString(PyExc_ValueError, "lengths array size must match num_sequences");
        return NULL;
    }

    if (num_sequences < 2)
    {
        PyErr_SetString(PyExc_ValueError, "Need at least 2 sequences");
        return NULL;
    }

    size_t *h_lengths = new size_t[num_sequences];
    for (size_t i = 0; i < num_sequences; i++)
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
        if (val <= 0 || (size_t)val > max_length)
        {
            delete[] h_lengths;
            PyErr_Format(PyExc_ValueError,
                         "length[%zu]=%lld out of range (1..%zu)", i, val, max_length);
            return NULL;
        }
        h_lengths[i] = (size_t)val;
    }

    float *sequences_data = (float *)PyArray_DATA(sequences_array);

    npy_intp out_dims[2] = {(npy_intp)num_sequences, (npy_intp)num_sequences};
    PyArrayObject *distance_matrix = (PyArrayObject *)PyArray_ZEROS(2, out_dims, NPY_FLOAT32, 0);
    if (distance_matrix == NULL)
    {
        delete[] h_lengths;
        return NULL;
    }

    float *distances_data = (float *)PyArray_DATA(distance_matrix);

    int result = opendba_dtw_pairwise_varlen(
        sequences_data, h_lengths, num_sequences, max_length,
        use_open_start, use_open_end,
        distances_data);

    delete[] h_lengths;

    if (result != 0)
    {
        Py_DECREF(distance_matrix);
        PyErr_SetString(PyExc_RuntimeError, "CUDA varlen batch DTW computation failed");
        return NULL;
    }

    return (PyObject *)distance_matrix;
}

/**
 * Python wrapper for opendba_dtw_cleanup
 */
static PyObject *py_dtw_cleanup(PyObject *self, PyObject *args)
{
    opendba_dtw_cleanup();
    Py_RETURN_NONE;
}

// Method definitions
static PyMethodDef DtwMethods[] = {
    {"dtw_distance", (PyCFunction)py_dtw_cuda, METH_VARARGS | METH_KEYWORDS,
     "Compute DTW distance between two sequences using CUDA.\n\n"
     "Parameters\n"
     "----------\n"
     "seq1 : np.ndarray\n"
     "    First sequence (1D float32 array)\n"
     "seq2 : np.ndarray\n"
     "    Second sequence (1D float32 array)\n"
     "use_open_start : bool, optional\n"
     "    Enable open start boundary (default: False)\n"
     "use_open_end : bool, optional\n"
     "    Enable open end boundary (default: False)\n\n"
     "Returns\n"
     "-------\n"
     "float\n"
     "    DTW distance between seq1 and seq2\n"},
    {"dtw_pairwise", (PyCFunction)py_dtw_pairwise, METH_VARARGS | METH_KEYWORDS,
     "Compute pairwise DTW distances for a batch of sequences using CUDA.\n\n"
     "This is much more efficient than computing distances one-by-one,\n"
     "as it amortizes GPU memory transfer overhead over many computations.\n\n"
     "Parameters\n"
     "----------\n"
     "sequences : np.ndarray\n"
     "    2D array of sequences (num_sequences, seq_length) in float32\n"
     "    All sequences must have the same length\n"
     "use_open_start : bool, optional\n"
     "    Enable open start boundary (default: False)\n"
     "use_open_end : bool, optional\n"
     "    Enable open end boundary (default: False)\n\n"
     "Returns\n"
     "-------\n"
     "np.ndarray\n"
     "    Distance matrix (num_sequences, num_sequences) with DTW distances\n"
     "    Matrix is symmetric with zeros on diagonal\n"},
    {"dtw_pairwise_varlen", (PyCFunction)py_dtw_pairwise_varlen, METH_VARARGS | METH_KEYWORDS,
     "Compute pairwise DTW distances for variable-length sequences using CUDA.\n\n"
     "Parameters\n"
     "----------\n"
     "sequences : np.ndarray\n"
     "    2D padded array (num_sequences, max_length) in float32\n"
     "lengths : np.ndarray\n"
     "    1D array of actual sequence lengths (int32 or int64)\n"
     "use_open_start : bool, optional\n"
     "    Enable open start boundary (default: False)\n"
     "use_open_end : bool, optional\n"
     "    Enable open end boundary (default: False)\n\n"
     "Returns\n"
     "-------\n"
     "np.ndarray\n"
     "    Distance matrix (num_sequences, num_sequences) with DTW distances\n"},
    {"cleanup", py_dtw_cleanup, METH_NOARGS,
     "Reset CUDA device and free all resources.\n\n"
     "This should be called when done using CUDA DTW to free GPU resources.\n"},
    {NULL, NULL, 0, NULL} // Sentinel
};

// Module definition
static struct PyModuleDef dtwmodule = {
    PyModuleDef_HEAD_INIT,
    "_cuda_dtw",
    "CUDA-accelerated Dynamic Time Warping (DTW) computation\n\n"
    "This module provides GPU-accelerated DTW distance calculation using CUDA.\n"
    "It supports open start and open end boundary conditions.\n",
    -1,
    DtwMethods};

// Module initialization function
PyMODINIT_FUNC PyInit__cuda_dtw(void)
{
    // Import NumPy API
    import_array();
    if (PyErr_Occurred())
    {
        return NULL;
    }

    PyObject *module = PyModule_Create(&dtwmodule);
    if (module == NULL)
    {
        return NULL;
    }

    // Add module-level constants
    PyModule_AddIntConstant(module, "__version_major__", 0);
    PyModule_AddIntConstant(module, "__version_minor__", 1);
    PyModule_AddStringConstant(module, "__version__", "0.1.0");

    return module;
}