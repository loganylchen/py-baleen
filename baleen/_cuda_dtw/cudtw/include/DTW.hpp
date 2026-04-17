#ifndef CUDTW_DTW_HPP
#define CUDTW_DTW_HPP

// Simplified cuDTW++ dispatcher for baleen.
// Only database-mode (query_type==0), non-subwarp kernels.
// Query is passed as a global-memory pointer (no constant memory).

#include "./kernels/SHFL_FULLDTW_127.cuh"
#include "./kernels/SHFL_FULLDTW_255.cuh"
#include "./kernels/SHFL_FULLDTW_511.cuh"
#include "./kernels/SHFL_FULLDTW_1023.cuh"
#include "./kernels/SHFL_FULLDTW_2047.cuh"

namespace FullDTW {

template <typename value_t, typename index_t>
__host__
bool dist(
    const value_t *Query,
    value_t *Subject,
    value_t *Dist,
    index_t num_entries,
    index_t num_features,
    cudaStream_t stream = 0)
{
    const dim3 grid(num_entries, 1, 1);
    const dim3 block(32, 1, 1);

    if (num_features == 127) {
        shfl_FullDTW_127<<<grid, block, 0, stream>>>(
            Query, Subject, Dist, num_entries, num_features);
        return true;
    }
    if (num_features == 255) {
        shfl_FullDTW_255<<<grid, block, 0, stream>>>(
            Query, Subject, Dist, num_entries, num_features);
        return true;
    }
    if (num_features == 511) {
        shfl_FullDTW_511<<<grid, block, 0, stream>>>(
            Query, Subject, Dist, num_entries, num_features);
        return true;
    }
    if (num_features == 1023) {
        shfl_FullDTW_1023<<<grid, block, 0, stream>>>(
            Query, Subject, Dist, num_entries, num_features);
        return true;
    }
    if (num_features == 2047) {
        shfl_FullDTW_2047<<<grid, block, 0, stream>>>(
            Query, Subject, Dist, num_entries, num_features);
        return true;
    }

    return false;  // unsupported length
}

} // namespace FullDTW

#endif
