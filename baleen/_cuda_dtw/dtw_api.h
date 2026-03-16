#ifndef DTW_C_API_H
#define DTW_C_API_H

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief 调用 OpenDBA 原版 CUDA DTW 计算两个序列的距离
     * @param seq1 主机端浮点序列1（float 类型）
     * @param len1 序列1长度
     * @param seq2 主机端浮点序列2（float 类型）
     * @param len2 序列2长度
     * @param use_open_start 是否启用 open start 边界
     * @param use_open_end 是否启用 open end 边界
     * @param out_distance 输出 DTW 距离（主机端，float 类型）
     * @return 0=成功，非0=错误码（1=内存分配失败，2=核函数启动失败，3=数据拷贝失败）
     */
    int opendba_dtw_cuda(
        const float *seq1, size_t len1,
        const float *seq2, size_t len2,
        int use_open_start,
        int use_open_end,
        float *out_distance);

    /**
     * @brief Compute pairwise DTW distances for a batch of sequences (all same length)
     * @param sequences Flattened array of sequences (num_sequences * seq_length floats)
     * @param num_sequences Number of sequences
     * @param seq_length Length of each sequence (all must be same length)
     * @param use_open_start Whether to use open start boundary
     * @param use_open_end Whether to use open end boundary
     * @param out_distances Output pairwise distance matrix (num_sequences * num_sequences floats)
     * @return 0=success, non-zero=error
     */
    int opendba_dtw_pairwise_batch(
        const float *sequences,
        size_t num_sequences,
        size_t seq_length,
        int use_open_start,
        int use_open_end,
        float *out_distances);

    /**
     * @brief 清理 CUDA 资源
     */
    void opendba_dtw_cleanup();

#ifdef __cplusplus
}
#endif

#endif // DTW_C_API_H