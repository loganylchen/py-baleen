"""
Accuracy of the cuDTW++ backend against the CPU reference.

The kernels compute full DTW over a fixed width, so everything here is about
the two places the GPU result can drift from true variable-length DTW: how
signals are padded/bucketed, and the hand-unrolled recurrences themselves.
"""

import numpy as np
import pytest

from baleen._cuda_dtw import (
    CUDA_AVAILABLE,
    _CUDTW_ACTIVE,
    _fit_to_kernel,
    _select_bucket,
    dtw_distance,
    dtw_multi_position_pairwise,
    dtw_pairwise,
    dtw_pairwise_varlen,
    estimate_gpu_memory,
)

BUCKETS = (127, 255, 511, 1023, 2047)

needs_cudtw = pytest.mark.skipif(
    not (CUDA_AVAILABLE and _CUDTW_ACTIVE), reason="cuDTW++ backend not available"
)


def _cpu_matrix(signals):
    n = len(signals)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = mat[j, i] = dtw_distance(signals[i], signals[j], use_cuda=False)
    return mat


def _signals(rng, lengths, offset):
    return [(offset + 8 * rng.standard_normal(int(n))).astype(np.float32) for n in lengths]


# ---------------------------------------------------------------------------
# Bucketing rules (pure Python, no GPU needed)
# ---------------------------------------------------------------------------

class TestBucketSelection:
    @pytest.mark.parametrize("min_len, max_len, expected", [
        (100, 100, 127),
        (127, 127, 127),      # equal lengths may fill the width
        (100, 126, 127),
        (100, 127, 255),      # mixed lengths need a spare padding slot
        (10, 1023, 2047),
        (2047, 2047, 2047),
        (10, 2046, 2047),
    ])
    def test_select_bucket(self, min_len, max_len, expected):
        assert _select_bucket(min_len, max_len) == expected

    def test_fit_leaves_fitting_groups_untouched(self):
        equal = [np.zeros(2047, np.float32) for _ in range(3)]
        mixed = [np.zeros(n, np.float32) for n in (50, 2046)]
        assert _fit_to_kernel(equal) is equal
        assert _fit_to_kernel(mixed) is mixed

    @pytest.mark.parametrize("lengths, expected", [
        ((50, 2047), [50, 2046]),
        ((50, 5000), [50, 2046]),
        ((2047, 3000), [2046, 2046]),
        ((3000, 3000), [2047, 2047]),   # equal lengths keep the full width
    ])
    def test_fit_caps_overlong_signals(self, lengths, expected):
        fitted = _fit_to_kernel([np.ones(n, np.float32) for n in lengths])
        assert [len(s) for s in fitted] == expected

    def test_estimate_uses_per_position_bucket(self):
        short = [np.zeros(100, np.float32)] * 4          # bucket 127
        long = [np.zeros(n, np.float32) for n in (600, 900)]  # bucket 1023
        expected = (4 * 127 + 2 * 1023) * 4 + (16 + 4) * 4 + 6 * 8
        assert estimate_gpu_memory([short, long]) == int(expected * 1.2)


# ---------------------------------------------------------------------------
# Variable-length groups must equal true DTW, in both matrix orientations
# ---------------------------------------------------------------------------

@needs_cudtw
class TestVariableLengthMatchesCPU:
    # offset 90 looks like pA-scale signal; offset 0 puts samples next to any
    # zero padding, which is where a padding value of 0 goes wrong.
    @pytest.mark.parametrize("offset", [0.0, 90.0])
    @pytest.mark.parametrize("bucket", BUCKETS)
    def test_ragged_group(self, bucket, offset):
        rng = np.random.default_rng(bucket)
        sigs = _signals(rng, rng.integers(max(2, bucket // 20), bucket, size=7), offset)
        gpu = dtw_multi_position_pairwise([sigs], use_cuda=True)[0]
        np.testing.assert_allclose(gpu, _cpu_matrix(sigs), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("offset", [0.0, 90.0])
    @pytest.mark.parametrize("bucket", BUCKETS[:-1])
    def test_full_width_signal_among_shorter(self, bucket, offset):
        rng = np.random.default_rng(bucket + 1)
        sigs = _signals(rng, [bucket, bucket // 3, bucket // 2, bucket - 1, bucket], offset)
        gpu = dtw_multi_position_pairwise([sigs], use_cuda=True)[0]
        np.testing.assert_allclose(gpu, _cpu_matrix(sigs), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("long_len", [2047, 3500])
    def test_overlong_signal_among_shorter(self, long_len):
        """Signals past the widest kernel are resampled; nothing else may change."""
        rng = np.random.default_rng(long_len)
        sigs = _signals(rng, [long_len, 80, 150, 400], 90.0)
        gpu = dtw_multi_position_pairwise([sigs], use_cuda=True)[0]
        np.testing.assert_allclose(gpu, _cpu_matrix(_fit_to_kernel(sigs)), rtol=1e-4, atol=1e-4)
        # The short signals' own block is exact DTW of the originals.
        np.testing.assert_allclose(gpu[1:, 1:], _cpu_matrix(sigs[1:]), rtol=1e-4, atol=1e-4)

    def test_positions_are_bucketed_independently(self):
        rng = np.random.default_rng(7)
        positions = [
            _signals(rng, [127, 40, 90], 90.0),
            _signals(rng, [1500, 300], 90.0),
            _signals(rng, [60], 90.0),
            _signals(rng, [255, 255, 255], 90.0),
        ]
        gpu = dtw_multi_position_pairwise(positions, use_cuda=True)
        for g, sigs in zip(gpu, positions):
            np.testing.assert_allclose(g, _cpu_matrix(sigs), rtol=1e-4, atol=1e-4)

    def test_varlen_and_single_pair_entry_points(self):
        rng = np.random.default_rng(11)
        sigs = _signals(rng, [255, 70, 180], 0.0)
        cpu = _cpu_matrix(sigs)
        np.testing.assert_allclose(dtw_pairwise_varlen(sigs, use_cuda=True), cpu, rtol=1e-4, atol=1e-4)
        for i, j in ((0, 1), (1, 0), (1, 2)):
            assert dtw_distance(sigs[i], sigs[j], use_cuda=True) == pytest.approx(cpu[i, j], rel=1e-4)

    def test_padding_is_never_a_cheap_match(self):
        """Samples far from 0 or huge in magnitude must still pay for every alignment."""
        a = np.array([1e9, 1e9], dtype=np.float32)
        b = np.array([0.0], dtype=np.float32)
        expected = np.sqrt(2.0) * 1e9
        assert dtw_distance(a, b, use_cuda=True) == pytest.approx(expected, rel=1e-4)
        assert dtw_distance(b, a, use_cuda=True) == pytest.approx(expected, rel=1e-4)
        mat = dtw_pairwise_varlen([a, b, -a], use_cuda=True)
        np.testing.assert_allclose(mat, _cpu_matrix([a, b, -a]), rtol=1e-4)

    def test_single_pair_resamples_overlong(self):
        rng = np.random.default_rng(13)
        a, b = _signals(rng, [3000, 200], 90.0)
        expected = dtw_distance(*_fit_to_kernel([a, b]), use_cuda=False)
        assert dtw_distance(a, b, use_cuda=True) == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Kernel recurrences: inputs whose optimal path runs through the cells that
# the wide kernels hand between threads / compute in their final step
# ---------------------------------------------------------------------------

@needs_cudtw
class TestKernelRecurrence:
    def test_2047_first_row_crosses_thread_boundary(self):
        """Row 1 must carry its cost across the first 64-column thread boundary."""
        rng = np.random.default_rng(3)
        query = rng.standard_normal(2047).astype(np.float32)
        query[0] = 0.0
        subject = rng.standard_normal(2047).astype(np.float32)
        # Thread 0 holds the boundary column plus samples 0..62.
        subject[:31] = 0.0     # matches query[0] for free
        subject[31:63] = 50.0  # must be paid for by any valid path
        seqs = np.stack([query, subject])
        np.testing.assert_allclose(dtw_pairwise(seqs, use_cuda=True), _cpu_matrix(seqs), rtol=1e-4)

    @pytest.mark.parametrize("bucket", [1023, 2047])
    def test_last_row_inside_last_thread(self, bucket):
        """The last row must pay for every subject sample in the last thread's columns."""
        per_thread = (bucket + 1) // 32
        first = per_thread * 31 - 1          # 0-based index of the last thread's first sample
        rng = np.random.default_rng(bucket)
        query = rng.standard_normal(bucket).astype(np.float32)
        query[-1] = 0.0
        subject = query.copy()
        subject[first:first + 16] = 50.0     # must be paid for by any valid path
        subject[first + 16:] = 0.0           # matches query[-1] for free
        seqs = np.stack([query, subject])
        np.testing.assert_allclose(dtw_pairwise(seqs, use_cuda=True), _cpu_matrix(seqs), rtol=1e-4)

    @pytest.mark.parametrize("bucket", BUCKETS)
    def test_matrix_is_symmetric(self, bucket):
        rng = np.random.default_rng(bucket + 2)
        sigs = _signals(rng, rng.integers(max(2, bucket // 20), bucket, size=6), 90.0)
        gpu = dtw_multi_position_pairwise([sigs], use_cuda=True)[0]
        np.testing.assert_allclose(gpu, gpu.T, rtol=1e-5)
