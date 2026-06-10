"""Tests for the DTW shim (baleen._dtw), which delegates to krill.

Covers the contract the pipeline relies on: distance/pairwise correctness,
symmetry, zero diagonal, batch consistency, input validation, backend
reporting, and (when a GPU is present) GPU/CPU agreement on signals within the
no-resample length cap.
"""

import numpy as np
import pytest

from baleen import _dtw


class TestDTWDistance:
    def test_identical_sequences_distance_zero(self):
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        assert _dtw.dtw_distance(seq, seq, use_cuda=False) == pytest.approx(0.0, abs=1e-6)

    def test_different_sequences_positive_distance(self):
        seq1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seq2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        assert _dtw.dtw_distance(seq1, seq2, use_cuda=False) > 0.0

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        seq1 = rng.standard_normal(50).astype(np.float32)
        seq2 = rng.standard_normal(50).astype(np.float32)
        d1 = _dtw.dtw_distance(seq1, seq2, use_cuda=False)
        d2 = _dtw.dtw_distance(seq2, seq1, use_cuda=False)
        assert d1 == pytest.approx(d2, abs=1e-5)

    def test_return_type_is_float(self):
        seq1 = np.array([1.0, 2.0], dtype=np.float32)
        seq2 = np.array([3.0, 4.0], dtype=np.float32)
        assert isinstance(_dtw.dtw_distance(seq1, seq2, use_cuda=False), float)

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError):
            _dtw.dtw_distance(
                np.array([], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                use_cuda=False,
            )


class TestDTWPairwiseVarlen:
    def test_shape_symmetry_diagonal(self):
        signals = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
        ]
        result = _dtw.dtw_pairwise_varlen(signals, use_cuda=False)
        assert result.shape == (3, 3)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-6)
        np.testing.assert_allclose(result, result.T, atol=1e-5)

    def test_consistent_with_dtw_distance(self):
        signals = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        ]
        matrix = _dtw.dtw_pairwise_varlen(signals, use_cuda=False)
        for i in range(3):
            for j in range(i + 1, 3):
                expected = _dtw.dtw_distance(signals[i], signals[j], use_cuda=False)
                np.testing.assert_allclose(matrix[i, j], expected, rtol=1e-5)

    def test_single_signal_raises(self):
        with pytest.raises(ValueError):
            _dtw.dtw_pairwise_varlen(
                [np.array([1.0, 2.0], dtype=np.float32)], use_cuda=False
            )


class TestMultiPositionBatch:
    def test_batch_matches_individual(self):
        rng = np.random.default_rng(42)
        position_signals = [
            [rng.standard_normal(int(rng.integers(5, 20))).astype(np.float32) for _ in range(4)],
            [rng.standard_normal(int(rng.integers(5, 20))).astype(np.float32) for _ in range(6)],
            [rng.standard_normal(int(rng.integers(5, 20))).astype(np.float32) for _ in range(3)],
        ]
        batch = _dtw.dtw_multi_position_pairwise(position_signals, use_cuda=False)
        assert len(batch) == 3
        for p, signals in enumerate(position_signals):
            expected = _dtw.dtw_pairwise_varlen(signals, use_cuda=False)
            np.testing.assert_allclose(batch[p], expected, rtol=1e-5,
                                       err_msg=f"position {p}")

    def test_shapes_symmetry_diagonal(self):
        rng = np.random.default_rng(99)
        position_signals = [
            [rng.standard_normal(int(rng.integers(3, 15))).astype(np.float32)
             for _ in range(int(rng.integers(2, 8)))]
            for _ in range(12)
        ]
        results = _dtw.dtw_multi_position_pairwise(position_signals, use_cuda=False)
        assert len(results) == 12
        for matrix, signals in zip(results, position_signals):
            n = len(signals)
            assert matrix.shape == (n, n)
            np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-6)
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-5)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            _dtw.dtw_multi_position_pairwise([], use_cuda=False)


class TestBackendReporting:
    def test_backend_is_gpu_or_cpu(self):
        assert _dtw.backend() in ("gpu", "cpu")

    def test_backend_matches_availability(self):
        if _dtw.CUDA_AVAILABLE:
            assert _dtw.backend() == "gpu"
        else:
            assert _dtw.backend() == "cpu"

    def test_is_available_matches_flag(self):
        assert _dtw.is_available() == _dtw.CUDA_AVAILABLE


class TestMemoryHelpers:
    def test_estimate_gpu_memory_positive(self):
        rng = np.random.default_rng(7)
        ps = [[rng.standard_normal(50).astype(np.float32) for _ in range(3)]]
        assert _dtw.estimate_gpu_memory(ps) > 0

    def test_per_device_memory_list(self):
        assert isinstance(_dtw.get_per_device_memory(), list)


@pytest.mark.skipif(not _dtw.CUDA_AVAILABLE, reason="GPU DTW not available")
class TestGPUMatchesCPU:
    """On signals within the no-resample cap, GPU and CPU must agree."""

    def test_gpu_matches_cpu_short(self):
        rng = np.random.default_rng(42)
        position_signals = [
            [rng.standard_normal(int(rng.integers(5, 200))).astype(np.float32) for _ in range(4)],
            [rng.standard_normal(int(rng.integers(5, 200))).astype(np.float32) for _ in range(6)],
        ]
        cpu = _dtw.dtw_multi_position_pairwise(position_signals, use_cuda=False)
        gpu = _dtw.dtw_multi_position_pairwise(position_signals, use_cuda=True)
        for p in range(len(position_signals)):
            np.testing.assert_allclose(gpu[p], cpu[p], rtol=1e-4, atol=1e-4,
                                       err_msg=f"position {p}")
