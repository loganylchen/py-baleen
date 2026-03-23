"""
Tests for DTW computation with CPU and GPU backends.

These tests verify:
1. dtw_distance() and dtw_pairwise() work on CPU via tslearn/numpy
2. Backend reporting is accurate
3. Numerical correctness against known DTW properties
4. Open-start/open-end boundary conditions work on CPU
5. Input validation works on both paths
6. use_cuda parameter controls backend selection
"""

import sys
import warnings

import numpy as np
import pytest


class TestDTWDistanceCPU:
    """dtw_distance() must work without CUDA by falling back to CPU."""

    def test_identical_sequences_distance_zero(self):
        from baleen._cuda_dtw import dtw_distance

        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        dist = dtw_distance(seq, seq)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_different_sequences_positive_distance(self):
        from baleen._cuda_dtw import dtw_distance

        seq1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seq2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        dist = dtw_distance(seq1, seq2)
        assert dist > 0.0

    def test_symmetry(self):
        from baleen._cuda_dtw import dtw_distance

        rng = np.random.default_rng(42)
        seq1 = rng.standard_normal(50).astype(np.float32)
        seq2 = rng.standard_normal(50).astype(np.float32)
        d1 = dtw_distance(seq1, seq2)
        d2 = dtw_distance(seq2, seq1)
        assert d1 == pytest.approx(d2, abs=1e-5)

    def test_different_lengths(self):
        from baleen._cuda_dtw import dtw_distance

        seq1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seq2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        dist = dtw_distance(seq1, seq2)
        assert isinstance(dist, float)
        assert dist >= 0.0

    def test_list_input(self):
        from baleen._cuda_dtw import dtw_distance

        dist = dtw_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_return_type_is_float(self):
        from baleen._cuda_dtw import dtw_distance

        seq1 = np.array([1.0, 2.0], dtype=np.float32)
        seq2 = np.array([3.0, 4.0], dtype=np.float32)
        dist = dtw_distance(seq1, seq2)
        assert isinstance(dist, float)

    def test_triangle_inequality(self):
        from baleen._cuda_dtw import dtw_distance

        rng = np.random.default_rng(123)
        a = rng.standard_normal(30).astype(np.float32)
        b = rng.standard_normal(30).astype(np.float32)
        c = rng.standard_normal(30).astype(np.float32)
        d_ac = dtw_distance(a, c)
        d_ab = dtw_distance(a, b)
        d_bc = dtw_distance(b, c)
        assert d_ac <= d_ab + d_bc + 1e-5


class TestDTWPairwiseCPU:
    """dtw_pairwise() must work without CUDA by falling back to CPU."""

    def test_pairwise_shape(self):
        from baleen._cuda_dtw import dtw_pairwise

        rng = np.random.default_rng(42)
        sequences = rng.standard_normal((5, 20)).astype(np.float32)
        result = dtw_pairwise(sequences)
        assert result.shape == (5, 5)

    def test_pairwise_diagonal_zero(self):
        from baleen._cuda_dtw import dtw_pairwise

        rng = np.random.default_rng(42)
        sequences = rng.standard_normal((4, 15)).astype(np.float32)
        result = dtw_pairwise(sequences)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-6)

    def test_pairwise_symmetric(self):
        from baleen._cuda_dtw import dtw_pairwise

        rng = np.random.default_rng(42)
        sequences = rng.standard_normal((4, 15)).astype(np.float32)
        result = dtw_pairwise(sequences)
        np.testing.assert_allclose(result, result.T, atol=1e-5)

    def test_pairwise_nonnegative(self):
        from baleen._cuda_dtw import dtw_pairwise

        rng = np.random.default_rng(42)
        sequences = rng.standard_normal((3, 10)).astype(np.float32)
        result = dtw_pairwise(sequences)
        assert np.all(result >= -1e-6)

    def test_pairwise_return_type(self):
        from baleen._cuda_dtw import dtw_pairwise

        sequences = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result = dtw_pairwise(sequences)
        assert isinstance(result, np.ndarray)

    def test_pairwise_consistent_with_distance(self):
        from baleen._cuda_dtw import dtw_distance, dtw_pairwise

        rng = np.random.default_rng(42)
        sequences = rng.standard_normal((3, 10)).astype(np.float32)
        matrix = dtw_pairwise(sequences)

        for i in range(3):
            for j in range(3):
                expected = dtw_distance(sequences[i], sequences[j])
                assert matrix[i, j] == pytest.approx(expected, abs=1e-4), (
                    f"matrix[{i},{j}]={matrix[i,j]} != dtw_distance={expected}"
                )


class TestInputValidationCPU:
    """Input validation must still work on the CPU path."""

    def test_empty_sequence_raises(self):
        from baleen._cuda_dtw import dtw_distance

        with pytest.raises(ValueError, match="empty"):
            dtw_distance(np.array([], dtype=np.float32), np.array([1.0], dtype=np.float32))

    def test_2d_sequence_raises(self):
        from baleen._cuda_dtw import dtw_distance

        with pytest.raises(ValueError, match="1-dimensional"):
            dtw_distance(
                np.array([[1.0, 2.0]], dtype=np.float32),
                np.array([1.0, 2.0], dtype=np.float32),
            )

    def test_pairwise_1d_raises(self):
        from baleen._cuda_dtw import dtw_pairwise

        with pytest.raises(ValueError, match="2D"):
            dtw_pairwise(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_pairwise_single_sequence_raises(self):
        from baleen._cuda_dtw import dtw_pairwise

        with pytest.raises(ValueError, match="at least 2"):
            dtw_pairwise(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

    def test_pairwise_zero_length_raises(self):
        from baleen._cuda_dtw import dtw_pairwise

        with pytest.raises(ValueError, match="0"):
            dtw_pairwise(np.zeros((3, 0), dtype=np.float32))


class TestBackendReporting:
    """Backend selection and reporting."""

    def test_backend_function_exists(self):
        from baleen._cuda_dtw import backend

        result = backend()
        assert isinstance(result, str)

    def test_backend_is_cuda_or_cpu(self):
        from baleen._cuda_dtw import backend

        assert backend() in ("cuda", "cpu")

    def test_backend_matches_cuda_available(self):
        from baleen._cuda_dtw import CUDA_AVAILABLE, backend

        if CUDA_AVAILABLE:
            assert backend() == "cuda"
        else:
            assert backend() == "cpu"

    def test_is_available_still_reports_cuda(self):
        from baleen._cuda_dtw import CUDA_AVAILABLE, is_available

        assert is_available() == CUDA_AVAILABLE


# ---------------------------------------------------------------------------
# use_cuda parameter
# ---------------------------------------------------------------------------

class TestUseCudaParameter:
    """use_cuda parameter must control backend dispatch."""

    def test_use_cuda_false_forces_cpu(self):
        """use_cuda=False must use CPU even if CUDA is available."""
        from baleen._cuda_dtw import dtw_distance

        seq1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seq2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dist = dtw_distance(seq1, seq2, use_cuda=False)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_use_cuda_none_auto_selects(self):
        """use_cuda=None (default) must auto-select based on availability."""
        from baleen._cuda_dtw import dtw_distance

        seq1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seq2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dist = dtw_distance(seq1, seq2, use_cuda=None)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_use_cuda_true_raises_without_gpu(self):
        """use_cuda=True must raise RuntimeError if CUDA is not available."""
        from baleen._cuda_dtw import CUDA_AVAILABLE, dtw_distance

        if not CUDA_AVAILABLE:
            with pytest.raises(RuntimeError, match="CUDA"):
                dtw_distance(
                    np.array([1.0, 2.0], dtype=np.float32),
                    np.array([1.0, 2.0], dtype=np.float32),
                    use_cuda=True,
                )

    def test_pairwise_use_cuda_false_forces_cpu(self):
        """use_cuda=False on dtw_pairwise must use CPU."""
        from baleen._cuda_dtw import dtw_pairwise

        sequences = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result = dtw_pairwise(sequences, use_cuda=False)
        assert result.shape == (3, 3)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-6)

    def test_pairwise_use_cuda_true_raises_without_gpu(self):
        """use_cuda=True on dtw_pairwise must raise if no CUDA."""
        from baleen._cuda_dtw import CUDA_AVAILABLE, dtw_pairwise

        if not CUDA_AVAILABLE:
            with pytest.raises(RuntimeError, match="CUDA"):
                dtw_pairwise(
                    np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                    use_cuda=True,
                )

    def test_cpu_and_default_produce_same_result(self):
        """use_cuda=False and use_cuda=None must produce identical results on CPU."""
        from baleen._cuda_dtw import CUDA_AVAILABLE, dtw_distance

        if not CUDA_AVAILABLE:
            rng = np.random.default_rng(99)
            seq1 = rng.standard_normal(30).astype(np.float32)
            seq2 = rng.standard_normal(30).astype(np.float32)
            d_auto = dtw_distance(seq1, seq2, use_cuda=None)
            d_cpu = dtw_distance(seq1, seq2, use_cuda=False)
            assert d_auto == pytest.approx(d_cpu, abs=1e-10)


# ---------------------------------------------------------------------------
# Open boundary conditions on CPU (actual computation, not just warnings)
# ---------------------------------------------------------------------------

class TestOpenBoundaryCPU:
    """Open-start/open-end boundary conditions must compute correctly on CPU."""

    def test_open_end_leq_standard(self):
        """open_end DTW distance must be <= standard DTW (free tail is cheaper)."""
        from baleen._cuda_dtw import dtw_distance

        # short prefix + long tail that diverges
        seq1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seq2 = np.array([1.0, 2.0, 3.0, 100.0, 200.0], dtype=np.float32)
        d_std = dtw_distance(seq1, seq2, use_cuda=False)
        d_open = dtw_distance(seq1, seq2, use_open_end=True, use_cuda=False)
        assert d_open <= d_std + 1e-6

    def test_open_start_leq_standard(self):
        """open_start DTW distance must be <= standard DTW (free head is cheaper)."""
        from baleen._cuda_dtw import dtw_distance

        seq1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seq2 = np.array([100.0, 200.0, 1.0, 2.0, 3.0], dtype=np.float32)
        d_std = dtw_distance(seq1, seq2, use_cuda=False)
        d_open = dtw_distance(seq1, seq2, use_open_start=True, use_cuda=False)
        assert d_open <= d_std + 1e-6

    def test_open_start_identical_prefix(self):
        """open_start with seq1 matching end of seq2 should give ~0 distance."""
        from baleen._cuda_dtw import dtw_distance

        seq1 = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        seq2 = np.array([99.0, 99.0, 5.0, 6.0, 7.0], dtype=np.float32)
        d = dtw_distance(seq1, seq2, use_open_start=True, use_cuda=False)
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_open_end_identical_suffix(self):
        """open_end with seq1 matching start of seq2 should give ~0 distance."""
        from baleen._cuda_dtw import dtw_distance

        seq1 = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        seq2 = np.array([5.0, 6.0, 7.0, 99.0, 99.0], dtype=np.float32)
        d = dtw_distance(seq1, seq2, use_open_end=True, use_cuda=False)
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_open_both_leq_either(self):
        """open_start + open_end raw cost must be <= either alone.

        When exactly one of open_start/open_end is set, the distance is
        normalized by len(seq1): d = sqrt(cost) / len1.  When both are set,
        no normalization: d = sqrt(cost).  To compare the underlying
        relaxation property we must undo normalization so we compare raw
        costs: sqrt(cost_both) <= sqrt(cost_start) and sqrt(cost_both) <=
        sqrt(cost_end).
        """
        from baleen._cuda_dtw import dtw_distance

        rng = np.random.default_rng(77)
        seq1 = rng.standard_normal(10).astype(np.float32)
        seq2 = rng.standard_normal(20).astype(np.float32)
        len1 = len(seq1)
        d_start = dtw_distance(seq1, seq2, use_open_start=True, use_cuda=False)
        d_end = dtw_distance(seq1, seq2, use_open_end=True, use_cuda=False)
        d_both = dtw_distance(seq1, seq2, use_open_start=True, use_open_end=True, use_cuda=False)
        # Undo normalization: single-open returns sqrt(cost)/len1,
        # both-open returns sqrt(cost).
        raw_start = d_start * len1  # = sqrt(cost_start)
        raw_end = d_end * len1      # = sqrt(cost_end)
        raw_both = d_both           # = sqrt(cost_both), already unnormalized
        assert raw_both <= raw_start + 1e-5
        assert raw_both <= raw_end + 1e-5

    def test_open_end_same_length_identical(self):
        """open_end with identical same-length sequences must be 0."""
        from baleen._cuda_dtw import dtw_distance

        seq = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        d = dtw_distance(seq, seq, use_open_end=True, use_cuda=False)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_open_start_same_length_identical(self):
        """open_start with identical same-length sequences must be 0."""
        from baleen._cuda_dtw import dtw_distance

        seq = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        d = dtw_distance(seq, seq, use_open_start=True, use_cuda=False)
        assert d == pytest.approx(0.0, abs=1e-6)


class TestOpenBoundaryPairwiseCPU:
    """Pairwise with open boundaries must be consistent with single-pair."""

    def test_pairwise_open_end_consistent_with_distance(self):
        """dtw_pairwise(open_end=True)[i,j] must match dtw_distance for upper triangle.

        Open-boundary DTW is NOT symmetric: dtw(a,b,open_end) != dtw(b,a,open_end)
        because seq1 (Y-axis) and seq2 (X-axis) have different roles.
        Pairwise computes (i,j) for i<j and mirrors to (j,i), so we only
        verify the upper triangle + diagonal.
        """
        from baleen._cuda_dtw import dtw_distance, dtw_pairwise

        rng = np.random.default_rng(42)
        sequences = rng.standard_normal((3, 10)).astype(np.float32)
        matrix = dtw_pairwise(sequences, use_open_end=True, use_cuda=False)

        for i in range(3):
            assert matrix[i, i] == pytest.approx(0.0, abs=1e-6)
            for j in range(i + 1, 3):
                expected = dtw_distance(sequences[i], sequences[j], use_open_end=True, use_cuda=False)
                assert matrix[i, j] == pytest.approx(expected, abs=1e-4), (
                    f"matrix[{i},{j}]={matrix[i,j]} != dtw_distance={expected}"
                )

    def test_pairwise_open_start_consistent_with_distance(self):
        """dtw_pairwise(open_start=True)[i,j] must match dtw_distance for upper triangle."""
        from baleen._cuda_dtw import dtw_distance, dtw_pairwise

        rng = np.random.default_rng(42)
        sequences = rng.standard_normal((3, 10)).astype(np.float32)
        matrix = dtw_pairwise(sequences, use_open_start=True, use_cuda=False)

        for i in range(3):
            assert matrix[i, i] == pytest.approx(0.0, abs=1e-6)
            for j in range(i + 1, 3):
                expected = dtw_distance(sequences[i], sequences[j], use_open_start=True, use_cuda=False)
                assert matrix[i, j] == pytest.approx(expected, abs=1e-4), (
                    f"matrix[{i},{j}]={matrix[i,j]} != dtw_distance={expected}"
                )

    def test_pairwise_open_end_shape(self):
        from baleen._cuda_dtw import dtw_pairwise

        sequences = np.random.default_rng(42).standard_normal((4, 8)).astype(np.float32)
        result = dtw_pairwise(sequences, use_open_end=True, use_cuda=False)
        assert result.shape == (4, 4)

    def test_pairwise_open_end_diagonal_zero(self):
        from baleen._cuda_dtw import dtw_pairwise

        sequences = np.random.default_rng(42).standard_normal((4, 8)).astype(np.float32)
        result = dtw_pairwise(sequences, use_open_end=True, use_cuda=False)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# dtw_pairwise_varlen on CPU
# ---------------------------------------------------------------------------

class TestDTWPairwiseVarlenCPU:

    def test_basic_variable_lengths(self):
        from baleen._cuda_dtw import dtw_pairwise_varlen

        signals = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
        ]
        result = dtw_pairwise_varlen(signals, use_cuda=False)
        assert result.shape == (3, 3)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-6)
        assert np.allclose(result, result.T)

    def test_consistent_with_dtw_distance(self):
        from baleen._cuda_dtw import dtw_distance, dtw_pairwise_varlen

        signals = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        ]
        matrix = dtw_pairwise_varlen(signals, use_cuda=False)
        for i in range(3):
            for j in range(i + 1, 3):
                expected = dtw_distance(signals[i], signals[j], use_cuda=False)
                np.testing.assert_allclose(
                    matrix[i, j], expected, rtol=1e-5,
                    err_msg=f"Mismatch at ({i},{j})",
                )

    def test_open_end_consistent_with_dtw_distance(self):
        from baleen._cuda_dtw import dtw_distance, dtw_pairwise_varlen

        signals = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0, 99.0, 99.0], dtype=np.float32),
            np.array([5.0, 6.0], dtype=np.float32),
        ]
        matrix = dtw_pairwise_varlen(signals, use_open_end=True, use_cuda=False)
        for i in range(3):
            for j in range(i + 1, 3):
                expected = dtw_distance(
                    signals[i], signals[j],
                    use_open_end=True, use_cuda=False,
                )
                np.testing.assert_allclose(
                    matrix[i, j], expected, rtol=1e-5,
                    err_msg=f"Mismatch at ({i},{j})",
                )

    def test_many_variable_length_signals(self):
        from baleen._cuda_dtw import dtw_pairwise_varlen

        rng = np.random.RandomState(42)
        signals = [
            rng.randn(rng.randint(5, 30)).astype(np.float32)
            for _ in range(15)
        ]
        matrix = dtw_pairwise_varlen(signals, use_cuda=False)
        assert matrix.shape == (15, 15)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-6)
        assert np.allclose(matrix, matrix.T)
        off_diag = matrix[np.triu_indices(15, k=1)]
        assert np.all(off_diag > 0.0)

    def test_single_signal_raises(self):
        from baleen._cuda_dtw import dtw_pairwise_varlen

        with pytest.raises(ValueError, match="at least 2"):
            dtw_pairwise_varlen(
                [np.array([1.0, 2.0], dtype=np.float32)],
                use_cuda=False,
            )

    def test_empty_signal_raises(self):
        from baleen._cuda_dtw import dtw_pairwise_varlen

        with pytest.raises(ValueError, match="non-empty"):
            dtw_pairwise_varlen(
                [
                    np.array([], dtype=np.float32),
                    np.array([1.0], dtype=np.float32),
                ],
                use_cuda=False,
            )

    def test_use_cuda_true_raises_without_gpu(self):
        from baleen._cuda_dtw import CUDA_AVAILABLE, dtw_pairwise_varlen

        if not CUDA_AVAILABLE:
            with pytest.raises(RuntimeError, match="CUDA"):
                dtw_pairwise_varlen(
                    [
                        np.array([1.0, 2.0], dtype=np.float32),
                        np.array([3.0, 4.0], dtype=np.float32),
                    ],
                    use_cuda=True,
                )

    def test_identical_signals_zero_distance(self):
        from baleen._cuda_dtw import dtw_pairwise_varlen

        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = dtw_pairwise_varlen([sig, sig.copy()], use_cuda=False)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# dtw_multi_position_pairwise on CPU
# ---------------------------------------------------------------------------

class TestMultiPositionBatchCPU:
    """dtw_multi_position_pairwise must produce same results as per-position calls."""

    def test_batch_matches_individual(self):
        """Batch result must match individual dtw_pairwise_varlen calls."""
        from baleen._cuda_dtw import dtw_multi_position_pairwise, dtw_pairwise_varlen

        rng = np.random.default_rng(42)
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


# ---------------------------------------------------------------------------
# dtw_multi_position_pairwise on GPU (conditional)
# ---------------------------------------------------------------------------

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
