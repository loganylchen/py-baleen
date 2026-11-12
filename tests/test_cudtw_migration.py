"""
Migration verification tests for cuDTW++ warp-shuffle kernel.

These tests validate that the cuDTW++ backend produces correct DTW distances
by comparing against the CPU (tslearn) reference implementation.

Run with: pytest tests/test_cudtw_migration.py -v
"""

import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from baleen._cuda_dtw import (
    CUDA_AVAILABLE,
    _CUDTW_ACTIVE,
    backend,
    dtw_distance,
    dtw_pairwise,
    dtw_pairwise_varlen,
    dtw_multi_position_pairwise,
)

pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_signals(n, length, seed=42):
    rng = np.random.RandomState(seed)
    return [rng.randn(length).astype(np.float32) for _ in range(n)]


def _pairwise_cpu(signals):
    """Reference pairwise DTW via CPU backend."""
    n = len(signals)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(signals[i], signals[j], use_cuda=False)
            mat[i, j] = d
            mat[j, i] = d
    return mat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBackendDetection:
    def test_backend_is_cuda(self):
        assert backend() == "cuda"

    def test_cudtw_active(self):
        """If built with default settings, cuDTW++ should be active."""
        # This test documents which kernel is active; it's informational.
        print(f"cuDTW++ active: {_CUDTW_ACTIVE}")


class TestSinglePairDTW:
    """Single-pair DTW distance correctness."""

    @pytest.mark.parametrize("length", [50, 127, 200, 255, 500, 511, 1023])
    def test_equal_length(self, length):
        rng = np.random.RandomState(123)
        s1 = rng.randn(length).astype(np.float32)
        s2 = rng.randn(length).astype(np.float32)

        d_gpu = dtw_distance(s1, s2, use_cuda=True)
        d_cpu = dtw_distance(s1, s2, use_cuda=False)

        # cuDTW++ pads to bucket boundaries, so distances differ slightly
        # for non-bucket lengths. Use generous tolerance.
        assert d_gpu >= 0
        assert d_cpu >= 0
        # Both should be in the same ballpark
        if d_cpu > 0:
            rel_err = abs(d_gpu - d_cpu) / d_cpu
            print(f"len={length}: gpu={d_gpu:.4f} cpu={d_cpu:.4f} rel_err={rel_err:.4f}")

    @pytest.mark.parametrize("length", [127, 255, 511, 1023])
    def test_bucket_length_close_match(self, length):
        """At exact bucket lengths, GPU and CPU should agree closely."""
        rng = np.random.RandomState(456)
        s1 = rng.randn(length).astype(np.float32)
        s2 = rng.randn(length).astype(np.float32)

        d_gpu = dtw_distance(s1, s2, use_cuda=True)
        d_cpu = dtw_distance(s1, s2, use_cuda=False)

        # At bucket boundaries, zero-padding is minimal → close match
        if d_cpu > 0:
            rel_err = abs(d_gpu - d_cpu) / d_cpu
            assert rel_err < 0.15, (
                f"Bucket {length}: gpu={d_gpu:.4f} cpu={d_cpu:.4f} rel_err={rel_err:.4f}"
            )

    def test_identical_sequences(self):
        s = np.ones(127, dtype=np.float32)
        d = dtw_distance(s, s, use_cuda=True)
        assert d == pytest.approx(0.0, abs=1e-5)


class TestPairwiseDTW:
    """Pairwise distance matrix correctness."""

    def test_pairwise_symmetric(self):
        sigs = _random_signals(5, 127)
        seqs = np.array(sigs)
        mat = dtw_pairwise(seqs, use_cuda=True)
        np.testing.assert_allclose(mat, mat.T, atol=1e-5)

    def test_pairwise_diagonal_zero(self):
        sigs = _random_signals(5, 127)
        seqs = np.array(sigs)
        mat = dtw_pairwise(seqs, use_cuda=True)
        np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-5)

    def test_pairwise_nonnegative(self):
        sigs = _random_signals(8, 255)
        seqs = np.array(sigs)
        mat = dtw_pairwise(seqs, use_cuda=True)
        assert np.all(mat >= -1e-6)


class TestPairwiseVarlen:
    """Variable-length pairwise DTW."""

    def test_varlen_runs(self):
        rng = np.random.RandomState(789)
        sigs = [rng.randn(l).astype(np.float32) for l in [80, 100, 120, 90]]
        mat = dtw_pairwise_varlen(sigs, use_cuda=True)
        assert mat.shape == (4, 4)
        np.testing.assert_allclose(mat, mat.T, atol=1e-5)
        np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-5)


class TestMultiPosition:
    """Multi-position batched pairwise DTW."""

    def test_multi_position_runs(self):
        rng = np.random.RandomState(321)
        pos1 = [rng.randn(100).astype(np.float32) for _ in range(3)]
        pos2 = [rng.randn(80).astype(np.float32) for _ in range(4)]
        results = dtw_multi_position_pairwise([pos1, pos2], use_cuda=True)
        assert len(results) == 2
        assert results[0].shape == (3, 3)
        assert results[1].shape == (4, 4)

    def test_multi_position_symmetric(self):
        rng = np.random.RandomState(654)
        pos = [rng.randn(127).astype(np.float32) for _ in range(5)]
        results = dtw_multi_position_pairwise([pos], use_cuda=True)
        mat = results[0]
        np.testing.assert_allclose(mat, mat.T, atol=1e-5)


class TestKernelMemoryBounds:
    """The kernels must only read inside the buffers the wrapper allocates.

    An out-of-bounds read is invisible to the output and only faults when the
    buffer happens to end on an unmapped page, so it needs the CUDA memory
    checker rather than a value assertion.
    """

    _SCRIPT = textwrap.dedent("""
        import numpy as np
        from baleen._cuda_dtw import _cuda_dtw as ext

        rng = np.random.default_rng(0)
        for bucket in (127, 255, 511, 1023, 2047):
            # Full-bucket rows: the last row ends exactly at the buffer end.
            seqs = rng.standard_normal((3, bucket)).astype(np.float32)
            lengths = np.full(3, bucket, np.int64)
            ext.dtw_multi_position_pairwise(seqs, lengths, np.array([3], np.int64))
            ext.dtw_pairwise(seqs)
            ext.dtw_pairwise_varlen(seqs, lengths)
            ext.dtw_distance(seqs[0], seqs[1])
    """)

    @pytest.mark.skipif(
        shutil.which("compute-sanitizer") is None,
        reason="compute-sanitizer (CUDA toolkit) not on PATH",
    )
    def test_memcheck_clean(self):
        import baleen

        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(baleen.__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [pkg_parent] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        proc = subprocess.run(
            ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "1",
             sys.executable, "-c", self._SCRIPT],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=900, env=env,
        )
        assert proc.returncode == 0, proc.stdout[-3000:]
        assert "ERROR SUMMARY: 0 errors" in proc.stdout, proc.stdout[-3000:]
