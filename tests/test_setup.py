"""
Tests for baleen package setup and installation.

These tests verify:
1. The package is importable after installation
2. The CUDA extension is optional (graceful fallback on non-GPU machines)
3. Package metadata is correct
4. The public API surface is as expected
"""

import importlib
import subprocess
import sys

import pytest


class TestPackageImport:
    """Test that the baleen package can be imported."""

    def test_import_baleen(self):
        """The top-level 'baleen' package must be importable."""
        import baleen

        assert baleen is not None

    def test_import_cuda_dtw_subpackage(self):
        """The _cuda_dtw subpackage must be importable (Python wrapper layer)."""
        from baleen import _cuda_dtw

        assert _cuda_dtw is not None

    def test_cuda_dtw_has_public_api(self):
        """The _cuda_dtw module must expose the expected public symbols."""
        from baleen import _cuda_dtw

        expected_names = ["dtw_distance", "dtw_pairwise", "cleanup", "is_available", "CUDA_AVAILABLE"]
        for name in expected_names:
            assert hasattr(_cuda_dtw, name), f"Missing public API: {name}"

    def test_is_available_returns_bool(self):
        """is_available() must return a boolean."""
        from baleen._cuda_dtw import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_cuda_available_is_bool(self):
        """CUDA_AVAILABLE must be a boolean."""
        from baleen._cuda_dtw import CUDA_AVAILABLE

        assert isinstance(CUDA_AVAILABLE, bool)


class TestCUDAGracefulFallback:
    """On machines without CUDA, the package must still import without error."""

    def test_no_crash_on_import(self):
        """Importing baleen._cuda_dtw must not raise on non-CUDA machines."""
        # This test always passes if we get here — import already succeeded.
        # The real test is that the import in the test above didn't crash.
        from baleen._cuda_dtw import CUDA_AVAILABLE

        # On this macOS dev machine, CUDA won't be available
        if sys.platform == "darwin":
            assert CUDA_AVAILABLE is False

    def test_dtw_distance_raises_without_cuda(self):
        """dtw_distance() must raise RuntimeError when CUDA is not available."""
        import numpy as np

        from baleen._cuda_dtw import CUDA_AVAILABLE, dtw_distance

        if not CUDA_AVAILABLE:
            with pytest.raises(RuntimeError, match="CUDA"):
                dtw_distance(np.array([1.0, 2.0], dtype=np.float32), np.array([1.0, 2.0], dtype=np.float32))

    def test_dtw_pairwise_raises_without_cuda(self):
        """dtw_pairwise() must raise RuntimeError when CUDA is not available."""
        import numpy as np

        from baleen._cuda_dtw import CUDA_AVAILABLE, dtw_pairwise

        if not CUDA_AVAILABLE:
            with pytest.raises(RuntimeError, match="CUDA"):
                dtw_pairwise(np.random.randn(3, 10).astype(np.float32))

    def test_cleanup_noop_without_cuda(self):
        """cleanup() must be a no-op (not raise) when CUDA is not available."""
        from baleen._cuda_dtw import cleanup

        # Should not raise even without CUDA
        cleanup()


class TestPackageMetadata:
    """Test that package metadata is correctly configured."""

    def test_pip_show_baleen(self):
        """pip show must recognize the installed package."""
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "baleen"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"pip show failed: {result.stderr}"
        assert "Name: baleen" in result.stdout

    def test_numpy_is_dependency(self):
        """numpy must be listed as a dependency."""
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "baleen"],
            capture_output=True,
            text=True,
        )
        # Check Requires line includes numpy
        for line in result.stdout.splitlines():
            if line.startswith("Requires:"):
                assert "numpy" in line, f"numpy not in Requires: {line}"
                break
        else:
            pytest.fail("No 'Requires:' line found in pip show output")
