"""
Tests for the Sakoe-Chiba band constraint in CUDA DTW.

These tests are CUDA-only; they are skipped when CUDA is not available.
The band is currently only honoured by the CUDA kernel — the CPU fallback
computes full DTW regardless of the ``sakoe_band`` argument.
"""

import numpy as np
import pytest

from baleen._cuda_dtw import (
    CUDA_AVAILABLE,
    _resolve_sakoe_band,
    dtw_distance,
    dtw_multi_position_pairwise,
    dtw_pairwise,
    dtw_pairwise_varlen,
)


requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA extension not built on this machine"
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _rand_seq(rng, n, scale=1.0):
    return rng.standard_normal(n).astype(np.float32) * scale


def test_resolve_sakoe_band_disabled():
    assert _resolve_sakoe_band(0, 1000) == 0
    assert _resolve_sakoe_band(0.0, 1000) == 0
    assert _resolve_sakoe_band(None, 1000) == 0
    assert _resolve_sakoe_band(-1.0, 1000) == 0


def test_resolve_sakoe_band_fractional():
    assert _resolve_sakoe_band(0.1, 1000) == 100
    assert _resolve_sakoe_band(0.5, 1000) == 500
    assert _resolve_sakoe_band(1.0, 1000) == 1000
    # Tiny fractions still yield at least 1 cell so the band is not degenerate.
    assert _resolve_sakoe_band(1e-4, 100) == 1


def test_resolve_sakoe_band_absolute():
    assert _resolve_sakoe_band(50, 1000) == 50
    assert _resolve_sakoe_band(500.0, 1000) == 500


@requires_cuda
def test_dtw_distance_band_disabled_matches_full(rng):
    s1 = _rand_seq(rng, 256)
    s2 = _rand_seq(rng, 256)
    d_full = dtw_distance(s1, s2, use_cuda=True)
    d_band_zero = dtw_distance(s1, s2, use_cuda=True, sakoe_band=0.0)
    assert d_band_zero == pytest.approx(d_full, rel=1e-6, abs=1e-6)


@requires_cuda
def test_dtw_distance_band_large_equals_full(rng):
    """A band wider than the signal must produce the full-DTW distance."""
    s1 = _rand_seq(rng, 128)
    s2 = _rand_seq(rng, 128)
    d_full = dtw_distance(s1, s2, use_cuda=True)
    # Band >= max(len1, len2) covers the whole matrix.
    d_band_wide = dtw_distance(s1, s2, use_cuda=True, sakoe_band=1.0)
    assert d_band_wide == pytest.approx(d_full, rel=1e-6, abs=1e-6)


@requires_cuda
def test_dtw_distance_band_small_is_not_smaller(rng):
    """
    A tighter Sakoe band forbids cells, so the optimal cost cannot decrease.
    For random signals of equal length the band must produce a finite value.
    """
    s1 = _rand_seq(rng, 512)
    s2 = _rand_seq(rng, 512)
    d_full = dtw_distance(s1, s2, use_cuda=True)
    d_band_narrow = dtw_distance(s1, s2, use_cuda=True, sakoe_band=0.1)
    assert np.isfinite(d_band_narrow)
    # Band-restricted cost is >= full cost (constraint cannot help).
    assert d_band_narrow >= d_full - 1e-6


@requires_cuda
def test_dtw_pairwise_band_disabled_matches_full(rng):
    n, L = 8, 128
    seqs = rng.standard_normal((n, L)).astype(np.float32)
    M_full = dtw_pairwise(seqs, use_cuda=True)
    M_zero = dtw_pairwise(seqs, use_cuda=True, sakoe_band=0.0)
    np.testing.assert_allclose(M_zero, M_full, rtol=1e-6, atol=1e-6)


@requires_cuda
def test_dtw_pairwise_band_wide_matches_full(rng):
    n, L = 6, 96
    seqs = rng.standard_normal((n, L)).astype(np.float32)
    M_full = dtw_pairwise(seqs, use_cuda=True)
    M_wide = dtw_pairwise(seqs, use_cuda=True, sakoe_band=1.0)
    np.testing.assert_allclose(M_wide, M_full, rtol=1e-6, atol=1e-6)


@requires_cuda
def test_dtw_pairwise_varlen_band_disabled(rng):
    signals = [_rand_seq(rng, n) for n in (80, 95, 110, 120)]
    M_full = dtw_pairwise_varlen(signals, use_cuda=True)
    M_zero = dtw_pairwise_varlen(signals, use_cuda=True, sakoe_band=0.0)
    np.testing.assert_allclose(M_zero, M_full, rtol=1e-6, atol=1e-6)


@requires_cuda
def test_dtw_multi_position_band_disabled(rng):
    positions = [
        [_rand_seq(rng, 60), _rand_seq(rng, 80), _rand_seq(rng, 90)],
        [_rand_seq(rng, 70), _rand_seq(rng, 75)],
    ]
    M_full = dtw_multi_position_pairwise(positions, use_cuda=True)
    M_zero = dtw_multi_position_pairwise(positions, use_cuda=True, sakoe_band=0.0)
    for a, b in zip(M_full, M_zero):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
