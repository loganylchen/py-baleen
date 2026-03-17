"""
CUDA-accelerated Dynamic Time Warping (DTW) module

This module provides GPU-accelerated DTW distance calculation with
automatic CPU fallback when CUDA is not available.

CPU backend strategy:
  - Standard DTW (no open boundaries): delegates to tslearn for performance
  - Open-boundary DTW: pure-numpy DP implementation matching CUDA kernel
    semantics, since tslearn does not support open_start / open_end
"""

import numpy as np
from typing import Union, Optional

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

try:
    from ._cuda_dtw import dtw_distance as _dtw_distance_cuda
    from ._cuda_dtw import dtw_pairwise as _dtw_pairwise_cuda
    from ._cuda_dtw import cleanup as _cuda_cleanup

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

try:
    from tslearn.metrics import dtw as _tslearn_dtw
    from tslearn.metrics import cdist_dtw as _tslearn_cdist_dtw

    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False

_BACKEND = "cuda" if CUDA_AVAILABLE else "cpu"

if _BACKEND == "cuda":
    print("🚀 DTW: GPU (CUDA) acceleration ENABLED")
else:
    print("💻 DTW: CPU implementation (tslearn + numpy open-boundary fallback)")


def backend() -> str:
    """
    Return the name of the active DTW backend.

    Returns
    -------
    str
        'cuda' if GPU acceleration is available,
        'cpu' if CPU fallback is active.
    """
    return _BACKEND


# ---------------------------------------------------------------------------
# CPU DTW implementation (pure numpy)
# ---------------------------------------------------------------------------

def _dtw_distance_cpu(
    seq1: np.ndarray,
    seq2: np.ndarray,
    use_open_start: bool = False,
    use_open_end: bool = False,
) -> float:
    """
    Compute DTW distance on CPU.

    When no open boundaries are requested, delegates to tslearn for
    performance.  When open_start or open_end is True, uses a pure-numpy
    DP that matches the CUDA kernel semantics exactly:
      - Cost function: squared Euclidean (a - b)^2
      - open_start: first row (i=0) costs are zero
      - open_end: last row (i=len1-1) rightward/diagonal moves are free
      - Normalization: sqrt(cost) / len1 when exactly one boundary is open,
        sqrt(cost) otherwise
    """
    if not use_open_start and not use_open_end:
        if not TSLEARN_AVAILABLE:
            raise RuntimeError(
                "tslearn is required for CPU DTW.\n"
                "Install it with: pip install tslearn"
            )
        s1_2d = seq1.reshape(-1, 1)
        s2_2d = seq2.reshape(-1, 1)
        return float(_tslearn_dtw(s1_2d, s2_2d))

    return _dtw_distance_open_boundary(seq1, seq2, use_open_start, use_open_end)


def _dtw_distance_open_boundary(
    seq1: np.ndarray,
    seq2: np.ndarray,
    use_open_start: bool,
    use_open_end: bool,
) -> float:
    """Pure-numpy DTW DP with open-boundary support (CUDA-compatible semantics)."""
    len1 = len(seq1)
    len2 = len(seq2)

    # Cost matrix: cost[i][j] = accumulated cost to align seq1[:i+1] with seq2[:j+1]
    # Using float64 for numerical stability in accumulation
    cost = np.full((len1, len2), np.inf, dtype=np.float64)

    # --- Fill first row (i=0) ---
    # open_start: diff = 0 for all positions in the first row
    if use_open_start:
        # No cost contribution from first row at all
        cost[0, 0] = 0.0
        for j in range(1, len2):
            cost[0, j] = cost[0, j - 1]  # rightward moves, zero cost
    else:
        diff_00 = float(seq1[0] - seq2[0])
        cost[0, 0] = diff_00 * diff_00
        for j in range(1, len2):
            diff = float(seq1[0] - seq2[j])
            cost[0, j] = cost[0, j - 1] + diff * diff

    # --- Fill first column (j=0) ---
    for i in range(1, len1):
        diff = float(seq1[i] - seq2[0])
        cost[i, 0] = cost[i - 1, 0] + diff * diff

    # --- Fill the rest of the cost matrix ---
    for i in range(1, len1):
        is_last_row = (i == len1 - 1)
        for j in range(1, len2):
            diff = float(seq1[i] - seq2[j])
            diff_sq = diff * diff

            # UP move: cost[i-1, j] + diff^2
            up_cost = cost[i - 1, j] + diff_sq

            # DIAGONAL move: cost[i-1, j-1] + diff^2
            diag_cost = cost[i - 1, j - 1] + diff_sq

            # RIGHT move: cost[i, j-1] + diff^2
            # But if open_end and this is the last row, rightward moves are free
            if use_open_end and is_last_row:
                right_cost = cost[i, j - 1]  # no additional cost
            else:
                right_cost = cost[i, j - 1] + diff_sq

            # Also, if open_end and last row, diagonal has no diff cost
            if use_open_end and is_last_row:
                diag_cost = cost[i - 1, j - 1]
                up_cost = cost[i - 1, j] + diff_sq  # UP still has cost

            cost[i, j] = min(up_cost, diag_cost, right_cost)

    # Final accumulated cost is at cost[len1-1, len2-1]
    final_cost = cost[len1 - 1, len2 - 1]

    # Normalization: when exactly one of open_start/open_end is set
    if (use_open_end and not use_open_start) or (not use_open_end and use_open_start):
        return float(np.sqrt(final_cost) / len1)
    else:
        return float(np.sqrt(final_cost))


# ---------------------------------------------------------------------------
# CPU pairwise DTW (loop-based for open boundary support)
# ---------------------------------------------------------------------------

def _dtw_pairwise_cpu(
    sequences: np.ndarray,
    use_open_start: bool = False,
    use_open_end: bool = False,
) -> np.ndarray:
    """
    Compute pairwise DTW distances on CPU.

    Standard DTW delegates to tslearn's cdist_dtw.  When open boundaries
    are requested, falls back to a loop over _dtw_distance_cpu.
    """
    if not use_open_start and not use_open_end:
        if not TSLEARN_AVAILABLE:
            raise RuntimeError(
                "tslearn is required for CPU DTW.\n"
                "Install it with: pip install tslearn"
            )
        dataset_3d = sequences[:, :, np.newaxis]
        result = _tslearn_cdist_dtw(dataset_3d)
        return np.asarray(result, dtype=np.float64)

    n = len(sequences)
    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _dtw_distance_cpu(
                sequences[i], sequences[j],
                use_open_start=use_open_start,
                use_open_end=use_open_end,
            )
            result[i, j] = d
            result[j, i] = d
    return result


# ---------------------------------------------------------------------------
# dtw_distance (public API)
# ---------------------------------------------------------------------------

def dtw_distance(
    seq1: Union[np.ndarray, list],
    seq2: Union[np.ndarray, list],
    use_open_start: bool = False,
    use_open_end: bool = False,
    use_cuda: Optional[bool] = None,
) -> float:
    """
    Compute DTW distance between two sequences.

    Parameters
    ----------
    seq1 : array-like
        First sequence (will be converted to float32 numpy array)
    seq2 : array-like
        Second sequence (will be converted to float32 numpy array)
    use_open_start : bool, optional
        Enable open start boundary condition (default: False).
    use_open_end : bool, optional
        Enable open end boundary condition (default: False).
    use_cuda : bool or None, optional
        Backend selection:
        - None (default): auto-select (CUDA if available, else CPU)
        - True: force CUDA, raises RuntimeError if unavailable
        - False: force CPU

    Returns
    -------
    float
        DTW distance between seq1 and seq2

    Raises
    ------
    RuntimeError
        If use_cuda=True and CUDA is not available
    ValueError
        If input sequences are invalid
    """
    # --- Input conversion ---
    if not isinstance(seq1, np.ndarray):
        seq1 = np.array(seq1, dtype=np.float32)
    else:
        seq1 = np.asarray(seq1, dtype=np.float32)

    if not isinstance(seq2, np.ndarray):
        seq2 = np.array(seq2, dtype=np.float32)
    else:
        seq2 = np.asarray(seq2, dtype=np.float32)

    # Ensure arrays are contiguous
    if not seq1.flags["C_CONTIGUOUS"]:
        seq1 = np.ascontiguousarray(seq1)
    if not seq2.flags["C_CONTIGUOUS"]:
        seq2 = np.ascontiguousarray(seq2)

    # --- Input validation ---
    if seq1.ndim != 1:
        raise ValueError(f"seq1 must be 1-dimensional, got shape {seq1.shape}")
    if seq2.ndim != 1:
        raise ValueError(f"seq2 must be 1-dimensional, got shape {seq2.shape}")

    if len(seq1) == 0 or len(seq2) == 0:
        raise ValueError("Sequences cannot be empty")

    # --- Backend dispatch ---
    if use_cuda is True:
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA backend requested but not available. "
                "Install with CUDA support or use use_cuda=False for CPU."
            )
        return _dtw_distance_cuda(
            seq1, seq2,
            use_open_start=int(use_open_start),
            use_open_end=int(use_open_end),
        )

    if use_cuda is False:
        # Force CPU
        return _dtw_distance_cpu(seq1, seq2, use_open_start, use_open_end)

    # use_cuda is None: auto-select
    if CUDA_AVAILABLE:
        return _dtw_distance_cuda(
            seq1, seq2,
            use_open_start=int(use_open_start),
            use_open_end=int(use_open_end),
        )

    return _dtw_distance_cpu(seq1, seq2, use_open_start, use_open_end)


# ---------------------------------------------------------------------------
# dtw_pairwise (public API)
# ---------------------------------------------------------------------------

def dtw_pairwise(
    sequences: Union[np.ndarray, list],
    use_open_start: bool = False,
    use_open_end: bool = False,
    use_cuda: Optional[bool] = None,
) -> np.ndarray:
    """
    Compute pairwise DTW distances for a batch of sequences.

    Parameters
    ----------
    sequences : array-like
        2D array of sequences with shape (num_sequences, seq_length).
        All sequences must have the same length.
        Will be converted to float32 if needed.
    use_open_start : bool, optional
        Enable open start boundary condition (default: False).
    use_open_end : bool, optional
        Enable open end boundary condition (default: False).
    use_cuda : bool or None, optional
        Backend selection:
        - None (default): auto-select (CUDA if available, else CPU)
        - True: force CUDA, raises RuntimeError if unavailable
        - False: force CPU

    Returns
    -------
    np.ndarray
        Distance matrix of shape (num_sequences, num_sequences).
        Matrix is symmetric with zeros on the diagonal.

    Raises
    ------
    RuntimeError
        If use_cuda=True and CUDA is not available
    ValueError
        If input sequences are invalid
    """
    # --- Input conversion ---
    if not isinstance(sequences, np.ndarray):
        sequences = np.array(sequences, dtype=np.float32)
    else:
        sequences = np.asarray(sequences, dtype=np.float32)

    # --- Input validation ---
    if sequences.ndim != 2:
        raise ValueError(f"sequences must be 2D array, got shape {sequences.shape}")

    if sequences.shape[0] < 2:
        raise ValueError(f"Need at least 2 sequences, got {sequences.shape[0]}")

    if sequences.shape[1] == 0:
        raise ValueError("Sequence length cannot be 0")

    # --- Backend dispatch ---
    if use_cuda is True:
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA backend requested but not available. "
                "Install with CUDA support or use use_cuda=False for CPU."
            )
        return _dtw_pairwise_cuda(
            sequences,
            use_open_start=int(use_open_start),
            use_open_end=int(use_open_end),
        )

    if use_cuda is False:
        # Force CPU
        return _dtw_pairwise_cpu(sequences, use_open_start, use_open_end)

    # use_cuda is None: auto-select
    if CUDA_AVAILABLE:
        return _dtw_pairwise_cuda(
            sequences,
            use_open_start=int(use_open_start),
            use_open_end=int(use_open_end),
        )

    return _dtw_pairwise_cpu(sequences, use_open_start, use_open_end)


# ---------------------------------------------------------------------------
# cleanup / is_available
# ---------------------------------------------------------------------------

def cleanup():
    """
    Reset CUDA device and free all GPU resources.

    No-op when running on CPU backend.
    """
    if not CUDA_AVAILABLE:
        return  # Nothing to cleanup on CPU

    _cuda_cleanup()


def is_available() -> bool:
    """
    Check if CUDA DTW extension is available.

    Returns
    -------
    bool
        True if CUDA extension is available, False otherwise
    """
    return CUDA_AVAILABLE


__all__ = [
    "dtw_distance",
    "dtw_pairwise",
    "cleanup",
    "is_available",
    "backend",
    "CUDA_AVAILABLE",
]
