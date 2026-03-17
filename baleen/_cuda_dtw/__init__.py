"""
CUDA-accelerated Dynamic Time Warping (DTW) module

This module provides GPU-accelerated DTW distance calculation.
"""

import numpy as np
from typing import Union, Optional

# Try to import the CUDA extension
try:
    from ._cuda_dtw import dtw_distance as _dtw_distance_cuda
    from ._cuda_dtw import dtw_pairwise as _dtw_pairwise_cuda
    from ._cuda_dtw import cleanup as _cuda_cleanup

    CUDA_AVAILABLE = True
except ImportError as e:
    CUDA_AVAILABLE = False
    _import_error = str(e)

# Log backend availability upon module import
if CUDA_AVAILABLE:
    print("🚀 DTW: GPU (CUDA) acceleration ENABLED")
else:
    print("💻 DTW: CPU implementation (CUDA not available)")


def dtw_distance(
    seq1: Union[np.ndarray, list],
    seq2: Union[np.ndarray, list],
    use_open_start: bool = False,
    use_open_end: bool = False,
) -> float:
    """
    Compute DTW distance between two sequences using CUDA acceleration.

    Parameters
    ----------
    seq1 : array-like
        First sequence (will be converted to float32 numpy array)
    seq2 : array-like
        Second sequence (will be converted to float32 numpy array)
    use_open_start : bool, optional
        Enable open start boundary condition (default: False)
    use_open_end : bool, optional
        Enable open end boundary condition (default: False)

    Returns
    -------
    float
        DTW distance between seq1 and seq2

    Raises
    ------
    RuntimeError
        If CUDA extension is not available
    ValueError
        If input sequences are invalid

    Examples
    --------
    >>> import numpy as np
    >>> from baleen._cuda_dtw import dtw_distance
    >>> seq1 = np.random.randn(100).astype(np.float32)
    >>> seq2 = np.random.randn(100).astype(np.float32)
    >>> distance = dtw_distance(seq1, seq2)
    >>> print(f"DTW distance: {distance}")
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            f"CUDA DTW extension is not available.\n"
            f"Import error: {_import_error}\n\n"
            f"The extension was not built during installation. This can happen if:\n"
            f"  1. CUDA Toolkit is not installed (check with: nvcc --version)\n"
            f"  2. The build process skipped the CUDA extension\n"
            f"  3. The package was installed in a different environment\n\n"
            f"To build with CUDA support:\n"
            f"  1. Install CUDA Toolkit from NVIDIA\n"
            f"  2. Ensure nvcc is in PATH\n"
            f"  3. Reinstall: pip uninstall baleen && pip install -e .\n\n"
            f"Check availability with: baleen._cuda_dtw.is_available()"
        )

    # Convert inputs to numpy arrays if needed
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

    # Validate shapes
    if seq1.ndim != 1:
        raise ValueError(f"seq1 must be 1-dimensional, got shape {seq1.shape}")
    if seq2.ndim != 1:
        raise ValueError(f"seq2 must be 1-dimensional, got shape {seq2.shape}")

    if len(seq1) == 0 or len(seq2) == 0:
        raise ValueError("Sequences cannot be empty")

    # Call the CUDA function
    return _dtw_distance_cuda(
        seq1, seq2, use_open_start=int(use_open_start), use_open_end=int(use_open_end)
    )


def dtw_pairwise(
    sequences: Union[np.ndarray, list], use_open_start: bool = False, use_open_end: bool = False
) -> np.ndarray:
    """
    Compute pairwise DTW distances for a batch of sequences using CUDA.

    This is significantly more efficient than calling dtw_distance() in a loop,
    as it:
    - Transfers all sequences to GPU in one batch
    - Computes multiple DTW pairs in parallel
    - Amortizes memory allocation/deallocation overhead

    Parameters
    ----------
    sequences : array-like
        2D array of sequences with shape (num_sequences, seq_length)
        All sequences must have the same length
        Will be converted to float32 if needed
    use_open_start : bool, optional
        Enable open start boundary condition (default: False)
    use_open_end : bool, optional
        Enable open end boundary condition (default: False)

    Returns
    -------
    np.ndarray
        Distance matrix of shape (num_sequences, num_sequences)
        Matrix is symmetric with zeros on the diagonal

    Raises
    ------
    RuntimeError
        If CUDA extension is not available
    ValueError
        If input sequences are invalid

    Examples
    --------
    >>> import numpy as np
    >>> from baleen._cuda_dtw import dtw_pairwise
    >>> # Generate 10 sequences of length 100
    >>> sequences = np.random.randn(10, 100).astype(np.float32)
    >>> distance_matrix = dtw_pairwise(sequences)
    >>> print(f"Distance matrix shape: {distance_matrix.shape}")
    >>> # distance_matrix[i, j] is the DTW distance between sequences[i] and sequences[j]
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            f"CUDA DTW extension is not available.\n"
            f"Import error: {_import_error}\n\n"
            f"Check availability with: baleen._cuda_dtw.is_available()"
        )

    # Convert to numpy array if needed
    if not isinstance(sequences, np.ndarray):
        sequences = np.array(sequences, dtype=np.float32)
    else:
        sequences = np.asarray(sequences, dtype=np.float32)

    # Validate input
    if sequences.ndim != 2:
        raise ValueError(f"sequences must be 2D array, got shape {sequences.shape}")

    if sequences.shape[0] < 2:
        raise ValueError(f"Need at least 2 sequences, got {sequences.shape[0]}")

    if sequences.shape[1] == 0:
        raise ValueError("Sequence length cannot be 0")

    # Call the CUDA function
    return _dtw_pairwise_cuda(
        sequences, use_open_start=int(use_open_start), use_open_end=int(use_open_end)
    )


def cleanup():
    """
    Reset CUDA device and free all GPU resources.

    This function should be called when you're done using the CUDA DTW
    functionality to properly clean up GPU resources.

    Examples
    --------
    >>> from baleen._cuda_dtw import cleanup
    >>> # After computing many DTW distances...
    >>> cleanup()
    """
    if not CUDA_AVAILABLE:
        return  # Nothing to cleanup if CUDA is not available

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


__all__ = ["dtw_distance", "dtw_pairwise", "cleanup", "is_available", "CUDA_AVAILABLE"]
