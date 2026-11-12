"""
CUDA-accelerated Dynamic Time Warping (DTW) module

This module provides GPU-accelerated DTW distance calculation with
automatic CPU fallback when CUDA is not available.

CPU backend: delegates to tslearn.
"""

import logging
import subprocess
import numpy as np
from typing import Union, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

try:
    from ._cuda_dtw import dtw_distance as _dtw_distance_cuda
    from ._cuda_dtw import dtw_pairwise as _dtw_pairwise_cuda
    from ._cuda_dtw import dtw_pairwise_varlen as _dtw_pairwise_varlen_cuda
    from ._cuda_dtw import cleanup as _cuda_cleanup

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

try:
    from ._cuda_dtw import dtw_multi_position_pairwise as _dtw_multi_position_cuda
except (ImportError, AttributeError):
    _dtw_multi_position_cuda = None

try:
    from tslearn.metrics import dtw as _tslearn_dtw
    from tslearn.metrics import cdist_dtw as _tslearn_cdist_dtw

    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False

_BACKEND = "cuda" if CUDA_AVAILABLE else "cpu"

# Detect cuDTW++ (v0.2+) vs legacy OpenDBA kernel
_CUDTW_ACTIVE = False
if CUDA_AVAILABLE:
    try:
        from ._cuda_dtw import __version__ as _cuda_ver
        _CUDTW_ACTIVE = "cudtw" in _cuda_ver
    except (ImportError, AttributeError):
        pass

if _BACKEND == "cuda":
    if _CUDTW_ACTIVE:
        _log.debug("DTW backend: cuda (cuDTW++ warp-shuffle)")
    else:
        _log.debug("DTW backend: cuda (legacy OpenDBA wavefront)")
else:
    _log.debug("DTW backend: cpu (tslearn fallback)")


def backend() -> str:
    """Return the name of the active DTW backend ('cuda' or 'cpu')."""
    return _BACKEND


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_CUDTW_LEN = 2047
_BUCKETS = (127, 255, 511, 1023, 2047)


def _select_bucket(min_len: int, max_len: int) -> int:
    """Kernel width for a group of signals; mirrors select_bucket() in cudtw_wrapper.cu.

    Signals of mixed lengths need one slot of padding beyond the longest one;
    signals of equal length may fill the width exactly.
    """
    need = max_len if min_len == max_len else max_len + 1
    for b in _BUCKETS:
        if need <= b:
            return b
    return _BUCKETS[-1]  # caller is expected to resample before reaching here


def _resample_signal(sig, target_len):
    """Resample a signal to target_len using scipy."""
    from scipy.signal import resample
    return resample(sig.astype(np.float64), target_len).astype(np.float32)


def _fit_to_kernel(signals: list) -> list:
    """Resample over-long signals so the group fits one cuDTW++ kernel.

    The widest kernel holds 2047 samples.  A group of mixed lengths must leave
    one slot for padding (see ``_select_bucket``), so its signals are capped at
    2046; a group of equal lengths is only capped at 2047.
    """
    lengths = {len(s) for s in signals}
    limit = _MAX_CUDTW_LEN if len(lengths) == 1 else _MAX_CUDTW_LEN - 1
    if max(lengths) <= limit:
        return signals
    return [_resample_signal(s, limit) if len(s) > limit else s for s in signals]


# ---------------------------------------------------------------------------
# CPU DTW implementation
# ---------------------------------------------------------------------------

def _dtw_distance_cpu(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute DTW distance on CPU via tslearn."""
    if not TSLEARN_AVAILABLE:
        raise RuntimeError(
            "tslearn is required for CPU DTW.\n"
            "Install it with: pip install tslearn"
        )
    s1_2d = seq1.reshape(-1, 1)
    s2_2d = seq2.reshape(-1, 1)
    return float(_tslearn_dtw(s1_2d, s2_2d))


def _dtw_pairwise_cpu(sequences: np.ndarray) -> np.ndarray:
    """Compute pairwise DTW distances on CPU via tslearn."""
    if not TSLEARN_AVAILABLE:
        raise RuntimeError(
            "tslearn is required for CPU DTW.\n"
            "Install it with: pip install tslearn"
        )
    dataset_3d = sequences[:, :, np.newaxis]
    result = _tslearn_cdist_dtw(dataset_3d)
    return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# dtw_distance (public API)
# ---------------------------------------------------------------------------

def dtw_distance(
    seq1: Union[np.ndarray, list],
    seq2: Union[np.ndarray, list],
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
    use_cuda : bool or None, optional
        Backend selection:
        - None (default): auto-select (CUDA if available, else CPU)
        - True: force CUDA, raises RuntimeError if unavailable
        - False: force CPU

    Returns
    -------
    float
        DTW distance between seq1 and seq2
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

    if not seq1.flags["C_CONTIGUOUS"]:
        seq1 = np.ascontiguousarray(seq1)
    if not seq2.flags["C_CONTIGUOUS"]:
        seq2 = np.ascontiguousarray(seq2)

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
        return _dtw_distance_gpu(seq1, seq2)

    if use_cuda is False:
        return _dtw_distance_cpu(seq1, seq2)

    # use_cuda is None: auto-select
    if CUDA_AVAILABLE:
        return _dtw_distance_gpu(seq1, seq2)

    return _dtw_distance_cpu(seq1, seq2)


def _dtw_distance_gpu(seq1: np.ndarray, seq2: np.ndarray) -> float:
    if _CUDTW_ACTIVE:
        seq1, seq2 = _fit_to_kernel([seq1, seq2])
    return _dtw_distance_cuda(seq1, seq2)


# ---------------------------------------------------------------------------
# dtw_pairwise (public API)
# ---------------------------------------------------------------------------

def dtw_pairwise(
    sequences: Union[np.ndarray, list],
    use_cuda: Optional[bool] = None,
) -> np.ndarray:
    """
    Compute pairwise DTW distances for a batch of equal-length sequences.

    Parameters
    ----------
    sequences : array-like
        2D array of sequences with shape (num_sequences, seq_length).
        All sequences must have the same length.
    use_cuda : bool or None, optional
        Backend selection (None=auto, True=force CUDA, False=force CPU).

    Returns
    -------
    np.ndarray
        Symmetric distance matrix of shape (num_sequences, num_sequences).
    """
    # --- Input conversion ---
    if not isinstance(sequences, np.ndarray):
        sequences_list = list(sequences)
        if sequences_list and hasattr(sequences_list[0], '__len__'):
            lengths = {len(s) for s in sequences_list}
            if len(lengths) > 1:
                raise ValueError(
                    f"All sequences must have the same length; got lengths {sorted(lengths)}"
                )
        sequences = np.array(sequences, dtype=np.float32)
    else:
        sequences = np.asarray(sequences, dtype=np.float32)

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
        return _dtw_pairwise_cuda(sequences)

    if use_cuda is False:
        return _dtw_pairwise_cpu(sequences)

    if CUDA_AVAILABLE:
        return _dtw_pairwise_cuda(sequences)

    return _dtw_pairwise_cpu(sequences)


def dtw_pairwise_varlen(
    signals: list[np.ndarray],
    use_cuda: Optional[bool] = None,
) -> np.ndarray:
    """
    Compute pairwise DTW distances for variable-length sequences.

    Parameters
    ----------
    signals : list of np.ndarray
        List of 1D float32 arrays (variable lengths allowed).
    use_cuda : bool or None
        Backend selection (None=auto, True=force CUDA, False=force CPU).

    Returns
    -------
    np.ndarray
        Symmetric distance matrix of shape (N, N).
    """
    if len(signals) < 2:
        raise ValueError(f"Need at least 2 signals, got {len(signals)}")

    prepped = [np.ascontiguousarray(np.asarray(s, dtype=np.float32)) for s in signals]
    lengths = np.array([len(s) for s in prepped], dtype=np.int64)

    if any(l == 0 for l in lengths):
        raise ValueError("All signals must be non-empty")

    want_cuda = use_cuda is True or (use_cuda is None and CUDA_AVAILABLE)

    if want_cuda:
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA backend requested but not available. "
                "Install with CUDA support or use use_cuda=False."
            )

        if _CUDTW_ACTIVE:
            prepped = _fit_to_kernel(prepped)
            lengths = np.array([len(s) for s in prepped], dtype=np.int64)

        max_len = int(lengths.max())
        n = len(prepped)
        padded = np.zeros((n, max_len), dtype=np.float32)
        for i, s in enumerate(prepped):
            padded[i, :len(s)] = s
        result = _dtw_pairwise_varlen_cuda(padded, lengths)
        return np.asarray(result, dtype=np.float64)

    n = len(prepped)
    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _dtw_distance_cpu(prepped[i], prepped[j])
            result[i, j] = d
            result[j, i] = d
    return result


# ---------------------------------------------------------------------------
# dtw_multi_position_pairwise (public API)
# ---------------------------------------------------------------------------

def dtw_multi_position_pairwise(
    position_signals: list[list[np.ndarray]],
    use_cuda: Optional[bool] = None,
    num_streams: int = 16,
    device_id: int = 0,
) -> list[np.ndarray]:
    """
    Batch-compute pairwise DTW distances for multiple positions in one GPU call.

    Parameters
    ----------
    position_signals : list of list of np.ndarray
        position_signals[p][r] is the 1D float32 signal for position p, read r.
    use_cuda : bool or None
        Backend selection (None=auto, True=force CUDA, False=force CPU).
    num_streams : int
        Number of CUDA streams for concurrent processing (default 16).
    device_id : int
        GPU device ID (default 0).

    Returns
    -------
    list of np.ndarray
        Distance matrices, one per position. Each is (n_p, n_p) float64.
    """
    if len(position_signals) < 1:
        raise ValueError("Need at least 1 position, got 0")

    prepped: list[list[np.ndarray]] = []
    counts: list[int] = []
    for pos_sigs in position_signals:
        ps = [np.ascontiguousarray(np.asarray(s, dtype=np.float32)) for s in pos_sigs]
        if any(len(s) == 0 for s in ps):
            raise ValueError("All signals must be non-empty")
        prepped.append(ps)
        counts.append(len(ps))

    want_cuda = use_cuda is True or (use_cuda is None and CUDA_AVAILABLE)

    if want_cuda:
        if not CUDA_AVAILABLE or _dtw_multi_position_cuda is None:
            raise RuntimeError(
                "CUDA backend requested but not available. "
                "Install with CUDA support or use use_cuda=False."
            )

        # Each position is its own kernel group.
        if _CUDTW_ACTIVE:
            prepped = [_fit_to_kernel(pos_sigs) if pos_sigs else pos_sigs
                       for pos_sigs in prepped]

        global_max_len = max(
            len(s) for pos_sigs in prepped for s in pos_sigs
        )
        total_seqs = sum(counts)

        padded = np.zeros((total_seqs, global_max_len), dtype=np.float32)
        lengths = np.empty(total_seqs, dtype=np.int64)
        idx = 0
        for pos_sigs in prepped:
            for s in pos_sigs:
                padded[idx, :len(s)] = s
                lengths[idx] = len(s)
                idx += 1

        counts_arr = np.array(counts, dtype=np.int64)

        flat_result = _dtw_multi_position_cuda(
            padded, lengths, counts_arr,
            num_cuda_streams=num_streams,
            device_id=device_id,
        )

        result_list: list[np.ndarray] = []
        offset = 0
        for n in counts:
            mat = np.asarray(flat_result[offset:offset + n * n], dtype=np.float64).reshape(n, n)
            result_list.append(mat)
            offset += n * n
        return result_list

    # CPU fallback
    result_list = []
    for pos_sigs in prepped:
        n = len(pos_sigs)
        if n < 2:
            result_list.append(np.zeros((n, n), dtype=np.float64))
            continue
        mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = _dtw_distance_cpu(pos_sigs[i], pos_sigs[j])
                mat[i, j] = d
                mat[j, i] = d
        result_list.append(mat)
    return result_list


# ---------------------------------------------------------------------------
# cleanup / is_available
# ---------------------------------------------------------------------------

def cleanup():
    """Reset CUDA device and free all GPU resources. No-op on CPU."""
    if not CUDA_AVAILABLE:
        return
    _cuda_cleanup()


def is_available() -> bool:
    """Check if CUDA DTW extension is available."""
    return CUDA_AVAILABLE


def estimate_gpu_memory(position_signals: list[list[np.ndarray]]) -> int:
    """Estimate GPU memory bytes for a multi-position pairwise DTW call.

    The cuDTW++ wrapper allocates:
      - d_subjects: sum(n_p * bucket_p) * 4 bytes (doubles as query source)
      - d_out:      sum(n_p^2) * 4 bytes (float32 squared cost)
    No cost matrix is allocated (cuDTW++ accumulates in registers).
    """
    total_seqs = sum(len(ps) for ps in position_signals)
    input_floats = 0
    for ps in position_signals:
        if not ps:
            continue
        lengths = [len(s) for s in ps]
        input_floats += len(ps) * _select_bucket(min(lengths), max(lengths))

    input_bytes   = input_floats * 4
    output_bytes  = sum(len(ps) ** 2 for ps in position_signals) * 4
    lengths_bytes = total_seqs * 8  # host-side h_lengths array

    total = input_bytes + output_bytes + lengths_bytes
    return int(total * 1.2)  # 20% headroom for stream/kernel overhead


def get_device_count() -> int:
    """Return number of visible CUDA devices."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().split('\n') if l.strip()])
    except Exception:
        pass
    return 1 if CUDA_AVAILABLE else 0


def get_per_device_memory() -> list[int]:
    """Return total GPU memory in bytes for each visible CUDA device."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            return [int(mb) * 1024 * 1024 for mb in lines]
    except Exception:
        pass
    if CUDA_AVAILABLE:
        return [8 * 1024 ** 3]  # default 8 GB
    return []


__all__ = [
    "dtw_distance",
    "dtw_pairwise",
    "dtw_pairwise_varlen",
    "dtw_multi_position_pairwise",
    "estimate_gpu_memory",
    "cleanup",
    "is_available",
    "backend",
    "get_device_count",
    "get_per_device_memory",
    "CUDA_AVAILABLE",
]
