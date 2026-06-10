"""DTW backend — thin delegation to krill's bundled DTW.

krill ships the same DTW kernels py-baleen historically maintained as the
``_cuda_dtw`` C-extension (cuDTW++ on GPU, CPU fallback otherwise).  This module
re-exports that surface with py-baleen's historical ``use_cuda=`` keyword
(krill uses ``use_gpu=``) and keeps the pure-Python GPU memory-planning helpers
the pipeline's chunking relies on (krill does not expose them).
"""
from __future__ import annotations

import logging
import subprocess
from typing import Optional

import numpy as np

try:
    import krill
except ModuleNotFoundError as exc:  # pragma: no cover - install-time guard
    raise ModuleNotFoundError(
        "baleen requires the 'krill' engine (DTW + eventalign), which is not on "
        "PyPI. Install it from the project index, e.g.:\n"
        "    pip install krill --no-deps "
        "--index-url https://loganylchen.github.io/krill-dist/cu122/simple/  "
        "(GPU/cu122)\n"
        "    pip install krill --no-deps "
        "--index-url https://loganylchen.github.io/krill-dist/simple/  (CPU)\n"
        "or use a prebuilt baleen Docker image."
    ) from exc

_log = logging.getLogger(__name__)

# True if krill's GPU DTW is usable right now (extension built + device present).
CUDA_AVAILABLE: bool = bool(krill.dtw_available())


def backend() -> str:
    """Return the backend krill auto-selection will use now ('gpu' or 'cpu')."""
    return krill.dtw_backend()


def is_available() -> bool:
    """Check if GPU DTW is usable right now."""
    return CUDA_AVAILABLE


def cleanup() -> None:
    """Release any cached GPU DTW resources."""
    krill.dtw_cleanup()


# ---------------------------------------------------------------------------
# DTW entry points (use_cuda -> use_gpu adapters over krill)
# ---------------------------------------------------------------------------

def dtw_distance(seq1, seq2, use_cuda: Optional[bool] = None) -> float:
    """DTW distance between two 1-D signals."""
    return float(krill.dtw_distance(seq1, seq2, use_gpu=use_cuda))


def dtw_pairwise(sequences, use_cuda: Optional[bool] = None) -> np.ndarray:
    """Pairwise DTW matrix for a batch of equal-length signals."""
    return krill.dtw_pairwise(sequences, use_gpu=use_cuda)


def dtw_pairwise_varlen(signals, use_cuda: Optional[bool] = None) -> np.ndarray:
    """Pairwise DTW matrix for variable-length signals."""
    return krill.dtw_pairwise_varlen(signals, use_gpu=use_cuda)


def dtw_multi_position_pairwise(
    position_signals,
    use_cuda: Optional[bool] = None,
    num_streams: int = 16,
    device_id: int = 0,
) -> list:
    """Pairwise DTW matrices for several positions in one batched call.

    Returns one (N_p, N_p) float64 matrix per position.
    """
    return krill.dtw_multi_position_pairwise(
        position_signals,
        use_gpu=use_cuda,
        num_streams=num_streams,
        device_id=device_id,
    )


# ---------------------------------------------------------------------------
# GPU memory planning (pure-Python; krill does not expose these)
# ---------------------------------------------------------------------------

_BUCKETS = (127, 255, 511, 1023, 2047)


def _select_bucket(n: int) -> int:
    for b in _BUCKETS:
        if n <= b:
            return b
    return _BUCKETS[-1]  # caller resamples before reaching here


def estimate_gpu_memory(position_signals: list[list[np.ndarray]]) -> int:
    """Estimate GPU memory bytes for a multi-position pairwise DTW call.

    The cuDTW++ kernel allocates:
      - d_subjects: total_seqs * global_bucket * 4 bytes
      - d_out:      sum(n_p^2) * 4 bytes (float32 squared cost)
    """
    total_seqs = sum(len(ps) for ps in position_signals)
    max_len = max(len(s) for ps in position_signals for s in ps)
    bucket = _select_bucket(max_len)

    input_bytes = total_seqs * bucket * 4
    output_bytes = sum(len(ps) ** 2 for ps in position_signals) * 4
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
    "CUDA_AVAILABLE",
    "backend",
    "is_available",
    "cleanup",
    "dtw_distance",
    "dtw_pairwise",
    "dtw_pairwise_varlen",
    "dtw_multi_position_pairwise",
    "estimate_gpu_memory",
    "get_device_count",
    "get_per_device_memory",
]
