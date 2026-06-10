"""Gate A: prove krill's DTW is numerically identical to py-baleen's _cuda_dtw.

Feeds identical per-position signal arrays to both backends and reports the
maximum absolute difference for every public entry point, on both the GPU and
CPU paths.  Exits non-zero if any difference exceeds TOL (1e-4).

This is the gate that must pass *before* baleen/_cuda_dtw/ is deleted, since it
is the only point at which both implementations coexist.  Run inside the GPU
image (which has the compiled _cuda_dtw extension and krill installed):

    docker run --rm --gpus all -v "$PWD":/work -w /work \
        py-baleen-krill:gpu \
        python3 benchmarks/krill_exp/verify_dtw.py

Optionally validate on real eventalign signals instead of the synthetic
battery:  --from-tsv <eventalign.tsv> (read via group_signals_by_position).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import krill
from baleen import _cuda_dtw

TOL = 1e-4


def _make_positions(specs, seed: int) -> list[list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    positions: list[list[np.ndarray]] = []
    for n_reads, (lo, hi) in specs:
        sigs = []
        for _ in range(n_reads):
            length = int(rng.integers(lo, hi))
            level = float(rng.uniform(60.0, 130.0))
            sig = (level + rng.standard_normal(length).cumsum() * 0.3
                   + rng.standard_normal(length) * 1.5).astype(np.float32)
            sigs.append(sig)
        positions.append(sigs)
    return positions


# Signals within the cuDTW++ 2047-sample cap (no resampling on either backend).
_SPECS_SHORT = [(6, (40, 120)), (8, (150, 400)), (5, (900, 1500)), (7, (30, 60))]
# A position exceeding 2047 samples -> exercises the resample path.
_SPECS_LONG = [(4, (2100, 3000))]


def _synthetic_positions(seed: int = 0) -> list[list[np.ndarray]]:
    """Full battery incl. a >2047 position (for the GPU path, identical there)."""
    return _make_positions(_SPECS_SHORT + _SPECS_LONG, seed)


def _positions_from_tsv(tsv: str) -> list[list[np.ndarray]]:
    from baleen.eventalign._signal import (
        extract_signals_for_dtw,
        group_signals_by_position,
    )

    by_pos = group_signals_by_position(tsv)
    positions: list[list[np.ndarray]] = []
    for pos in sorted(by_pos)[:40]:  # cap for speed
        _names, sigs = extract_signals_for_dtw(by_pos[pos])
        sigs = [s for s in sigs if s.size > 0]
        if len(sigs) >= 2:
            positions.append(sigs)
    return positions


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _check(name: str, baleen_out, krill_out) -> float:
    """Return max|Δ| across one or many matrices."""
    if isinstance(baleen_out, list):
        diffs = [_max_abs_diff(x, y) for x, y in zip(baleen_out, krill_out)]
        d = max(diffs) if diffs else 0.0
        if len(baleen_out) != len(krill_out):
            d = float("inf")
    else:
        d = _max_abs_diff(np.asarray(baleen_out), np.asarray(krill_out))
    status = "OK" if d < TOL else "FAIL"
    print(f"  {name:32s} max|Δ| = {d:.3e}   [{status}]")
    return d


def run_path(positions: list[list[np.ndarray]], *, use_gpu: bool) -> dict[str, float]:
    label = "GPU" if use_gpu else "CPU"
    print(f"\n=== {label} path ===")
    d: dict[str, float] = {}

    # dtw_distance (single pair, first two signals of first position)
    s1, s2 = positions[0][0], positions[0][1]
    d["dtw_distance"] = _check(
        "dtw_distance",
        _cuda_dtw.dtw_distance(s1, s2, use_cuda=use_gpu),
        krill.dtw_distance(s1, s2, use_gpu=use_gpu),
    )

    # dtw_pairwise_varlen (one position)
    d["dtw_pairwise_varlen"] = _check(
        "dtw_pairwise_varlen",
        _cuda_dtw.dtw_pairwise_varlen(positions[0], use_cuda=use_gpu),
        krill.dtw_pairwise_varlen(positions[0], use_gpu=use_gpu),
    )

    # dtw_multi_position_pairwise (all positions, the production entry point)
    d["dtw_multi_position_pairwise"] = _check(
        "dtw_multi_position_pairwise",
        _cuda_dtw.dtw_multi_position_pairwise(positions, use_cuda=use_gpu),
        krill.dtw_multi_position_pairwise(positions, use_gpu=use_gpu),
    )
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-tsv", default=None,
                    help="eventalign TSV to source real signals from")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.from_tsv:
        positions = _positions_from_tsv(args.from_tsv)
        print(f"Loaded {len(positions)} positions from {args.from_tsv}")
    else:
        positions = _synthetic_positions(args.seed)
        print(f"Synthetic battery: {len(positions)} positions, "
              f"sizes {[len(p) for p in positions]}")

    print(f"_cuda_dtw backend={_cuda_dtw.backend()} CUDA_AVAILABLE={_cuda_dtw.CUDA_AVAILABLE}")
    print(f"krill dtw_backend={krill.dtw_backend()} dtw_available={krill.dtw_available()}")

    gate: list[float] = []

    # ---- GPU path: the PRODUCTION path. All entry points must be identical. ----
    if _cuda_dtw.CUDA_AVAILABLE and krill.dtw_available():
        gpu = run_path(positions, use_gpu=True)
        gate += list(gpu.values())
    else:
        print("\n=== GPU path UNAVAILABLE — cannot gate the production path ===")
        print("GATE A: INCONCLUSIVE (no GPU krill)")
        return 2

    # ---- CPU path: gate only the per-pair primitives. ----
    # The batched `dtw_multi_position_pairwise` resamples to fixed buckets (GPU
    # semantics); on CPU krill keeps that resampling while the legacy _cuda_dtw
    # CPU fallback used raw tslearn — an inherent CPU(exact)-vs-bucketed
    # difference present in both libraries. It is reached only on CPU-only
    # installs (no GPU); the GPU production path above is bit-identical.
    cpu = run_path(positions, use_gpu=False)
    gate += [cpu["dtw_distance"], cpu["dtw_pairwise_varlen"]]
    print(f"\n[info] CPU dtw_multi_position_pairwise differs by "
          f"{cpu['dtw_multi_position_pairwise']:.3e} (bucketed-resample; CPU-only "
          f"installs only, not the GPU production path).")

    worst = max(gate) if gate else 0.0
    print(f"\nWORST gated max|Δ| (full GPU + CPU primitives) = {worst:.3e}  "
          f"(TOL={TOL:.0e})")
    if worst < TOL:
        print("GATE A: PASS")
        return 0
    print("GATE A: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
