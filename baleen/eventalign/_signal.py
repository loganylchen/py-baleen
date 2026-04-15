from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class EventalignRow:
    contig: str
    position: int
    reference_kmer: str
    read_name: str
    strand: str
    event_index: int
    event_level_mean: float
    event_stdv: float
    event_duration: float
    model_predict: float
    model_stdv: float
    samples: NDArray[np.float32]
    start_idx: Optional[int]
    end_idx: Optional[int]


@dataclass
class PositionSignals:
    contig: str
    position: int
    reference_kmer: str
    read_signals: dict[str, NDArray[np.float32]]
    read_names: list[str] = field(default_factory=list)


def _parse_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    return int(value)


def _parse_float(value: Optional[str], default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _parse_samples(value: Optional[str]) -> NDArray[np.float32]:
    if value is None or value.strip() == "":
        return np.array([], dtype=np.float32)
    try:
        return np.fromstring(value, dtype=np.float32, sep=",")
    except ValueError:
        return np.array(
            [float(token) for token in value.split(",") if token != ""],
            dtype=np.float32,
        )


def parse_eventalign(tsv_path: Path) -> Generator[EventalignRow, None, None]:
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            return
        col_idx = {name: i for i, name in enumerate(header_line.rstrip("\n\r").split("\t"))}
        _g = col_idx.get
        ic, ip, ik, ir = _g("contig"), _g("position"), _g("reference_kmer"), _g("read_name")
        ist, iei = _g("strand"), _g("event_index")
        ielm, ies, ied = _g("event_level_mean"), _g("event_stdv"), _g("event_duration")
        imp, ims = _g("model_predict"), _g("model_stdv")
        isa, isi, iend = _g("samples"), _g("start_idx"), _g("end_idx")

        for line in handle:
            f = line.rstrip("\n\r").split("\t")
            n = len(f)
            yield EventalignRow(
                contig=f[ic] if ic is not None and ic < n and f[ic] else "",
                position=int(f[ip]) if ip is not None and ip < n and f[ip] else 0,
                reference_kmer=f[ik] if ik is not None and ik < n and f[ik] else "",
                read_name=f[ir] if ir is not None and ir < n and f[ir] else "",
                strand=f[ist] if ist is not None and ist < n and f[ist] else "",
                event_index=int(f[iei]) if iei is not None and iei < n and f[iei] else 0,
                event_level_mean=float(f[ielm]) if ielm is not None and ielm < n and f[ielm] else 0.0,
                event_stdv=float(f[ies]) if ies is not None and ies < n and f[ies] else 0.0,
                event_duration=float(f[ied]) if ied is not None and ied < n and f[ied] else 0.0,
                model_predict=float(f[imp]) if imp is not None and imp < n and f[imp] else 0.0,
                model_stdv=float(f[ims]) if ims is not None and ims < n and f[ims] else 0.0,
                samples=_parse_samples(f[isa] if isa is not None and isa < n else None),
                start_idx=int(f[isi]) if isi is not None and isi < n and f[isi] else None,
                end_idx=int(f[iend]) if iend is not None and iend < n and f[iend] else None,
            )


def group_signals_by_position(tsv_path: Path) -> dict[int, PositionSignals]:
    """Group eventalign samples by genomic position and read.

    Notes
    -----
    Events for the same ``(read_name, position)`` are concatenated in
    ascending ``start_idx`` order (= temporal order in the raw signal).

    For RNA nanopore, f5c eventalign writes events in ascending
    ``event_index`` order within a position, which is the *reverse* of
    temporal order (lower ``start_idx`` = earlier in time, but higher
    ``event_index`` for RNA because the strand threads 3'→5').  Sorting
    by ``start_idx`` corrects this.

    ``start_idx`` is always present because ``run_eventalign`` calls f5c
    with ``--signal-index`` unconditionally.  A missing ``start_idx``
    raises ``RuntimeError``.
    """

    grouped: dict[int, PositionSignals] = {}
    # Store (start_idx, samples) tuples for sorting
    pending: defaultdict[int, defaultdict[str, list[tuple[Optional[int], NDArray[np.float32]]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    t0 = time.perf_counter()
    n_rows = 0

    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            return grouped
        col_idx = {name: i for i, name in enumerate(header_line.rstrip("\n\r").split("\t"))}
        ic = col_idx["contig"]
        ip = col_idx["position"]
        ik = col_idx["reference_kmer"]
        ir = col_idx["read_name"]
        isi = col_idx.get("start_idx")
        isa = col_idx["samples"]

        for line in handle:
            n_rows += 1
            f = line.rstrip("\n\r").split("\t")
            contig = f[ic]
            position = int(f[ip])
            reference_kmer = f[ik]
            read_name = f[ir]
            si_str = f[isi] if isi is not None else ""
            start_idx: Optional[int] = int(si_str) if si_str else None
            samples = _parse_samples(f[isa])

            # Shift from 0-based first-of-kmer to 1-based center-of-kmer
            shifted = position + len(reference_kmer) // 2 + 1
            if shifted not in grouped:
                grouped[shifted] = PositionSignals(
                    contig=contig,
                    position=shifted,
                    reference_kmer=reference_kmer,
                    read_signals={},
                    read_names=[],
                )

            pos_signals = grouped[shifted]
            if read_name not in pos_signals.read_signals:
                pos_signals.read_signals[read_name] = np.array([], dtype=np.float32)
                pos_signals.read_names.append(read_name)

            pending[shifted][read_name].append((start_idx, samples))

    for position, per_read in pending.items():
        for read_name, chunks in per_read.items():
            if not chunks:
                grouped[position].read_signals[read_name] = np.array([], dtype=np.float32)
                continue

            # start_idx is always present (f5c is called with --signal-index).
            # Sorting ascending = temporal order; for RNA, file order (event_index
            # ascending) is the *reverse* of temporal order.
            if any(idx is None for idx, _ in chunks):
                raise RuntimeError(
                    f"Events at position {position} for read '{read_name}' are missing "
                    "start_idx. Ensure f5c was run with --signal-index."
                )
            chunks.sort(key=lambda x: x[0])
            sorted_signals: list[NDArray[np.float32]] = [s for _, s in chunks]
            grouped[position].read_signals[read_name] = np.concatenate(sorted_signals).astype(np.float32, copy=False)

    total_pairs = sum(len(ps.read_names) for ps in grouped.values())
    elapsed = time.perf_counter() - t0
    logger.info(
        "Parsed %d rows → %d positions, %d read-position pairs from %s (%.1fs)",
        n_rows, len(grouped), total_pairs, tsv_path, elapsed,
    )
    return grouped


def extract_signals_for_dtw(
    position_signals: PositionSignals,
) -> tuple[list[str], list[NDArray[np.float32]]]:
    read_names = list(position_signals.read_names)
    signals: list[NDArray[np.float32]] = []
    for read_name in read_names:
        signal = np.asarray(position_signals.read_signals[read_name], dtype=np.float32)
        if signal.ndim != 1:
            signal = signal.reshape(-1)
        signals.append(signal)
    return read_names, signals


def extract_signals_for_dtw_padded(
    all_positions: dict[int, PositionSignals],
    target_position: int,
    padding: int,
) -> tuple[list[str], list[NDArray[np.float32]]]:
    """Extract per-read signals with neighboring-position padding.

    For each read present at *target_position*, concatenate the signal from
    positions ``[target + padding, ..., target, ..., target - padding]`` (in
    descending genomic-position order = temporal order for RNA).  For RNA
    nanopore (3'→5'), higher genomic position is encountered earlier in the
    raw signal.  Neighbor positions where the read has no signal are simply
    skipped — no zero-fill is applied.

    Parameters
    ----------
    all_positions : dict[int, PositionSignals]
        Complete position→signals mapping (from ``group_signals_by_position``).
    target_position : int
        The centre position to extract.
    padding : int
        Number of flanking positions on each side.  ``padding=0`` is equivalent
        to the plain :func:`extract_signals_for_dtw`.

    Returns
    -------
    tuple[list[str], list[NDArray[np.float32]]]
        Read names and their padded signal arrays.
    """
    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding}")

    if target_position not in all_positions:
        return [], []

    center = all_positions[target_position]
    read_names = list(center.read_names)

    if padding == 0:
        return extract_signals_for_dtw(center)

    # For RNA nanopore (3'→5'), higher genomic position = earlier in time.
    # Iterate descending so the concatenated signal is in temporal order.
    window_positions = list(range(target_position + padding, target_position - padding - 1, -1))

    signals: list[NDArray[np.float32]] = []
    for read_name in read_names:
        chunks: list[NDArray[np.float32]] = []
        for pos in window_positions:
            pos_data = all_positions.get(pos)
            if pos_data is None:
                continue
            sig = pos_data.read_signals.get(read_name)
            if sig is None:
                continue
            arr = np.asarray(sig, dtype=np.float32)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            if arr.size > 0:
                chunks.append(arr)

        if chunks:
            signals.append(np.concatenate(chunks))
        else:
            signals.append(np.array([], dtype=np.float32))

    return read_names, signals


def get_common_positions(
    native_signals: dict[int, PositionSignals],
    ivt_signals: dict[int, PositionSignals],
) -> list[int]:
    common = sorted(set(native_signals).intersection(ivt_signals))
    logger.info(
        "Found %d common positions out of %d native, %d ivt positions",
        len(common),
        len(native_signals),
        len(ivt_signals),
    )
    return common
