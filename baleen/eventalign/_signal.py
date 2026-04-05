from __future__ import annotations

import csv
import logging
import time
import typing
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
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            row_dict = typing.cast(dict[str, Optional[str]], row)
            yield EventalignRow(
                contig=row_dict.get("contig", "") or "",
                position=_parse_int(row_dict.get("position"), default=0) or 0,
                reference_kmer=row_dict.get("reference_kmer", "") or "",
                read_name=row_dict.get("read_name", "") or "",
                strand=row_dict.get("strand", "") or "",
                event_index=_parse_int(row_dict.get("event_index"), default=0) or 0,
                event_level_mean=_parse_float(row_dict.get("event_level_mean"), default=0.0),
                event_stdv=_parse_float(row_dict.get("event_stdv"), default=0.0),
                event_duration=_parse_float(row_dict.get("event_duration"), default=0.0),
                model_predict=_parse_float(row_dict.get("model_predict"), default=0.0),
                model_stdv=_parse_float(row_dict.get("model_stdv"), default=0.0),
                samples=_parse_samples(row_dict.get("samples")),
                start_idx=_parse_int(row_dict.get("start_idx"), default=None),
                end_idx=_parse_int(row_dict.get("end_idx"), default=None),
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
    for event in parse_eventalign(tsv_path):
        n_rows += 1
        # Shift from 0-based first-of-kmer to 1-based center-of-kmer
        shifted = event.position + len(event.reference_kmer) // 2 + 1
        if shifted not in grouped:
            grouped[shifted] = PositionSignals(
                contig=event.contig,
                position=shifted,
                reference_kmer=event.reference_kmer,
                read_signals={},
                read_names=[],
            )

        pos_signals = grouped[shifted]
        if event.read_name not in pos_signals.read_signals:
            pos_signals.read_signals[event.read_name] = np.array([], dtype=np.float32)
            pos_signals.read_names.append(event.read_name)

        pending[shifted][event.read_name].append((event.start_idx, event.samples))

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
