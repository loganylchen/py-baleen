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
    return np.array([float(token) for token in value.split(",") if token != ""], dtype=np.float32)


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
    RNA nanopore signal is generated in 3'→5' direction, i.e. reversed relative
    to the reference/transcript 5'→3' sequence direction. Eventalign row order is
    already in raw signal order, so events for a given ``(read_name, position)``
    are concatenated strictly in file order. Any sequence-direction reversal is
    intentionally left to downstream callers/pipeline steps.
    """

    grouped: dict[int, PositionSignals] = {}
    pending: defaultdict[int, defaultdict[str, list[NDArray[np.float32]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    t0 = time.perf_counter()
    n_rows = 0
    for event in parse_eventalign(tsv_path):
        n_rows += 1
        if event.position not in grouped:
            grouped[event.position] = PositionSignals(
                contig=event.contig,
                position=event.position,
                reference_kmer=event.reference_kmer,
                read_signals={},
                read_names=[],
            )

        pos_signals = grouped[event.position]
        if event.read_name not in pos_signals.read_signals:
            pos_signals.read_signals[event.read_name] = np.array([], dtype=np.float32)
            pos_signals.read_names.append(event.read_name)

        pending[event.position][event.read_name].append(event.samples)

    for position, per_read in pending.items():
        for read_name, chunks in per_read.items():
            if not chunks:
                grouped[position].read_signals[read_name] = np.array([], dtype=np.float32)
                continue
            grouped[position].read_signals[read_name] = np.concatenate(chunks).astype(np.float32, copy=False)

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
