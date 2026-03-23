from pathlib import Path
from collections.abc import Sequence
from typing import Union

import numpy as np
import pytest

from baleen.eventalign._signal import (
    PositionSignals,
    extract_signals_for_dtw,
    extract_signals_for_dtw_padded,
    get_common_positions,
    group_signals_by_position,
    parse_eventalign,
)

EVENTALIGN_HEADER = (
    "contig\tposition\treference_kmer\tread_name\tstrand\tevent_index\t"
    "event_level_mean\tevent_stdv\tevent_duration\tmodel_predict\tmodel_stdv\t"
    "samples\tstart_idx\tend_idx"
)


def write_test_eventalign(
    tmp_path: Path,
    filename: str,
    rows: Sequence[Union[dict[str, str], str]],
) -> Path:
    path = tmp_path / filename
    columns = EVENTALIGN_HEADER.split("\t")
    with path.open("w", encoding="utf-8") as handle:
        _ = handle.write(EVENTALIGN_HEADER + "\n")
        for row in rows:
            if isinstance(row, str):
                _ = handle.write(row + "\n")
            else:
                values = [str(row.get(col, "")) for col in columns]
                _ = handle.write("\t".join(values) + "\n")
    return path


def _base_row(**overrides: object) -> dict[str, str]:
    row: dict[str, str] = {
        "contig": "chr1",
        "position": "100",
        "reference_kmer": "ACGTA",
        "read_name": "read_001",
        "strand": "+",
        "event_index": "0",
        "event_level_mean": "75.1",
        "event_stdv": "1.2",
        "event_duration": "0.004",
        "model_predict": "74.9",
        "model_stdv": "1.1",
        "samples": "1.0,2.0,3.0",
        "start_idx": "10",
        "end_idx": "13",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


class TestParseEventalign:
    def test_parse_single_row(self, tmp_path: Path) -> None:
        path = write_test_eventalign(tmp_path, "single.tsv", [_base_row()])
        rows = list(parse_eventalign(path))

        assert len(rows) == 1
        row = rows[0]
        assert row.contig == "chr1"
        assert row.position == 100
        assert row.reference_kmer == "ACGTA"
        assert row.read_name == "read_001"
        assert row.strand == "+"
        assert row.event_index == 0
        assert row.event_level_mean == 75.1
        assert row.event_stdv == 1.2
        assert row.event_duration == 0.004
        assert row.model_predict == 74.9
        assert row.model_stdv == 1.1
        np.testing.assert_array_almost_equal(
            row.samples,
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )
        assert row.start_idx == 10
        assert row.end_idx == 13

    def test_parse_multiple_rows(self, tmp_path: Path) -> None:
        rows_in = [_base_row(read_name=f"read_{i:03d}", event_index=i, position=100 + i) for i in range(5)]
        path = write_test_eventalign(tmp_path, "multi.tsv", rows_in)
        rows_out = list(parse_eventalign(path))

        assert len(rows_out) == 5
        assert rows_out[0].read_name == "read_000"
        assert rows_out[-1].read_name == "read_004"
        assert rows_out[-1].position == 104

    def test_parse_samples(self, tmp_path: Path) -> None:
        path = write_test_eventalign(tmp_path, "samples.tsv", [_base_row(samples="1.0,2.0,3.0")])
        row = next(parse_eventalign(path))
        np.testing.assert_array_almost_equal(row.samples, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_parse_signal_index(self, tmp_path: Path) -> None:
        path = write_test_eventalign(tmp_path, "signal_index.tsv", [_base_row(start_idx=21, end_idx=42)])
        row = next(parse_eventalign(path))
        assert row.start_idx == 21
        assert row.end_idx == 42

    def test_parse_no_signal_index(self, tmp_path: Path) -> None:
        header = (
            "contig\tposition\treference_kmer\tread_name\tstrand\tevent_index\t"
            "event_level_mean\tevent_stdv\tevent_duration\tmodel_predict\tmodel_stdv\tsamples"
        )
        path = tmp_path / "no_index.tsv"
        with path.open("w", encoding="utf-8") as handle:
            _ = handle.write(header + "\n")
            _ = handle.write("chr1\t100\tACGTA\tread_001\t+\t0\t75.1\t1.2\t0.004\t74.9\t1.1\t1.0,2.0\n")

        row = next(parse_eventalign(path))
        assert row.start_idx is None
        assert row.end_idx is None

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        path = write_test_eventalign(tmp_path, "empty.tsv", [])
        assert list(parse_eventalign(path)) == []


class TestGroupSignalsByPosition:
    def test_single_read_single_position(self, tmp_path: Path) -> None:
        path = write_test_eventalign(tmp_path, "single_pos.tsv", [_base_row()])
        grouped = group_signals_by_position(path)

        assert list(grouped.keys()) == [100]
        pos = grouped[100]
        assert pos.position == 100
        assert pos.read_names == ["read_001"]
        np.testing.assert_array_almost_equal(
            pos.read_signals["read_001"],
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

    def test_multiple_reads_same_position(self, tmp_path: Path) -> None:
        rows = [
            _base_row(read_name="read_001", samples="1,2"),
            _base_row(read_name="read_002", samples="3,4"),
            _base_row(read_name="read_003", samples="5,6"),
        ]
        path = write_test_eventalign(tmp_path, "same_pos_multi_reads.tsv", rows)
        grouped = group_signals_by_position(path)

        pos = grouped[100]
        assert len(pos.read_signals) == 3
        assert pos.read_names == ["read_001", "read_002", "read_003"]

    def test_multiple_positions(self, tmp_path: Path) -> None:
        rows = [
            _base_row(position=100, read_name="read_001"),
            _base_row(position=200, read_name="read_002"),
            _base_row(position=300, read_name="read_003"),
        ]
        path = write_test_eventalign(tmp_path, "multi_pos.tsv", rows)
        grouped = group_signals_by_position(path)

        assert sorted(grouped.keys()) == [100, 200, 300]

    def test_same_read_multiple_events_concatenated(self, tmp_path: Path) -> None:
        """Events are concatenated in file order when start_idx matches event_index."""
        rows = [
            _base_row(read_name="read_001", position=100, event_index=0, start_idx=10, samples="1,2,3"),
            _base_row(read_name="read_001", position=100, event_index=1, start_idx=20, samples="4,5,6"),
        ]
        path = write_test_eventalign(tmp_path, "concat.tsv", rows)
        grouped = group_signals_by_position(path)

        np.testing.assert_array_almost_equal(
            grouped[100].read_signals["read_001"],
            np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        )

    def test_concatenate_sorted_by_start_idx(self, tmp_path: Path) -> None:
        """Events should be concatenated in start_idx order, not file order."""
        rows = [
            _base_row(read_name="read_001", position=100, start_idx=30, samples="7,8,9"),
            _base_row(read_name="read_001", position=100, start_idx=10, samples="1,2,3"),
            _base_row(read_name="read_001", position=100, start_idx=20, samples="4,5,6"),
        ]
        path = write_test_eventalign(tmp_path, "sorted.tsv", rows)
        grouped = group_signals_by_position(path)

        # Should be sorted by start_idx: 10, 20, 30 → [1,2,3,4,5,6,7,8,9]
        np.testing.assert_array_almost_equal(
            grouped[100].read_signals["read_001"],
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32),
        )

    def test_concatenate_mixed_start_idx_none(self, tmp_path: Path) -> None:
        """Events missing start_idx raise RuntimeError (f5c must be run with --signal-index)."""
        header = EVENTALIGN_HEADER
        path = tmp_path / "mixed.tsv"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(header + "\n")
            handle.write("chr1\t100\tACGTA\tread_001\t+\t0\t75.1\t1.2\t0.004\t74.9\t1.1\t4,5,6\t20\t26\n")
            # Row with no start_idx (empty)
            handle.write("chr1\t100\tACGTA\tread_001\t+\t1\t75.1\t1.2\t0.004\t74.9\t1.1\t7,8\t\t\n")
            handle.write("chr1\t100\tACGTA\tread_001\t+\t2\t75.1\t1.2\t0.004\t74.9\t1.1\t1,2,3\t10\t16\n")

        with pytest.raises(RuntimeError, match="start_idx"):
            group_signals_by_position(path)

    def test_concatenate_all_start_idx_none(self, tmp_path: Path) -> None:
        """All events missing start_idx raise RuntimeError."""
        header = EVENTALIGN_HEADER
        path = tmp_path / "all_none.tsv"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(header + "\n")
            handle.write("chr1\t100\tACGTA\tread_001\t+\t0\t75.1\t1.2\t0.004\t74.9\t1.1\t7,8,9\t\t\n")
            handle.write("chr1\t100\tACGTA\tread_001\t+\t1\t75.1\t1.2\t0.004\t74.9\t1.1\t1,2,3\t\t\n")
            handle.write("chr1\t100\tACGTA\tread_001\t+\t2\t75.1\t1.2\t0.004\t74.9\t1.1\t4,5,6\t\t\n")

        with pytest.raises(RuntimeError, match="start_idx"):
            group_signals_by_position(path)

    def test_read_order_preserved(self, tmp_path: Path) -> None:
        rows = [
            _base_row(read_name="read_B", samples="1"),
            _base_row(read_name="read_A", samples="2"),
            _base_row(read_name="read_C", samples="3"),
        ]
        path = write_test_eventalign(tmp_path, "read_order.tsv", rows)
        grouped = group_signals_by_position(path)

        assert grouped[100].read_names == ["read_B", "read_A", "read_C"]

    def test_kmer_from_first_event(self, tmp_path: Path) -> None:
        rows = [
            _base_row(reference_kmer="AAAAA", read_name="read_001", event_index=0),
            _base_row(reference_kmer="CCCCC", read_name="read_002", event_index=1),
        ]
        path = write_test_eventalign(tmp_path, "kmer.tsv", rows)
        grouped = group_signals_by_position(path)

        assert grouped[100].reference_kmer == "AAAAA"


class TestExtractSignalsForDTW:
    def test_extract_returns_matching_order(self) -> None:
        position = PositionSignals(
            contig="chr1",
            position=100,
            reference_kmer="ACGTA",
            read_signals={
                "read_1": np.array([1.0, 2.0], dtype=np.float32),
                "read_2": np.array([3.0], dtype=np.float32),
            },
            read_names=["read_1", "read_2"],
        )
        read_names, signals = extract_signals_for_dtw(position)

        assert read_names == ["read_1", "read_2"]
        assert len(signals) == 2
        np.testing.assert_array_almost_equal(signals[0], np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(signals[1], np.array([3.0], dtype=np.float32))

    def test_extract_signal_dtypes(self) -> None:
        read_1 = np.asarray([1.0, 2.0], dtype=np.float32)
        read_2 = np.asarray([[3.0], [4.0]], dtype=np.float32)
        position = PositionSignals(
            contig="chr1",
            position=100,
            reference_kmer="ACGTA",
            read_signals={
                "read_1": read_1,
                "read_2": read_2,
            },
            read_names=["read_1", "read_2"],
        )
        _, signals = extract_signals_for_dtw(position)

        for signal in signals:
            assert signal.dtype == np.float32
            assert signal.ndim == 1

    def test_extract_empty_position(self) -> None:
        position = PositionSignals(
            contig="chr1",
            position=100,
            reference_kmer="ACGTA",
            read_signals={},
            read_names=[],
        )
        read_names, signals = extract_signals_for_dtw(position)

        assert read_names == []
        assert signals == []


class TestGetCommonPositions:
    def _dummy(self, pos: int) -> PositionSignals:
        return PositionSignals(
            contig="chr1",
            position=pos,
            reference_kmer="AAAAA",
            read_signals={},
            read_names=[],
        )

    def test_full_overlap(self) -> None:
        native = {100: self._dummy(100), 200: self._dummy(200)}
        ivt = {100: self._dummy(100), 200: self._dummy(200)}
        assert get_common_positions(native, ivt) == [100, 200]

    def test_partial_overlap(self) -> None:
        native = {100: self._dummy(100), 200: self._dummy(200), 300: self._dummy(300)}
        ivt = {200: self._dummy(200), 300: self._dummy(300), 400: self._dummy(400)}
        assert get_common_positions(native, ivt) == [200, 300]

    def test_no_overlap(self) -> None:
        native = {100: self._dummy(100)}
        ivt = {200: self._dummy(200)}
        assert get_common_positions(native, ivt) == []

    def test_sorted_output(self) -> None:
        native = {300: self._dummy(300), 100: self._dummy(100), 200: self._dummy(200)}
        ivt = {200: self._dummy(200), 100: self._dummy(100), 300: self._dummy(300)}
        assert get_common_positions(native, ivt) == [100, 200, 300]


class TestExtractSignalsForDTWPadded:
    def _make_positions(self) -> dict[int, PositionSignals]:
        return {
            100: PositionSignals(
                contig="chr1", position=100, reference_kmer="AAAAA",
                read_signals={
                    "r1": np.array([1.0, 2.0], dtype=np.float32),
                    "r2": np.array([10.0], dtype=np.float32),
                },
                read_names=["r1", "r2"],
            ),
            101: PositionSignals(
                contig="chr1", position=101, reference_kmer="AACGT",
                read_signals={
                    "r1": np.array([3.0, 4.0], dtype=np.float32),
                    "r2": np.array([20.0, 30.0], dtype=np.float32),
                },
                read_names=["r1", "r2"],
            ),
            102: PositionSignals(
                contig="chr1", position=102, reference_kmer="CGTAA",
                read_signals={
                    "r1": np.array([5.0], dtype=np.float32),
                },
                read_names=["r1"],
            ),
        }

    def test_padding_zero_same_as_unpadded(self) -> None:
        positions = self._make_positions()
        names_p, sigs_p = extract_signals_for_dtw_padded(positions, 100, padding=0)
        names_u, sigs_u = extract_signals_for_dtw(positions[100])
        assert names_p == names_u
        for a, b in zip(sigs_p, sigs_u):
            np.testing.assert_array_equal(a, b)

    def test_padding_one_concatenates_neighbors(self) -> None:
        # Descending position order (RNA temporal order): 102 → 101 → 100
        positions = self._make_positions()
        names, sigs = extract_signals_for_dtw_padded(positions, 101, padding=1)
        assert names == ["r1", "r2"]
        np.testing.assert_array_almost_equal(
            sigs[0], np.array([5.0, 3.0, 4.0, 1.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_array_almost_equal(
            sigs[1], np.array([20.0, 30.0, 10.0], dtype=np.float32),
        )

    def test_padding_skips_missing_neighbor(self) -> None:
        # target=100, padding=1 → window 101 → 100 → 99 (99 missing)
        positions = self._make_positions()
        names, sigs = extract_signals_for_dtw_padded(positions, 100, padding=1)
        assert names == ["r1", "r2"]
        np.testing.assert_array_almost_equal(
            sigs[0], np.array([3.0, 4.0, 1.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_array_almost_equal(
            sigs[1], np.array([20.0, 30.0, 10.0], dtype=np.float32),
        )

    def test_padding_read_absent_in_neighbor(self) -> None:
        # target=102, padding=1 → window 103 → 102 → 101; r2 absent at center
        positions = self._make_positions()
        names, sigs = extract_signals_for_dtw_padded(positions, 102, padding=1)
        assert names == ["r1"]
        np.testing.assert_array_almost_equal(
            sigs[0], np.array([5.0, 3.0, 4.0], dtype=np.float32),
        )

    def test_padding_large_window(self) -> None:
        # Large padding; only positions 100/101/102 exist, descending: 102 → 101 → 100
        positions = self._make_positions()
        names, sigs = extract_signals_for_dtw_padded(positions, 101, padding=5)
        assert names == ["r1", "r2"]
        np.testing.assert_array_almost_equal(
            sigs[0], np.array([5.0, 3.0, 4.0, 1.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_array_almost_equal(
            sigs[1], np.array([20.0, 30.0, 10.0], dtype=np.float32),
        )

    def test_padding_negative_raises(self) -> None:
        import pytest
        positions = self._make_positions()
        with pytest.raises(ValueError, match="padding must be >= 0"):
            extract_signals_for_dtw_padded(positions, 100, padding=-1)

    def test_target_not_in_positions(self) -> None:
        positions = self._make_positions()
        names, sigs = extract_signals_for_dtw_padded(positions, 999, padding=1)
        assert names == []
        assert sigs == []

    def test_only_center_position_exists(self) -> None:
        positions = {
            50: PositionSignals(
                contig="chr1", position=50, reference_kmer="TTTTT",
                read_signals={"r1": np.array([7.0, 8.0], dtype=np.float32)},
                read_names=["r1"],
            ),
        }
        names, sigs = extract_signals_for_dtw_padded(positions, 50, padding=2)
        assert names == ["r1"]
        np.testing.assert_array_almost_equal(
            sigs[0], np.array([7.0, 8.0], dtype=np.float32),
        )
