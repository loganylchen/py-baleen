from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

pipeline_mod = importlib.import_module("baleen.eventalign._pipeline")
ContigResult = pipeline_mod.ContigResult
PipelineMetadata = pipeline_mod.PipelineMetadata
PositionResult = pipeline_mod.PositionResult
_compute_pairwise_distances = pipeline_mod._compute_pairwise_distances
load_results = pipeline_mod.load_results
run_pipeline = pipeline_mod.run_pipeline
save_results = pipeline_mod.save_results


EVENTALIGN_HEADER = (
    "contig\tposition\treference_kmer\tread_name\tstrand\tevent_index\t"
    "event_level_mean\tevent_stdv\tevent_duration\tmodel_predict\tmodel_stdv\t"
    "samples\tstart_idx\tend_idx"
)


def _write_eventalign_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = EVENTALIGN_HEADER.split("\t")
    with path.open("w", encoding="utf-8") as handle:
        _ = handle.write(EVENTALIGN_HEADER + "\n")
        for row in rows:
            values = [str(row.get(col, "")) for col in columns]
            _ = handle.write("\t".join(values) + "\n")


def _event_row(**overrides: object) -> dict[str, str]:
    base: dict[str, str] = {
        "contig": "ctg1",
        "position": "10",
        "reference_kmer": "AAAAA",
        "read_name": "read_001",
        "strand": "t",
        "event_index": "0",
        "event_level_mean": "75.1",
        "event_stdv": "1.2",
        "event_duration": "0.004",
        "model_predict": "74.9",
        "model_stdv": "1.1",
        "samples": "1.0,2.0",
        "start_idx": "10",
        "end_idx": "12",
    }
    base.update({k: str(v) for k, v in overrides.items()})
    return base


def _create_test_bam(
    tmp_path: Path,
    bam_name: str,
    contig_reads: dict[str, list[tuple[int, str]]],
    contigs_info: list[tuple[str, int]],
) -> Path:
    pysam_mod = importlib.import_module("pysam")

    header = {
        "HD": {"VN": "1.0", "SO": "unsorted"},
        "SQ": [{"SN": name, "LN": length} for name, length in contigs_info],
    }
    contig_to_id = {name: i for i, (name, _) in enumerate(contigs_info)}

    unsorted_bam = tmp_path / f"{bam_name}.unsorted.bam"
    sorted_bam = tmp_path / f"{bam_name}.bam"

    with pysam_mod.AlignmentFile(str(unsorted_bam), "wb", header=header) as bam:
        for contig, reads in contig_reads.items():
            for i, (pos, seq) in enumerate(reads):
                aln = pysam_mod.AlignedSegment()
                aln.query_name = f"{bam_name}_{contig}_read_{i}"
                aln.query_sequence = seq
                aln.flag = 0
                aln.reference_id = contig_to_id[contig]
                aln.reference_start = int(pos)
                aln.mapping_quality = 60
                aln.cigarstring = f"{len(seq)}M"
                aln.query_qualities = pysam_mod.qualitystring_to_array("I" * len(seq))
                bam.write(aln)

    pysam_mod.sort("-o", str(sorted_bam), str(unsorted_bam))
    pysam_mod.index(str(sorted_bam))
    unsorted_bam.unlink()
    return sorted_bam


class TestPositionResult:
    def test_creation(self) -> None:
        matrix = np.zeros((3, 3), dtype=np.float64)
        result = PositionResult(
            position=123,
            reference_kmer="ACGTA",
            n_native_reads=1,
            n_ivt_reads=2,
            native_read_names=["native_1"],
            ivt_read_names=["ivt_1", "ivt_2"],
            distance_matrix=matrix,
        )
        assert result.position == 123
        assert result.distance_matrix.shape == (3, 3)


class TestContigResult:
    def test_creation(self) -> None:
        contig_result = ContigResult(
            contig="ctg1",
            native_depth=20.0,
            ivt_depth=21.0,
            positions={},
        )
        assert contig_result.contig == "ctg1"
        assert contig_result.positions == {}


class TestPipelineMetadata:
    def test_creation(self) -> None:
        metadata = PipelineMetadata(
            f5c_version="1.6",
            min_depth=15,
            use_cuda=None,
            padding=0,
            n_contigs_total=5,
            n_contigs_passed_filter=2,
            n_contigs_skipped=3,
            filter_results=[],
        )
        assert metadata.f5c_version == "1.6"
        assert metadata.n_contigs_skipped == 3


class TestComputePairwiseDistances:
    def test_standard_dtw_uses_batch_cdist(self) -> None:
        """Standard DTW (no open boundaries, CPU) should use batch cdist_dtw, not loop."""
        signals = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1.5, 2.5, 3.5], dtype=np.float32),
            np.array([2.0, 3.0, 4.0], dtype=np.float32),
        ]
        with patch("baleen.eventalign._pipeline._dtw_distance") as mock_dtw:
            matrix = _compute_pairwise_distances(
                signals,
                use_cuda=False,
                use_open_start=False,
                use_open_end=False,
            )
        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 0.0)
        assert np.allclose(matrix, matrix.T)
        mock_dtw.assert_not_called()

    def test_standard_dtw_variable_length_matches_pairwise_loop(self) -> None:
        """Batch cdist_dtw with variable-length signals matches individual DTW calls."""
        signals = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
        ]
        matrix = _compute_pairwise_distances(
            signals,
            use_cuda=False,
            use_open_start=False,
            use_open_end=False,
        )
        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 0.0)
        assert np.allclose(matrix, matrix.T)
        from tslearn.metrics import dtw as tslearn_dtw
        for i in range(3):
            for j in range(i + 1, 3):
                expected = tslearn_dtw(
                    signals[i].reshape(-1, 1),
                    signals[j].reshape(-1, 1),
                )
                np.testing.assert_allclose(
                    matrix[i, j], expected, rtol=1e-5,
                    err_msg=f"Mismatch at ({i},{j})",
                )

    def test_open_end_uses_loop_path(self) -> None:
        """Open-boundary DTW must use per-pair loop (no batch shortcut)."""
        signals = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
        ]

        def fake_dtw(a: NDArray[np.float32], b: NDArray[np.float32], **_: object) -> float:
            return float(abs(len(a) - len(b)))

        with patch("baleen.eventalign._pipeline._dtw_distance", side_effect=fake_dtw) as mock_dtw:
            matrix = _compute_pairwise_distances(
                signals,
                use_cuda=None,
                use_open_start=False,
                use_open_end=True,
            )

        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 0.0)
        assert matrix[0, 1] == 1.0
        assert matrix[1, 2] == 2.0
        assert np.allclose(matrix, matrix.T)
        assert mock_dtw.call_count == 3

    def test_open_start_uses_loop_path(self) -> None:
        """Open-start DTW must use per-pair loop."""
        signals = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0, 5.0], dtype=np.float32),
        ]

        def fake_dtw(a: NDArray[np.float32], b: NDArray[np.float32], **_: object) -> float:
            return 42.0

        with patch("baleen.eventalign._pipeline._dtw_distance", side_effect=fake_dtw) as mock_dtw:
            matrix = _compute_pairwise_distances(
                signals,
                use_cuda=None,
                use_open_start=True,
                use_open_end=False,
            )

        assert matrix.shape == (2, 2)
        assert matrix[0, 1] == 42.0
        assert mock_dtw.call_count == 1

    def test_cuda_uses_loop_path(self) -> None:
        """CUDA backend must use per-pair loop (CUDA expects equal-length)."""
        signals = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ]

        def fake_dtw(a: NDArray[np.float32], b: NDArray[np.float32], **_: object) -> float:
            return 99.0

        with patch("baleen.eventalign._pipeline._dtw_distance", side_effect=fake_dtw) as mock_dtw:
            matrix = _compute_pairwise_distances(
                signals,
                use_cuda=True,
                use_open_start=False,
                use_open_end=False,
            )

        assert matrix.shape == (2, 2)
        assert matrix[0, 1] == 99.0
        assert mock_dtw.call_count == 1

    def test_two_signals_batch_path(self) -> None:
        """Batch path works correctly with minimum 2 signals."""
        signals = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0], dtype=np.float32),
        ]
        matrix = _compute_pairwise_distances(
            signals,
            use_cuda=False,
            use_open_start=False,
            use_open_end=False,
        )
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 0.0
        assert matrix[1, 1] == 0.0
        assert matrix[0, 1] > 0.0
        assert matrix[0, 1] == matrix[1, 0]

    def test_many_signals_batch_correctness(self) -> None:
        """Batch path is correct for a larger number of variable-length signals."""
        rng = np.random.RandomState(42)
        signals = [
            rng.randn(rng.randint(5, 30)).astype(np.float32)
            for _ in range(20)
        ]
        matrix = _compute_pairwise_distances(
            signals,
            use_cuda=False,
            use_open_start=False,
            use_open_end=False,
        )
        assert matrix.shape == (20, 20)
        assert np.allclose(np.diag(matrix), 0.0)
        assert np.allclose(matrix, matrix.T)
        off_diag = matrix[np.triu_indices(20, k=1)]
        assert np.all(off_diag > 0.0)


class TestRunPipeline:
    def test_success_single_contig_two_positions(self, tmp_path: Path) -> None:
        native_bam = _create_test_bam(
            tmp_path,
            "native",
            {"ctg1": [(0, "A" * 20), (0, "A" * 20)]},
            [("ctg1", 20)],
        )
        ivt_bam = _create_test_bam(
            tmp_path,
            "ivt",
            {"ctg1": [(0, "C" * 20), (0, "C" * 20)]},
            [("ctg1", 20)],
        )

        native_rows = [
            _event_row(position=10, read_name="n1", samples="1,2"),
            _event_row(position=10, read_name="n2", samples="1,2,3"),
            _event_row(position=20, read_name="n1", samples="3,4"),
            _event_row(position=20, read_name="n2", samples="3,4,5"),
        ]
        ivt_rows = [
            _event_row(position=10, read_name="i1", samples="2,3"),
            _event_row(position=10, read_name="i2", samples="2,3,4"),
            _event_row(position=20, read_name="i1", samples="4,5"),
            _event_row(position=20, read_name="i2", samples="4,5,6"),
        ]

        native_tsv = tmp_path / "native_eventalign.tsv"
        ivt_tsv = tmp_path / "ivt_eventalign.tsv"
        _write_eventalign_tsv(native_tsv, native_rows)
        _write_eventalign_tsv(ivt_tsv, ivt_rows)

        def fake_run_eventalign(
            bam: Path,
            ref_fasta: Path,
            fastq: Path,
            blow5: Path,
            output_tsv: Path,
            **kwargs: object,
        ) -> Path:
            del ref_fasta, blow5, kwargs, bam
            src = native_tsv if "native" in str(fastq) else ivt_tsv
            output_tsv.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            return output_tsv

        def fake_dtw(a: NDArray[np.float32], b: NDArray[np.float32], **kwargs: object) -> float:
            del kwargs
            return float(abs(len(a) - len(b)) + np.mean(np.abs(a[0] - b[0])))

        with (
            patch("baleen.eventalign._pipeline._f5c.check_f5c", return_value="1.6"),
            patch("baleen.eventalign._pipeline._f5c.index_fastq_blow5"),
            patch("baleen.eventalign._pipeline._f5c.index_blow5"),
            patch("baleen.eventalign._pipeline._f5c.run_eventalign", side_effect=fake_run_eventalign),
            patch("baleen.eventalign._pipeline._dtw_distance", side_effect=fake_dtw),
        ):
            results, metadata = run_pipeline(
                native_bam=native_bam,
                native_fastq=tmp_path / "native.fastq",
                native_blow5=tmp_path / "native.blow5",
                ivt_bam=ivt_bam,
                ivt_fastq=tmp_path / "ivt.fastq",
                ivt_blow5=tmp_path / "ivt.blow5",
                ref_fasta=tmp_path / "ref.fa",
                min_depth=1,
            )

        assert metadata.f5c_version == "1.6"
        assert metadata.n_contigs_passed_filter == 1
        assert set(results.keys()) == {"ctg1"}
        contig_result = results["ctg1"]
        assert set(contig_result.positions.keys()) == {10, 20}
        pos10 = contig_result.positions[10]
        assert pos10.n_native_reads == 2
        assert pos10.n_ivt_reads == 2
        assert pos10.distance_matrix.shape == (4, 4)

    def test_empty_results_when_no_contig_passes_filter(self, tmp_path: Path) -> None:
        native_bam = _create_test_bam(
            tmp_path,
            "native",
            {"ctg1": [(0, "A" * 20)]},
            [("ctg1", 20)],
        )
        ivt_bam = _create_test_bam(
            tmp_path,
            "ivt",
            {"ctg1": [(0, "C" * 20)]},
            [("ctg1", 20)],
        )

        with (
            patch("baleen.eventalign._pipeline._f5c.check_f5c", return_value="1.6"),
            patch("baleen.eventalign._pipeline._f5c.index_fastq_blow5"),
            patch("baleen.eventalign._pipeline._f5c.index_blow5"),
            patch("baleen.eventalign._pipeline._f5c.run_eventalign") as mock_run_eventalign,
        ):
            results, metadata = run_pipeline(
                native_bam=native_bam,
                native_fastq=tmp_path / "native.fastq",
                native_blow5=tmp_path / "native.blow5",
                ivt_bam=ivt_bam,
                ivt_fastq=tmp_path / "ivt.fastq",
                ivt_blow5=tmp_path / "ivt.blow5",
                ref_fasta=tmp_path / "ref.fa",
                min_depth=10,
            )

        assert results == {}
        assert metadata.n_contigs_passed_filter == 0
        mock_run_eventalign.assert_not_called()

    def test_f5c_not_found_raises_runtime_error(self, tmp_path: Path) -> None:
        native_bam = _create_test_bam(
            tmp_path,
            "native",
            {"ctg1": [(0, "A" * 20)]},
            [("ctg1", 20)],
        )
        ivt_bam = _create_test_bam(
            tmp_path,
            "ivt",
            {"ctg1": [(0, "C" * 20)]},
            [("ctg1", 20)],
        )

        with patch("baleen.eventalign._pipeline._f5c.check_f5c", side_effect=RuntimeError("f5c not found")):
            with pytest.raises(RuntimeError, match="f5c not found"):
                _ = run_pipeline(
                    native_bam=native_bam,
                    native_fastq=tmp_path / "native.fastq",
                    native_blow5=tmp_path / "native.blow5",
                    ivt_bam=ivt_bam,
                    ivt_fastq=tmp_path / "ivt.fastq",
                    ivt_blow5=tmp_path / "ivt.blow5",
                    ref_fasta=tmp_path / "ref.fa",
                    min_depth=1,
                )

    def test_output_dir_triggers_save(self, tmp_path: Path) -> None:
        native_bam = _create_test_bam(
            tmp_path,
            "native",
            {"ctg1": [(0, "A" * 20), (0, "A" * 20)]},
            [("ctg1", 20)],
        )
        ivt_bam = _create_test_bam(
            tmp_path,
            "ivt",
            {"ctg1": [(0, "C" * 20), (0, "C" * 20)]},
            [("ctg1", 20)],
        )

        native_tsv = tmp_path / "native_eventalign.tsv"
        ivt_tsv = tmp_path / "ivt_eventalign.tsv"
        _write_eventalign_tsv(native_tsv, [_event_row(position=10, read_name="n1", samples="1,2")])
        _write_eventalign_tsv(ivt_tsv, [_event_row(position=10, read_name="i1", samples="2,3")])

        def fake_run_eventalign(
            bam: Path,
            ref_fasta: Path,
            fastq: Path,
            blow5: Path,
            output_tsv: Path,
            **kwargs: object,
        ) -> Path:
            del bam, ref_fasta, blow5, kwargs
            src = native_tsv if "native" in str(fastq) else ivt_tsv
            output_tsv.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            return output_tsv

        out_dir = tmp_path / "results"
        with (
            patch("baleen.eventalign._pipeline._f5c.check_f5c", return_value="1.6"),
            patch("baleen.eventalign._pipeline._f5c.index_fastq_blow5"),
            patch("baleen.eventalign._pipeline._f5c.index_blow5"),
            patch("baleen.eventalign._pipeline._f5c.run_eventalign", side_effect=fake_run_eventalign),
            patch("baleen.eventalign._pipeline._dtw_distance", return_value=1.23),
        ):
            _ = run_pipeline(
                native_bam=native_bam,
                native_fastq=tmp_path / "native.fastq",
                native_blow5=tmp_path / "native.blow5",
                ivt_bam=ivt_bam,
                ivt_fastq=tmp_path / "ivt.fastq",
                ivt_blow5=tmp_path / "ivt.blow5",
                ref_fasta=tmp_path / "ref.fa",
                min_depth=1,
                output_dir=out_dir,
            )

        output_file = out_dir / "pipeline_results.pkl"
        assert output_file.exists()


class TestSaveLoadResults:
    def test_round_trip_pickle(self, tmp_path: Path) -> None:
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        position = PositionResult(
            position=10,
            reference_kmer="AAAAA",
            n_native_reads=1,
            n_ivt_reads=1,
            native_read_names=["n1"],
            ivt_read_names=["i1"],
            distance_matrix=matrix,
        )
        results = {
            "ctg1": ContigResult(
                contig="ctg1",
                native_depth=2.0,
                ivt_depth=2.0,
                positions={10: position},
            )
        }
        metadata = PipelineMetadata(
            f5c_version="1.6",
            min_depth=1,
            use_cuda=None,
            padding=0,
            n_contigs_total=1,
            n_contigs_passed_filter=1,
            n_contigs_skipped=0,
            filter_results=[],
        )

        output_path = tmp_path / "pipeline.pkl"
        save_results(results, metadata, output_path)
        loaded_results, loaded_meta = load_results(output_path)

        assert set(loaded_results.keys()) == {"ctg1"}
        np.testing.assert_allclose(
            loaded_results["ctg1"].positions[10].distance_matrix,
            matrix,
        )
        assert loaded_meta.f5c_version == "1.6"
