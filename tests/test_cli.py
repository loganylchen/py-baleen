"""Tests for baleen CLI module."""

import argparse
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import CLI functions
from baleen.cli import (
    _add_aggregate_args,
    _add_run_args,
    _cmd_aggregate,
    _cmd_run,
    _validate_input_files,
    main,
)


@pytest.fixture
def sample_args_run():
    """Create sample args for the 'run' sub-command."""
    args = argparse.Namespace(
        native_bam="native.bam",
        native_fastq="native.fq.gz",
        native_blow5="native.blow5",
        ivt_bam="ivt.bam",
        ivt_fastq="ivt.fq.gz",
        ivt_blow5="ivt.blow5",
        ref="ref.fa",
        output_dir="test_output",
        padding=0,
        min_depth=15,
        min_mapq=0,
        cuda=False,
        no_cuda=False,
        open_start=False,
        open_end=False,
        hmm_params=None,
        no_hmm=False,
        no_rna=False,
        kmer_model=None,
        threads=1,
        no_primary_only=False,
        keep_temp=False,
        no_read_bam=True,
    )
    return args


@pytest.fixture
def sample_args_aggregate():
    """Create sample args for the 'aggregate' sub-command."""
    args = argparse.Namespace(
        input="pipeline_results.pkl",
        output="sites.tsv",
        score_field="p_mod_hmm",
        hmm_params=None,
        no_read_bam=True,
        ref=None,
    )
    return args


class TestValidateInputFiles:
    """Tests for _validate_input_files function."""

    def test_all_files_exist(self, sample_args_run, tmp_path):
        """Test validation passes when all files exist."""
        # Create all required files and update args to use tmp_path
        for attr in ["native_bam", "native_fastq", "native_blow5", "ivt_bam", "ivt_fastq", "ivt_blow5", "ref"]:
            path = tmp_path / getattr(sample_args_run, attr)
            path.touch()
            setattr(sample_args_run, attr, str(path))

        # Should not raise
        _validate_input_files(sample_args_run)

    def test_missing_native_bam(self, sample_args_run):
        """Test validation fails when native_bam is missing."""
        with pytest.raises(SystemExit) as exc_info:
            _validate_input_files(sample_args_run)
        assert "file not found" in str(exc_info.value)

    def test_missing_ivt_blow5(self, sample_args_run):
        """Test validation fails when ivt_blow5 is missing."""
        # Create some files but not ivt_blow5
        for attr in ["native_bam", "native_fastq", "native_blow5", "ivt_bam", "ivt_fastq", "ref"]:
            path = Path(getattr(sample_args_run, attr))
            path.touch()

        with pytest.raises(SystemExit) as exc_info:
            _validate_input_files(sample_args_run)
        assert "ivt.blow5" in str(exc_info.value)


class TestAddRunArgs:
    """Tests for _add_run_args function."""

    def test_adds_required_arguments(self):
        """Test that all required arguments are added."""
        parser = argparse.ArgumentParser()
        _add_run_args(parser)

        # Test parsing with required args
        args = parser.parse_args(
            [
                "--native-bam", "n.bam",
                "--native-fastq", "n.fq",
                "--native-blow5", "n.blow5",
                "--ivt-bam", "i.bam",
                "--ivt-fastq", "i.fq",
                "--ivt-blow5", "i.blow5",
                "--ref", "r.fa",
            ]
        )

        assert args.native_bam == "n.bam"
        assert args.ref == "r.fa"
        assert args.min_depth == 15  # default
        assert args.cuda is False
        assert args.no_cuda is False

    def test_cuda_mutual_exclusion(self):
        """Test that --cuda and --no-cuda cannot both be specified."""
        parser = argparse.ArgumentParser()
        _add_run_args(parser)

        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "run",
                    "--cuda",
                    "--no-cuda",
                    "--native-bam", "n.bam",
                    "--native-fastq", "n.fq",
                    "--native-blow5", "n.blow5",
                    "--ivt-bam", "i.bam",
                    "--ivt-fastq", "i.fq",
                    "--ivt-blow5", "i.blow5",
                    "--ref", "r.fa",
                ]
            )


class TestAddAggregateArgs:
    """Tests for _add_aggregate_args function."""

    def test_adds_aggregate_arguments(self):
        """Test that aggregate arguments are added correctly."""
        parser = argparse.ArgumentParser()
        _add_aggregate_args(parser)

        args = parser.parse_args(["-i", "in.pkl", "-o", "out.tsv"])

        assert args.input == "in.pkl"
        assert args.output == "out.tsv"
        assert args.score_field == "p_mod_hmm"  # default

    def test_score_field_choices(self):
        """Test that score_field accepts valid choices."""
        parser = argparse.ArgumentParser()
        _add_aggregate_args(parser)

        for field in ["p_mod_hmm", "p_mod_knn", "p_mod_raw"]:
            args = parser.parse_args(["-i", "in.pkl", "-o", "out.tsv", "--score-field", field])
            assert args.score_field == field

    def test_score_field_invalid_choice(self):
        """Test that invalid score_field is rejected."""
        parser = argparse.ArgumentParser()
        _add_aggregate_args(parser)

        with pytest.raises(SystemExit):
            parser.parse_args(["-i", "in.pkl", "-o", "out.tsv", "--score-field", "invalid"])


class TestCmdRun:
    """Tests for _cmd_run function."""

    @patch("baleen.cli.run_pipeline")
    @patch("baleen.cli._validate_input_files")
    @patch("baleen.cli.compute_sequential_modification_probabilities")
    @patch("baleen.cli.aggregate_all")
    @patch("baleen.cli.write_site_tsv")
    @patch("baleen.cli.save_results")
    def test_run_args_forwarded_correctly(
        self,
        mock_save,
        mock_write,
        mock_agg,
        mock_hmm,
        mock_validate,
        mock_pipeline,
        sample_args_run,
        tmp_path,
    ):
        """Test that args are forwarded correctly to run_pipeline."""
        sample_args_run.output_dir = str(tmp_path)

        # Mock the pipeline to return empty results
        mock_pipeline.return_value = ({}, Mock())

        _cmd_run(sample_args_run)

        # Verify run_pipeline was called with correct args
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]

        assert call_kwargs["native_bam"] == sample_args_run.native_bam
        assert call_kwargs["ref_fasta"] == sample_args_run.ref
        assert call_kwargs["min_depth"] == sample_args_run.min_depth
        assert call_kwargs["padding"] == sample_args_run.padding
        assert call_kwargs["min_mapq"] == sample_args_run.min_mapq

    @patch("baleen.cli.run_pipeline")
    @patch("baleen.cli._validate_input_files")
    @patch("baleen.cli.compute_sequential_modification_probabilities")
    @patch("baleen.cli.aggregate_all")
    @patch("baleen.cli.write_site_tsv")
    @patch("baleen.cli.save_results")
    def test_run_creates_output_dir(
        self,
        mock_save,
        mock_write,
        mock_agg,
        mock_hmm,
        mock_validate,
        mock_pipeline,
        sample_args_run,
        tmp_path,
    ):
        """Test that output directory is created if it doesn't exist."""
        output_path = tmp_path / "new_output"
        sample_args_run.output_dir = str(output_path)

        mock_pipeline.return_value = ({}, Mock())

        _cmd_run(sample_args_run)

        assert output_path.exists()
        assert output_path.is_dir()

    @patch("baleen.cli.run_pipeline")
    @patch("baleen.cli._validate_input_files")
    @patch("baleen.cli.compute_sequential_modification_probabilities")
    @patch("baleen.cli.aggregate_all")
    @patch("baleen.cli.write_site_tsv")
    @patch("baleen.cli.save_results")
    def test_run_cuda_flag_handling(
        self,
        mock_save,
        mock_write,
        mock_agg,
        mock_hmm,
        mock_validate,
        mock_pipeline,
        sample_args_run,
        tmp_path,
    ):
        """Test that CUDA flag is handled correctly."""
        sample_args_run.output_dir = str(tmp_path)
        sample_args_run.cuda = True
        sample_args_run.no_cuda = False

        mock_pipeline.return_value = ({}, Mock())

        _cmd_run(sample_args_run)

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["use_cuda"] is True


class TestCmdAggregate:
    """Tests for _cmd_aggregate function."""

    @patch("baleen.cli.load_results")
    @patch("baleen.cli.compute_sequential_modification_probabilities")
    @patch("baleen.cli.aggregate_all")
    @patch("baleen.cli.write_site_tsv")
    def test_aggregate_with_valid_input(
        self, mock_write, mock_agg, mock_hmm, mock_load, sample_args_aggregate, tmp_path
    ):
        """Test aggregate command with valid input."""
        # Create mock results
        from baleen.eventalign import ContigModificationResult

        mock_result = Mock(spec=ContigModificationResult)
        mock_results = {"chr1": mock_result}
        mock_load.return_value = mock_results

        mock_hmm.return_value = Mock()
        mock_agg.return_value = []
        mock_write.return_value = None

        _cmd_aggregate(sample_args_aggregate)

        # Verify load_results was called
        mock_load.assert_called_once_with(sample_args_aggregate.input)

        # Verify write_site_tsv was called
        mock_write.assert_called_once()


class TestMain:
    """Tests for main function."""

    @patch("sys.argv", ["baleen", "--help"])
    def test_main_help(self, capsys):
        """Test that --help exits successfully."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_main_no_args(self, capsys):
        """Test that main with no args prints help."""
        with patch("sys.argv", ["baleen"]):
            with pytest.raises(SystemExit):
                main()
        captured = capsys.readouterr()
        # Should contain usage information
        output = (captured.err + captured.out).lower()
        assert "usage:" in output

    @patch("baleen.cli._cmd_run")
    def test_main_dispatches_run(self, mock_cmd_run):
        """Test that main dispatches to run sub-command."""
        with patch(
            "sys.argv",
            [
                "baleen",
                "run",
                "--native-bam",
                "n.bam",
                "--native-fastq",
                "n.fq",
                "--native-blow5",
                "n.blow5",
                "--ivt-bam",
                "i.bam",
                "--ivt-fastq",
                "i.fq",
                "--ivt-blow5",
                "i.blow5",
                "--ref",
                "r.fa",
            ],
        ):
            main()
        mock_cmd_run.assert_called_once()

    @patch("baleen.cli._cmd_aggregate")
    def test_main_dispatches_aggregate(self, mock_cmd_agg):
        """Test that main dispatches to aggregate sub-command."""
        with patch("sys.argv", ["baleen", "aggregate", "-i", "in.pkl", "-o", "out.tsv"]):
            main()
        mock_cmd_agg.assert_called_once()
