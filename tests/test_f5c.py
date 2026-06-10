from pathlib import Path
import subprocess
from typing import cast
from unittest.mock import Mock, patch

import pytest

import baleen.eventalign._f5c as f5c_mod


@pytest.fixture(autouse=True)
def reset_f5c_cache():
    setattr(f5c_mod, "_f5c_version", None)
    yield
    setattr(f5c_mod, "_f5c_version", None)


class TestCheckF5c:
    def test_check_f5c_success(self):
        mock_result = Mock(stdout="f5c v1.6\n", stderr="")
        with patch("baleen.eventalign._f5c.subprocess.run", return_value=mock_result) as mock_run:
            assert f5c_mod.check_f5c() == "1.6"
            mock_run.assert_called_once_with(
                ["f5c", "--version"],
                check=True,
                capture_output=True,
                text=True,
            )

    def test_check_f5c_not_found(self) -> None:
        with patch("baleen.eventalign._f5c.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="f5c not found"):
                _ = f5c_mod.check_f5c()

    @pytest.mark.parametrize(
        "version_output,expected",
        [
            ("f5c v1.6", "1.6"),
            ("f5c v1.6.1", "1.6.1"),
            ("f5c 1.6", "1.6"),
            ("f5c V2.0.3", "2.0.3"),
        ],
    )
    def test_check_f5c_version_parse_variants(self, version_output: str, expected: str) -> None:
        mock_result = Mock(stdout=version_output, stderr="")
        with patch("baleen.eventalign._f5c.subprocess.run", return_value=mock_result):
            assert f5c_mod.check_f5c() == expected

    def test_check_f5c_caching(self):
        mock_result = Mock(stdout="f5c v1.6\n", stderr="")
        with patch("baleen.eventalign._f5c.subprocess.run", return_value=mock_result) as mock_run:
            assert f5c_mod.check_f5c() == "1.6"
            assert f5c_mod.check_f5c() == "1.6"
            assert mock_run.call_count == 1


class TestGetF5cVersion:
    def test_version_tuple(self) -> None:
        with patch("baleen.eventalign._f5c.check_f5c", return_value="1.6"):
            assert f5c_mod.get_f5c_version() == (1, 6)

    def test_version_tuple_three_parts(self) -> None:
        with patch("baleen.eventalign._f5c.check_f5c", return_value="1.6.1"):
            assert f5c_mod.get_f5c_version() == (1, 6, 1)


class TestIsIndexed:
    def test_indexed_true(self, tmp_path: Path):
        fastq = tmp_path / "reads.fastq"
        _ = fastq.write_text("@r\nACGT\n+\n####\n", encoding="utf-8")
        readdb = tmp_path / "reads.fastq.index.readdb"
        _ = readdb.write_text("indexed", encoding="utf-8")
        assert f5c_mod.is_indexed(fastq) is True

    def test_indexed_false_missing(self, tmp_path: Path):
        fastq = tmp_path / "reads.fastq"
        _ = fastq.write_text("@r\nACGT\n+\n####\n", encoding="utf-8")
        assert f5c_mod.is_indexed(fastq) is False

    def test_indexed_false_empty(self, tmp_path: Path):
        fastq = tmp_path / "reads.fastq"
        _ = fastq.write_text("@r\nACGT\n+\n####\n", encoding="utf-8")
        readdb = tmp_path / "reads.fastq.index.readdb"
        _ = readdb.write_text("", encoding="utf-8")
        assert f5c_mod.is_indexed(fastq) is False


class TestIsBlow5Indexed:
    def test_blow5_indexed_true(self, tmp_path: Path):
        blow5 = tmp_path / "reads.blow5"
        _ = blow5.write_text("x", encoding="utf-8")
        idx = tmp_path / "reads.blow5.idx"
        _ = idx.write_text("ok", encoding="utf-8")
        assert f5c_mod.is_blow5_indexed(blow5) is True

    def test_blow5_indexed_false_missing(self, tmp_path: Path):
        blow5 = tmp_path / "reads.blow5"
        _ = blow5.write_text("x", encoding="utf-8")
        assert f5c_mod.is_blow5_indexed(blow5) is False

    def test_blow5_indexed_false_empty(self, tmp_path: Path):
        blow5 = tmp_path / "reads.blow5"
        _ = blow5.write_text("x", encoding="utf-8")
        idx = tmp_path / "reads.blow5.idx"
        _ = idx.write_text("", encoding="utf-8")
        assert f5c_mod.is_blow5_indexed(blow5) is False


class TestIndexFastqBlow5:
    def test_index_runs_f5c(self, tmp_path: Path):
        fastq = tmp_path / "reads.fastq"
        blow5 = tmp_path / "reads.blow5"
        _ = fastq.write_text("x", encoding="utf-8")
        _ = blow5.write_text("x", encoding="utf-8")

        with patch("baleen.eventalign._f5c.subprocess.run") as mock_run:
            f5c_mod.index_fastq_blow5(fastq, blow5)

        mock_run.assert_called_once_with(
            ["f5c", "index", "--slow5", str(blow5), str(fastq)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_index_skips_if_already_indexed(self, tmp_path: Path):
        fastq = tmp_path / "reads.fastq"
        blow5 = tmp_path / "reads.blow5"
        _ = fastq.write_text("x", encoding="utf-8")
        _ = blow5.write_text("x", encoding="utf-8")
        readdb = tmp_path / "reads.fastq.index.readdb"
        _ = readdb.write_text("ok", encoding="utf-8")

        with patch("baleen.eventalign._f5c.subprocess.run") as mock_run:
            f5c_mod.index_fastq_blow5(fastq, blow5)

        mock_run.assert_not_called()

    def test_index_failure(self, tmp_path: Path):
        fastq = tmp_path / "reads.fastq"
        blow5 = tmp_path / "reads.blow5"
        _ = fastq.write_text("x", encoding="utf-8")
        _ = blow5.write_text("x", encoding="utf-8")
        err = subprocess.CalledProcessError(
            returncode=1,
            cmd=["f5c", "index"],
            stderr="index failed",
        )

        with patch("baleen.eventalign._f5c.subprocess.run", side_effect=err):
            with pytest.raises(RuntimeError, match="index failed"):
                _ = f5c_mod.index_fastq_blow5(fastq, blow5)


class TestIndexBlow5:
    def test_index_blow5_runs_slow5tools(self, tmp_path: Path):
        blow5 = tmp_path / "reads.blow5"
        _ = blow5.write_text("x", encoding="utf-8")

        with patch("baleen.eventalign._f5c.subprocess.run") as mock_run:
            f5c_mod.index_blow5(blow5)

        mock_run.assert_called_once_with(
            ["slow5tools", "index", str(blow5)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_index_blow5_skips_when_indexed(self, tmp_path: Path):
        blow5 = tmp_path / "reads.blow5"
        _ = blow5.write_text("x", encoding="utf-8")
        idx = tmp_path / "reads.blow5.idx"
        _ = idx.write_text("ok", encoding="utf-8")

        with patch("baleen.eventalign._f5c.subprocess.run") as mock_run:
            f5c_mod.index_blow5(blow5)

        mock_run.assert_not_called()

    def test_index_blow5_failure(self, tmp_path: Path):
        blow5 = tmp_path / "reads.blow5"
        _ = blow5.write_text("x", encoding="utf-8")
        err = subprocess.CalledProcessError(
            returncode=1,
            cmd=["slow5tools", "index"],
            stderr="slow5tools failed",
        )

        with patch("baleen.eventalign._f5c.subprocess.run", side_effect=err):
            with pytest.raises(RuntimeError, match="slow5tools failed"):
                _ = f5c_mod.index_blow5(blow5)


class TestRunEventalign:
    def test_eventalign_basic_command(self, tmp_path: Path):
        bam = tmp_path / "in.bam"
        ref = tmp_path / "ref.fa"
        fastq = tmp_path / "reads.fastq"
        blow5 = tmp_path / "reads.blow5"
        out = tmp_path / "out.tsv"

        with patch("baleen.eventalign._f5c.subprocess.run") as mock_run:
            result = f5c_mod.run_eventalign(bam, ref, fastq, blow5, out)

        assert result == out
        called_cmd = cast(list[str], mock_run.call_args.args[0])
        assert called_cmd[:2] == ["f5c", "eventalign"]
        assert "--rna" in called_cmd
        assert "--samples" in called_cmd
        assert "--signal-index" in called_cmd
        assert "--scale-events" in called_cmd
        assert "--print-read-names" in called_cmd

    def test_eventalign_no_rna(self, tmp_path: Path):
        out = tmp_path / "out.tsv"
        with patch("baleen.eventalign._f5c.subprocess.run") as mock_run:
            _ = f5c_mod.run_eventalign("a.bam", "r.fa", "q.fastq", "s.blow5", out, rna=False)
        called_cmd = cast(list[str], mock_run.call_args.args[0])
        assert "--rna" not in called_cmd

    def test_eventalign_with_kmer_model(self, tmp_path: Path):
        out = tmp_path / "out.tsv"
        with patch("baleen.eventalign._f5c.subprocess.run") as mock_run:
            _ = f5c_mod.run_eventalign(
                "a.bam",
                "r.fa",
                "q.fastq",
                "s.blow5",
                out,
                kmer_model="rna004.model",
            )
        called_cmd = cast(list[str], mock_run.call_args.args[0])
        assert "--kmer-model" in called_cmd
        idx = called_cmd.index("--kmer-model")
        assert called_cmd[idx + 1] == "rna004.model"

    def test_eventalign_with_extra_args(self, tmp_path: Path):
        out = tmp_path / "out.tsv"
        with patch("baleen.eventalign._f5c.subprocess.run") as mock_run:
            _ = f5c_mod.run_eventalign(
                "a.bam",
                "r.fa",
                "q.fastq",
                "s.blow5",
                out,
                extra_args=["--threads", "4", "--iop", "32"],
            )
        called_cmd = cast(list[str], mock_run.call_args.args[0])
        assert called_cmd[-4:] == ["--threads", "4", "--iop", "32"]

    def test_eventalign_failure(self, tmp_path: Path):
        out = tmp_path / "out.tsv"
        err = subprocess.CalledProcessError(
            returncode=1,
            cmd=["f5c", "eventalign"],
            stderr="eventalign failed",
        )

        with patch("baleen.eventalign._f5c.subprocess.run", side_effect=err):
            with pytest.raises(RuntimeError, match="eventalign failed"):
                _ = f5c_mod.run_eventalign("a.bam", "r.fa", "q.fastq", "s.blow5", out)

    def test_eventalign_output_file_created(self, tmp_path: Path):
        out = tmp_path / "out.tsv"
        with patch("baleen.eventalign._f5c.subprocess.run") as mock_run:
            _ = f5c_mod.run_eventalign("a.bam", "r.fa", "q.fastq", "s.blow5", out)

        assert out.exists()
        assert "stdout" in mock_run.call_args.kwargs
