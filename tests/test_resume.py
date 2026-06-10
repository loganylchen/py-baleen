"""Tests for the resume helpers in baleen.eventalign._pipeline.

Covers the three small pure helpers used by ``run_pipeline_streaming(resume=True)``:

- ``_compute_resume_fingerprint``  → deterministic dict shape
- ``_write_resume_fingerprint`` + ``_validate_resume_compatibility``
- ``_scan_completed_contigs`` → counts rows + significance from per-contig TSVs

The full streaming pipeline is not exercised here (it requires f5c + real
BAMs); these helpers are independently testable and cover the actual
resume decision logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baleen.eventalign._pipeline import (
    _RESUME_PARAMS_FILENAME,
    ContigSummary,
    _compute_resume_fingerprint,
    _sanitize_contig_filename,
    _scan_completed_contigs,
    _validate_resume_compatibility,
    _write_resume_fingerprint,
)


def _make_inputs(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in (
        "native_bam", "native_fastq", "native_blow5",
        "ivt_bam", "ivt_fastq", "ivt_blow5",
        "ref_fasta",
    ):
        p = tmp_path / f"{name}.dat"
        p.write_bytes(b"x" * 16)
        paths[name] = p
    return paths


def _baseline_fingerprint(paths: dict[str, Path]) -> dict:
    return _compute_resume_fingerprint(
        native_bam=paths["native_bam"],
        native_fastq=paths["native_fastq"],
        native_blow5=paths["native_blow5"],
        ivt_bam=paths["ivt_bam"],
        ivt_fastq=paths["ivt_fastq"],
        ivt_blow5=paths["ivt_blow5"],
        ref_fasta=paths["ref_fasta"],
        min_depth=15.0,
        depth_mode="read_count",
        padding=1,
        min_mapq=20,
        primary_only=True,
        subsample=True,
        subsample_n=300,
        legacy_scoring=False,
        mod_threshold=0.9,
        write_bam=True,
        run_hmm=True,
        target_contigs=None,
        read_intersection=True,
        pore="rna002",
    )


class TestComputeFingerprint:
    def test_round_trip_json(self, tmp_path: Path) -> None:
        fp = _baseline_fingerprint(_make_inputs(tmp_path))
        # Must be json-serializable.
        s = json.dumps(fp, sort_keys=True)
        assert json.loads(s) == fp
        assert fp["schema_version"] == 2
        assert set(fp["inputs"]) == {
            "native_bam", "native_fastq", "native_blow5",
            "ivt_bam", "ivt_fastq", "ivt_blow5", "ref_fasta",
        }

    def test_changing_param_changes_fingerprint(self, tmp_path: Path) -> None:
        paths = _make_inputs(tmp_path)
        fp1 = _baseline_fingerprint(paths)
        # Change one param.
        fp2 = _compute_resume_fingerprint(
            native_bam=paths["native_bam"],
            native_fastq=paths["native_fastq"],
            native_blow5=paths["native_blow5"],
            ivt_bam=paths["ivt_bam"],
            ivt_fastq=paths["ivt_fastq"],
            ivt_blow5=paths["ivt_blow5"],
            ref_fasta=paths["ref_fasta"],
            min_depth=20.0,  # changed
            depth_mode="read_count",
            padding=1,
            min_mapq=20,
            primary_only=True,
            subsample=True,
            subsample_n=300,
            legacy_scoring=False,
            mod_threshold=0.9,
            write_bam=True,
            run_hmm=True,
            target_contigs=None,
            read_intersection=True,
            pore="rna002",
        )
        assert fp1 != fp2

    def test_changing_input_size_changes_fingerprint(self, tmp_path: Path) -> None:
        paths = _make_inputs(tmp_path)
        fp1 = _baseline_fingerprint(paths)
        # Modify content + size of a tracked file.
        paths["native_bam"].write_bytes(b"y" * 999)
        fp2 = _baseline_fingerprint(paths)
        assert fp1["inputs"]["native_bam"] != fp2["inputs"]["native_bam"]


class TestValidateResumeCompatibility:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        per = tmp_path / "per_contig"
        per.mkdir()
        with pytest.raises(RuntimeError, match="not found"):
            _validate_resume_compatibility(per, _baseline_fingerprint(_make_inputs(tmp_path)))

    def test_matching_passes(self, tmp_path: Path) -> None:
        paths = _make_inputs(tmp_path)
        per = tmp_path / "per_contig"
        per.mkdir()
        fp = _baseline_fingerprint(paths)
        _write_resume_fingerprint(per, fp)
        # No exception.
        _validate_resume_compatibility(per, fp)

    def test_mismatch_lists_diffs(self, tmp_path: Path) -> None:
        paths = _make_inputs(tmp_path)
        per = tmp_path / "per_contig"
        per.mkdir()
        fp_old = _baseline_fingerprint(paths)
        _write_resume_fingerprint(per, fp_old)
        # Build a new fingerprint with mod_threshold changed.
        fp_new = _compute_resume_fingerprint(
            native_bam=paths["native_bam"],
            native_fastq=paths["native_fastq"],
            native_blow5=paths["native_blow5"],
            ivt_bam=paths["ivt_bam"],
            ivt_fastq=paths["ivt_fastq"],
            ivt_blow5=paths["ivt_blow5"],
            ref_fasta=paths["ref_fasta"],
            min_depth=15.0,
            depth_mode="read_count",
            padding=1,
            min_mapq=20,
            primary_only=True,
            subsample=True,
            subsample_n=300,
            legacy_scoring=False,
            mod_threshold=0.5,  # changed
            write_bam=True,
            run_hmm=True,
            target_contigs=None,
            read_intersection=True,
            pore="rna002",
        )
        with pytest.raises(RuntimeError, match="mod_threshold"):
            _validate_resume_compatibility(per, fp_new)

    def test_write_is_atomic(self, tmp_path: Path) -> None:
        paths = _make_inputs(tmp_path)
        per = tmp_path / "per_contig"
        per.mkdir()
        fp = _baseline_fingerprint(paths)
        _write_resume_fingerprint(per, fp)
        on_disk = json.loads((per / _RESUME_PARAMS_FILENAME).read_text())
        assert on_disk == fp
        # No leftover .tmp file.
        leftovers = list(per.glob("*.tmp"))
        assert leftovers == []


def _write_tsv(path: Path, padj_values: list[float]) -> None:
    """Tiny helper that mimics the columns inspected by _scan_completed_contigs."""
    header = "contig\tposition\treference_kmer\tpadj\n"
    with path.open("w") as fh:
        fh.write(header)
        for i, p in enumerate(padj_values):
            fh.write(f"ctg\t{i}\tAAAAA\t{p}\n")


class TestScanCompletedContigs:
    def test_picks_up_completed_contig(self, tmp_path: Path) -> None:
        per = tmp_path / "per_contig"
        per.mkdir()
        safe = _sanitize_contig_filename("ctg_a")
        _write_tsv(per / f"{safe}.tsv", padj_values=[0.001, 0.5, 0.04])
        (per / f"{safe}.bam").write_bytes(b"")  # marker; not parsed
        result = _scan_completed_contigs(per, ["ctg_a"], write_bam=True)
        assert "ctg_a" in result
        s: ContigSummary = result["ctg_a"]
        assert s.n_sites == 3
        assert s.n_significant == 2  # padj < 0.05
        assert s.tsv_path == per / f"{safe}.tsv"
        assert s.bam_path == per / f"{safe}.bam"

    def test_missing_tsv_skipped(self, tmp_path: Path) -> None:
        per = tmp_path / "per_contig"
        per.mkdir()
        result = _scan_completed_contigs(per, ["ctg_a"], write_bam=False)
        assert result == {}

    def test_missing_bam_skipped_when_write_bam(self, tmp_path: Path) -> None:
        per = tmp_path / "per_contig"
        per.mkdir()
        safe = _sanitize_contig_filename("ctg_a")
        _write_tsv(per / f"{safe}.tsv", padj_values=[0.1])
        # No .bam → must be considered incomplete when write_bam=True.
        result = _scan_completed_contigs(per, ["ctg_a"], write_bam=True)
        assert result == {}
        # When write_bam=False, .bam absence is fine.
        result2 = _scan_completed_contigs(per, ["ctg_a"], write_bam=False)
        assert "ctg_a" in result2
        assert result2["ctg_a"].bam_path is None

    def test_unsafe_contig_name_uses_sanitized_path(self, tmp_path: Path) -> None:
        per = tmp_path / "per_contig"
        per.mkdir()
        unsafe = "weird/name..with*chars"
        safe = _sanitize_contig_filename(unsafe)
        # Path-traversal characters are stripped; the resulting file
        # lives inside per_contig_dir.
        assert "/" not in safe
        out = (per / f"{safe}.tsv").resolve()
        assert out.parent == per.resolve()
        _write_tsv(out, padj_values=[0.01])
        result = _scan_completed_contigs(per, [unsafe], write_bam=False)
        assert unsafe in result
        # The summary keeps the original contig name, not the sanitized one.
        assert result[unsafe].contig_name == unsafe
