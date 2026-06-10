"""End-to-end integration test: DTW → HMM → aggregation → TSV + BAM.

Chains all four pipeline stages on synthetic data to verify the full
output path without requiring external tools (f5c, real BAM files).
"""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

hier = importlib.import_module("baleen.eventalign._hierarchical")
pipeline = importlib.import_module("baleen.eventalign._pipeline")
agg = importlib.import_module("baleen.eventalign._aggregation")
read_bam = importlib.import_module("baleen.eventalign._read_bam")

PositionResult = pipeline.PositionResult
ContigResult = pipeline.ContigResult


def _make_block_distance_matrix(
    n_native: int,
    n_ivt: int,
    within_native: float = 1.0,
    within_ivt: float = 1.0,
    between: float = 5.0,
    noise: float = 0.1,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(42)
    n = n_native + n_ivt
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if i < n_native and j < n_native:
                d = within_native
            elif i >= n_native and j >= n_native:
                d = within_ivt
            else:
                d = between
            d += rng.normal(0, noise)
            d = max(d, 0.0)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _make_homogeneous_matrix(
    n_native: int,
    n_ivt: int,
    base_dist: float = 1.0,
    noise: float = 0.05,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(99)
    n = n_native + n_ivt
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = base_dist + rng.normal(0, noise)
            d = max(d, 0.0)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _build_multi_contig_data(
    n_contigs: int = 2,
    n_positions: int = 8,
    n_native: int = 15,
    n_ivt: int = 10,
    modified_positions_per_contig: dict[str, set[int]] | None = None,
    seed: int = 42,
) -> dict[str, ContigResult]:
    """Build synthetic multi-contig data with known modified positions."""
    if modified_positions_per_contig is None:
        modified_positions_per_contig = {}
    rng = np.random.RandomState(seed)
    results = {}
    for c_idx in range(n_contigs):
        contig_name = f"contig_{c_idx}"
        modified = modified_positions_per_contig.get(contig_name, set())
        native_names = [f"nat_{contig_name}_{i}" for i in range(n_native)]
        ivt_names = [f"ivt_{contig_name}_{i}" for i in range(n_ivt)]
        positions = {}
        for p_idx in range(n_positions):
            pos = 100 + p_idx
            if p_idx in modified:
                dm = _make_block_distance_matrix(
                    n_native, n_ivt, within_native=1.5, within_ivt=1.0,
                    between=6.0, noise=0.15, rng=rng,
                )
            else:
                dm = _make_homogeneous_matrix(
                    n_native, n_ivt, base_dist=1.0, noise=0.05, rng=rng,
                )
            positions[pos] = PositionResult(
                position=pos, reference_kmer="AACGU",
                n_native_reads=n_native, n_ivt_reads=n_ivt,
                native_read_names=list(native_names),
                ivt_read_names=list(ivt_names),
                distance_matrix=dm,
            )
        results[contig_name] = ContigResult(
            contig=contig_name, native_depth=float(n_native),
            ivt_depth=float(n_ivt), positions=positions,
        )
    return results


class TestEndToEndPipeline:
    """DTW (synthetic) → HMM → aggregation → TSV → BAM round-trip."""

    def test_full_pipeline_single_contig(self):
        """Single contig through HMM → aggregation → TSV."""
        contig_results = _build_multi_contig_data(
            n_contigs=1, n_positions=10,
            modified_positions_per_contig={"contig_0": {2, 5, 7}},
        )

        # Stage 1: HMM
        hmm_results = {}
        for contig, cr in contig_results.items():
            hmm_results[contig] = hier.compute_sequential_modification_probabilities(cr)

        assert len(hmm_results) == 1
        cmr = hmm_results["contig_0"]
        assert len(cmr.position_stats) == 10

        # Every position should have p_mod_hmm values (not all NaN)
        for ps in cmr.position_stats.values():
            assert not np.all(np.isnan(ps.p_mod_hmm))

        # Stage 2: Aggregation with FDR
        sites = agg.aggregate_all(hmm_results)
        assert len(sites) > 0
        # padj should be set (FDR-corrected)
        assert all(s.padj <= 1.0 for s in sites)
        assert all(s.padj >= 0.0 for s in sites)

        # Stage 3: Write TSV
        with tempfile.TemporaryDirectory() as tmp:
            tsv_path = Path(tmp) / "sites.tsv"
            agg.write_site_tsv(sites, tsv_path)
            assert tsv_path.exists()
            lines = tsv_path.read_text().splitlines()
            assert len(lines) == len(sites) + 1  # header + data

    def test_full_pipeline_multi_contig(self):
        """Two contigs: FDR correction spans both."""
        contig_results = _build_multi_contig_data(
            n_contigs=2, n_positions=8,
            modified_positions_per_contig={
                "contig_0": {1, 3},
                "contig_1": {0, 5},
            },
        )

        # HMM
        hmm_results = {}
        for contig, cr in contig_results.items():
            hmm_results[contig] = hier.compute_sequential_modification_probabilities(cr)

        assert len(hmm_results) == 2

        # Aggregation
        sites = agg.aggregate_all(hmm_results)
        # Should have sites from both contigs
        contigs_seen = {s.contig for s in sites}
        assert "contig_0" in contigs_seen
        assert "contig_1" in contigs_seen

        # FDR correction applied across all sites
        pvalues = [s.pvalue for s in sites]
        padjs = [s.padj for s in sites]
        # padj >= pvalue (BH can only increase or keep equal)
        for pv, pa in zip(pvalues, padjs):
            assert pa >= pv - 1e-12

    def test_full_pipeline_with_bam_output(self):
        """HMM results → mod-BAM round-trip with MM/ML tags."""
        import pysam

        n_native = 15
        n_ivt = 10
        contig_results = _build_multi_contig_data(
            n_contigs=1, n_positions=5,
            n_native=n_native, n_ivt=n_ivt,
            modified_positions_per_contig={"contig_0": {1, 3}},
        )

        hmm_results = {}
        for contig, cr in contig_results.items():
            hmm_results[contig] = hier.compute_sequential_modification_probabilities(cr)

        with tempfile.TemporaryDirectory() as tmp:
            # Create reference FASTA
            seq_len = 3000
            ref_path = Path(tmp) / "ref.fa"
            ref_path.write_text(">contig_0\n" + "A" * seq_len + "\n")
            pysam.faidx(str(ref_path))

            # Create synthetic input BAMs with reads aligned to contig_0
            native_bam_path = Path(tmp) / "native.bam"
            ivt_bam_path = Path(tmp) / "ivt.bam"

            header = pysam.AlignmentHeader.from_dict({
                "HD": {"VN": "1.6", "SO": "coordinate"},
                "SQ": [{"SN": "contig_0", "LN": seq_len}],
            })

            for bam_path_i, prefix, n_reads in [
                (native_bam_path, "nat_contig_0", n_native),
                (ivt_bam_path, "ivt_contig_0", n_ivt),
            ]:
                with pysam.AlignmentFile(str(bam_path_i), "wb", header=header) as bf:
                    for i in range(n_reads):
                        a = pysam.AlignedSegment(bf.header)
                        a.query_name = f"{prefix}_{i}"
                        a.flag = 0
                        a.reference_id = 0
                        a.reference_start = 0
                        a.mapping_quality = 60
                        a.query_sequence = "A" * 200
                        a.cigar = [(0, 200)]  # 200M
                        a.query_qualities = pysam.qualitystring_to_array("I" * 200)
                        bf.write(a)
                # Sort and index
                sorted_path = str(bam_path_i) + ".sorted.bam"
                pysam.sort("-o", sorted_path, str(bam_path_i))
                Path(sorted_path).rename(bam_path_i)
                pysam.index(str(bam_path_i))

            out_bam = Path(tmp) / "read_results.bam"
            result_path = read_bam.write_mod_bam(
                hmm_results, native_bam_path, ivt_bam_path, ref_path, out_bam,
            )
            assert result_path.exists()
            assert Path(str(result_path) + ".bai").exists()

            # Read back and verify records exist
            df = read_bam.load_read_results(result_path)
            assert len(df) > 0
            assert "contig_0" in df["contig"].values
            assert set(df["is_native"].unique()) == {True, False}

    def test_modified_positions_have_higher_scores(self):
        """Modified positions should have higher mod_ratio than unmodified."""
        contig_results = _build_multi_contig_data(
            n_contigs=1, n_positions=10,
            modified_positions_per_contig={"contig_0": {2, 4, 6, 8}},
            seed=123,
        )

        hmm_results = {}
        for contig, cr in contig_results.items():
            hmm_results[contig] = hier.compute_sequential_modification_probabilities(cr)

        sites = agg.aggregate_all(hmm_results)
        modified_positions = {102, 104, 106, 108}

        mod_ratios = [s.mod_ratio for s in sites if s.position in modified_positions]
        unmod_ratios = [s.mod_ratio for s in sites if s.position not in modified_positions]

        # Modified positions should on average have higher mod_ratio
        if mod_ratios and unmod_ratios:
            assert np.mean(mod_ratios) > np.mean(unmod_ratios)
