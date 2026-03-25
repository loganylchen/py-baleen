"""Read-level modification probability output in BAM format.

Writes per-read ``p_mod_hmm`` values as pseudo-alignments in a sorted,
indexed BAM file.  Each record represents one read at one genomic position.

Custom tags:
    MP:f  — HMM-smoothed modification probability (0.0–1.0)
    RG:Z  — Read group: ``native`` or ``ivt``
    KM:Z  — Original RNA kmer (preserves U bases)

Public API
----------
load_read_results
    Load read-level results into a DataFrame, optionally filtered by region.
load_read_results_iter
    Memory-efficient iterator over read-level results.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Union

import numpy as np
import pandas as pd
import pysam

if TYPE_CHECKING:
    from baleen.eventalign._hierarchical import ContigModificationResult
    from baleen.eventalign._pipeline import ContigResult

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def write_read_bam(
    hierarchical_results: dict[str, "ContigModificationResult"],
    contig_results: dict[str, "ContigResult"] | None,
    ref_fasta: PathLike,
    output_path: PathLike,
) -> Path:
    """Write per-read p_mod_hmm to a sorted, indexed BAM file.

    Parameters
    ----------
    hierarchical_results
        Per-contig HMM pipeline output (contains p_mod_hmm arrays).
    contig_results
        Per-contig DTW results (contains read names).
    ref_fasta
        Reference FASTA (used for BAM header @SQ lines).
    output_path
        Destination path for the BAM file.

    Returns
    -------
    Path
        Path to the written (sorted, indexed) BAM file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build header from reference FASTA
    header = _build_header(ref_fasta, hierarchical_results)

    # Write unsorted BAM to temp file, then sort + index
    tmp_dir = out.parent
    tmp_unsorted = tmp_dir / f".{out.name}.unsorted.bam"

    try:
        n_records = 0
        with pysam.AlignmentFile(str(tmp_unsorted), "wb", header=header) as bam_out:
            for contig in sorted(hierarchical_results.keys()):
                cmr = hierarchical_results[contig]
                cr = contig_results.get(contig) if contig_results is not None else None

                for pos in sorted(cmr.position_stats.keys()):
                    ps = cmr.position_stats[pos]

                    # Get read names and kmer from ContigResult if available,
                    # otherwise fall back to PositionStats fields.
                    if cr is not None:
                        pr = cr.positions.get(pos)
                        if pr is None:
                            continue
                        kmer = pr.reference_kmer
                        native_names = pr.native_read_names
                        ivt_names = pr.ivt_read_names
                    else:
                        kmer = ps.reference_kmer
                        native_names = ps.native_read_names
                        ivt_names = ps.ivt_read_names

                    kmer_dna = kmer.replace("U", "T").replace("u", "t")
                    cigar_len = len(kmer)

                    # Native reads: indices 0..n_native-1
                    for i, name in enumerate(native_names):
                        p_mod = float(ps.p_mod_hmm[i])
                        if math.isnan(p_mod):
                            continue
                        a = _make_record(
                            bam_out, name, contig, pos, kmer_dna, kmer,
                            cigar_len, p_mod, "native",
                        )
                        bam_out.write(a)
                        n_records += 1

                    # IVT reads: indices n_native..n_total-1
                    n_native_offset = len(native_names)
                    for j, name in enumerate(ivt_names):
                        p_mod = float(ps.p_mod_hmm[n_native_offset + j])
                        if math.isnan(p_mod):
                            continue
                        a = _make_record(
                            bam_out, name, contig, pos, kmer_dna, kmer,
                            cigar_len, p_mod, "ivt",
                        )
                        bam_out.write(a)
                        n_records += 1

        # Sort and index; clean up output on failure to avoid corrupt files
        success = False
        try:
            try:
                pysam.sort("-o", str(out), str(tmp_unsorted))
            except Exception as exc:
                raise RuntimeError(f"Failed to sort BAM file {out}.") from exc
            try:
                pysam.index(str(out))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to index BAM file {out}. "
                    "Records may not be properly sorted."
                ) from exc
            success = True
        finally:
            if not success and out.exists():
                out.unlink()

        logger.info(
            "Wrote %d read-level records to %s", n_records, out,
        )
        return out

    finally:
        if tmp_unsorted.exists():
            tmp_unsorted.unlink()


def _build_header(
    ref_fasta: PathLike,
    hierarchical_results: dict[str, "ContigModificationResult"],
) -> dict:
    """Build BAM header from reference FASTA or fallback to contig names."""
    sq_lines: list[dict[str, Any]] = []

    ref_path = Path(ref_fasta)
    if ref_path.exists():
        fasta_contigs: set[str] = set()
        try:
            fa_handle = pysam.FastaFile(str(ref_path))
        except Exception as exc:
            raise RuntimeError(
                f"Cannot open reference FASTA {ref_path}. "
                "Ensure the file is indexed (run: samtools faidx <ref.fa>)."
            ) from exc
        with fa_handle as fa:
            for name in fa.references:
                sq_lines.append({"SN": name, "LN": fa.get_reference_length(name)})
                fasta_contigs.add(name)
        # Also include result contigs not present in the FASTA
        for contig in sorted(hierarchical_results.keys()):
            if contig not in fasta_contigs:
                sq_lines.append({"SN": contig, "LN": 0})
    else:
        # Fallback: use contig names from results with length 0
        for contig in sorted(hierarchical_results.keys()):
            sq_lines.append({"SN": contig, "LN": 0})

    return {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": sq_lines,
        "RG": [
            {"ID": "native", "SM": "native"},
            {"ID": "ivt", "SM": "ivt"},
        ],
        "PG": [{"ID": "baleen", "PN": "baleen"}],
    }


def _make_record(
    bam_out: pysam.AlignmentFile,
    read_name: str,
    contig: str,
    position: int,
    kmer_dna: str,
    kmer_rna: str,
    cigar_len: int,
    p_mod: float,
    read_group: str,
) -> pysam.AlignedSegment:
    """Create a single BAM record."""
    a = pysam.AlignedSegment(bam_out.header)
    a.query_name = read_name
    a.flag = 0
    a.reference_id = bam_out.get_tid(contig)
    a.reference_start = position - 1
    a.mapping_quality = 255
    a.cigar = [(0, cigar_len)]  # M operation
    a.query_sequence = kmer_dna
    a.query_qualities = None
    a.set_tag("MP", p_mod, "f")
    a.set_tag("RG", read_group, "Z")
    a.set_tag("KM", kmer_rna, "Z")
    return a


def load_read_results(
    bam_path: PathLike,
    contig: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> "pd.DataFrame":
    """Load read-level results into a DataFrame.

    Parameters
    ----------
    bam_path
        Path to the read-level BAM file.
    contig
        Filter to this contig (optional).
    start, end
        Filter to this region within *contig* (0-based, optional).

    Returns
    -------
    pd.DataFrame
        Columns: contig, position, kmer, read_name, is_native, p_mod_hmm.

    Warning
    -------
    Calling without region filters on human-scale data may produce
    a DataFrame with 100M+ rows.  Use ``load_read_results_iter()``
    for memory-efficient streaming.
    """
    records = list(load_read_results_iter(bam_path, contig, start, end))
    if not records:
        return pd.DataFrame(
            columns=["contig", "position", "kmer", "read_name", "is_native", "p_mod_hmm"],
        )
    return pd.DataFrame.from_records(records)


def load_read_results_iter(
    bam_path: PathLike,
    contig: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate read-level results as dicts.

    Memory-efficient alternative to ``load_read_results()``.

    Parameters
    ----------
    bam_path
        Path to the read-level BAM file.
    contig, start, end
        Optional region filter (0-based coordinates).

    Yields
    ------
    dict
        Keys: contig, position, kmer, read_name, is_native, p_mod_hmm.
    """
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        if contig is not None:
            iterator = bam.fetch(contig, start, end)
        else:
            iterator = bam.fetch()

        for read in iterator:
            yield {
                "contig": read.reference_name,
                "position": read.reference_start + 1,
                "kmer": read.get_tag("KM"),
                "read_name": read.query_name,
                "is_native": read.get_tag("RG") == "native",
                "p_mod_hmm": float(read.get_tag("MP")),
            }
