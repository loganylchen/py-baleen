"""Read-level modification probability output in standard mod-BAM format.

Writes per-read modification probabilities as ``MM:Z`` and ``ML:B:C`` tags
on copies of the original input BAM alignments (one record per read, not
per position).  This format is compatible with downstream tools such as
``modkit``, ``modbamtools``, and IGV's mod-BAM visualization.

Tags written:
    MM:Z  — Delta-encoded modified base positions (``N+?`` = unknown mod type)
    ML:B:C — Per-position uint8 modification probabilities (0–255)
    RG:Z  — Read group: ``native`` or ``ivt``

Public API
----------
write_mod_bam
    Write mod-BAM from HMM results + original BAM files.
load_read_results
    Load read-level results into a DataFrame, optionally filtered by region.
load_read_results_iter
    Memory-efficient iterator over read-level results.
"""

from __future__ import annotations

import array
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Union

import numpy as np
import pandas as pd
import pysam

if TYPE_CHECKING:
    from baleen.eventalign._hierarchical import ContigModificationResult

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def write_mod_bam(
    hierarchical_results: dict[str, "ContigModificationResult"],
    native_bam: PathLike,
    ivt_bam: PathLike,
    ref_fasta: PathLike,
    output_path: PathLike,
) -> Path:
    """Write mod-BAM with MM/ML tags from HMM results + original BAM reads.

    For each read in the HMM results, copies the original alignment from
    the input BAM and appends MM:Z / ML:B:C tags encoding per-base
    modification probabilities, plus an RG:Z tag for sample identity.

    Parameters
    ----------
    hierarchical_results
        Per-contig HMM pipeline output (contains p_mod_hmm arrays).
    native_bam
        Path to the original native BAM file.
    ivt_bam
        Path to the original IVT control BAM file.
    ref_fasta
        Reference FASTA (used for BAM header).
    output_path
        Destination path for the output BAM file.

    Returns
    -------
    Path
        Path to the written (sorted, indexed) BAM file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Build reverse index — read_name -> [(contig, genomic_pos, p_mod, is_native)]
    read_positions: dict[str, list[tuple[str, int, float, bool]]] = defaultdict(list)
    for contig, cmr in hierarchical_results.items():
        for pos, ps in cmr.position_stats.items():
            for i, name in enumerate(ps.native_read_names):
                p = float(ps.p_mod_hmm[i])
                if not math.isnan(p):
                    read_positions[name].append((contig, pos, p, True))

            for j, name in enumerate(ps.ivt_read_names):
                p = float(ps.p_mod_hmm[ps.n_native + j])
                if not math.isnan(p):
                    read_positions[name].append((contig, pos, p, False))

    if not read_positions:
        logger.warning("No reads with valid p_mod_hmm found; skipping mod-BAM output")
        return out

    logger.info(
        "Building mod-BAM for %d reads across %d contigs",
        len(read_positions), len(hierarchical_results),
    )

    # Step 2: Build header from the native BAM (preserves all @SQ lines)
    header = _build_header_from_bam(native_bam, ref_fasta)

    # Step 3: Scan both BAMs, copy reads with MM/ML tags
    tmp_unsorted = out.parent / f".{out.name}.unsorted.bam"
    try:
        n_records = 0
        n_tagged = 0
        seen_reads: set[str] = set()

        with pysam.AlignmentFile(str(tmp_unsorted), "wb", header=header) as bam_out:
            for bam_path, is_native, rg_label in [
                (native_bam, True, "native"),
                (ivt_bam, False, "ivt"),
            ]:
                count, tagged = _scan_and_write(
                    bam_path, bam_out, read_positions, is_native, rg_label, seen_reads,
                )
                n_records += count
                n_tagged += tagged

        # Sort and index
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
            "Wrote %d reads to mod-BAM (%d with MM/ML tags): %s",
            n_records, n_tagged, out,
        )
        return out

    finally:
        if tmp_unsorted.exists():
            tmp_unsorted.unlink()


def _build_header_from_bam(
    bam_path: PathLike,
    ref_fasta: PathLike,
) -> pysam.AlignmentHeader:
    """Build output BAM header from the input BAM, adding RG lines."""
    with pysam.AlignmentFile(str(bam_path), "rb") as bam_in:
        hdr_dict = bam_in.header.to_dict()

    # Ensure read group entries exist
    hdr_dict.setdefault("RG", [])
    existing_rg_ids = {rg["ID"] for rg in hdr_dict["RG"]}
    for rg_id in ("native", "ivt"):
        if rg_id not in existing_rg_ids:
            hdr_dict["RG"].append({"ID": rg_id, "SM": rg_id})

    # Add baleen to PG
    hdr_dict.setdefault("PG", [])
    pg_ids = {pg.get("ID") for pg in hdr_dict["PG"]}
    if "baleen" not in pg_ids:
        hdr_dict["PG"].append({"ID": "baleen", "PN": "baleen"})

    return pysam.AlignmentHeader.from_dict(hdr_dict)


def _scan_and_write(
    bam_path: PathLike,
    bam_out: pysam.AlignmentFile,
    read_positions: dict[str, list[tuple[str, int, float, bool]]],
    is_native: bool,
    rg_label: str,
    seen_reads: set[str],
) -> tuple[int, int]:
    """Scan an input BAM, copy reads that appear in HMM results with MM/ML tags.

    Returns (n_written, n_tagged).
    """
    n_written = 0
    n_tagged = 0
    n_skipped_mapping = 0

    with pysam.AlignmentFile(str(bam_path), "rb") as bam_in:
        for read in bam_in.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            name = read.query_name
            if name not in read_positions:
                continue
            if name in seen_reads:
                continue
            seen_reads.add(name)

            # Collect HMM positions for this read (matching native/ivt)
            entries = [
                (contig, gpos, p)
                for contig, gpos, p, entry_is_native in read_positions[name]
                if entry_is_native == is_native
            ]
            if not entries:
                continue

            # Map genomic positions -> query positions via aligned_pairs
            # aligned_pairs returns (query_pos, ref_pos), ref_pos is 0-based
            ref_to_query: dict[int, int | None] = {}
            for qry, ref in read.get_aligned_pairs():
                if ref is not None:
                    ref_to_query[ref] = qry

            # Build (query_pos, p_mod) pairs, skipping deletions / unmapped
            qpos_pmod: list[tuple[int, float]] = []
            for _contig, gpos, p in entries:
                ref_pos_0 = gpos - 1  # eventalign 1-based -> 0-based
                qp = ref_to_query.get(ref_pos_0)
                if qp is None:
                    n_skipped_mapping += 1
                    continue
                qpos_pmod.append((qp, p))

            # Create output record (copy from original alignment)
            a = pysam.AlignedSegment(bam_out.header)
            a.query_name = read.query_name
            a.flag = read.flag
            a.reference_id = bam_out.get_tid(read.reference_name)
            a.reference_start = read.reference_start
            a.mapping_quality = read.mapping_quality
            a.cigar = read.cigartuples
            a.query_sequence = read.query_sequence
            a.query_qualities = read.query_qualities
            a.next_reference_id = -1
            a.next_reference_start = -1
            a.template_length = 0

            # Add MM/ML tags if we have mapped positions
            if qpos_pmod:
                qpos_pmod.sort(key=lambda x: x[0])
                mm_str, ml_vals = _build_mm_ml(qpos_pmod)
                a.set_tag("MM", mm_str, "Z")
                a.set_tag("ML", ml_vals)
                n_tagged += 1

            a.set_tag("RG", rg_label, "Z")
            bam_out.write(a)
            n_written += 1

    if n_skipped_mapping > 0:
        logger.debug(
            "Skipped %d position mappings (deletions/outside alignment) in %s reads",
            n_skipped_mapping, rg_label,
        )

    return n_written, n_tagged


def _build_mm_ml(
    qpos_pmod: list[tuple[int, float]],
) -> tuple[str, list[int]]:
    """Build MM:Z string and ML:B:C array from sorted (query_pos, p_mod) pairs.

    Uses ``N+?`` (unknown modification on any base) with delta encoding.
    """
    deltas = []
    prev = -1
    for qp, _p in qpos_pmod:
        deltas.append(qp - prev - 1)
        prev = qp

    mm_str = "N+?," + ",".join(str(d) for d in deltas) + ";"
    ml_vals = array.array("B", [min(255, round(p * 255)) for _qp, p in qpos_pmod])
    return mm_str, ml_vals


# ---------------------------------------------------------------------------
# Loading / iteration
# ---------------------------------------------------------------------------


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
        Path to the mod-BAM file.
    contig
        Filter to this contig (optional).
    start, end
        Filter to this region within *contig* (0-based, optional).

    Returns
    -------
    pd.DataFrame
        Columns: contig, position, read_name, is_native, p_mod_hmm.
    """
    records = list(load_read_results_iter(bam_path, contig, start, end))
    if not records:
        return pd.DataFrame(
            columns=["contig", "position", "read_name", "is_native", "p_mod_hmm"],
        )
    return pd.DataFrame.from_records(records)


def load_read_results_iter(
    bam_path: PathLike,
    contig: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate read-level results as dicts from a mod-BAM file.

    Parses MM:Z / ML:B:C tags to reconstruct per-position modification
    probabilities.  Falls back to legacy MP:f tag format if MM is absent.

    Parameters
    ----------
    bam_path
        Path to the mod-BAM file.
    contig, start, end
        Optional region filter (0-based coordinates).

    Yields
    ------
    dict
        Keys: contig, position, read_name, is_native, p_mod_hmm.
    """
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        if contig is not None:
            iterator = bam.fetch(contig, start, end)
        else:
            iterator = bam.fetch()

        for read in iterator:
            if read.is_unmapped:
                continue

            read_name = read.query_name
            ref_name = read.reference_name

            # Determine read group
            try:
                rg = read.get_tag("RG")
            except KeyError:
                rg = "native"
            is_native = rg == "native"

            # Legacy format: single MP:f tag per record (one record per position)
            try:
                mp_tag = read.get_tag("MP")
                # Legacy record: one position per record
                try:
                    kmer = read.get_tag("KM")
                except KeyError:
                    kmer = None
                result = {
                    "contig": ref_name,
                    "position": read.reference_start + 1,
                    "read_name": read_name,
                    "is_native": is_native,
                    "p_mod_hmm": float(mp_tag),
                }
                if kmer is not None:
                    result["kmer"] = kmer
                yield result
                continue
            except KeyError:
                pass

            # Mod-BAM format: MM:Z + ML:B:C tags
            try:
                mm_str = read.get_tag("MM")
                ml_vals = read.get_tag("ML")
            except KeyError:
                continue  # No modification data on this read

            # Parse MM tag to get query positions
            query_positions = _parse_mm_tag(mm_str)
            if len(query_positions) != len(ml_vals):
                logger.warning(
                    "MM/ML length mismatch for read %s: %d vs %d",
                    read_name, len(query_positions), len(ml_vals),
                )
                continue

            # Map query positions back to reference positions
            # aligned_pairs: (query_pos, ref_pos)
            query_to_ref: dict[int, int] = {}
            for qry, ref in read.get_aligned_pairs():
                if qry is not None and ref is not None:
                    query_to_ref[qry] = ref

            for qpos, ml_val in zip(query_positions, ml_vals):
                ref_pos = query_to_ref.get(qpos)
                if ref_pos is None:
                    continue  # insertion position, skip

                genomic_pos = ref_pos + 1  # 0-based -> 1-based
                p_mod = ml_val / 255.0

                # Apply region filter for mod-BAM (the read spans the region
                # but individual positions may be outside it)
                if start is not None and ref_pos < start:
                    continue
                if end is not None and ref_pos >= end:
                    continue

                yield {
                    "contig": ref_name,
                    "position": genomic_pos,
                    "read_name": read_name,
                    "is_native": is_native,
                    "p_mod_hmm": p_mod,
                }


def _parse_mm_tag(mm_str: str) -> list[int]:
    """Parse MM:Z tag string into absolute query positions.

    Supports the ``N+?,d1,d2,...;`` format (unknown mod on any base).
    """
    # Strip trailing semicolons and split modification entries
    # We only handle the first entry (N+?)
    mm_str = mm_str.strip().rstrip(";")
    if not mm_str:
        return []

    # Split on semicolons for multiple mod types; take first
    parts = mm_str.split(";")[0]

    # Format: "N+?,d1,d2,..." — skip the base+mod prefix
    tokens = parts.split(",")
    if len(tokens) < 2:
        return []

    # First token is "N+?" or similar base+mod specifier
    deltas = []
    for t in tokens[1:]:
        t = t.strip()
        if t:
            deltas.append(int(t))

    # Convert deltas to absolute positions
    positions = []
    prev = -1
    for d in deltas:
        pos = prev + d + 1
        positions.append(pos)
        prev = pos

    return positions
