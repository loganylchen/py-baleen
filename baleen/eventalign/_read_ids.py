"""Read-ID intersection helpers.

f5c silently drops BAM reads whose UUIDs are not present in the BLOW5
signal file.  When a sample's BAM/FASTQ/BLOW5 are mildly out of sync
(common when the FASTQ was re-basecalled from a different POD5 set, or
when the BAM was produced before all signals were transferred), this
manifests as eventalign outputs with fewer reads than the BAM split
suggested — and downstream subsampling biases towards whichever reads
happened to survive.

These helpers compute the per-condition intersection
``reads(BAM) ∩ reads(FASTQ) ∩ reads(BLOW5)`` so that every stage of
the pipeline operates on the exact same read set, and the
``--subsample-n`` budget is drawn from reads that will actually
produce signals.

Intersection sets are written to ``<file>.txt`` (newline-separated
UUIDs) under a caller-chosen directory so they can be passed across
the ``ProcessPoolExecutor`` spawn boundary by path instead of by
serialised ``set[str]`` — important when there are millions of
reads and thousands of contigs.
"""

from __future__ import annotations

import gzip
import importlib
import logging
import os
from pathlib import Path
from typing import Optional, Union, cast

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# --- BAM ---------------------------------------------------------------------


def read_ids_from_bam(
    bam_path: PathLike,
    *,
    primary_only: bool = True,
    min_mapq: int = 0,
) -> set[str]:
    """Return the set of read query-names in *bam_path* surviving the
    same primary/min_mapq filters used later by the pipeline."""
    import pysam  # local import keeps module import cheap

    bam_path = Path(bam_path)
    ids: set[str] = set()
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped:
                continue
            if primary_only and (read.is_secondary or read.is_supplementary):
                continue
            if read.mapping_quality < min_mapq:
                continue
            qn = read.query_name
            if qn is not None:
                ids.add(qn)
    return ids


# --- FASTQ -------------------------------------------------------------------


def _readdb_path_for(fastq: PathLike) -> Path:
    """f5c writes the readdb adjacent to the FASTQ as ``<fastq>.index.readdb``."""
    return Path(f"{fastq}.index.readdb")


def read_ids_from_readdb(readdb_path: PathLike) -> set[str]:
    """Parse f5c's ``.index.readdb`` (read_id<TAB>path per line)."""
    readdb_path = Path(readdb_path)
    ids: set[str] = set()
    with readdb_path.open("rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            # Format is: read_id<TAB>path.  Some readdb files contain only
            # the read_id (no path).  Splitting on first whitespace handles
            # both cases.
            rid = line.split("\t", 1)[0]
            if rid:
                ids.add(rid)
    return ids


def read_ids_from_fastq(fastq_path: PathLike) -> set[str]:
    """Parse a FASTQ (optionally gzipped) and return the set of read IDs.

    Used as a fallback when no ``.index.readdb`` companion exists.
    Read IDs are the first whitespace-delimited token of each header line.
    """
    fastq_path = Path(fastq_path)
    opener = gzip.open if str(fastq_path).endswith(".gz") else open
    ids: set[str] = set()
    line_no = 0
    with opener(str(fastq_path), "rt") as fh:  # type: ignore[arg-type]
        for line in fh:
            if line_no % 4 == 0:
                if not line.startswith("@"):
                    raise ValueError(
                        f"Malformed FASTQ at line {line_no + 1} of {fastq_path}: "
                        f"expected header starting with '@'"
                    )
                rid = line[1:].split(None, 1)[0].rstrip()
                if rid:
                    ids.add(rid)
            line_no += 1
    return ids


def read_ids_from_fastq_with_readdb(fastq_path: PathLike) -> set[str]:
    """Prefer the cheap ``.index.readdb`` if present, else parse the FASTQ.

    f5c always creates the readdb during indexing, so by the time the
    pipeline reaches the intersection step the cheap path is virtually
    always taken.  The FASTQ fallback exists for unit tests and ad-hoc
    callers that compute intersections without running f5c first.
    """
    readdb = _readdb_path_for(fastq_path)
    if readdb.exists():
        return read_ids_from_readdb(readdb)
    return read_ids_from_fastq(fastq_path)


# --- BLOW5 -------------------------------------------------------------------


def read_ids_from_blow5(blow5_path: PathLike) -> set[str]:
    """Enumerate read IDs in a BLOW5/SLOW5 file via pyslow5.

    ``pyslow5.Open(path, 'r').get_read_ids()`` returns ``(ids, n_reads)``
    where ``ids`` is a Python list of UUID strings.
    """
    pyslow5 = importlib.import_module("pyslow5")
    s5 = pyslow5.Open(str(blow5_path), "r")
    try:
        ids_list, _n = s5.get_read_ids()
    finally:
        # pyslow5.Open returns a Cython object with no documented context
        # manager; it cleans up on GC, but explicit del is harmless.
        del s5
    return set(ids_list)


# --- Intersection driver -----------------------------------------------------


def compute_condition_intersection(
    *,
    bam: PathLike,
    fastq: PathLike,
    blow5: PathLike,
    primary_only: bool = True,
    min_mapq: int = 0,
    label: str = "",
) -> set[str]:
    """Compute ``reads(BAM) ∩ reads(FASTQ) ∩ reads(BLOW5)`` for one condition."""
    bam_ids = read_ids_from_bam(bam, primary_only=primary_only, min_mapq=min_mapq)
    fq_ids = read_ids_from_fastq_with_readdb(fastq)
    blow5_ids = read_ids_from_blow5(blow5)

    inter = bam_ids & fq_ids & blow5_ids
    prefix = f"[{label}] " if label else ""
    logger.info(
        "%sread-id intersection: bam=%d fastq=%d blow5=%d -> %d",
        prefix, len(bam_ids), len(fq_ids), len(blow5_ids), len(inter),
    )
    if not inter:
        logger.warning(
            "%sread-id intersection is empty; pipeline will produce no output. "
            "Check that BAM/FASTQ/BLOW5 come from the same basecalling run.",
            prefix,
        )
    else:
        # Surface the most likely failure modes — reads that the BAM
        # mentioned but BLOW5 lacked are the f5c-silently-drops case.
        bam_minus_blow5 = len(bam_ids - blow5_ids)
        if bam_minus_blow5:
            logger.info(
                "%s  bam \\ blow5 = %d reads will be excluded "
                "(no signal available)",
                prefix, bam_minus_blow5,
            )
    return inter


def write_read_ids(ids: set[str], path: PathLike) -> Path:
    """Persist a set of read IDs to *path*, newline-separated, atomically."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        # Sorted output gives a reproducible file (helpful for diffs/hashes).
        for rid in sorted(ids):
            fh.write(rid)
            fh.write("\n")
    os.replace(tmp, path)
    return path


def load_read_ids(path: Optional[PathLike]) -> Optional[set[str]]:
    """Inverse of ``write_read_ids``.  Returns ``None`` when *path* is None
    so callers can short-circuit when the intersection feature is off."""
    if path is None:
        return None
    p = Path(path)
    with p.open("r") as fh:
        return {line.rstrip("\n") for line in fh if line.strip()}
