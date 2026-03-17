"""Utilities for invoking the f5c eventalign command-line tool.

This module provides a small wrapper around f5c/slow5tools subprocess calls
used by the eventalign pipeline.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Optional, Union, cast

logger = logging.getLogger(__name__)

_f5c_version: Optional[str] = None
PathLike = Union[str, Path]
_VERSION_PATTERN = re.compile(r"\bf5c\s+v?(\d+(?:\.\d+)*)\b", re.IGNORECASE)


def check_f5c() -> str:
    """Check f5c availability and cache its version.

    Returns
    -------
    str
        Installed f5c version string (for example ``"1.6"``).

    Raises
    ------
    RuntimeError
        If f5c is not available in ``PATH``.
    RuntimeError
        If version output cannot be parsed.
    """
    global _f5c_version

    if _f5c_version is not None:
        return _f5c_version

    try:
        result = subprocess.run(
            ["f5c", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("f5c not found in PATH. Please install f5c.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = cast(Optional[str], exc.stderr)
        stderr_text = (stderr or "").strip()
        raise RuntimeError(f"f5c not found or failed to run: {stderr_text}") from exc

    version_output = (result.stdout or result.stderr or "").strip()
    match = _VERSION_PATTERN.search(version_output)
    if match is None:
        raise RuntimeError(f"Could not parse f5c version from output: {version_output!r}")

    version = match.group(1)
    _f5c_version = version
    return version


def get_f5c_version() -> tuple[int, ...]:
    """Return parsed f5c version components.

    Returns
    -------
    tuple of int
        Parsed version tuple (for example ``(1, 6)`` or ``(1, 6, 1)``).

    Raises
    ------
    RuntimeError
        If f5c cannot be detected or version cannot be parsed.
    """
    version_str = _f5c_version if _f5c_version is not None else check_f5c()
    return tuple(int(part) for part in version_str.split("."))


def is_indexed(fastq: PathLike) -> bool:
    """Check whether f5c FASTQ index exists and is non-empty.

    Parameters
    ----------
    fastq : str or pathlib.Path
        FASTQ file used for f5c indexing.

    Returns
    -------
    bool
        ``True`` when ``<fastq>.index.readdb`` exists and has non-zero size,
        otherwise ``False``.
    """
    fastq_path = Path(fastq)
    index_path = fastq_path.with_name(f"{fastq_path.name}.index.readdb")
    return index_path.exists() and index_path.stat().st_size > 0


def is_blow5_indexed(blow5: PathLike) -> bool:
    """Check whether SLOW5/BLOW5 index exists and is non-empty.

    Parameters
    ----------
    blow5 : str or pathlib.Path
        BLOW5 file to check.

    Returns
    -------
    bool
        ``True`` when ``<blow5>.idx`` exists and has non-zero size,
        otherwise ``False``.
    """
    blow5_path = Path(blow5)
    idx_path = blow5_path.with_name(f"{blow5_path.name}.idx")
    return idx_path.exists() and idx_path.stat().st_size > 0


def index_fastq_blow5(fastq: PathLike, blow5: PathLike) -> None:
    """Index FASTQ against BLOW5 using f5c.

    Parameters
    ----------
    fastq : str or pathlib.Path
        FASTQ file path.
    blow5 : str or pathlib.Path
        BLOW5 file path.

    Raises
    ------
    RuntimeError
        If f5c indexing command fails.
    """
    fastq_path = Path(fastq)
    blow5_path = Path(blow5)

    if is_indexed(fastq_path):
        logger.info("Skipping f5c index; FASTQ already indexed: %s", fastq_path)
        return

    cmd = ["f5c", "index", "--slow5", str(blow5_path), str(fastq_path)]
    logger.debug("Running command: %s", " ".join(cmd))

    t0 = time.perf_counter()
    try:
        _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = cast(Optional[str], exc.stderr)
        stderr_text = (stderr or "").strip()
        raise RuntimeError(f"f5c index failed: {stderr_text}") from exc
    logger.debug("f5c index completed in %.1fs: %s", time.perf_counter() - t0, fastq_path)


def index_blow5(blow5: PathLike) -> None:
    """Create BLOW5 index using slow5tools.

    Parameters
    ----------
    blow5 : str or pathlib.Path
        BLOW5 file path.

    Raises
    ------
    RuntimeError
        If slow5tools indexing command fails.
    """
    blow5_path = Path(blow5)

    if is_blow5_indexed(blow5_path):
        logger.info("Skipping slow5tools index; BLOW5 already indexed: %s", blow5_path)
        return

    cmd = ["slow5tools", "index", str(blow5_path)]
    logger.debug("Running command: %s", " ".join(cmd))

    t0 = time.perf_counter()
    try:
        _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = cast(Optional[str], exc.stderr)
        stderr_text = (stderr or "").strip()
        raise RuntimeError(f"slow5tools index failed: {stderr_text}") from exc
    logger.debug("slow5tools index completed in %.1fs: %s", time.perf_counter() - t0, blow5_path)


def run_eventalign(
    bam: PathLike,
    ref_fasta: PathLike,
    fastq: PathLike,
    blow5: PathLike,
    output_tsv: PathLike,
    *,
    rna: bool = True,
    kmer_model: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
) -> Path:
    """Run ``f5c eventalign`` and write TSV output.

    Parameters
    ----------
    bam : str or pathlib.Path
        Input BAM file.
    ref_fasta : str or pathlib.Path
        Reference FASTA file.
    fastq : str or pathlib.Path
        Input FASTQ file.
    blow5 : str or pathlib.Path
        Input BLOW5 file.
    output_tsv : str or pathlib.Path
        Output eventalign TSV path.
    rna : bool, optional
        If ``True``, include the ``--rna`` flag.
    kmer_model : str, optional
        Optional k-mer model path/name passed via ``--kmer-model``.
    extra_args : list of str, optional
        Additional command-line arguments appended as-is.

    Returns
    -------
    pathlib.Path
        Output TSV path.

    Raises
    ------
    RuntimeError
        If the f5c eventalign command fails.
    """
    bam_path = Path(bam)
    ref_fasta_path = Path(ref_fasta)
    fastq_path = Path(fastq)
    blow5_path = Path(blow5)
    output_path = Path(output_tsv)

    cmd = [
        "f5c",
        "eventalign",
        "-b",
        str(bam_path),
        "-g",
        str(ref_fasta_path),
        "-r",
        str(fastq_path),
        "--slow5",
        str(blow5_path),
        "--samples",
        "--signal-index",
        "--scale-events",
        "--print-read-names",
    ]

    if rna:
        cmd.append("--rna")
    if kmer_model is not None:
        cmd.extend(["--kmer-model", kmer_model])
    if extra_args:
        cmd.extend(extra_args)

    logger.debug("Running command: %s", " ".join(cmd))

    t0 = time.perf_counter()
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            _ = subprocess.run(
                cmd,
                check=True,
                stdout=output_file,
                stderr=subprocess.PIPE,
                text=True,
            )
    except subprocess.CalledProcessError as exc:
        stderr = cast(Optional[str], exc.stderr)
        stderr_text = (stderr or "").strip()
        raise RuntimeError(f"f5c eventalign failed: {stderr_text}") from exc

    elapsed = time.perf_counter() - t0
    size_kb = output_path.stat().st_size / 1024
    logger.debug("f5c eventalign completed in %.1fs (output: %.1f KB): %s", elapsed, size_kb, output_path)
    return output_path
