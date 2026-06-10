"""krill-backed eventalign — produces f5c-eventalign-format TSV.

Replaces the external f5c CLI with krill's in-process aligner.  For every
primary, forward-mapped read in the contig BAM, the read's raw signal is
aligned to its mapped reference subsequence (HMM confidence disabled: dense,
skip-free), and the result is written in f5c's 16-column ``--samples`` TSV
format so the rest of the pipeline (``group_signals_by_position`` -> DTW ->
V1/V2/V3 -> aggregation) runs unchanged.

Coordinate convention (verified against real f5c output, RNA002 5-mer):
    f5c ``position`` = 0-based index of the FIRST base of the k-mer.
    krill ``position`` = central base (kmer_center=2 for a 5-mer).
    => f5c_position = krill_position - aligner.kmer_center
``group_signals_by_position`` then applies its usual ``+ len(kmer)//2 + 1``
shift to align predicted sites with the reference.

samples column: f5c with ``--scale-events`` writes per-sample pA.  We write
pA = (raw + offset) * range / digitisation (krill's convention), so units
match.  A residual per-read scale/shift vs f5c is symmetric across native/IVT
and is absorbed by the per-position empirical-Bayes / mixture calibration
downstream.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, Union, cast

import numpy as np
import pyfastx
import pysam
import pyslow5

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

DEFAULT_PORE = "rna002"

# f5c eventalign --samples header (16 cols).  group_signals_by_position only
# reads: contig, position, reference_kmer, read_name, start_idx, samples.
_HEADER = (
    "contig\tposition\treference_kmer\tread_name\tstrand\tevent_index\t"
    "event_level_mean\tevent_stdv\tevent_length\tmodel_kmer\tmodel_mean\t"
    "model_stdv\tstandardized_level\tstart_idx\tend_idx\tsamples"
)

_krill_version: Optional[str] = None
# Aligner construction loads the pore model; cache per (pore) within a process.
_ALIGNER_CACHE: dict[str, object] = {}
_REF_CACHE: dict[str, object] = {}


def check_krill() -> str:
    """Verify krill is importable and return a version string.

    Returns
    -------
    str
        krill version (or ``"unknown"`` if krill exposes no ``__version__``).

    Raises
    ------
    RuntimeError
        If krill cannot be imported.
    """
    global _krill_version
    if _krill_version is not None:
        return _krill_version
    try:
        import krill
    except ModuleNotFoundError as exc:  # pragma: no cover - install-time guard
        raise RuntimeError(
            "krill not found. baleen's eventalign engine requires the 'krill' "
            "package (not on PyPI). Install it from the project index, e.g. "
            "`pip install krill --no-deps --index-url "
            "https://loganylchen.github.io/krill-dist/cu122/simple/` (GPU) or "
            "the /simple/ index (CPU), or use a prebuilt baleen Docker image."
        ) from exc
    version = getattr(krill, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as _pkg_version
            version = _pkg_version("krill")
        except Exception:  # noqa: BLE001
            version = "unknown"
    _krill_version = str(version)
    return _krill_version


def _get_aligner(pore: str):
    import krill

    if pore not in _ALIGNER_CACHE:
        _ALIGNER_CACHE[pore] = krill.Aligner(
            pore=pore,
            use_gpu=False,
            hmm_confidence=False,
            keep_kmer_skips=False,
        )
        logger.info("krill Aligner(pore=%s) constructed", pore)
    return _ALIGNER_CACHE[pore]


def _get_ref(ref_fasta: PathLike):
    key = str(ref_fasta)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = pyfastx.Fasta(key)
    return _REF_CACHE[key]


def is_blow5_indexed(blow5: PathLike) -> bool:
    """Check whether a SLOW5/BLOW5 ``.idx`` exists and is non-empty."""
    blow5_path = Path(blow5)
    idx_path = blow5_path.with_name(f"{blow5_path.name}.idx")
    return idx_path.exists() and idx_path.stat().st_size > 0


def index_blow5(blow5: PathLike) -> None:
    """Create a BLOW5 index using slow5tools (pyslow5 random access needs it).

    Raises
    ------
    RuntimeError
        If the slow5tools indexing command fails.
    """
    blow5_path = Path(blow5)
    if is_blow5_indexed(blow5_path):
        logger.info("Skipping slow5tools index; BLOW5 already indexed: %s", blow5_path)
        return
    cmd = ["slow5tools", "index", str(blow5_path)]
    logger.debug("Running command: %s", " ".join(cmd))
    try:
        _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = cast(Optional[str], exc.stderr)
        raise RuntimeError(f"slow5tools index failed: {(stderr or '').strip()}") from exc


def run_eventalign(
    bam: PathLike,
    ref_fasta: PathLike,
    fastq: PathLike,           # unused (kept for signature parity with f5c)
    blow5: PathLike,
    output_tsv: PathLike,
    *,
    rna: bool = True,          # unused; krill pore drives chemistry
    kmer_model: Optional[str] = None,  # unused
    extra_args: Optional[list[str]] = None,  # unused
    min_mapq: int = 0,
    pore: str = DEFAULT_PORE,
) -> Path:
    """Align every primary, forward-mapped read in *bam* with krill.

    Drop-in replacement for the former ``_f5c.run_eventalign``: writes an
    f5c-format eventalign TSV with HMM confidence disabled (dense, skip-free).

    Returns
    -------
    pathlib.Path
        Output TSV path.
    """
    bam_path = Path(bam)
    out_path = Path(output_tsv)
    aligner = _get_aligner(pore)
    kmer_center = int(aligner.kmer_center)
    ref = _get_ref(ref_fasta)

    s5 = pyslow5.Open(str(blow5), "r")
    n_reads = n_rows = n_no_signal = n_failed = n_reverse = 0

    tmp_path = out_path.with_suffix(".tmp")
    t0 = time.perf_counter()
    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bamf, \
                tmp_path.open("w", encoding="utf-8") as out:
            out.write(_HEADER + "\n")
            for aln in bamf.fetch(until_eof=True):
                if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                    continue
                if aln.mapping_quality < min_mapq:
                    continue
                if aln.is_reverse:
                    # Direct-RNA on a transcriptome ref should map forward;
                    # skip (and count) reverse alignments for safety.
                    n_reverse += 1
                    continue

                rid = aln.query_name
                contig = aln.reference_name
                rs, re_ = aln.reference_start, aln.reference_end  # 0-based half-open
                ref_sub = str(ref[contig][rs:re_].seq)
                if not ref_sub:
                    continue

                try:
                    rd = s5.get_read(rid, pA=False)
                except Exception:  # noqa: BLE001
                    rd = None
                if rd is None:
                    n_no_signal += 1
                    continue

                raw = np.asarray(rd["signal"], dtype=np.float32)
                digit = float(rd["digitisation"])
                offset = float(rd["offset"])
                rng = float(rd["range"])
                sr = float(rd["sampling_rate"])
                pA = (raw + offset) * (rng / digit)

                res = aligner.align({
                    "read_id": rid, "sequence": ref_sub, "signal": raw,
                    "digitisation": digit, "offset": offset, "range": rng,
                    "sample_rate": sr, "start": rs,
                })[0]
                n_reads += 1
                if res["status"] != 0:
                    n_failed += 1
                    continue

                P = res["position"]
                RK = res["reference_kmer"]
                EI = res["event_index"]
                ELM = res["event_level_mean"]
                ESD = res["event_stdv"]
                ELN = res["event_length"]
                MK = res["model_kmer"]
                MM = res["model_mean"]
                MSD = res["model_stdv"]
                SL = res["standardized_level"]
                SI = res["start_idx"]
                END = res["end_idx"]

                for i in range(int(P.size)):
                    si = int(SI[i])
                    ei = int(END[i])
                    seg = pA[si:ei]
                    if seg.size == 0:
                        continue
                    f5c_pos = int(P[i]) - kmer_center
                    if f5c_pos < 0:
                        continue
                    samples = ",".join(np.char.mod("%.3f", seg))
                    sl = SL[i]
                    sl_str = "" if (sl != sl) else f"{float(sl):.2f}"  # NaN -> ""
                    out.write(
                        f"{contig}\t{f5c_pos}\t{RK[i]}\t{rid}\tt\t{int(EI[i])}\t"
                        f"{float(ELM[i]):.2f}\t{float(ESD[i]):.3f}\t{float(ELN[i]):.5f}\t"
                        f"{MK[i]}\t{float(MM[i]):.2f}\t{float(MSD[i]):.2f}\t{sl_str}\t"
                        f"{si}\t{ei}\t{samples}\n"
                    )
                    n_rows += 1
        tmp_path.replace(out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        s5.close()

    logger.debug(
        "krill eventalign %s: %d reads, %d rows in %.1fs "
        "(no_signal=%d, failed=%d, reverse=%d)",
        bam_path.name, n_reads, n_rows, time.perf_counter() - t0,
        n_no_signal, n_failed, n_reverse,
    )
    return out_path
