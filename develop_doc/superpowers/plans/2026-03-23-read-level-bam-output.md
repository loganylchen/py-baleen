# Read-Level BAM Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sorted, indexed BAM file output containing per-read modification probabilities (`p_mod_hmm`) for every read at every position, enabling region-based random access and downstream analysis.

**Architecture:** New module `baleen/eventalign/_read_bam.py` with `write_read_bam()` (internal writer) and `load_read_results()` / `load_read_results_iter()` (public readers). CLI integration via `--no-read-bam` flag. Each BAM record is a 1-kmer pseudo-alignment with `MP:f` tag for the modification probability and `RG:Z` tag for native/ivt classification.

**Tech Stack:** pysam (already a dependency), numpy, pandas

**Spec:** `docs/superpowers/specs/2026-03-23-read-level-bam-output-design.md`

---

### Task 1: Core writer — `write_read_bam()`

**Files:**
- Create: `baleen/eventalign/_read_bam.py`
- Test: `tests/test_read_bam.py`

- [ ] **Step 1: Write test for basic BAM writing**

Create `tests/test_read_bam.py` with a test that builds synthetic `ContigResult` + `ContigModificationResult`, calls `write_read_bam()`, and verifies the BAM contains the expected records.

```python
from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import numpy as np
import pysam
import pytest

pipeline = importlib.import_module("baleen.eventalign._pipeline")
hier = importlib.import_module("baleen.eventalign._hierarchical")

PositionResult = pipeline.PositionResult
ContigResult = pipeline.ContigResult


def _make_synthetic_data():
    """Build minimal ContigResult + ContigModificationResult for testing."""
    n_native, n_ivt = 3, 2
    n_total = n_native + n_ivt

    # Distance matrix (not used by writer, but required by dataclass)
    mat = np.ones((n_total, n_total), dtype=np.float64)
    np.fill_diagonal(mat, 0.0)

    pr = PositionResult(
        position=100,
        reference_kmer="AACGU",
        n_native_reads=n_native,
        n_ivt_reads=n_ivt,
        native_read_names=["nat_0", "nat_1", "nat_2"],
        ivt_read_names=["ivt_0", "ivt_1"],
        distance_matrix=mat,
    )

    cr = ContigResult(
        contig="ecoli23S",
        native_depth=30.0,
        ivt_depth=20.0,
        positions={100: pr},
    )

    # Build PositionStats with known p_mod_hmm values
    ps = hier.PositionStats(
        position=100,
        reference_kmer="AACGU",
        coverage_class=hier.CoverageClass.HIGH,
        n_ivt=n_ivt,
        n_native=n_native,
        mu_raw=1.0, sigma_raw=0.5,
        mu_shrunk=1.0, sigma_shrunk=0.5,
        scores=np.zeros(n_total),
        z_scores=np.zeros(n_total),
        p_null=np.ones(n_total),
        p_mod_raw=np.array([0.8, 0.9, 0.7, 0.1, 0.05]),
        mixture_pi=0.5, mixture_null_gate=False, gate_weight=1.0,
        p_mod_knn=np.array([0.75, 0.85, 0.65, 0.1, 0.05]),
        p_mod_hmm=np.array([0.85, 0.92, 0.78, 0.05, 0.02]),
    )

    cmr = hier.ContigModificationResult(
        contig="ecoli23S",
        position_stats={100: ps},
        native_trajectories=[],
        ivt_trajectories=[],
        global_mu=1.0,
        global_sigma=0.5,
    )

    return {"ecoli23S": cr}, {"ecoli23S": cmr}


def test_write_read_bam_basic():
    """write_read_bam produces a sorted, indexed BAM with correct records."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    contig_results, hmm_results = _make_synthetic_data()

    with tempfile.TemporaryDirectory() as tmp:
        # Create a minimal FASTA for header
        ref_path = Path(tmp) / "ref.fa"
        ref_path.write_text(">ecoli23S\n" + "A" * 3000 + "\n")

        bam_path = Path(tmp) / "read_results.bam"
        result_path = read_bam.write_read_bam(
            hmm_results, contig_results, ref_path, bam_path,
        )

        # Verify BAM file and index exist
        assert result_path.exists()
        assert Path(str(result_path) + ".bai").exists()

        # Read back and verify records
        with pysam.AlignmentFile(str(result_path), "rb") as bam:
            records = list(bam.fetch())

        assert len(records) == 5  # 3 native + 2 ivt

        # Check first native read
        r0 = records[0]
        assert r0.query_name == "nat_0"
        assert r0.reference_name == "ecoli23S"
        assert r0.reference_start == 100
        assert abs(r0.get_tag("MP") - 0.85) < 1e-5
        assert r0.get_tag("RG") == "native"
        assert r0.get_tag("KM") == "AACGU"
        # SEQ should have U->T conversion
        assert "U" not in r0.query_sequence
        assert r0.query_sequence == "AACGT"
        # CIGAR should match kmer length
        assert r0.cigarstring == "5M"

        # Check last IVT read
        r4 = records[4]
        assert r4.query_name == "ivt_1"
        assert r4.get_tag("RG") == "ivt"
        assert abs(r4.get_tag("MP") - 0.02) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_read_bam.py::test_write_read_bam_basic -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'baleen.eventalign._read_bam'`

- [ ] **Step 3: Implement `write_read_bam()`**

Create `baleen/eventalign/_read_bam.py`:

```python
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
    hierarchical_results: dict[str, ContigModificationResult],
    contig_results: dict[str, ContigResult],
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
                cr = contig_results.get(contig)
                if cr is None:
                    continue

                for pos in sorted(cmr.position_stats.keys()):
                    ps = cmr.position_stats[pos]
                    pr = cr.positions.get(pos)
                    if pr is None:
                        continue

                    kmer = pr.reference_kmer
                    kmer_dna = kmer.replace("U", "T").replace("u", "t")
                    cigar_len = len(kmer)

                    # Native reads: indices 0..n_native-1
                    for i, name in enumerate(pr.native_read_names):
                        p_mod = float(ps.p_mod_hmm[i])
                        if math.isnan(p_mod):
                            continue
                        a = _make_record(
                            bam_out, name, contig, pos, kmer_dna, kmer,
                            cigar_len, p_mod, "native", header,
                        )
                        bam_out.write(a)
                        n_records += 1

                    # IVT reads: indices n_native..n_total-1
                    for j, name in enumerate(pr.ivt_read_names):
                        p_mod = float(ps.p_mod_hmm[ps.n_native + j])
                        if math.isnan(p_mod):
                            continue
                        a = _make_record(
                            bam_out, name, contig, pos, kmer_dna, kmer,
                            cigar_len, p_mod, "ivt", header,
                        )
                        bam_out.write(a)
                        n_records += 1

        # Sort and index
        pysam.sort("-o", str(out), str(tmp_unsorted))
        pysam.index(str(out))

        logger.info(
            "Wrote %d read-level records to %s", n_records, out,
        )
        return out

    finally:
        if tmp_unsorted.exists():
            tmp_unsorted.unlink()


def _build_header(
    ref_fasta: PathLike,
    hierarchical_results: dict[str, ContigModificationResult],
) -> dict:
    """Build BAM header from reference FASTA or fallback to contig names."""
    sq_lines: list[dict[str, Any]] = []

    ref_path = Path(ref_fasta)
    if ref_path.exists():
        with pysam.FastaFile(str(ref_path)) as fa:
            for name in fa.references:
                sq_lines.append({"SN": name, "LN": fa.get_reference_length(name)})
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
    header: dict,
) -> pysam.AlignedSegment:
    """Create a single BAM record."""
    a = pysam.AlignedSegment(bam_out.header)
    a.query_name = read_name
    a.flag = 0
    a.reference_id = bam_out.get_tid(contig)
    a.reference_start = position
    a.mapping_quality = 255
    a.cigar = [(0, cigar_len)]  # M operation
    a.query_sequence = kmer_dna
    a.query_qualities = None
    a.set_tag("MP", p_mod, "f")
    a.set_tag("RG", read_group, "Z")
    a.set_tag("KM", kmer_rna, "Z")
    return a
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_read_bam.py::test_write_read_bam_basic -v`
Expected: PASS

- [ ] **Step 5: Write test for NaN handling**

Add to `tests/test_read_bam.py`:

```python
def test_write_read_bam_skips_nan():
    """Records with NaN p_mod_hmm are omitted from the BAM."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    contig_results, hmm_results = _make_synthetic_data()

    # Set one native read's p_mod_hmm to NaN
    ps = hmm_results["ecoli23S"].position_stats[100]
    ps.p_mod_hmm[1] = np.nan  # nat_1 becomes NaN

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.fa"
        ref_path.write_text(">ecoli23S\n" + "A" * 3000 + "\n")

        bam_path = Path(tmp) / "read_results.bam"
        read_bam.write_read_bam(hmm_results, contig_results, ref_path, bam_path)

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            records = list(bam.fetch())

        # 5 total - 1 NaN = 4
        assert len(records) == 4
        names = [r.query_name for r in records]
        assert "nat_1" not in names
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_read_bam.py -v`
Expected: Both tests PASS

- [ ] **Step 7: Commit**

```bash
git add baleen/eventalign/_read_bam.py tests/test_read_bam.py
git commit -m "feat: add write_read_bam() for per-read BAM output"
```

---

### Task 2: Public readers — `load_read_results()` and `load_read_results_iter()`

**Files:**
- Modify: `baleen/eventalign/_read_bam.py`
- Test: `tests/test_read_bam.py`

- [ ] **Step 1: Write tests for load functions**

Add to `tests/test_read_bam.py`:

```python
def _write_test_bam(tmp_dir):
    """Helper: write a test BAM and return its path."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")
    contig_results, hmm_results = _make_synthetic_data()

    ref_path = Path(tmp_dir) / "ref.fa"
    ref_path.write_text(">ecoli23S\n" + "A" * 3000 + "\n")

    bam_path = Path(tmp_dir) / "read_results.bam"
    read_bam.write_read_bam(hmm_results, contig_results, ref_path, bam_path)
    return bam_path


def test_load_read_results_full():
    """load_read_results returns a DataFrame with all records."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    with tempfile.TemporaryDirectory() as tmp:
        bam_path = _write_test_bam(tmp)
        df = read_bam.load_read_results(bam_path)

    assert len(df) == 5
    assert set(df.columns) == {"contig", "position", "kmer", "read_name", "is_native", "p_mod_hmm"}
    assert df["is_native"].sum() == 3
    assert (~df["is_native"]).sum() == 2
    assert df["contig"].iloc[0] == "ecoli23S"
    assert df["kmer"].iloc[0] == "AACGU"  # Original RNA kmer from KM tag


def test_load_read_results_region_filter():
    """load_read_results with region filter returns subset."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    with tempfile.TemporaryDirectory() as tmp:
        bam_path = _write_test_bam(tmp)

        # Query region that includes position 100
        df = read_bam.load_read_results(bam_path, contig="ecoli23S", start=99, end=105)
        assert len(df) == 5

        # Query region that excludes position 100
        df_empty = read_bam.load_read_results(bam_path, contig="ecoli23S", start=200, end=300)
        assert len(df_empty) == 0


def test_load_read_results_iter():
    """load_read_results_iter yields dicts with correct keys."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")

    with tempfile.TemporaryDirectory() as tmp:
        bam_path = _write_test_bam(tmp)
        records = list(read_bam.load_read_results_iter(bam_path))

    assert len(records) == 5
    r0 = records[0]
    assert set(r0.keys()) == {"contig", "position", "kmer", "read_name", "is_native", "p_mod_hmm"}
    assert isinstance(r0["p_mod_hmm"], float)
    assert isinstance(r0["is_native"], bool)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_read_bam.py -k "load" -v`
Expected: FAIL — `AttributeError: module has no attribute 'load_read_results'`

- [ ] **Step 3: Implement load functions**

Add to `baleen/eventalign/_read_bam.py`:

```python
def load_read_results(
    bam_path: PathLike,
    contig: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
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
                "position": read.reference_start,
                "kmer": read.get_tag("KM"),
                "read_name": read.query_name,
                "is_native": read.get_tag("RG") == "native",
                "p_mod_hmm": float(read.get_tag("MP")),
            }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_read_bam.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add baleen/eventalign/_read_bam.py tests/test_read_bam.py
git commit -m "feat: add load_read_results() and load_read_results_iter()"
```

---

### Task 3: Export public API

**Files:**
- Modify: `baleen/eventalign/__init__.py`

- [ ] **Step 1: Write test for public import**

Add to `tests/test_read_bam.py`:

```python
def test_public_api_exports():
    """load_read_results and load_read_results_iter are importable from baleen.eventalign."""
    from baleen.eventalign import load_read_results, load_read_results_iter

    assert callable(load_read_results)
    assert callable(load_read_results_iter)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_read_bam.py::test_public_api_exports -v`
Expected: FAIL — `ImportError: cannot import name 'load_read_results'`

- [ ] **Step 3: Add exports to `__init__.py`**

In `baleen/eventalign/__init__.py`, add the import and update `__all__`:

```python
# Add import (near the other _aggregation imports):
from baleen.eventalign._read_bam import load_read_results, load_read_results_iter

# Add to __all__ list:
    "load_read_results",
    "load_read_results_iter",
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_read_bam.py::test_public_api_exports -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add baleen/eventalign/__init__.py tests/test_read_bam.py
git commit -m "feat: export load_read_results from public API"
```

---

### Task 4: CLI integration — `baleen run`

**Files:**
- Modify: `baleen/cli.py`
- Test: `tests/test_read_bam.py`

- [ ] **Step 1: Write test for CLI flag**

Add to `tests/test_read_bam.py`:

```python
def test_cli_no_read_bam_flag():
    """The --no-read-bam flag is accepted by the argparse parser."""
    from baleen.cli import main
    import argparse

    # Verify the flag exists in the parser
    parser = argparse.ArgumentParser()
    from baleen.cli import _add_run_args
    _add_run_args(parser)

    # Should parse without error
    args = parser.parse_args(
        ["--native-bam", "a", "--native-fastq", "b", "--native-blow5", "c",
         "--ivt-bam", "d", "--ivt-fastq", "e", "--ivt-blow5", "f",
         "--ref", "g", "--no-read-bam"]
    )
    assert args.no_read_bam is True

    # Default should be False
    args2 = parser.parse_args(
        ["--native-bam", "a", "--native-fastq", "b", "--native-blow5", "c",
         "--ivt-bam", "d", "--ivt-fastq", "e", "--ivt-blow5", "f",
         "--ref", "g"]
    )
    assert args2.no_read_bam is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_read_bam.py::test_cli_no_read_bam_flag -v`
Expected: FAIL — `error: unrecognized arguments: --no-read-bam`

- [ ] **Step 3: Add `--no-read-bam` flag to `_add_run_args()`**

In `baleen/cli.py`, add to the "miscellaneous" argument group (after `--keep-temp`, around line 118):

```python
    misc.add_argument(
        "--no-read-bam", action="store_true", default=False,
        help="Skip writing per-read BAM output (read_results.bam)",
    )
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_read_bam.py::test_cli_no_read_bam_flag -v`
Expected: PASS

- [ ] **Step 5: Add `write_read_bam` call to `_cmd_run()`**

In `baleen/cli.py`, in the `_cmd_run()` function:

1. Add import at the top of the function (alongside existing imports):
```python
    from baleen.eventalign._read_bam import write_read_bam
```

2. After the site aggregation block (after line 253 `logger.info("Aggregation done in %.1fs", agg_time)`), add:

```python
    # Step 4: Write per-read BAM
    if not args.no_read_bam:
        logger.info("Writing per-read BAM...")
        t0 = time.perf_counter()
        bam_path = output_dir / "read_results.bam"
        write_read_bam(hmm_results, contig_results, args.ref, bam_path)
        bam_time = time.perf_counter() - t0
        logger.info("Per-read BAM done in %.1fs", bam_time)
    else:
        bam_path = None
```

3. Update step numbering in log messages: `"Step 1/3"` → `"Step 1/4"`, `"Step 2/3"` → `"Step 2/4"`, `"Step 3/3"` → `"Step 3/4"`.

4. Add BAM path to the results summary:
```python
    if bam_path:
        logger.info("  Read results:      %s", bam_path)
```

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/test_read_bam.py tests/test_cli.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add baleen/cli.py tests/test_read_bam.py
git commit -m "feat: add --no-read-bam CLI flag and write_read_bam integration"
```

---

### Task 5: CLI integration — `baleen aggregate`

**Files:**
- Modify: `baleen/cli.py`

- [ ] **Step 1: Add `write_read_bam` to `_cmd_aggregate()`**

In `baleen/cli.py`, in `_cmd_aggregate()`:

1. Add `--no-read-bam` to `_add_aggregate_args()`:
```python
    parser.add_argument(
        "--no-read-bam", action="store_true", default=False,
        help="Skip writing per-read BAM output",
    )
    parser.add_argument(
        "--ref", type=str, default=None,
        help="Reference FASTA (required for per-read BAM output)",
    )
```

2. Add import and call in `_cmd_aggregate()` after `write_site_tsv(sites, args.output)`:
```python
    from baleen.eventalign._read_bam import write_read_bam

    if not args.no_read_bam:
        if args.ref is None:
            logger.warning("Skipping read-level BAM: --ref not provided")
        else:
            bam_path = Path(args.output).parent / "read_results.bam"
            write_read_bam(hmm_results, contig_results, args.ref, bam_path)
            logger.info("Wrote per-read BAM to %s", bam_path)
```

- [ ] **Step 2: Run existing CLI tests**

Run: `pytest tests/test_cli.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add baleen/cli.py
git commit -m "feat: add write_read_bam to aggregate command"
```

---

### Task 6: Multi-contig and edge case tests

**Files:**
- Modify: `tests/test_read_bam.py`

- [ ] **Step 1: Write multi-contig test**

```python
def test_write_read_bam_multi_contig():
    """BAM correctly handles multiple contigs sorted by (contig, position)."""
    read_bam = importlib.import_module("baleen.eventalign._read_bam")
    n = 3  # reads per group

    def _make_pos(pos, kmer="ACGTA"):
        mat = np.ones((n + n, n + n), dtype=np.float64)
        np.fill_diagonal(mat, 0.0)
        return PositionResult(
            position=pos, reference_kmer=kmer,
            n_native_reads=n, n_ivt_reads=n,
            native_read_names=[f"nat_{pos}_{i}" for i in range(n)],
            ivt_read_names=[f"ivt_{pos}_{i}" for i in range(n)],
            distance_matrix=mat,
        )

    def _make_ps(pos):
        return hier.PositionStats(
            position=pos, reference_kmer="ACGTA",
            coverage_class=hier.CoverageClass.HIGH,
            n_ivt=n, n_native=n,
            mu_raw=1.0, sigma_raw=0.5, mu_shrunk=1.0, sigma_shrunk=0.5,
            scores=np.zeros(2 * n), z_scores=np.zeros(2 * n),
            p_null=np.ones(2 * n),
            p_mod_raw=np.full(2 * n, 0.5),
            mixture_pi=0.5, mixture_null_gate=False, gate_weight=1.0,
            p_mod_knn=np.full(2 * n, 0.5),
            p_mod_hmm=np.full(2 * n, 0.5),
        )

    contig_results = {
        "chrB": ContigResult("chrB", 30, 20, {200: _make_pos(200), 300: _make_pos(300)}),
        "chrA": ContigResult("chrA", 30, 20, {100: _make_pos(100)}),
    }
    hmm_results = {
        "chrB": hier.ContigModificationResult("chrB", {200: _make_ps(200), 300: _make_ps(300)}, [], [], 1.0, 0.5),
        "chrA": hier.ContigModificationResult("chrA", {100: _make_ps(100)}, [], [], 1.0, 0.5),
    }

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.fa"
        ref_path.write_text(">chrA\n" + "A" * 1000 + "\n>chrB\n" + "A" * 1000 + "\n")

        bam_path = Path(tmp) / "read_results.bam"
        read_bam.write_read_bam(hmm_results, contig_results, ref_path, bam_path)

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            records = list(bam.fetch())

        # 3 positions × (3 native + 3 ivt) = 18 records
        assert len(records) == 18

        # Verify sorted by (contig, position)
        prev_contig, prev_pos = "", -1
        for r in records:
            if r.reference_name == prev_contig:
                assert r.reference_start >= prev_pos
            prev_contig = r.reference_name
            prev_pos = r.reference_start

        # Region query should work
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            chrA_records = list(bam.fetch("chrA"))
        assert len(chrA_records) == 6  # 1 position × 6 reads
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_read_bam.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_read_bam.py
git commit -m "test: add multi-contig and edge case tests for read BAM"
```

---

### Task 7: Export from top-level `baleen` package

**Files:**
- Modify: `baleen/__init__.py`

- [ ] **Step 1: Check current top-level exports**

Read `baleen/__init__.py` to see what's currently re-exported.

- [ ] **Step 2: Add `load_read_results` and `load_read_results_iter` to top-level exports**

Add to `baleen/__init__.py` in the import and `__all__` list:

```python
from baleen.eventalign import load_read_results, load_read_results_iter
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add baleen/__init__.py
git commit -m "feat: export load_read_results from top-level baleen package"
```
