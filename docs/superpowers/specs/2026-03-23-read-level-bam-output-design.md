# Read-Level BAM Output for Baleen

## Problem

Baleen computes per-read modification probabilities (`p_mod_hmm`) during the hierarchical pipeline but only persists site-level aggregated results as TSV. Users need read-level results for visualization (genome browsers), single-molecule trajectory analysis, and custom statistical pipelines. At human transcriptome scale (10K+ contigs, millions of positions, 50-200 reads/position), the storage format must support efficient region-based random access and good compression.

## Decision

Store read-level modification probabilities in a **sorted, indexed BAM file** (`read_results.bam` + `.bam.bai`). Each BAM record represents one read at one position with `p_mod_hmm` in a custom float tag.

### Why BAM over alternatives

- **HDF5**: No region-range queries without custom indexing; adds `h5py` dependency; not loadable in genome browsers.
- **Parquet**: Excellent for bulk scans but no efficient region queries; adds `pyarrow` dependency.
- **BAM**: Region queries via index (O(log n)); pysam already a dependency; samtools/IGV interop; excellent compression (~10x).

## BAM Record Structure

Each record is a 1-kmer pseudo-alignment:

| BAM Field | Content | Example |
|-----------|---------|---------|
| `QNAME` | Read name | `read_0a3f_...` |
| `FLAG` | `0` (mapped forward) | `0` |
| `RNAME` | Contig name | `ecoli23S` |
| `POS` | Genomic position (0-based in BAM, 1-based in SAM text) | `745` (BAM) / `746` (SAM) |
| `MAPQ` | `255` (not meaningful) | `255` |
| `CIGAR` | `{len(kmer)}M` (derived from `reference_kmer`) | `5M` |
| `SEQ` | Reference kmer with U→T conversion (BAM only supports ACGTN) | `AACGT` |
| `QUAL` | `*` (not applicable) | `*` |

**Coordinate system**: Pipeline positions from f5c eventalign are 0-based. pysam uses 0-based coordinates internally. No conversion needed — positions are written directly to pysam.

**RNA bases**: BAM only supports the DNA alphabet (A, C, G, T, N). All `U` bases in `reference_kmer` are converted to `T` when writing to SEQ. The original RNA kmer is preserved in a custom tag (`KM:Z`).

**CIGAR length**: Derived from `len(reference_kmer)` at runtime, not hardcoded. This ensures compatibility with future nanopore chemistries that may use different kmer sizes.

### Custom tags

| Tag | Type | Description |
|-----|------|-------------|
| `MP:f` | float | HMM-smoothed modification probability (0.0-1.0) |
| `RG:Z` | string | Read group: `native` or `ivt` |
| `KM:Z` | string | Original RNA kmer (preserves U bases) |

### BAM header

- `@SQ` lines: contig names and lengths from reference FASTA (or from saved metadata if FASTA unavailable)
- `@RG`: two read groups — `ID:native` and `ID:ivt`
- `@PG`: baleen version and pipeline parameters

### IVT reads

Both native and IVT reads are included in the BAM. IVT reads carry `RG:Z:ivt` and generally have low `p_mod_hmm` values (~0). Including them allows users to verify that the IVT control behaves as expected. Users can filter to native-only with `samtools view -r native`.

## Output Files

```
output_dir/
├── site_results.tsv          # Transcript-level (12-column TSV, unchanged)
├── read_results.bam          # Read-level modification probabilities
├── read_results.bam.bai      # BAM index for region queries
└── pipeline_results.pkl      # Raw DTW matrices (unchanged)
```

## Python API

### New module: `baleen/eventalign/_read_bam.py`

**Writing (internal — not exported):**

```python
def write_read_bam(
    hierarchical_results: dict[str, ContigModificationResult],
    contig_results: dict[str, ContigResult],
    ref_fasta: PathLike,
    output_path: PathLike,
) -> Path:
    """Write per-read p_mod_hmm to a sorted, indexed BAM file.

    Iterates all positions, pairs read names from ContigResult with
    p_mod_hmm values from PositionStats, writes BAM records sorted
    by (contig, position), then indexes.

    Per-read records with NaN p_mod_hmm are omitted.
    If the output file already exists, it is overwritten.

    Returns path to the written BAM file.
    """
```

**Reading (public — exported from `baleen.eventalign`):**

```python
def load_read_results(
    bam_path: PathLike,
    contig: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
    """Load read-level results, optionally filtered by region.

    Returns DataFrame with columns:
        contig, position, kmer, read_name, is_native, p_mod_hmm

    Warning: calling without region filters on human-scale data may
    produce a DataFrame with 100M+ rows. For large files, use region
    filters or iterate with load_read_results_iter().
    """

def load_read_results_iter(
    bam_path: PathLike,
    contig: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate read-level results as dicts, one per BAM record.

    Memory-efficient alternative to load_read_results() for large files.
    Each dict has keys: contig, position, kmer, read_name, is_native, p_mod_hmm.
    """
```

### Export from `baleen/eventalign/__init__.py`

Add `load_read_results` and `load_read_results_iter` to public exports. `write_read_bam` stays internal.

## Pipeline Integration

### Data flow

```
run_pipeline()  →  pipeline_results.pkl
       ↓
compute_sequential_modification_probabilities()  (per contig, sequential)
       ↓
aggregate_all()          (sequential)
       ↓
write_site_tsv()         →  site_results.tsv
       ↓
write_read_bam()         →  read_results.bam + .bai
```

`aggregate_all()` and `write_read_bam()` run **sequentially** (not in parallel). Both are fast relative to DTW and HMM computation, so parallelizing them is not worth the complexity.

### CLI changes

- `baleen run`: Writes `read_results.bam` by default. Add `--no-read-bam` flag to skip.
- `baleen aggregate`: Also writes `read_results.bam` (since re-running HMM with different params changes `p_mod_hmm`).

### Sorting

Records are written pre-sorted by `(contig, position)` since the pipeline already iterates contigs and positions in order. After writing all records to an unsorted BAM, `pysam.sort()` produces the final sorted BAM and `pysam.index()` creates the `.bai` index. The temporary unsorted BAM is deleted.

### NaN handling

Per-read `p_mod_hmm` values that are NaN (HMM skipped due to insufficient trajectory length) cause that individual read's record to be omitted. Other reads at the same position with valid values are still written.

### Reference fallback

If `ref_fasta` is available, BAM header `@SQ` lines come from its `.fai` index (auto-created by pysam if missing). If the FASTA is unavailable (e.g., re-running aggregation on a different machine), contig names and lengths are taken from the saved metadata.

### Error handling

- **Output directory missing**: Created automatically (`mkdir -p` equivalent).
- **Output file exists**: Overwritten without warning (same behavior as `write_site_tsv`).
- **Contig mismatch**: Only contigs present in both the results and the reference are written. Extra contigs in the reference are included in the header but have no records. Extra contigs in the results (not in reference) use length 0 in the header.
- **Index failure**: If `pysam.index()` fails (e.g., records not properly sorted), raise `RuntimeError` with a descriptive message.

## Usage Examples

```python
from baleen.eventalign import load_read_results

# Region query: all reads at ecoli23S positions 700-800
df = load_read_results("output/read_results.bam", "ecoli23S", 700, 800)

# Full scan (use with caution on large files)
df = load_read_results("output/read_results.bam")

# Memory-efficient iteration for large files
from baleen.eventalign import load_read_results_iter
for record in load_read_results_iter("output/read_results.bam", "chr1", 0, 10000):
    if record["is_native"] and record["p_mod_hmm"] > 0.8:
        print(f"{record['read_name']} at {record['position']}: {record['p_mod_hmm']:.3f}")
```

```bash
# Command-line region query
samtools view output/read_results.bam ecoli23S:700-800

# Filter native reads only
samtools view -r native output/read_results.bam

# Count reads per contig
samtools idxstats output/read_results.bam
```

## Estimated Sizes

| Scale | Positions | Reads/pos | Uncompressed | BAM compressed |
|-------|-----------|-----------|-------------|----------------|
| E. coli rRNA | 10K | 50 | ~50 MB | ~5 MB |
| Bacterial transcriptome | 100K | 50 | ~500 MB | ~50 MB |
| Human transcriptome | 1M+ | 100 | ~10 GB | ~2-5 GB |
