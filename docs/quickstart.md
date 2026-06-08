# Quick Start

This page runs the full pipeline on a native/IVT pair and inspects the output.
For a conceptual walkthrough, see the [Pipeline Overview](guide/overview.md).

## 1. Prepare inputs

You need, for **each** condition (native and IVT control):

| File | Description |
|------|-------------|
| BAM (sorted + indexed) | Alignments to the reference transcriptome. |
| FASTQ (`.gz`) | Basecalled reads. |
| BLOW5 | Raw signal. |

…plus a single reference FASTA (indexed with `.fai`). See
[Inputs](guide/inputs.md) for exact requirements and indexing commands.

## 2. Run the pipeline

```bash
baleen run \
    --native-bam native.bam \
    --native-fastq native.fq.gz \
    --native-blow5 native.blow5 \
    --ivt-bam ivt.bam \
    --ivt-fastq ivt.fq.gz \
    --ivt-blow5 ivt.blow5 \
    --ref ref.fa \
    -o results/
```

This produces:

| Output | Description |
|--------|-------------|
| `results/site_results.tsv` | Per-site modification calls (mod ratio, credible interval, p-value, BH-adjusted p-value, effect size, coverage, stoichiometry). |
| `results/read_results.bam` | Per-read modification probabilities in standard mod-BAM format (`MM`/`ML` tags). |

See [Outputs](guide/outputs.md) for the full column reference and mod-BAM tag layout.

## 3. Inspect site calls

```bash
column -t -s $'\t' results/site_results.tsv | head
```

A minimal significance filter (see [Interpreting & Filtering Results](guide/filtering.md)
for why p-value alone is not enough at high coverage):

```bash
awk -F'\t' 'NR==1 || ($8 < 0.05 && $9 > 0)' results/site_results.tsv > significant_sites.tsv
#                          padj<0.05   effect_size>0
```

## 4. Inspect read-level calls

```bash
# Summarise per-read modification calls
modkit summary results/read_results.bam

# Or load into pandas via the Python API
python - <<'PY'
from baleen import load_read_results
df = load_read_results("results/read_results.bam")
print(df.head())
PY
```

## Common adjustments

```bash
# Use 16 workers and cap each contig at 100 reads per condition
baleen run ... --threads 16 --subsample-n 100

# Restrict to a few transcripts
baleen run ... --target ENST00000001,ENST00000002

# Force CPU (no GPU available)
baleen run ... --no-cuda

# Resume an interrupted run
baleen run ... -o results/ --resume
```

The full flag list is in the [CLI Reference](guide/cli.md).

## Python API equivalent

```python
from baleen import run_pipeline_streaming

output_paths, metadata = run_pipeline_streaming(
    native_bam="native.bam",
    native_fastq="native.fq.gz",
    native_blow5="native.blow5",
    ivt_bam="ivt.bam",
    ivt_fastq="ivt.fq.gz",
    ivt_blow5="ivt.blow5",
    ref_fasta="ref.fa",
    output_dir="results/",
    threads=8,
)

print(output_paths["site_tsv"])        # results/site_results.tsv
print(output_paths["read_bam"])        # results/read_results.bam
print(output_paths["n_significant"])   # number of padj < 0.05 sites
```

!!! warning "Return shape"
    `run_pipeline_streaming` returns a **2-tuple** `(output_paths, metadata)`.
    `output_paths` is a dict with keys `site_tsv`, `read_bam`,
    `per_contig_dir`, `n_total_sites`, and `n_significant`.
