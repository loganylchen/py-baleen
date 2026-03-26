# Baleen

**Hierarchical Bayesian framework for RNA modification detection from nanopore direct RNA sequencing**

Baleen detects RNA modifications by comparing ionic current signals between native and IVT (in vitro transcribed) nanopore reads. It uses CUDA-accelerated Dynamic Time Warping (DTW) to compute signal distances and a three-stage hierarchical pipeline (Empirical-Bayes null estimation, anchored mixture EM, gap-aware HMM smoothing) to call per-read and per-site modification probabilities.

## Key Features

- **CUDA-accelerated DTW** — Batched multi-position GPU kernel processes all positions per contig in a single launch with 16 concurrent CUDA streams. Automatic CPU fallback via tslearn.
- **Three-stage hierarchical modification calling**
  - **V1**: Robust IVT null estimation with coverage-adaptive three-level shrinkage (position → local window → global)
  - **V2**: Anchored two-component mixture EM with continuous soft gating (replaces hard binary thresholds)
  - **V3**: Gap-aware Hidden Markov Model with forward-backward spatial smoothing along read trajectories
- **Standard mod-BAM output** — Per-read modification probabilities in `MM:Z` / `ML:B:C` tags, compatible with [modkit](https://github.com/nanoporetech/modkit), [modbamtools](https://github.com/regalab/modbamtools), and IGV
- **Streaming architecture** — Fuses DTW → HMM → aggregation per contig, discarding distance matrices after inference to minimize memory usage
- **Flexible HMM training** — Unsupervised (default), semi-supervised (Platt-scaling calibrator), or fully supervised (MLE transitions + KDE emissions) modes

## Installation

### From source

```bash
# With CUDA (auto-detected if nvcc is available)
pip install .

# CPU only (skip CUDA compilation)
BALEEN_NO_CUDA=1 pip install .
```

### Docker

Pre-built images are available on Docker Hub:

```bash
# CPU
docker pull loganylchen/py-baleen-cpu:latest

# GPU (requires NVIDIA Container Toolkit)
docker pull loganylchen/py-baleen-gpu:latest
```

### Prerequisites

- Python >= 3.9
- [f5c](https://github.com/hasindu2008/f5c) (>= v1.4) on `PATH` for event alignment
- CUDA toolkit (optional, for GPU-accelerated DTW)

## Quick Start

### Full pipeline

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
| `results/site_results.tsv` | Per-site modification calls with BH-adjusted p-values, mod ratios, and 95% credible intervals |
| `results/read_results.bam` | Per-read modification probabilities in standard mod-BAM format (MM/ML tags) |

### Re-aggregate with different parameters

```bash
baleen aggregate \
    -i results/pipeline_results.pkl \
    -o results/sites_rerun.tsv \
    --hmm-params trained_hmm.json
```

### Docker usage

```bash
docker run --rm -v $(pwd):/data loganylchen/py-baleen-cpu:latest run \
    --native-bam /data/native.bam \
    --native-fastq /data/native.fq.gz \
    --native-blow5 /data/native.blow5 \
    --ivt-bam /data/ivt.bam \
    --ivt-fastq /data/ivt.fq.gz \
    --ivt-blow5 /data/ivt.blow5 \
    --ref /data/ref.fa \
    -o /data/results/
```

## Pipeline Overview

```
Native reads ──┐                                              ┌── site_results.tsv
               ├── f5c eventalign ── Signal grouping ──┐      │   (per-site mod calls)
IVT reads ─────┘                                       │      │
                                                        ▼      │
Reference ──────────────────────────── Pairwise DTW ──────┐    │
                                       (CUDA / CPU)       │    │
                                                          ▼    │
                                                     ┌─────────┤
                                                     │ V1: Empirical-Bayes null
                                                     │     + hierarchical shrinkage
                                                     │         │
                                                     │ V2: Anchored mixture EM
                                                     │     + soft gating
                                                     │         │
                                                     │ V3: Gap-aware HMM
                                                     │     forward-backward
                                                     └─────────┤
                                                               │
                                              Beta-Binomial ◄──┘
                                              aggregation
                                              + BH FDR ──────────┬── site_results.tsv
                                                                 └── read_results.bam
                                                                     (mod-BAM, MM/ML tags)
```

## Input Requirements

| Input | Format | Description |
|-------|--------|-------------|
| `--native-bam` | BAM (indexed) | Native direct RNA sequencing alignments |
| `--native-fastq` | FASTQ (.gz) | Native basecalled reads |
| `--native-blow5` | BLOW5 | Native raw signal file |
| `--ivt-bam` | BAM (indexed) | IVT control alignments |
| `--ivt-fastq` | FASTQ (.gz) | IVT basecalled reads |
| `--ivt-blow5` | BLOW5 | IVT raw signal file |
| `--ref` | FASTA (indexed) | Reference transcriptome |

All BAM files must be sorted and indexed (`.bai`). The reference FASTA must be indexed (`.fai`).

## Output Format

### Site-level TSV (`site_results.tsv`)

| Column | Description |
|--------|-------------|
| `contig` | Reference contig name |
| `position` | 1-based genomic position (center of k-mer) |
| `kmer` | Reference k-mer at this position |
| `mod_ratio` | Estimated modification fraction (Beta-Binomial MAP) |
| `ci_low`, `ci_high` | 95% credible interval |
| `pvalue` | Mann-Whitney U test p-value (native vs IVT) |
| `padj` | Benjamini-Hochberg adjusted p-value |
| `effect_size` | Median p_mod difference (native - IVT) |
| `n_native`, `n_ivt` | Read coverage |

### Read-level mod-BAM (`read_results.bam`)

Standard [SAM specification](https://samtools.github.io/hts-specs/SAMtags.pdf) modification tags:

- **`MM:Z`** — Delta-encoded modified base positions (`N+?` format)
- **`ML:B:C`** — Per-position uint8 modification probabilities (0–255)
- **`RG:Z`** — Read group: `native` or `ivt`

Compatible with:
- `modkit summary read_results.bam`
- `modbamtools plot -b read_results.bam`
- IGV (View → Color By → Base Modification)

## CLI Reference

### `baleen run`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threads` | 8 | Parallel workers for contig processing |
| `--padding` | 1 | Flanking positions for signal concatenation |
| `--min-depth` | 15 | Minimum read depth per contig (both native & IVT) |
| `--min-mapq` | 0 | Minimum mapping quality filter |
| `--cuda` / `--no-cuda` | auto | Force CUDA or CPU for DTW |
| `--open-start` / `--open-end` | off | Open-boundary DTW alignment |
| `--hmm-params` | unsupervised | Path to trained HMM parameters JSON |
| `--no-hmm` | off | Skip HMM smoothing (output V2 scores only) |
| `--target` | all | Contig name, comma-separated list, or file |
| `--no-read-bam` | off | Skip mod-BAM output |
| `--keep-intermediate` | off | Save per-contig DTW results |

### `baleen aggregate`

Re-run HMM and/or aggregation on previously saved pipeline results without recomputing DTW distances.

```bash
baleen aggregate -i results/pipeline_results.pkl -o sites.tsv \
    --hmm-params trained_hmm.json \
    --native-bam native.bam --ivt-bam ivt.bam --ref ref.fa
```

## Python API

```python
from baleen import run_pipeline_streaming, load_read_results

# Run pipeline
hmm_results, sites, metadata = run_pipeline_streaming(
    native_bam="native.bam",
    native_fastq="native.fq.gz",
    native_blow5="native.blow5",
    ivt_bam="ivt.bam",
    ivt_fastq="ivt.fq.gz",
    ivt_blow5="ivt.blow5",
    ref_fasta="ref.fa",
    threads=8,
)

# Load read-level results from mod-BAM
df = load_read_results("results/read_results.bam", contig="chr1")
```

## Testing

```bash
pip install ".[test]"
pytest
```

## Citation

*Manuscript in preparation.*

## License

MIT
