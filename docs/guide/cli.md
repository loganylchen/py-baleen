# CLI Reference

Baleen exposes two sub-commands:

| Command | Purpose |
|---------|---------|
| [`baleen run`](#baleen-run) | Full pipeline: read-ID intersection → krill eventalign → DTW → HMM → site aggregation. |
| [`baleen aggregate`](#baleen-aggregate) | Re-run HMM and/or site aggregation from a saved `.pkl`, skipping DTW. |

```bash
baleen --help
baleen run --help
baleen aggregate --help
```

## `baleen run`

### Required inputs

| Flag | Description |
|------|-------------|
| `--native-bam` | Native BAM (sorted + indexed). |
| `--native-fastq` | Native FASTQ (`.fq.gz`). |
| `--native-blow5` | Native BLOW5 signal. |
| `--ivt-bam` | IVT control BAM. |
| `--ivt-fastq` | IVT control FASTQ. |
| `--ivt-blow5` | IVT control BLOW5 signal. |
| `--ref` | Reference FASTA (indexed with `.fai`). |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output-dir` | `baleen_output` | Output directory. |

### Pipeline parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--padding` | `1` | Flanking positions concatenated into each DTW window. |
| `--min-depth` | `15` | Minimum depth per contig; interpretation set by `--depth-mode`. |
| `--depth-mode` | `read_count` | `read_count`: total mapped reads on the contig ≥ `--min-depth`. `mean_coverage`: per-base coverage averaged over all positions ≥ `--min-depth`. |
| `--min-mapq` | `0` | Minimum mapping quality. |
| `--threads` | `8` | Parallel workers for contig processing. |
| `--target` | — | Contig name, comma-separated list, or path to a file with one contig per line. |
| `--keep-intermediate` | off | Keep the per-contig intermediate directory after merge. |
| `--resume` | off | Reuse per-contig slices under `<output_dir>/per_contig/`. Aborts if the saved parameter fingerprint disagrees; starts fresh if missing. |
| `--no-subsample` | off | Disable per-condition, per-contig read subsampling. |
| `--subsample-n` | `300` | Max reads per condition per contig. |
| `--legacy-scoring` | off | Per-position EM calibration (less sensitive at low stoichiometry). |
| `--mod-threshold` | `0.9` | Per-read `P(mod)` above which a read is counted modified. |

### DTW options

| Flag | Default | Description |
|------|---------|-------------|
| `--cuda [DEVICES]` | auto-detect | CUDA device(s): `0`, `0,1`, `0-3`, or `all`. |
| `--no-cuda` | off | Force the CPU backend. |
| `--gpu-memory-limit BYTES` | auto-detect | GPU memory budget for concurrent DTW workers. |

### HMM options

| Flag | Default | Description |
|------|---------|-------------|
| `--hmm-params` | 3-state unsupervised | Path to a trained HMM parameters JSON. See [HMM Training Modes](hmm-training.md). |
| `--no-hmm` | off | Skip HMM smoothing; output V2 scores only. |

### eventalign options

| Flag | Default | Description |
|------|---------|-------------|
| `--pore` | `rna002` | krill pore model for eventalign. |
| `--no-rna` | off | Disable RNA mode for eventalign. |
| `--kmer-model` | — | Reserved; currently unused by the krill engine. |

### Miscellaneous

| Flag | Default | Description |
|------|---------|-------------|
| `--no-primary-only` | off | Include secondary/supplementary alignments. |
| `--keep-temp` | off | Do not clean up temporary files. |
| `--no-read-bam` | off | Skip writing `read_results.bam`. |
| `--no-read-intersection` | off | Skip the BAM ∩ FASTQ ∩ BLOW5 read-ID intersection. See [Inputs › Read-ID intersection](inputs.md#read-id-intersection). |

### Example

```bash
baleen run \
    --native-bam native.bam --native-fastq native.fq.gz --native-blow5 native.blow5 \
    --ivt-bam ivt.bam --ivt-fastq ivt.fq.gz --ivt-blow5 ivt.blow5 \
    --ref ref.fa -o results/ \
    --threads 16 --subsample-n 100
```

## `baleen aggregate`

Re-runs the HMM and/or site aggregation from a saved pipeline pickle without
recomputing DTW — useful for sweeping `--mod-threshold` or applying trained HMM
parameters.

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | required | Saved pipeline results (`.pkl`). |
| `-o`, `--output` | required | Output TSV path. |
| `--score-field` | `p_mod_hmm` | Per-read score to aggregate: `p_mod_hmm`, `p_mod_knn`, or `p_mod_raw`. |
| `--hmm-params` | — | Trained HMM JSON; re-runs the HMM before aggregation. |
| `--no-read-bam` | off | Skip writing mod-BAM output. |
| `--ref` | — | Reference FASTA (required for mod-BAM output). |
| `--native-bam` | — | Native BAM (required for mod-BAM output). |
| `--ivt-bam` | — | IVT BAM (required for mod-BAM output). |
| `--legacy-scoring` | off | Per-position EM calibration (legacy). |
| `--mod-threshold` | `0.9` | Per-read `P(mod)` modified-call threshold. |

### Example

```bash
baleen aggregate \
    -i results/pipeline_results.pkl \
    -o results/sites.tsv \
    --mod-threshold 0.8
```
