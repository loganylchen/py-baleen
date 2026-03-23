# Instruction for AI Agents: RNA Modification Calling with Baleen

This document tells you everything you need to know to write Snakemake rules that run the **baleen** pipeline for detecting RNA modifications from Oxford Nanopore direct RNA sequencing (DRS) data.

---

## 1. What Baleen Does

Baleen detects RNA base modifications (m6A, Ψ, etc.) by comparing two samples:

- **Native RNA** — real cellular RNA that may contain modifications
- **IVT RNA** (in vitro transcribed) — modification-free synthetic RNA used as control

The pipeline works by:
1. Aligning raw nanopore signals to the reference with **f5c eventalign**
2. Computing pairwise **Dynamic Time Warping (DTW)** distances between native and IVT signals at each genomic position
3. Running a three-stage hierarchical Bayesian + HMM inference to call modification probabilities per read
4. Aggregating per-read probabilities to site-level statistics (mean, median, stoichiometry estimate, p-value, FDR)

The final output is a **TSV file of modification sites** with per-read BAM output as an optional companion.

---

## 2. Required Inputs (Per Sample Pair)

You need **six input files** — three for native, three for IVT:

| File | Description | Format |
|------|-------------|--------|
| `native.bam` | Reads aligned to reference | Sorted, indexed BAM |
| `native.fq.gz` | Raw reads | FASTQ (gzip OK) |
| `native.blow5` | Raw electrical signal | BLOW5 (or SLOW5) |
| `ivt.bam` | IVT reads aligned to reference | Sorted, indexed BAM |
| `ivt.fq.gz` | IVT raw reads | FASTQ (gzip OK) |
| `ivt.blow5` | IVT raw signal | BLOW5 (or SLOW5) |
| `ref.fa` | Reference FASTA | Indexed with `samtools faidx` |

### How to Get BLOW5 Files

Raw nanopore signal comes off the instrument as **POD5** (newer) or **FAST5** (older). Convert to BLOW5 using `slow5tools`:

```bash
# From POD5
slow5tools convert --from pod5 /path/to/pod5_dir/ -o output.blow5

# From FAST5
slow5tools f2s /path/to/fast5_dir/ -o output.blow5
```

### How to Get BAM Files

Align FASTQ reads to the reference using `minimap2` with RNA-appropriate settings:

```bash
# Direct RNA sequencing (dRNA-seq)
minimap2 -ax splice -uf -k14 --secondary=no ref.fa reads.fq.gz \
  | samtools sort -o aligned.bam
samtools index aligned.bam
```

For transcriptome reference (instead of genome), use:
```bash
minimap2 -ax map-ont --secondary=no ref.fa reads.fq.gz \
  | samtools sort -o aligned.bam
samtools index aligned.bam
```

---

## 3. Docker Images

Two images are provided — choose based on hardware:

### CPU Image (`Dockerfile.cpu`)
- Base: `python:3.11-slim`
- Includes: f5c v1.6 (CPU binary) + baleen (CPU-only, no CUDA)
- Build: `docker build -f Dockerfile.cpu -t baleen:cpu .`
- Use when: no GPU available, or small datasets

### GPU Image (`Dockerfile.gpu`)
- Base: `nvidia/cuda:12.6.3-devel-ubuntu22.04` (build) / `nvidia/cuda:12.6.3-runtime-ubuntu22.04` (runtime)
- Includes: f5c v1.6 (CUDA binary) + baleen (with CUDA DTW extension)
- Build: `docker build -f Dockerfile.gpu -t baleen:gpu .`
- Use when: GPU available; significantly faster for large datasets (thousands of positions)

The entrypoint is `python3 -m baleen` (equivalent to the `baleen` CLI command).

**Singularity/Apptainer** (for HPC without Docker):
```bash
singularity build baleen_cpu.sif docker://your-registry/baleen:cpu
singularity build baleen_gpu.sif docker://your-registry/baleen:gpu
```

---

## 4. CLI Reference

```
baleen run
  --native-bam   native.bam          # required
  --native-fastq native.fq.gz        # required
  --native-blow5 native.blow5        # required
  --ivt-bam      ivt.bam             # required
  --ivt-fastq    ivt.fq.gz           # required
  --ivt-blow5    ivt.blow5           # required
  --ref          ref.fa              # required
  -o             results/            # output directory

  # Key optional parameters
  --padding      1     # flanking positions for DTW signal window (default: 1)
  --min-depth    15    # minimum reads per contig to process (default: 15)
  --min-mapq     0     # minimum mapping quality (default: 0)
  --threads      4     # parallel workers for contig processing (default: 1)
  --hmm-params   params.json  # trained HMM parameters (default: unsupervised)
  --no-hmm              # skip HMM, output V2 kNN scores only
  --cuda                # force GPU
  --no-cuda             # force CPU
  --no-read-bam         # skip per-read BAM output
  --keep-temp           # keep intermediate eventalign TSV files
```

### Outputs (in `results/`)

| File | Description |
|------|-------------|
| `site_results.tsv` | Site-level modification calls (main output) |
| `pipeline_results.pkl` | Raw DTW results (for re-running HMM/aggregation) |
| `read_results.bam` | Per-read modification probabilities (custom tags: `MP:f`, `RG:Z`, `KM:Z`) |

### `site_results.tsv` columns

| Column | Description |
|--------|-------------|
| `contig` | Reference contig/chromosome |
| `position` | 0-based genomic position |
| `kmer` | Reference 5-mer (RNA, with U) |
| `n_native` | Number of native reads |
| `n_ivt` | Number of IVT reads |
| `mean_p_mod` | Mean HMM modification probability across native reads |
| `median_p_mod` | Median HMM modification probability |
| `stoichiometry` | Fraction of native reads with p_mod > 0.5 |
| `pvalue` | Mann-Whitney U test (native vs IVT) |
| `padj` | Benjamini-Hochberg adjusted p-value |

Significant sites: `padj < 0.05`.

---

## 5. Snakemake Rules

Below is a complete, working set of rules. Adapt paths and wildcards to your project structure.

```python
# config.yaml (sample entries)
# samples:
#   sample1:
#     native_blow5: data/sample1/native.blow5
#     ivt_blow5: data/sample1/ivt.blow5
#     native_fastq: data/sample1/native.fq.gz
#     ivt_fastq: data/sample1/ivt.fq.gz
# ref: data/ref.fa


# ── Rule 1: Index reference ─────────────────────────────────────────
rule faidx_ref:
    input:
        ref = config["ref"]
    output:
        fai = config["ref"] + ".fai"
    shell:
        "samtools faidx {input.ref}"


# ── Rule 2: Align native reads ───────────────────────────────────────
rule align_native:
    input:
        fastq = lambda wc: config["samples"][wc.sample]["native_fastq"],
        ref   = config["ref"],
        fai   = config["ref"] + ".fai",
    output:
        bam = "results/{sample}/native.bam",
        bai = "results/{sample}/native.bam.bai",
    threads: 8
    shell:
        """
        minimap2 -ax splice -uf -k14 --secondary=no -t {threads} \
            {input.ref} {input.fastq} \
          | samtools sort -@ {threads} -o {output.bam}
        samtools index {output.bam}
        """


# ── Rule 3: Align IVT reads ──────────────────────────────────────────
rule align_ivt:
    input:
        fastq = lambda wc: config["samples"][wc.sample]["ivt_fastq"],
        ref   = config["ref"],
        fai   = config["ref"] + ".fai",
    output:
        bam = "results/{sample}/ivt.bam",
        bai = "results/{sample}/ivt.bam.bai",
    threads: 8
    shell:
        """
        minimap2 -ax splice -uf -k14 --secondary=no -t {threads} \
            {input.ref} {input.fastq} \
          | samtools sort -@ {threads} -o {output.bam}
        samtools index {output.bam}
        """


# ── Rule 4: Convert raw signal to BLOW5 (if needed) ─────────────────
# Skip this rule if BLOW5 files already exist.
rule pod5_to_blow5:
    input:
        pod5_dir = "data/{sample}/{condition}_pod5/"
    output:
        blow5 = "data/{sample}/{condition}.blow5"
    shell:
        "slow5tools convert --from pod5 {input.pod5_dir} -o {output.blow5}"


# ── Rule 5: Run baleen (CPU) ─────────────────────────────────────────
rule baleen_run_cpu:
    input:
        native_bam   = "results/{sample}/native.bam",
        native_bai   = "results/{sample}/native.bam.bai",
        native_fastq = lambda wc: config["samples"][wc.sample]["native_fastq"],
        native_blow5 = lambda wc: config["samples"][wc.sample]["native_blow5"],
        ivt_bam      = "results/{sample}/ivt.bam",
        ivt_bai      = "results/{sample}/ivt.bam.bai",
        ivt_fastq    = lambda wc: config["samples"][wc.sample]["ivt_fastq"],
        ivt_blow5    = lambda wc: config["samples"][wc.sample]["ivt_blow5"],
        ref          = config["ref"],
        fai          = config["ref"] + ".fai",
    output:
        sites  = "results/{sample}/baleen/site_results.tsv",
        pkl    = "results/{sample}/baleen/pipeline_results.pkl",
        bam    = "results/{sample}/baleen/read_results.bam",
    params:
        outdir    = "results/{sample}/baleen",
        threads   = 4,
        min_depth = config.get("min_depth", 15),
        padding   = config.get("padding", 1),
    log:
        "logs/baleen/{sample}.log"
    container:
        "docker://your-registry/baleen:cpu"
    threads: 4
    shell:
        """
        baleen run \
            --native-bam   {input.native_bam} \
            --native-fastq {input.native_fastq} \
            --native-blow5 {input.native_blow5} \
            --ivt-bam      {input.ivt_bam} \
            --ivt-fastq    {input.ivt_fastq} \
            --ivt-blow5    {input.ivt_blow5} \
            --ref          {input.ref} \
            -o             {params.outdir} \
            --threads      {params.threads} \
            --min-depth    {params.min_depth} \
            --padding      {params.padding} \
        2>&1 | tee {log}
        """


# ── Rule 5b: Run baleen (GPU) ────────────────────────────────────────
# Use this instead of Rule 5 if CUDA is available.
rule baleen_run_gpu:
    input:
        native_bam   = "results/{sample}/native.bam",
        native_bai   = "results/{sample}/native.bam.bai",
        native_fastq = lambda wc: config["samples"][wc.sample]["native_fastq"],
        native_blow5 = lambda wc: config["samples"][wc.sample]["native_blow5"],
        ivt_bam      = "results/{sample}/ivt.bam",
        ivt_bai      = "results/{sample}/ivt.bam.bai",
        ivt_fastq    = lambda wc: config["samples"][wc.sample]["ivt_fastq"],
        ivt_blow5    = lambda wc: config["samples"][wc.sample]["ivt_blow5"],
        ref          = config["ref"],
        fai          = config["ref"] + ".fai",
    output:
        sites = "results/{sample}/baleen/site_results.tsv",
        pkl   = "results/{sample}/baleen/pipeline_results.pkl",
        bam   = "results/{sample}/baleen/read_results.bam",
    params:
        outdir    = "results/{sample}/baleen",
        threads   = 4,
        min_depth = config.get("min_depth", 15),
        padding   = config.get("padding", 1),
    log:
        "logs/baleen/{sample}.log"
    container:
        "docker://your-registry/baleen:gpu"
    resources:
        nvidia_gpu = 1
    threads: 4
    shell:
        """
        baleen run \
            --native-bam   {input.native_bam} \
            --native-fastq {input.native_fastq} \
            --native-blow5 {input.native_blow5} \
            --ivt-bam      {input.ivt_bam} \
            --ivt-fastq    {input.ivt_fastq} \
            --ivt-blow5    {input.ivt_blow5} \
            --ref          {input.ref} \
            -o             {params.outdir} \
            --threads      {params.threads} \
            --min-depth    {params.min_depth} \
            --padding      {params.padding} \
            --cuda \
        2>&1 | tee {log}
        """


# ── Rule 6: Re-aggregate with different HMM params (optional) ────────
# Useful when comparing HMM training modes without re-running DTW.
rule baleen_aggregate:
    input:
        pkl        = "results/{sample}/baleen/pipeline_results.pkl",
        hmm_params = "models/{hmm_model}.json",
        ref        = config["ref"],
    output:
        sites = "results/{sample}/baleen_{hmm_model}/site_results.tsv",
        bam   = "results/{sample}/baleen_{hmm_model}/read_results.bam",
    log:
        "logs/baleen_aggregate/{sample}_{hmm_model}.log"
    container:
        "docker://your-registry/baleen:cpu"
    shell:
        """
        baleen aggregate \
            -i {input.pkl} \
            -o {output.sites} \
            --hmm-params {input.hmm_params} \
            --ref {input.ref} \
        2>&1 | tee {log}
        """


# ── Rule 7: All targets ──────────────────────────────────────────────
rule all:
    input:
        expand("results/{sample}/baleen/site_results.tsv",
               sample=config["samples"])
```

### Important Snakemake Notes

- **BAI files must be declared as inputs** — baleen validates BAMs at startup; missing index causes failure.
- **Reference FAI must also be declared** — `_build_header()` in baleen calls `pysam.FastaFile`, which requires an index.
- **BLOW5 index files** (`.blow5.idx`) are created automatically by baleen internally; you do not need to declare them.
- **FASTQ index files** (`.fq.gz.index.readdb`) are also created internally by baleen via f5c; no separate rule needed.
- `--threads` controls parallel contig processing (Python `ProcessPoolExecutor`), not f5c threads. For most datasets, 4–8 is sufficient.
- If running in a containerized environment without internet access, build and push Docker images to a private registry first.

---

## 6. Benchmarking

### Recommended Benchmark Datasets

| Dataset | Modification | Source |
|---------|-------------|--------|
| HEK293T dRNA | m6A | PRJNA523503 (Garalde et al.) — native vs IVT |
| HEK293T dRNA | m6A | PRJNA694660 (Liu et al.) — with m6A-seq ground truth |
| Yeast dRNA | Ψ, m6A | PRJNA694671 — synthetic spike-in dataset |
| Synthetic spike-in | m6A | Custom oligos with known stoichiometry |
| METTL3 KO vs WT | m6A | Any METTL3 knockout cell line dRNA dataset |

For a clean benchmark: use IVT RNA as the universal negative control and known high-confidence m6A sites (from miCLIP, m6A-seq, or DRACH motif intersection with METTL3 ChIP-seq) as the positive set.

### Competing Tools to Compare Against

| Tool | Method | Modification | GitHub/Repo |
|------|--------|-------------|-------------|
| **m6Anet** | Multiple instance learning on per-read features | m6A | github.com/comprna/m6anet |
| **Nanocompore** | GMM on current level differences (native vs IVT) | Any | github.com/tleonardi/nanocompore |
| **xPore** | Gaussian mixture model (native vs IVT) | Any | github.com/GoekeLab/xpore |
| **CHEUI** | CNN on raw signal (native vs IVT) | m6A, Ψ | github.com/novoalab/CHEUI |
| **EpiNano** | SVM on per-position base-calling error features | m6A | github.com/enovoa/EpiNano |
| **ELIGOS2** | Logistic regression on error features | Any | github.com/CMB-BNU/Eligos2 |
| **Dorado (modBAM)** | Neural network in basecaller | m6A, 5mC | github.com/nanoporetech/dorado |

### Evaluation Metrics

```python
from sklearn.metrics import roc_auc_score, average_precision_score

# Load baleen output
import pandas as pd
sites = pd.read_csv("results/sample/baleen/site_results.tsv", sep="\t")

# Merge with ground truth (True/False labels per position)
merged = sites.merge(ground_truth, on=["contig", "position"])

# Metrics
auroc = roc_auc_score(merged["is_modified"], merged["mean_p_mod"])
auprc = average_precision_score(merged["is_modified"], merged["mean_p_mod"])
```

Standard metrics:
- **AUROC** — site-level, using `mean_p_mod` or `stoichiometry` as score
- **AUPRC** — preferred when positives are rare (typical for modification calling)
- **Precision/Recall at FDR 5%** — using `padj < 0.05` as threshold
- **Read-level AUROC** — using per-read `p_mod_hmm` from `read_results.bam`

For read-level evaluation, load the BAM with `baleen.load_read_results()`:

```python
from baleen import load_read_results
df = load_read_results("results/sample/baleen/read_results.bam")
# columns: contig, position, kmer, read_name, is_native, p_mod_hmm
native_df = df[df["is_native"]]
```

### Benchmarking Config Example

```yaml
# benchmark_config.yaml
ref: data/hg38_transcriptome.fa

samples:
  hek293_rep1:
    native_blow5: data/hek293_rep1/native.blow5
    native_fastq: data/hek293_rep1/native.fq.gz
    ivt_blow5:    data/hek293_rep1/ivt.blow5
    ivt_fastq:    data/hek293_rep1/ivt.fq.gz

# Parameters to sweep for benchmarking
min_depth: 20      # increase for higher precision
padding: 1         # 1 = default; try 0 or 2 to compare
```

---

## 7. Key Technical Details

These are non-obvious things that affect correctness:

**RNA directionality:** For RNA nanopore sequencing, the strand threads through the pore 3'→5'. This means higher genomic positions are encountered **earlier** in time. Within a single genomic position, f5c eventalign stores events in `event_index` ascending order, which is the **reverse** of temporal order (`start_idx` ascending = temporal order). Baleen handles this correctly by sorting by `start_idx` within positions and iterating neighbor positions in descending genomic order for signal concatenation.

**`--signal-index` is required:** Baleen always calls f5c with `--signal-index`. This flag is mandatory for correct temporal ordering. If you call f5c independently (outside of baleen), you must also include `--signal-index --samples --scale-events --print-read-names`.

**Reference must be indexed:** Run `samtools faidx ref.fa` before running baleen. An unindexed reference causes a `RuntimeError` with a helpful message.

**BLOW5 vs SLOW5:** Both formats are supported. BLOW5 is the binary (faster), SLOW5 is text. Prefer BLOW5.

**Contig parallelism:** `--threads N` processes N contigs in parallel. Contigs below `--min-depth` are silently skipped. If your reference has many short contigs, increase `--min-depth` to avoid spending time on low-coverage regions.

**Re-running HMM only:** The `baleen aggregate` sub-command loads the saved `.pkl` and re-runs only the HMM + aggregation steps. Use this to try different HMM parameter files without re-computing DTW distances (which is the expensive step).
