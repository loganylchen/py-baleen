# Inputs

Baleen compares two conditions — a **native** sample and an **IVT** (in vitro
transcribed, unmodified) control. For **each** condition you supply three files,
plus a single shared reference FASTA.

## Required files per condition

| File | Flag (native / IVT) | Requirements |
|------|---------------------|--------------|
| BAM | `--native-bam` / `--ivt-bam` | Aligned to the reference, **coordinate-sorted and indexed** (`.bai`). |
| FASTQ | `--native-fastq` / `--ivt-fastq` | Basecalled reads, bgzip-compressed (`.fq.gz`). |
| BLOW5 | `--native-blow5` / `--ivt-blow5` | Raw signal in BLOW5/SLOW5 format, indexed (`.blow5.idx`). |

## Shared reference

| File | Flag | Requirements |
|------|------|--------------|
| Reference FASTA | `--ref` | The transcriptome the BAM was aligned to, indexed with `samtools faidx` (`.fai`). |

## Indexing commands

```bash
# BAM: sort then index
samtools sort -o native.bam native.unsorted.bam
samtools index native.bam

# Reference FASTA
samtools faidx ref.fa

# BLOW5 signal index (slow5tools) — produces nanopore.blow5.idx
slow5tools index native.blow5
```

!!! note "No event-alignment index step"
    The krill engine reads the BLOW5 signal directly via pyslow5, so it only
    needs the `slow5tools index` above. There is no separate FASTQ read-index
    step — FASTQ read IDs are parsed straight from the FASTQ headers.

## Read-ID intersection

eventalign **silently drops** any BAM read whose UUID is absent from the
BLOW5 signal file. If your BAM contains reads that have no corresponding raw
signal (a common result of separate basecalling/alignment and signal-export
steps), those reads survive BAM parsing but vanish during event alignment.

Without correction this biases everything computed before eventalign runs:

- **depth statistics** count reads that will never produce signal,
- **`--min-depth` filtering** keeps or drops contigs against the wrong count,
- **subsampling** (`--subsample-n`) selects from a read set larger than the one
  that actually yields events.

To prevent this, Baleen computes, **independently for each condition**, the
intersection:

```
reads(BAM) ∩ reads(FASTQ) ∩ reads(BLOW5)
```

Every downstream stage — contig statistics, the `--min-depth` filter,
subsampling, and the per-contig BAM split — is gated on this intersection, so
the read set Baleen reasons about is exactly the one eventalign will align.

### How read IDs are enumerated

| Source | Method |
|--------|--------|
| BAM | Iterate alignments, collect `query_name`. |
| FASTQ | First whitespace-delimited token of each `@` header. A leftover f5c `<fastq>.index.readdb` is **ignored** — krill never creates one, and the f5c single-BLOW5 form (`*<TAB>blow5_path`) carries no read ids. |
| BLOW5 | `pyslow5.Open(path).get_read_ids()`. |

The intersection runs by default. Disable it with `--no-read-intersection` if
you have already guaranteed that all three files share an identical read set
(for example, on simulated data).

!!! warning "Empty intersection"
    If the three-way intersection for a condition is empty, Baleen emits a
    warning — this almost always means a mismatched file set (wrong BLOW5, or a
    BAM aligned from a different basecall) rather than a real absence of shared
    reads.

## Next steps

Run the pipeline with these inputs — see the [Quick Start](../quickstart.md) and
the [CLI Reference](cli.md). For what comes out the other end, see
[Outputs](outputs.md).
