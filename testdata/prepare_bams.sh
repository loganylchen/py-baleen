#!/usr/bin/env bash
# Align FASTQ reads to reference and produce sorted+indexed BAMs for all
# stoichiometry levels in testdata/.
#
# Requirements:
#   - minimap2  (apt install minimap2  / conda install minimap2)
#   - samtools  (apt install samtools  / conda install samtools)
#
# Usage:
#   cd testdata/
#   bash prepare_bams.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REF="${SCRIPT_DIR}/ref.fa"

# Check dependencies
for cmd in minimap2 samtools; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Please install it first." >&2
        exit 1
    fi
done

# Index reference if needed
if [ ! -f "${REF}.fai" ]; then
    echo "Indexing reference..."
    samtools faidx "$REF"
fi

# Process each stoichiometry level
for stoich_dir in "${SCRIPT_DIR}"/0.* "${SCRIPT_DIR}"/1.0; do
    [ -d "$stoich_dir" ] || continue
    stoich=$(basename "$stoich_dir")

    for sample_dir in "$stoich_dir"/data/*/; do
        [ -d "$sample_dir" ] || continue
        sample=$(basename "$sample_dir")

        fastq_dir="${sample_dir}fastq"
        fastq=$(find "$fastq_dir" -name "*.fq.gz" -o -name "*.fastq.gz" 2>/dev/null | head -1)

        if [ -z "$fastq" ]; then
            echo "SKIP: No FASTQ found in $fastq_dir"
            continue
        fi

        out_bam="${sample_dir}${sample}.bam"

        if [ -f "$out_bam" ] && [ -f "${out_bam}.bai" ]; then
            echo "SKIP: ${stoich}/${sample} (BAM exists)"
            continue
        fi

        echo "Aligning ${stoich}/${sample}..."

        # minimap2 -a: SAM output
        # -k 14: shorter k-mer for short nanopore RNA reads
        # --secondary=no: no secondary alignments
        minimap2 -a -k 14 --secondary=no -t 4 "$REF" "$fastq" 2>/dev/null \
            | samtools sort -@ 2 -o "$out_bam"

        samtools index "$out_bam"

        # Quick stats
        n_mapped=$(samtools view -c -F 4 "$out_bam")
        echo "  -> ${n_mapped} mapped reads -> ${out_bam}"
    done
done

echo "Done."
