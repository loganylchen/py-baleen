#!/usr/bin/env bash
# Run baleen benchmark across all stoichiometry levels (0.0–1.0).
#
# This script assumes you are already inside the baleen environment
# (e.g. singularity shell / singularity exec).
#
# Usage:
#   singularity exec -B /SSD --nv baleen.sif bash testdata/run_benchmark.sh /path/to/testdata
#
#   # Or if already inside the container:
#   bash run_benchmark.sh /path/to/testdata
#
# Arguments:
#   $1  Path to testdata directory containing {0.0..1.0}/data/ and ref.fa
#
# Environment variables:
#   BENCHMARK_THREADS     Pipeline threads (default: 4)
#   BALEEN_RUN_ARGS       Extra baleen args (default: empty = auto-detect CUDA)
#   BALEEN_MOD_THRESHOLD  Per-read P(mod) threshold (default: uses baleen default 0.99)
set -euo pipefail

# Resolve testdata directory
DATA_DIR="${1:?Usage: $0 /path/to/testdata}"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

STOICH_LEVELS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
THREADS="${BENCHMARK_THREADS:-4}"
RUN_ARGS="${BALEEN_RUN_ARGS:-}"
MOD_THRESHOLD="${BALEEN_MOD_THRESHOLD:-}"

TOTAL=${#STOICH_LEVELS[@]}
DONE=0

for stoich in "${STOICH_LEVELS[@]}"; do
    NATIVE="$DATA_DIR/$stoich/data/native_1"
    IVT="$DATA_DIR/$stoich/data/control_1"

    # Verify inputs exist
    if [[ ! -f "$NATIVE/native_1.bam" ]]; then
        echo "SKIP $stoich: native BAM not found at $NATIVE/native_1.bam"
        continue
    fi

    # Threshold arg (if set)
    THRESH_ARG=()
    if [[ -n "$MOD_THRESHOLD" ]]; then
        THRESH_ARG=(--mod-threshold "$MOD_THRESHOLD")
    fi

    COMMON_ARGS=(
        --native-bam "$NATIVE/native_1.bam"
        --native-fastq "$NATIVE/fastq/pass.fq.gz"
        --native-blow5 "$NATIVE/blow5/nanopore.blow5"
        --ivt-bam "$IVT/control_1.bam"
        --ivt-fastq "$IVT/fastq/pass.fq.gz"
        --ivt-blow5 "$IVT/blow5/nanopore.blow5"
        --ref "$DATA_DIR/ref.fa"
        --threads "$THREADS"
        --keep-intermediate
        --no-read-bam
        "${THRESH_ARG[@]}"
        $RUN_ARGS
    )

    # --- New scoring (contig-level global alternative) ---
    if [[ -f "$DATA_DIR/$stoich/output/site_results.tsv" ]]; then
        echo "SKIP $stoich new: output already exists"
    else
        echo "=== [$((DONE * 2 + 1))/$((TOTAL * 2))] Running $stoich (new scoring) ==="
        mkdir -p "$DATA_DIR/$stoich/output"
        baleen run "${COMMON_ARGS[@]}" -o "$DATA_DIR/$stoich/output"
    fi

    # --- Legacy scoring (per-position EM) ---
    if [[ -f "$DATA_DIR/$stoich/output_legacy/site_results.tsv" ]]; then
        echo "SKIP $stoich legacy: output already exists"
    else
        echo "=== [$((DONE * 2 + 2))/$((TOTAL * 2))] Running $stoich (legacy scoring) ==="
        mkdir -p "$DATA_DIR/$stoich/output_legacy"
        baleen run "${COMMON_ARGS[@]}" -o "$DATA_DIR/$stoich/output_legacy" --legacy-scoring
    fi

    DONE=$((DONE + 1))
done

echo ""
echo "=== Benchmark complete ($DONE stoichiometry levels) ==="
echo "Results in: $DATA_DIR/{0.0..1.0}/output/"

# Run evaluation
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo ""
echo "=== Running evaluation ==="
python "$SCRIPT_DIR/evaluate_benchmark.py" "$DATA_DIR"
