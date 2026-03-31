#!/usr/bin/env bash
# Run baleen benchmark across all stoichiometry levels (0.0–1.0) via Docker.
#
# Usage:
#   # CPU (default):
#   bash testdata/run_benchmark.sh
#
#   # GPU:
#   BALEEN_IMAGE=btrspg/py-baleen:dev bash testdata/run_benchmark.sh
#
# Environment variables:
#   BALEEN_IMAGE        Docker image (default: btrspg/py-baleen-cpu:dev)
#   BENCHMARK_THREADS   Pipeline threads (default: 4)
#   BALEEN_EXTRA_ARGS   Extra docker args, e.g. "--gpus all" for GPU
#   BALEEN_RUN_ARGS     Extra baleen args, e.g. "" to remove --no-cuda
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STOICH_LEVELS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
THREADS="${BENCHMARK_THREADS:-4}"

# Docker settings
IMAGE="${BALEEN_IMAGE:-btrspg/py-baleen-cpu:dev}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
EXTRA_DOCKER_ARGS="${BALEEN_EXTRA_ARGS:-}"
# Default: --no-cuda for CPU image; set BALEEN_RUN_ARGS="" to auto-detect
RUN_ARGS="${BALEEN_RUN_ARGS:---no-cuda}"

run_baleen() {
    docker run --rm \
        --user "$HOST_UID:$HOST_GID" \
        -v "$SCRIPT_DIR:/data" \
        $EXTRA_DOCKER_ARGS \
        "$IMAGE" \
        run "$@"
}

TOTAL=${#STOICH_LEVELS[@]}
DONE=0

for stoich in "${STOICH_LEVELS[@]}"; do
    NATIVE="/data/$stoich/data/native_1"
    IVT="/data/$stoich/data/control_1"

    # Verify inputs exist on host
    if [[ ! -f "$SCRIPT_DIR/$stoich/data/native_1/native_1.bam" ]]; then
        echo "SKIP $stoich: native BAM not found"
        continue
    fi

    COMMON_ARGS=(
        --native-bam "$NATIVE/native_1.bam"
        --native-fastq "$NATIVE/fastq/pass.fq.gz"
        --native-blow5 "$NATIVE/blow5/nanopore.blow5"
        --ivt-bam "$IVT/control_1.bam"
        --ivt-fastq "$IVT/fastq/pass.fq.gz"
        --ivt-blow5 "$IVT/blow5/nanopore.blow5"
        --ref /data/ref.fa
        --threads "$THREADS"
        --no-read-bam
        $RUN_ARGS
    )

    # --- New scoring (contig-level global alternative) ---
    if [[ -f "$SCRIPT_DIR/$stoich/output/site_results.tsv" ]]; then
        echo "SKIP $stoich new: output already exists"
    else
        echo "=== [$((DONE * 2 + 1))/$((TOTAL * 2))] Running $stoich (new scoring) ==="
        mkdir -p "$SCRIPT_DIR/$stoich/output"
        run_baleen "${COMMON_ARGS[@]}" -o "/data/$stoich/output"
    fi

    # --- Legacy scoring (per-position EM) ---
    if [[ -f "$SCRIPT_DIR/$stoich/output_legacy/site_results.tsv" ]]; then
        echo "SKIP $stoich legacy: output already exists"
    else
        echo "=== [$((DONE * 2 + 2))/$((TOTAL * 2))] Running $stoich (legacy scoring) ==="
        mkdir -p "$SCRIPT_DIR/$stoich/output_legacy"
        run_baleen "${COMMON_ARGS[@]}" -o "/data/$stoich/output_legacy" --legacy-scoring
    fi

    DONE=$((DONE + 1))
done

echo ""
echo "=== Benchmark complete ($DONE stoichiometry levels) ==="
echo "Run: python testdata/evaluate_benchmark.py"
