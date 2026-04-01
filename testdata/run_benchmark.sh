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
set -euo pipefail

# Resolve testdata directory
DATA_DIR="${1:?Usage: $0 /path/to/testdata}"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

STOICH_LEVELS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
THRESHOLDS=(0.9 0.95 0.99 0.999)
SCORING_MODES=(new legacy)
THREADS="${BENCHMARK_THREADS:-4}"
RUN_ARGS="${BALEEN_RUN_ARGS:-}"

N_STOICH=${#STOICH_LEVELS[@]}
N_THRESH=${#THRESHOLDS[@]}
N_MODES=${#SCORING_MODES[@]}
TOTAL=$((N_STOICH * N_THRESH * N_MODES))
STEP=0

for stoich in "${STOICH_LEVELS[@]}"; do
    NATIVE="$DATA_DIR/$stoich/data/native_1"
    IVT="$DATA_DIR/$stoich/data/control_1"

    # Verify inputs exist
    if [[ ! -f "$NATIVE/native_1.bam" ]]; then
        echo "SKIP $stoich: native BAM not found at $NATIVE/native_1.bam"
        STEP=$((STEP + N_THRESH * N_MODES))
        continue
    fi

    for thresh in "${THRESHOLDS[@]}"; do
        for mode in "${SCORING_MODES[@]}"; do
            STEP=$((STEP + 1))

            if [[ "$mode" == "legacy" ]]; then
                OUT_DIR="$DATA_DIR/$stoich/output_legacy_t${thresh}"
                EXTRA_ARGS=(--legacy-scoring)
            else
                OUT_DIR="$DATA_DIR/$stoich/output_t${thresh}"
                EXTRA_ARGS=()
            fi

            if [[ -f "$OUT_DIR/site_results.tsv" ]]; then
                echo "SKIP [$STEP/$TOTAL] $stoich $mode t=$thresh: output already exists"
                continue
            fi

            echo "=== [$STEP/$TOTAL] Running $stoich ($mode scoring, threshold=$thresh) ==="
            mkdir -p "$OUT_DIR"
            baleen run \
                --native-bam "$NATIVE/native_1.bam" \
                --native-fastq "$NATIVE/fastq/pass.fq.gz" \
                --native-blow5 "$NATIVE/blow5/nanopore.blow5" \
                --ivt-bam "$IVT/control_1.bam" \
                --ivt-fastq "$IVT/fastq/pass.fq.gz" \
                --ivt-blow5 "$IVT/blow5/nanopore.blow5" \
                --ref "$DATA_DIR/ref.fa" \
                --threads "$THREADS" \
                --keep-intermediate \
                --no-read-bam \
                --mod-threshold "$thresh" \
                "${EXTRA_ARGS[@]}" \
                $RUN_ARGS \
                -o "$OUT_DIR"
        done
    done
done

echo ""
echo "=== Benchmark complete ($TOTAL runs across $N_STOICH stoichiometry levels x $N_THRESH thresholds x $N_MODES scoring modes) ==="

# Run evaluation
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo ""
echo "=== Running evaluation ==="
python "$SCRIPT_DIR/evaluate_benchmark.py" "$DATA_DIR"
