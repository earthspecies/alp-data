#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --nodelist=slurm-cpu-48vcpu-384gb-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/logs/%x_%j.log
#SBATCH --job-name=benchmark-packed
#SBATCH --mail-type=FAIL

# Benchmark packed datasets against the live datasets they were frozen from.
# Packed and live run back to back per configuration on the same node so
# they see the same bucket conditions. See scripts/benchmarks/benchmark_packed.py.
#
#   PACKS="gs://.../beans/validation-native gs://.../fasd13/all-32k" sbatch jobs/benchmark_packed.sh

set -euo pipefail

ALP_DATA_DIR=${ALP_DATA_DIR:-$HOME/alp-data}
EXPORTS=${EXPORTS:-gs://esp-ci-cd-tests/esp-data-tests/exports}
PACKS=${PACKS:-"$EXPORTS/beans/validation-native $EXPORTS/fasd13/all-32k"}
WORKERS=${WORKERS:-0,4,16,48}
PREFETCH=${PREFETCH:-2,8}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_BATCHES=${MAX_BATCHES:-40}
WARMUP_BATCHES=${WARMUP_BATCHES:-4}
SEQUENTIAL_SAMPLES=${SEQUENTIAL_SAMPLES:-200}
MODES=${MODES:-dataloader,sequential}
SIDES=${SIDES:-packed,live}
TAG=${TAG:-}
OUT_DIR=${OUT_DIR:-$HOME/outputs/benchmarks}
UPLOAD=${UPLOAD:-gs://esp-ci-cd-tests/esp-data-tests/benchmark_packed}

mkdir -p "$OUT_DIR" "$HOME/logs"
cd "$ALP_DATA_DIR"
uv sync --group benchmark

PACK_ARGS=()
for p in $PACKS; do PACK_ARGS+=(--pack "$p"); done

echo "=== benchmark started $(date) on $(hostname) ==="
echo "packs=$PACKS workers=$WORKERS prefetch=$PREFETCH batch_size=$BATCH_SIZE max_batches=$MAX_BATCHES"

srun uv run --group benchmark python scripts/benchmarks/benchmark_packed.py \
    "${PACK_ARGS[@]}" \
    --workers "$WORKERS" \
    --prefetch "$PREFETCH" \
    --batch-size "$BATCH_SIZE" \
    --max-batches "$MAX_BATCHES" \
    --warmup-batches "$WARMUP_BATCHES" \
    --sequential-samples "$SEQUENTIAL_SAMPLES" \
    --modes "$MODES" \
    --sides "$SIDES" \
    --tag "$TAG" \
    --out "$OUT_DIR/benchmark_packed_${SLURM_JOB_ID}.csv" \
    --upload "$UPLOAD" \
    "$@"

echo "=== benchmark finished $(date) ==="
