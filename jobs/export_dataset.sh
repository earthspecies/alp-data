#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --nodelist=slurm-cpu-48vcpu-384gb-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/logs/%x_%j.log
#SBATCH --job-name=export-dataset
#SBATCH --mail-type=FAIL

# Freeze a configured dataset with alp_data.export and verify it against the
# live dataset. Every knob is an environment variable; see
# scripts/dataset_exports/README.md.
#
#   CONFIG=scripts/dataset_exports/configs/beans_validation_16k.yaml \
#   OUT=gs://esp-ci-cd-tests/esp-data-tests/exports/beans/validation-16k \
#   WORKERS=32 sbatch jobs/export_dataset.sh

set -euo pipefail

ALP_DATA_DIR=${ALP_DATA_DIR:-$HOME/alp-data}
CONFIG=${CONFIG:-scripts/dataset_exports/configs/beans_validation_16k.yaml}
OUT=${OUT:-gs://esp-ci-cd-tests/esp-data-tests/exports/beans/validation-16k}
FORMAT=${FORMAT:-pack}                       # pack | hf
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-1000}
AUDIO_FORMAT=${AUDIO_FORMAT:-flac}           # flac | wav
ON_ERROR=${ON_ERROR:-raise}                  # raise | skip
VERIFY=${VERIFY:-200}                        # samples to compare with the live dataset
VERIFY_WORKERS=${VERIFY_WORKERS:-4}          # spawned reader processes for the check
SUMMARY_DIR=${SUMMARY_DIR:-$HOME/outputs/exports}

mkdir -p "$SUMMARY_DIR" "$HOME/logs"
cd "$ALP_DATA_DIR"
uv sync

echo "=== export started $(date) on $(hostname) ==="
echo "config=$CONFIG out=$OUT format=$FORMAT workers=$WORKERS samples_per_shard=$SAMPLES_PER_SHARD"

srun uv run python scripts/dataset_exports/export_dataset.py \
    --config "$CONFIG" \
    --out "$OUT" \
    --format "$FORMAT" \
    --num-workers "$WORKERS" \
    --samples-per-shard "$SAMPLES_PER_SHARD" \
    --audio-format "$AUDIO_FORMAT" \
    --on-error "$ON_ERROR" \
    --verify "$VERIFY" \
    --verify-workers "$VERIFY_WORKERS" \
    --summary "$SUMMARY_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.json" \
    "$@"

echo "=== export finished $(date) ==="
