#!/usr/bin/env bash
#SBATCH --job-name=resample-pteroset
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/%u/logs/resample_pteroset_%A_%a.log
#SBATCH --error=/home/%u/logs/resample_pteroset_%A_%a.err

# ───────────────────────────────────────────────────────────────────
# Resample PteroSet recordings (192 kHz WAV) to mono 16-bit PCM WAV,
# mirroring the recordings/ layout.
#
# Array index → target rate:
#   0  -> audio_16k/  (16 kHz, librosa kaiser_best)
#   1  -> audio_32k/  (32 kHz, librosa kaiser_best)
#
# Listing/IO happen on the compute node (reads/writes GCS only). Submit from
# the login node:
#
#   ssh slurm-login 'mkdir -p ~/logs && cd ~/esp-data-dev && sbatch jobs/resample_pteroset.sh'
# ───────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/esp-data-dev"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo)}"
echo "GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"

ROOT="gs://esp-data-ingestion/pteroset/v0.1.0"
SRC="$ROOT/recordings/"

RATES=(16000 32000)
DESTS=("$ROOT/audio_16k/" "$ROOT/audio_32k/")

IDX=${SLURM_ARRAY_TASK_ID:-0}
SR=${RATES[$IDX]}
DST=${DESTS[$IDX]}

echo "=== PteroSet resample array task $IDX ($SR Hz) at $(date) ==="
echo "Source: $SRC"
echo "Dest:   $DST"
echo "CPUs:   ${SLURM_CPUS_PER_TASK:-?}"

uv run --script scripts/resample_to_sr.py \
    --source-prefix "$SRC" \
    --dest-prefix "$DST" \
    --target-sr "$SR" \
    --workers "${SLURM_CPUS_PER_TASK:-48}" \
    --skip-existing

echo "=== Finished array task $IDX at $(date) ==="
