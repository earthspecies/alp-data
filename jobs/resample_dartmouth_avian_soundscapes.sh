#!/usr/bin/env bash
#SBATCH --job-name=resample-dartmouth
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/%u/logs/resample_dartmouth_%A_%a.log
#SBATCH --error=/home/%u/logs/resample_dartmouth_%A_%a.err

# ───────────────────────────────────────────────────────────────────
# Resample Dartmouth Avian Soundscapes recordings (32 kHz FLAC) to mono
# 16-bit PCM WAV, mirroring the recordings/ layout.
#
# Array index → target rate:
#   0  -> audio_16k/  (16 kHz, librosa kaiser_best)
#   1  -> audio_32k/  (32 kHz, re-encode only; source is already 32 kHz)
#
# Listing/IO happen on the compute node (reads/writes GCS only — no external
# internet needed). Submit from the login node:
#
#   ssh slurm-login 'mkdir -p ~/logs && cd ~/esp-data-dev && sbatch jobs/resample_dartmouth_avian_soundscapes.sh'
# ───────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/esp-data-dev"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo)}"
echo "GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"

ROOT="gs://esp-data-ingestion/dartmouth-avian-soundscapes/v0.1.0"
SRC="$ROOT/recordings/"

RATES=(16000 32000)
DESTS=("$ROOT/audio_16k/" "$ROOT/audio_32k/")

IDX=${SLURM_ARRAY_TASK_ID:-0}
SR=${RATES[$IDX]}
DST=${DESTS[$IDX]}

echo "=== Dartmouth resample array task $IDX ($SR Hz) at $(date) ==="
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
