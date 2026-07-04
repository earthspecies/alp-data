#!/usr/bin/env bash
#SBATCH --job-name=noise-clips
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/%u/logs/noise_clips_%A_%a.log
#SBATCH --error=/home/%u/logs/noise_clips_%A_%a.err

# ───────────────────────────────────────────────────────────────────
# Extract inter-call "noise" windows (no annotated vocalisations) from the two
# bird detection datasets and upload 32 kHz clips to
# gs://foundation-model-data/audio_32k/noise/<dataset>/.
#
# Array index → dataset:
#   0  dartmouth_avian_soundscapes
#   1  pteroset
#
# Reads the 32 kHz audio + embedded selection tables from GCS (no internet
# needed); heavy download, runs on a Slurm cpu node. Submit:
#   ssh slurm-login 'mkdir -p ~/logs && cd ~/esp-data-dev && sbatch jobs/build_noise_clips.sh'
# ───────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/esp-data-dev"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo)}"
echo "GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"

DATASETS=(dartmouth_avian_soundscapes pteroset)
IDX=${SLURM_ARRAY_TASK_ID:-0}
DS=${DATASETS[$IDX]}

echo "=== noise clips for $DS at $(date) (cpus=${SLURM_CPUS_PER_TASK:-?}) ==="
uv run --script scripts/build_noise_clips.py \
    --dataset "$DS" \
    --workers "${SLURM_CPUS_PER_TASK:-48}"
echo "=== finished $DS at $(date) ==="
