#!/usr/bin/env bash
#SBATCH --job-name=validate-wavcaps
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/home/%u/logs/validate_wavcaps_%j.log
#SBATCH --error=/home/%u/logs/validate_wavcaps_%j.err

# ───────────────────────────────────────────────────────────────────
# Load + validate the 32 kHz audio for every clip in the completed
# AudioSkillsXL wavcaps split (reads ~57 GB from AudioSet on GCS). Runs on a
# Slurm cpu node so it never stresses the 14 GB dev VM. Submit:
#   ssh slurm-login 'mkdir -p ~/logs && cd ~/esp-data-dev && sbatch jobs/validate_wavcaps_audio.sh'
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$HOME/esp-data-dev"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo)}"

echo "=== validating wavcaps 32kHz audio at $(date) (cpus=${SLURM_CPUS_PER_TASK:-?}) ==="
uv run --script scripts/validate_wavcaps_audio.py \
    --rate-col 32khz_path \
    --expect-sr 32000 \
    --workers "${SLURM_CPUS_PER_TASK:-48}" \
    --out "$HOME/wavcaps_audio_validation_failures.csv"
echo "=== finished at $(date) ==="
