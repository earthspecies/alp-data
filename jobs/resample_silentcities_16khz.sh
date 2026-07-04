#!/bin/bash
#SBATCH --job-name=resample-sc-16k
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output="/home/%u/logs/resample_sc16k_%j.log"
#SBATCH --error="/home/%u/logs/resample_sc16k_%j.err"

# ───────────────────────────────────────────────────────────────────
# Resample silentcities 48kHz FLAC → 16kHz WAV (librosa kaiser_best).
#
# USAGE:
#   1. Generate the manifest (once, from login node):
#        uv run --script scripts/build_silentcities_manifest.py
#      or to resume an interrupted run:
#        uv run --script scripts/build_silentcities_manifest.py -- --skip-existing
#
#   2. Submit:
#        mkdir -p ~/logs && sbatch jobs/resample_silentcities_16khz.sh
# ───────────────────────────────────────────────────────────────────

set -euo pipefail
cd /home/david_earthspecies_org/esp-data-dev

MANIFEST="data/silentcities_v1_plus_avex_manifest.txt"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found at $MANIFEST"
    echo "Run first:  uv run --script scripts/build_silentcities_manifest.py"
    exit 1
fi

echo "Manifest: $MANIFEST ($(wc -l < "$MANIFEST") files)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"

uv run python scripts/resample_silentcities_16khz.py \
    "$MANIFEST" \
    --workers "${SLURM_CPUS_PER_TASK}"
