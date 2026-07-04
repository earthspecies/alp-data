#!/usr/bin/env bash
#SBATCH --job-name=stage2-f0-export
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output="/home/%u/logs/stage2_f0_export_%j.log"
#SBATCH --error="/home/%u/logs/stage2_f0_export_%j.err"

set -euo pipefail

WORKDIR="/home/david_earthspecies_org/esp-data-dev"
CONFIG="$WORKDIR/esp-research/projects/NatureLM-audio-v1.5/configs/datasets/stage2_train_v1.yml"
OUTPUT_DIR="$WORKDIR/data/exports/stage2_f0"

echo "=== Stage2 F0 backend export ==="
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Config: ${CONFIG}"
echo "Output dir: ${OUTPUT_DIR}"

cd "$WORKDIR"
mkdir -p "$HOME/logs"
mkdir -p "$OUTPUT_DIR"

uv run python -u scripts/export_stage2_f0_backends.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --format both

echo "Finished: $(date)"
echo "Exported files:"
ls -lh "$OUTPUT_DIR"/f0_bioacoustic_16khz_f0_* || true
