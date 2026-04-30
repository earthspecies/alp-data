#!/usr/bin/env bash
#SBATCH --job-name=s2-var-backend
#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --time=48:00:00
#SBATCH --output="/home/%u/logs/%A.log"

set -euo pipefail

WORKDIR="."
MANIFEST="${WORKDIR}/esp-research/projects/NatureLM-audio-v1.5/configs/datasets/manifest_train_stage2_variations.yml"
OUTPUT_DIR="${WORKDIR}/data/exports/stage2_variations_backend"

echo "=== Stage2 variations full backend export ==="
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Manifest: ${MANIFEST}"
echo "Output dir: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

uv run python -u scripts/export_chain_backend.py \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --sample-rate 16000 \
  --preserve-backend-columns \
  --save-entries

echo "Finished: $(date)"
echo "Exported files:"
ls -lh "${OUTPUT_DIR}/full_chain.jsonl" || true
ls -lh "${OUTPUT_DIR}/entries" | head || true
