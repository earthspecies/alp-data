#!/usr/bin/env bash
#SBATCH --job-name=validate-wingbeats-audio
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# Audit audio completeness for MosquitoSound + InsectSound on GCS.

set -euo pipefail

REPO_ROOT=$(realpath ../..)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
uv sync --reinstall-package esp-data --reinstall-package avex

OUT_DIR="${OUT_DIR:-/home/${USER}/wingbeats_audio_audit}"
mkdir -p "${OUT_DIR}"

srun -n 1 uv run python scripts/data_preprocessing_scripts/monster_monash/validate_wingbeats_audio.py \
    --out-dir "${OUT_DIR}"

echo "Done. Report: ${OUT_DIR}/wingbeats_audio_audit.json"
