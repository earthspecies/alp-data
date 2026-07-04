#!/usr/bin/env bash
#SBATCH --job-name=test-obs-org-load
#SBATCH --partition=t4
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=2:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# Iterate the full ObservationOrg `all` split at 32 kHz; report load
# success rate, sample-rate audit, and schema completeness.

set -euo pipefail

REPO_ROOT=$(realpath ../..)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
uv sync --reinstall-package esp-data --reinstall-package avex

OUT_DIR="${OUT_DIR:-${HOME}/observation_org_load_test_$(date +%Y%m%d_%H%M%S)}"
WORKERS="${WORKERS:-8}"
MAX_ROWS="${MAX_ROWS:--1}"

mkdir -p "${OUT_DIR}"

srun -n 1 uv run python scripts/test_observation_org_load.py \
    --out-dir "${OUT_DIR}" \
    --workers "${WORKERS}" \
    --max-rows "${MAX_ROWS}"

echo "Done. Results: ${OUT_DIR}/"
