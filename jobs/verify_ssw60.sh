#!/usr/bin/env bash
#SBATCH --job-name=verify-ssw60
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# Post-upload verification: GCS existence audit + multimodal decode smoke.
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
cd "${REPO_ROOT}"
# --extra video pulls PyAV so the video decode-smoke can run.
uv sync --extra video --reinstall-package esp-data

srun -n 1 uv run python scripts/data_preprocessing_scripts/ssw60/verify_ssw60.py
echo "Done."
