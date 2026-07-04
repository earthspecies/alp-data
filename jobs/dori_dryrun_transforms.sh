#!/usr/bin/env bash
#SBATCH --job-name=dori-dryrun-transforms
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# Metadata-only dry-run of the stage3.5 DORI transform chains (no audio).

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT="${REPO_ROOT}/scripts/data_preprocessing_scripts/dori_dryrun_transforms.py"
# Add the NatureLM project root to PYTHONPATH so `import data.transforms`
# resolves the project-only transforms (drop_null_or_empty_string,
# select_columns, set_columns, ...) directly from source — no project
# install (which would git-build esp-data and fail on git-less cpu nodes).
PROJECT_DIR="${REPO_ROOT}/esp-research/projects/NatureLM-audio-v1.5"
export PYTHONPATH="${PROJECT_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"

# Build the local esp-data into the venv from the esp-data-dev root (NOT the
# project dir — the project's pyproject pulls esp-data from git).
cd "${REPO_ROOT}"
export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
uv sync --reinstall-package esp-data --reinstall-package avex

srun -n 1 uv run python "${SCRIPT}"

echo "Done."
