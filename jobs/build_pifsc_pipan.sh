#!/usr/bin/env bash
#SBATCH --job-name=build-pifsc-pipan
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# Build the standalone PIFSC PIPAN manifest CSVs.
# Metadata-only: parses XWAV headers + joins GBIF, no audio re-processing.

set -euo pipefail

REPO_ROOT=$(realpath ../..)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
uv sync --reinstall-package esp-data --reinstall-package avex

OUT_DIR="${OUT_DIR:-${HOME}/pifsc_pipan_staging}"
WORKERS="${WORKERS:-8}"
mkdir -p "${OUT_DIR}"

srun -n 1 uv run python scripts/data_preprocessing_scripts/pifsc_pipan/build_pifsc_pipan.py \
    --out-dir "${OUT_DIR}" \
    --workers "${WORKERS}" \
    --upload

echo "Done. Staging dir: ${OUT_DIR}"
echo "Uploaded to: gs://esp-data-ingestion/pifsc-pipan/v0.1.0/"
