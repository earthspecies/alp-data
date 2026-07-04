#!/usr/bin/env bash
#SBATCH --job-name=validate-xc-strong-audio
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# Stream the XCStrong manifest + bulk-list audio_16k/32k/mp3 on GCS;
# report any xc_ids missing from each shard.

set -euo pipefail

REPO_ROOT=$(realpath ../..)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
uv sync --reinstall-package esp-data --reinstall-package avex

OUT_DIR="${OUT_DIR:-/home/${USER}/xc_strong_audio_audit}"
mkdir -p "${OUT_DIR}"

srun -n 1 uv run python scripts/data_preprocessing_scripts/xeno_canto_strong/validate_audio_present.py \
    --out-dir "${OUT_DIR}"

echo "Done. Report: ${OUT_DIR}/xc_strong_audio_audit.json + xc_strong_missing_audio.csv"
