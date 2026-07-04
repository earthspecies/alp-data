#!/usr/bin/env bash
#SBATCH --job-name=build-mosquito-sound
#SBATCH --partition=t4
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# Build MosquitoSound: download .npy from HF, write 279,566 clips × 3 SRs
# as FLAC, manifest CSVs, upload to GCS. ~30 min build + multi-hour upload.

set -euo pipefail

REPO_ROOT=$(realpath ../..)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
uv sync --reinstall-package esp-data --reinstall-package avex

# Scratch staging (~10 GB peak). NFS would work too but local /scratch is faster.
WORK_DIR="${WORK_DIR:-/scratch/${USER:-$LOGNAME}/monster_monash_staging_${SLURM_JOB_ID:-manual}}"
mkdir -p "${WORK_DIR}"

srun -n 1 uv run python scripts/data_preprocessing_scripts/monster_monash/build_wingbeats.py \
    --dataset MosquitoSound \
    --work-dir "${WORK_DIR}" \
    --workers 8 \
    --upload \
    --clean-audio-after-upload

echo "Done. Staging dir cleaned (audio removed post-upload)."
