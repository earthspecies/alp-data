#!/usr/bin/env bash
#SBATCH --job-name=build-idle-oo
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# ---------------------------------------------------------------------------
# Build + stage the IDLE-OO Camera Traps image benchmark into
# gs://esp-data-ingestion/idle_oo_camera_traps/v0.1.0/.
#
# Steps (all on Slurm scratch — never the dev VM):
#   1. Download the HuggingFace parquet (imageomics/IDLE-OO-Camera-Traps),
#      write embedded images out, build manifest CSVs (build_idle_oo.py).
#   2. Upload images + manifests to GCS via gsutil -m rsync / cp.
#
# Small dataset (~2,590 images); resumable rsync only copies missing objects.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
cd "${REPO_ROOT}"
uv sync --reinstall-package esp-data

GCS_ROOT="gs://esp-data-ingestion/idle_oo_camera_traps/v0.1.0"
SCRATCH="/scratch/${USER:-$LOGNAME}/idle_oo"
OUT="${SCRATCH}/staging"
export HF_HOME="${SCRATCH}/hf"
mkdir -p "${SCRATCH}" "${OUT}" "${HF_HOME}"

echo "=== 1. download + build manifests ==="
srun -n 1 uv run python scripts/data_preprocessing_scripts/idle_oo_camera_traps/build_idle_oo.py \
    --out "${OUT}" \
    --gcs-root "${GCS_ROOT}" \
    --hf-cache "${HF_HOME}"

echo "=== 2. upload images + manifests to GCS ==="
echo "[$(date +%H:%M:%S)] images ..."
gsutil -m -q rsync -d -r "${OUT}/images" "${GCS_ROOT}/images"
echo "[$(date +%H:%M:%S)] manifests ..."
gsutil -m -q cp "${OUT}"/*.csv "${GCS_ROOT}/"

echo "=== 3. counts ==="
echo "images:  $(gsutil ls "${GCS_ROOT}/images/**.jpg" 2>/dev/null | wc -l)"
echo "[$(date +%H:%M:%S)] DONE -> ${GCS_ROOT}/"
