#!/usr/bin/env bash
#SBATCH --job-name=build-newt
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# ---------------------------------------------------------------------------
# Build + stage the NeWT (Natural World Tasks) image benchmark into
# gs://esp-data-ingestion/newt/v0.1.0/.
#
# Steps (all on Slurm scratch — never the dev VM):
#   1. Download the public NeWT images (~4 GB) + labels tarballs from AWS
#      Open Data, verify md5, untar.
#   2. Build the unified + per-split manifest CSVs (build_newt.py).
#   3. Upload images + manifests to GCS via gsutil -m rsync / cp.
#
# Resumable: download/untar skip-if-present; rsync only copies missing objects.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
cd "${REPO_ROOT}"
uv sync --reinstall-package esp-data

GCS_ROOT="gs://esp-data-ingestion/newt/v0.1.0"
SCRATCH="/scratch/${USER:-$LOGNAME}/newt"
SRC="${SCRATCH}"                 # untarred root (holds newt2021_images/ + newt2021_labels.csv)
OUT="${SCRATCH}/staging"         # manifests
IMG_TAR="${SCRATCH}/newt2021_images.tar.gz"
LBL_TAR="${SCRATCH}/newt2021_labels.csv.tar.gz"
IMG_URL="https://ml-inat-competition-datasets.s3.amazonaws.com/newt/newt2021_images.tar.gz"
LBL_URL="https://ml-inat-competition-datasets.s3.amazonaws.com/newt/newt2021_labels.csv.tar.gz"
IMG_MD5="b04a56a5b1ffda87f16e6d4f81f9d38e"
LBL_MD5="4cb26d0ee085904887b1ca14dcb893e7"
mkdir -p "${SCRATCH}" "${OUT}"

verify_md5() {
    local fp="$1" want="$2" got
    got=$(md5sum "${fp}" | awk '{print $1}')
    if [ "${got}" != "${want}" ]; then
        echo "ERROR: md5 mismatch for ${fp} (got ${got}, want ${want})"; exit 1
    fi
}

echo "=== 1. download + verify + untar ==="
if [ ! -f "${SRC}/newt2021_labels.csv" ]; then
    if [ ! -f "${LBL_TAR}" ]; then
        echo "[$(date +%H:%M:%S)] downloading labels ..."
        curl -fSL --retry 8 --retry-delay 10 -o "${LBL_TAR}" "${LBL_URL}"
    fi
    verify_md5 "${LBL_TAR}" "${LBL_MD5}"
    tar -xzf "${LBL_TAR}" -C "${SCRATCH}"
fi
if [ ! -d "${SRC}/newt2021_images" ]; then
    if [ ! -f "${IMG_TAR}" ]; then
        echo "[$(date +%H:%M:%S)] downloading images (~4 GB) ..."
        curl -fSL --retry 8 --retry-delay 10 -o "${IMG_TAR}" "${IMG_URL}"
    fi
    echo "[$(date +%H:%M:%S)] verifying md5 ..."
    verify_md5 "${IMG_TAR}" "${IMG_MD5}"
    echo "[$(date +%H:%M:%S)] extracting images ..."
    tar -xzf "${IMG_TAR}" -C "${SCRATCH}"
else
    echo "[$(date +%H:%M:%S)] already extracted at ${SRC}"
fi

echo "=== 2. build manifests ==="
srun -n 1 uv run python scripts/data_preprocessing_scripts/newt/build_newt.py \
    --src "${SRC}" \
    --out "${OUT}" \
    --gcs-root "${GCS_ROOT}"

echo "=== 3. upload images + manifests to GCS ==="
echo "[$(date +%H:%M:%S)] images ..."
gsutil -m -q rsync -r "${SRC}/newt2021_images" "${GCS_ROOT}/images"
echo "[$(date +%H:%M:%S)] manifests ..."
gsutil -m -q cp "${OUT}"/*.csv "${GCS_ROOT}/"

echo "=== 4. counts ==="
echo "images:  $(gsutil ls "${GCS_ROOT}/images/**.jpg" 2>/dev/null | wc -l)"
echo "[$(date +%H:%M:%S)] DONE -> ${GCS_ROOT}/"
