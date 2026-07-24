#!/usr/bin/env bash
#SBATCH --job-name=build-nabirds
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# ---------------------------------------------------------------------------
# Build + stage the NABirds species-classification image dataset into
# gs://esp-data-ingestion/nabirds/v0.1.0/.
#
# NABirds is GATED (Cornell usage agreement), so it cannot be curl'd
# unattended. Download nabirds.tar.gz once from
#   https://dl.allaboutbirds.org/nabirds
# (after agreeing to the terms) and pre-stage it at:
#   /scratch/$USER/nabirds/nabirds.tar.gz
# This job is skip-if-present and errors clearly if the tar is absent.
#
# Steps (all on Slurm scratch — never the dev VM):
#   1. Untar the pre-staged tarball.
#   2. Roll categories up to species, GBIF-link, build manifests (build_nabirds.py).
#   3. Upload images + manifests to GCS via gsutil -m rsync / cp.
#
# Resumable: untar skips if present; rsync only copies missing objects.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
cd "${REPO_ROOT}"
uv sync --reinstall-package esp-data

GCS_ROOT="gs://esp-data-ingestion/nabirds/v0.1.0"
SCRATCH="/scratch/${USER:-$LOGNAME}/nabirds"
SRC="${SCRATCH}/nabirds"         # untarred root (holds images/ + *.txt)
OUT="${SCRATCH}/staging"         # manifests
TAR="${TAR:-${SCRATCH}/nabirds.tar.gz}"  # pre-staged gated tarball
GBIF_CACHE="${SCRATCH}/gbif_animals.tsv"
mkdir -p "${SCRATCH}" "${OUT}"

echo "=== 1. locate + untar pre-staged tarball ==="
if [ ! -f "${SRC}/images.txt" ]; then
    if [ ! -f "${TAR}" ]; then
        echo "ERROR: NABirds is gated and no pre-staged tarball was found at ${TAR}."
        echo "       Download it from https://dl.allaboutbirds.org/nabirds (agree to the"
        echo "       usage terms) and place it there, or set TAR=<path> and resubmit."
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] extracting ${TAR} ..."
    tar -xzf "${TAR}" -C "${SCRATCH}"
    # tarball may extract into nabirds/ or flat; normalise so ${SRC}/images.txt exists
    if [ ! -f "${SRC}/images.txt" ] && [ -f "${SCRATCH}/images.txt" ]; then
        mkdir -p "${SRC}"
        mv "${SCRATCH}"/{images.txt,image_class_labels.txt,classes.txt,hierarchy.txt,train_test_split.txt,images} "${SRC}/" 2>/dev/null || true
    fi
else
    echo "[$(date +%H:%M:%S)] already extracted at ${SRC}"
fi

echo "=== 2. build manifests (roll-up + GBIF-link) ==="
srun -n 1 uv run python scripts/data_preprocessing_scripts/nabirds/build_nabirds.py \
    --src "${SRC}" \
    --out "${OUT}" \
    --gcs-root "${GCS_ROOT}" \
    --gbif-cache "${GBIF_CACHE}"

echo "=== 3. upload images + manifests to GCS ==="
echo "[$(date +%H:%M:%S)] images ..."
gsutil -m -q rsync -r "${SRC}/images" "${GCS_ROOT}/images"
echo "[$(date +%H:%M:%S)] manifests ..."
gsutil -m -q cp "${OUT}"/*.csv "${GCS_ROOT}/"

echo "=== 4. counts ==="
echo "images:  $(gsutil ls "${GCS_ROOT}/images/**.jpg" 2>/dev/null | wc -l)"
echo "[$(date +%H:%M:%S)] DONE -> ${GCS_ROOT}/"
