#!/usr/bin/env bash
#SBATCH --job-name=build-ssw60
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# ---------------------------------------------------------------------------
# Build + stage the SSW60 multimodal (audio + video + image) bird dataset
# into gs://esp-data-ingestion/ssw60/v0.1.0/.
#
# Steps (all on Slurm scratch — never the dev VM):
#   1. Download the 31 GB public S3 tarball, verify md5, untar.
#   2. GBIF-link the 60 taxa + build per-modality/unified manifests +
#      resample audio_ml to 16 kHz / 32 kHz mirrors (build_ssw60.py).
#   3. Upload media (audio + mirrors, video, images) and manifests to GCS
#      via gsutil -m rsync.
#
# Resumable: download/untar/resample all skip-if-present; rsync only
# copies missing/changed objects.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
cd "${REPO_ROOT}"
uv sync --reinstall-package esp-data

GCS_ROOT="gs://esp-data-ingestion/ssw60/v0.1.0"
SCRATCH="/scratch/${USER:-$LOGNAME}/ssw60"
SRC="${SCRATCH}/ssw60"          # untarred root
OUT="${SCRATCH}/staging"        # manifests + resampled mirrors
TAR="${SCRATCH}/ssw60.tar.gz"
TAR_URL="https://ml-inat-competition-datasets.s3.amazonaws.com/ssw60/ssw60.tar.gz"
TAR_MD5="af0a54ea1a897d130d91be8ffe0de81c"
GBIF_CACHE="${SCRATCH}/gbif_animals.tsv"
mkdir -p "${SCRATCH}" "${OUT}"

echo "=== 1. download + verify + untar ==="
if [ ! -f "${SRC}/taxa.csv" ]; then
    if [ ! -f "${TAR}" ]; then
        echo "[$(date +%H:%M:%S)] downloading tarball (~31 GB) ..."
        curl -fSL --retry 8 --retry-delay 10 -o "${TAR}" "${TAR_URL}"
    fi
    echo "[$(date +%H:%M:%S)] verifying md5 ..."
    GOT=$(md5sum "${TAR}" | awk '{print $1}')
    if [ "${GOT}" != "${TAR_MD5}" ]; then
        echo "ERROR: md5 mismatch (got ${GOT}, want ${TAR_MD5})"; exit 1
    fi
    echo "[$(date +%H:%M:%S)] extracting ..."
    tar -xzf "${TAR}" -C "${SCRATCH}"
    # tarball may extract into ssw60/ or flat; normalise so ${SRC}/taxa.csv exists
    if [ ! -f "${SRC}/taxa.csv" ] && [ -f "${SCRATCH}/taxa.csv" ]; then
        mkdir -p "${SRC}"
        mv "${SCRATCH}"/{taxa.csv,*_ml.csv,images_*.csv,audio_ml,video_ml,images_inat,images_nabirds} "${SRC}/" 2>/dev/null || true
    fi
else
    echo "[$(date +%H:%M:%S)] already extracted at ${SRC}"
fi

echo "=== 2. build manifests + resample audio ==="
srun -n 1 uv run python scripts/data_preprocessing_scripts/ssw60/build_ssw60.py \
    --src "${SRC}" \
    --out "${OUT}" \
    --workers 8 \
    --gcs-root "${GCS_ROOT}" \
    --gbif-cache "${GBIF_CACHE}"

echo "=== 3. upload media + manifests to GCS ==="
echo "[$(date +%H:%M:%S)] audio (originals) ..."
gsutil -m -q rsync -r "${SRC}/audio_ml"        "${GCS_ROOT}/audio"
echo "[$(date +%H:%M:%S)] audio 16k / 32k mirrors ..."
gsutil -m -q rsync -r "${OUT}/audio_16k"       "${GCS_ROOT}/audio_16k"
gsutil -m -q rsync -r "${OUT}/audio_32k"       "${GCS_ROOT}/audio_32k"
echo "[$(date +%H:%M:%S)] video ..."
gsutil -m -q rsync -r "${SRC}/video_ml"        "${GCS_ROOT}/video"
echo "[$(date +%H:%M:%S)] images ..."
gsutil -m -q rsync -r "${SRC}/images_inat"     "${GCS_ROOT}/images_inat"
gsutil -m -q rsync -r "${SRC}/images_nabirds"  "${GCS_ROOT}/images_nabirds"
echo "[$(date +%H:%M:%S)] manifests ..."
gsutil -m -q cp "${OUT}"/*.csv "${GCS_ROOT}/"

echo "=== 4. counts ==="
echo "audio:    $(gsutil ls "${GCS_ROOT}/audio/**.wav" 2>/dev/null | wc -l)"
echo "audio16k: $(gsutil ls "${GCS_ROOT}/audio_16k/**.wav" 2>/dev/null | wc -l)"
echo "audio32k: $(gsutil ls "${GCS_ROOT}/audio_32k/**.wav" 2>/dev/null | wc -l)"
echo "video:    $(gsutil ls "${GCS_ROOT}/video/**.mp4" 2>/dev/null | wc -l)"
echo "img_inat: $(gsutil ls "${GCS_ROOT}/images_inat/**.jpg" 2>/dev/null | wc -l)"
echo "img_nabd: $(gsutil ls "${GCS_ROOT}/images_nabirds/**.jpg" 2>/dev/null | wc -l)"
echo "[$(date +%H:%M:%S)] DONE -> ${GCS_ROOT}/"
