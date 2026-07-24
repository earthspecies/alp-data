#!/usr/bin/env bash
#SBATCH --job-name=build-mammalnet
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# ---------------------------------------------------------------------------
# Build + stage the MammalNet behavior benchmark (behavior recognition, video)
# into gs://esp-data-ingestion/mammalnet/v0.1.0/.
#
# ⚠️ VIDEOS ARE NOT REDISTRIBUTED. MammalNet publishes only the annotations
# (annotation.tar, public) — the clips are YouTube-sourced and the old
# trimmed_videos.tar.gz S3 object now returns AccessDenied. The trimmed clips
# referenced by the manifests must therefore be PRE-STAGED at
#   ${SCRATCH}/trimmed_videos/<youtube_id[_seg]>.mp4
# (e.g. via a yt-dlp + ffmpeg-trim pass over the clip ids in the annotation).
# This job downloads the (open) annotations, builds manifests, then stages ONLY
# the referenced clips found under trimmed_videos/ and uploads them. It errors
# clearly if no clips are present.
#
# Steps (Slurm scratch only):
#   1. Download annotations from the public MammalNet S3, untar.
#   2. Build behavior manifests (build_mammalnet.py) for SPLITS (default test).
#   3. Stage the referenced pre-staged clips and upload to GCS.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
cd "${REPO_ROOT}"
uv sync --reinstall-package esp-data

GCS_ROOT="gs://esp-data-ingestion/mammalnet/v0.1.0"
SCRATCH="/scratch/${USER:-$LOGNAME}/mammalnet"
SRC="${SCRATCH}"
OUT="${SCRATCH}/staging"
SPLITS="${SPLITS:-test}"
ANN_TAR="${SCRATCH}/annotation.tar"
ANN_URL="https://mammalnet.s3.amazonaws.com/annotation.tar"
mkdir -p "${SCRATCH}" "${OUT}"

echo "=== 1. download annotations (public) ==="
if [ ! -d "${SRC}/annotation" ]; then
    [ -f "${ANN_TAR}" ] || curl -fSL --retry 8 --retry-delay 10 -o "${ANN_TAR}" "${ANN_URL}"
    tar -xf "${ANN_TAR}" -C "${SCRATCH}"
fi
if [ ! -d "${SRC}/trimmed_videos" ]; then
    echo "ERROR: no trimmed clips at ${SRC}/trimmed_videos. MammalNet does not"
    echo "       redistribute videos (they are YouTube-sourced). Pre-stage the"
    echo "       clips there (yt-dlp + ffmpeg-trim over the annotation clip ids)"
    echo "       and resubmit. Annotations were downloaded to ${SRC}/annotation."
    exit 1
fi

echo "=== 2. build manifests (splits: ${SPLITS}) ==="
srun -n 1 uv run python scripts/data_preprocessing_scripts/mammalnet/build_mammalnet.py \
    --src "${SRC}" --out "${OUT}" --gcs-root "${GCS_ROOT}" --splits ${SPLITS}

echo "=== 3. stage referenced clips + upload ==="
mkdir -p "${OUT}/video"
while read -r aid; do
    f=$(find "${SRC}/trimmed_videos" -name "${aid}.mp4" -print -quit 2>/dev/null || true)
    [ -n "${f}" ] && cp -n "${f}" "${OUT}/video/${aid}.mp4"
done < "${OUT}/upload_clip_ids.txt"
echo "[$(date +%H:%M:%S)] staged $(ls "${OUT}/video" | wc -l) clips; uploading ..."
gsutil -m -q rsync -r "${OUT}/video" "${GCS_ROOT}/video"
gsutil -m -q cp "${OUT}"/mammalnet_*.csv "${GCS_ROOT}/"

echo "=== 4. counts ==="
echo "videos: $(gsutil ls "${GCS_ROOT}/video/**.mp4" 2>/dev/null | wc -l)"
echo "[$(date +%H:%M:%S)] DONE -> ${GCS_ROOT}/"
