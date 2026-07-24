#!/usr/bin/env bash
#SBATCH --job-name=build-mammalps
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# Long walltime + small footprint: the bottleneck is the ~0.9 MB/s Zenodo
# download (~26 h for 82 GiB), not CPU. /scratch is node-local so a single job
# must complete the whole download in one run (no cross-job resume).
# ---------------------------------------------------------------------------
# Build + stage the MammAlps Benchmark I audiovisual behavior benchmark into
# gs://esp-data-ingestion/mammalps/v0.1.0/.
#
# MammAlps is OPEN on Zenodo (MIT). We download mammalps_v1.zip once (a single
# sequential curl is far more reliable on the cluster than thousands of range
# requests), then build_mammalps.py selectively extracts (stdlib zipfile) only
# the metadata CSVs + per-clip source files it needs, ffmpeg-muxes video+audio
# and trims each [start_s,end_s] segment, and we upload the trimmed clips.
#
# Needs ~90 GiB scratch for the zip (selective extraction to temp is small).
# Requires ffmpeg on PATH. Resumable: download + existing trimmed clips skip.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
cd "${REPO_ROOT}"
uv sync --reinstall-package esp-data

GCS_ROOT="gs://esp-data-ingestion/mammalps/v0.1.0"
SCRATCH="/scratch/${USER:-$LOGNAME}/mammalps"
OUT="${SCRATCH}/staging"
SPLITS="${SPLITS:-test}"
ZIP="${SCRATCH}/mammalps_v1.zip"
ZIP_URL="https://zenodo.org/records/15040901/files/mammalps_v1.zip"
LABELS_JSON="${SCRATCH}/labels_mapping_b1.json"
LABELS_URL="https://raw.githubusercontent.com/eceo-epfl/MammAlps/main/evaluation/labels_mapping_b1.json"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${SCRATCH}" "${OUT}" "${TMPDIR}"

echo "=== 1. download zip (~82 GiB, resume if partial) + labels ==="
[ -f "${LABELS_JSON}" ] || curl -fSL --retry 8 --retry-delay 10 -o "${LABELS_JSON}" "${LABELS_URL}"
# Always resume (-C -): completes a partial from a prior run on this node's
# scratch, or starts fresh. Tolerate a non-zero exit (e.g. 416 when already
# complete); the central-directory validation below is the real gate.
echo "[$(date +%H:%M:%S)] downloading/resuming mammalps_v1.zip ..."
curl -fSL --retry 20 --retry-delay 15 -C - -o "${ZIP}" "${ZIP_URL}" \
    || echo "[$(date +%H:%M:%S)] curl exited $? (already complete or transient) — validating"
echo "[$(date +%H:%M:%S)] zip size: $(du -h "${ZIP}" | cut -f1)"
echo "[$(date +%H:%M:%S)] validating zip central directory ..."
uv run python -c "import zipfile,sys; print('zip OK:', len(zipfile.ZipFile(sys.argv[1]).namelist()), 'members')" "${ZIP}"

echo "=== 2. selective extract + trim (splits: ${SPLITS}) ==="
srun -n 1 uv run python scripts/data_preprocessing_scripts/mammalps/build_mammalps.py \
    --zip "${ZIP}" --labels-json "${LABELS_JSON}" \
    --out "${OUT}" --gcs-root "${GCS_ROOT}" --splits ${SPLITS}

echo "=== 3. upload trimmed clips + manifests ==="
gsutil -m -q rsync -r "${OUT}/video" "${GCS_ROOT}/video"
gsutil -m -q cp "${OUT}"/mammalps_*.csv "${GCS_ROOT}/"

echo "=== 4. counts ==="
echo "videos: $(gsutil ls "${GCS_ROOT}/video/**.mp4" 2>/dev/null | wc -l)"
echo "[$(date +%H:%M:%S)] DONE -> ${GCS_ROOT}/"
