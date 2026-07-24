#!/usr/bin/env bash
#SBATCH --job-name=build-ak
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# ---------------------------------------------------------------------------
# Build + stage the Animal Kingdom (AR) action benchmark into
# gs://esp-data-ingestion/animal_kingdom/v0.1.0/.
#
# Animal Kingdom is GATED (SUTD usage agreement, https://forms.office.com/r/WCtC0FRWpA).
# Download the action_recognition archive(s) once and pre-stage on scratch:
#   /scratch/$USER/animal_kingdom/  (extract to action_recognition/ with the
#   AR video clips + annotation). This job is skip-if-present and errors if
#   the extracted tree is absent.
#
# Steps: untar (if archives present) -> build manifests (build_animal_kingdom.py,
# requested SPLITS, default test) -> stage referenced clips -> upload to GCS.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export UV_PROJECT_ENVIRONMENT="/scratch/${USER:-$LOGNAME}/venvs/${SLURM_JOB_ID:-manual}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
cd "${REPO_ROOT}"
uv sync --reinstall-package esp-data

GCS_ROOT="gs://esp-data-ingestion/animal_kingdom/v0.1.0"
SCRATCH="/scratch/${USER:-$LOGNAME}/animal_kingdom"
SRC="${SRC:-${SCRATCH}/action_recognition}"
OUT="${SCRATCH}/staging"
SPLITS="${SPLITS:-test}"
mkdir -p "${SCRATCH}" "${OUT}"

echo "=== 1. locate pre-staged AR tree ==="
# Extract any pre-staged archives (video.tar.gz / annotation*) if present.
for tar in "${SCRATCH}"/*.tar.gz "${SCRATCH}"/*.tar; do
    [ -e "${tar}" ] || continue
    echo "[$(date +%H:%M:%S)] extracting ${tar} ..."
    tar -xf "${tar}" -C "${SCRATCH}"
done
if [ ! -d "${SRC}" ]; then
    echo "ERROR: Animal Kingdom is gated; no extracted AR tree at ${SRC}."
    echo "       Download from https://forms.office.com/r/WCtC0FRWpA, extract to"
    echo "       ${SRC} (with the AR clips + annotation), or set SRC=<path>, and resubmit."
    exit 1
fi

echo "=== 2. build manifests (splits: ${SPLITS}) ==="
srun -n 1 uv run python scripts/data_preprocessing_scripts/animal_kingdom/build_animal_kingdom.py \
    --src "${SRC}" --out "${OUT}" --gcs-root "${GCS_ROOT}" --splits ${SPLITS}

echo "=== 3. stage referenced clips + upload ==="
mkdir -p "${OUT}/video"
python3 - "${OUT}" "${SRC}" <<'PY'
import csv, glob, os, shutil, sys
out, src = sys.argv[1], sys.argv[2]
ids = set()
for csvf in glob.glob(os.path.join(out, "animal_kingdom_*.csv")):
    with open(csvf) as f:
        for row in csv.DictReader(f):
            ids.add(row["asset_id"])
staged = 0
for aid in ids:
    hits = glob.glob(os.path.join(src, "**", f"{aid}.mp4"), recursive=True)
    if hits:
        dst = os.path.join(out, "video", f"{aid}.mp4")
        if not os.path.exists(dst):
            shutil.copy(hits[0], dst)
        staged += 1
print(f"staged {staged}/{len(ids)} clips")
PY
gsutil -m -q rsync -r "${OUT}/video" "${GCS_ROOT}/video"
gsutil -m -q cp "${OUT}"/animal_kingdom_*.csv "${GCS_ROOT}/"

echo "=== 4. counts ==="
echo "videos: $(gsutil ls "${GCS_ROOT}/video/**.mp4" 2>/dev/null | wc -l)"
echo "[$(date +%H:%M:%S)] DONE -> ${GCS_ROOT}/"
