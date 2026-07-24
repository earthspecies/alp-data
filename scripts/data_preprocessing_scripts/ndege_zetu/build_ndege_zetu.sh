#!/usr/bin/env bash
#SBATCH --job-name=ndege-build
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Build Ndege Zetu (Dryad d51c5b0c7) into
# gs://esp-data-ingestion/ndege-zetu/v0.1.0/. Uses the zip already on NFS:
# extracts (if needed), writes 16k+32k WAV mirrors (+MP3 originals) and a
# durations.csv, builds weak WABAD-shaped manifests, uploads. Runs on Slurm.
#   sbatch scripts/data_preprocessing_scripts/ndege_zetu/build_ndege_zetu.sh
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${HOME}/esp-data-dev"

GCS="gs://esp-data-ingestion/ndege-zetu/v0.1.0"
STAGE="${HOME}/esp-data-staging/ndege_zetu"
WORK="${STAGE}/work"
MIRRORS="${STAGE}/mirrors"
SCRIPT="scripts/data_preprocessing_scripts/ndege_zetu/build_ndege_zetu.py"

export CLOUDSDK_CONFIG="$(mktemp -d)"   # attached SA for gsutil (+ gcsfs)
echo "Node: $(hostname)"
echo "ok" | gsutil cp - "${GCS}/.auth_probe" && gsutil rm "${GCS}/.auth_probe" && echo "auth probe OK"

# 1. Extract audio + annotations if not already present.
if [ ! -d "${WORK}/audio" ]; then unzip -o -q "${STAGE}/ndege-zetu.zip" "audio/*" -d "${WORK}/"; fi
if [ ! -d "${WORK}/annotations" ]; then unzip -o -q "${STAGE}/ndege-zetu.zip" "annotations/*" -d "${WORK}/"; fi
echo "mp3s: $(find "${WORK}/audio" -iname '*.mp3' | wc -l)"

# 2. Resample MP3 -> 16k + 32k WAV mirrors (+copy MP3) and write durations.csv.
rm -rf "${MIRRORS}"; mkdir -p "${MIRRORS}"
uv run python "${SCRIPT}" resample \
    --audio-dir "${WORK}/audio" --out-root "${MIRRORS}" --workers "${SLURM_CPUS_PER_TASK:-16}"

# 3. Build weak manifests (needs durations.csv from step 2).
uv run python "${SCRIPT}" manifests \
    --anno-dir "${WORK}/annotations" --durations-csv "${MIRRORS}/durations.csv" \
    --out-dir "${MIRRORS}/manifests"

# 4. Upload audio mirrors + manifests.
gsutil -m -q rsync -r "${MIRRORS}/audio"     "${GCS}/audio"
gsutil -m -q rsync -r "${MIRRORS}/audio_16k" "${GCS}/audio_16k"
gsutil -m -q rsync -r "${MIRRORS}/audio_32k" "${GCS}/audio_32k"
gsutil -m -q cp "${MIRRORS}/manifests"/*.csv "${GCS}/"

echo "final listing:"; gsutil ls "${GCS}/"
echo "Done."
