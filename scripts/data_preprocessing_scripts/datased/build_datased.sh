#!/usr/bin/env bash
#SBATCH --job-name=datased-build
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Build DataSED (Zenodo 15346092) audio + manifests into
# gs://esp-data-ingestion/datased/v0.1.0/. Extracts the WAVs from the
# already-downloaded zip on NFS, uploads originals, writes 16k + 32k
# pre-resampled mirrors, and (re)builds + uploads the manifests.
# Runs on Slurm (not the dev VM) to avoid OOM during audio processing.
#   sbatch scripts/data_preprocessing_scripts/datased/build_datased.sh
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${HOME}/esp-data-dev"

GCS="gs://esp-data-ingestion/datased/v0.1.0"
STAGE="${HOME}/esp-data-staging/datased"
ZIP="${STAGE}/datased.zip"
WORK="${STAGE}/work"
SCRIPT="scripts/data_preprocessing_scripts/datased/build_datased.py"

echo "Node: $(hostname)  work=${WORK}"

# GCS auth: the node's ambient credential is a stale *user* ADC that cannot
# reauthenticate non-interactively (ReauthUnattendedError). Point
# CLOUDSDK_CONFIG at an empty dir so gsutil ignores it and uses the node's
# attached service account via the GCE metadata server.
export CLOUDSDK_CONFIG="$(mktemp -d)"
echo "auth probe -> ${GCS}/.auth_probe"
echo "ok" | gsutil cp - "${GCS}/.auth_probe"
gsutil rm "${GCS}/.auth_probe"
echo "auth probe OK"

# 1. Extract WAVs + ground-truth CSVs (flat) from the zip.
mkdir -p "${WORK}/audio" "${WORK}/csv"
unzip -o -j "${ZIP}" "*/SED_wav/*.wav" -d "${WORK}/audio"
unzip -o -j "${ZIP}" "*/SED_ground_truth/*.csv" -d "${WORK}/csv"
echo "extracted $(ls "${WORK}/audio" | wc -l) wavs, $(ls "${WORK}/csv" | wc -l) csvs"

# 2. Upload originals (44.1 kHz). Also an early GCS-auth check.
gsutil -m -q rsync -r "${WORK}/audio" "${GCS}/audio"
echo "uploaded originals -> ${GCS}/audio"

# 3. Write 16 kHz + 32 kHz mirrors.
uv run python "${SCRIPT}" resample \
    --audio-dir "${WORK}/audio" --out-root "${WORK}" --workers "${SLURM_CPUS_PER_TASK:-16}"

# 4. Upload mirrors.
gsutil -m -q rsync -r "${WORK}/audio_16k" "${GCS}/audio_16k"
gsutil -m -q rsync -r "${WORK}/audio_32k" "${GCS}/audio_32k"
echo "uploaded 16k + 32k mirrors"

# 5. (Re)build manifests from the extracted CSVs and upload.
uv run python "${SCRIPT}" manifests --csv-dir "${WORK}/csv" --out-dir "${WORK}/manifests"
gsutil -m -q cp "${WORK}/manifests"/*.csv "${GCS}/"

echo "final listing:"
gsutil ls "${GCS}/"
echo "Done."
