#!/usr/bin/env bash
#SBATCH --job-name=delphinid-build
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Build the delphinid-whistle detection dataset (Dryad z34tmpgq6) into
# gs://esp-data-ingestion/delphinid-whistles/v0.1.0/. Uses the zips already
# on NFS: extracts (if needed), writes 16k+32k mirrors (+48k originals)
# flattened to <site>/<basename>, builds manifests, uploads. Runs on Slurm
# (big merged aquarium wavs) — not the dev VM.
#   sbatch scripts/data_preprocessing_scripts/delphinid_whistles/build_delphinid_whistles.sh
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${HOME}/esp-data-dev"

GCS="gs://esp-data-ingestion/delphinid-whistles/v0.1.0"
STAGE="${HOME}/esp-data-staging/delphinid"
WORK="${STAGE}/work"
MIRRORS="${STAGE}/mirrors"
TRAIN_ROOT="${WORK}/Training_Audio_and_Anntoations"
TEST_ROOT="${WORK}/Testing_Audio_and_Annotations"
SCRIPT="scripts/data_preprocessing_scripts/delphinid_whistles/build_delphinid_whistles.py"

# Ignore stale user ADC; use node attached SA (gsutil + gcsfs).
export CLOUDSDK_CONFIG="$(mktemp -d)"
echo "Node: $(hostname)  work=${WORK}"
echo "ok" | gsutil cp - "${GCS}/.auth_probe" && gsutil rm "${GCS}/.auth_probe" && echo "auth probe OK"

# 1. Extract trees if not already present.
if [ ! -d "${TRAIN_ROOT}" ]; then unzip -o -q "${STAGE}/training.zip" -d "${WORK}/"; fi
if [ ! -d "${TEST_ROOT}" ]; then unzip -o -q "${STAGE}/testing.zip" -d "${WORK}/"; fi
echo "wavs: $(find "${WORK}" -iname '*.wav' | wc -l)"

# 2. Resample every wav -> 16k + 32k mirrors (+ copy 48k original), flattened.
rm -rf "${MIRRORS}"; mkdir -p "${MIRRORS}"
uv run python "${SCRIPT}" resample \
    --audio-root "${WORK}" --out-root "${MIRRORS}" --workers "${SLURM_CPUS_PER_TASK:-16}"

# 3. Build manifests.
uv run python "${SCRIPT}" manifests \
    --train-root "${TRAIN_ROOT}" --test-root "${TEST_ROOT}" --out-dir "${MIRRORS}/manifests"

# 4. Upload audio mirrors + manifests.
gsutil -m -q rsync -r "${MIRRORS}/audio"     "${GCS}/audio"
gsutil -m -q rsync -r "${MIRRORS}/audio_16k" "${GCS}/audio_16k"
gsutil -m -q rsync -r "${MIRRORS}/audio_32k" "${GCS}/audio_32k"
gsutil -m -q cp "${MIRRORS}/manifests"/*.csv "${GCS}/"

echo "final listing:"; gsutil ls "${GCS}/"
echo "Done."
