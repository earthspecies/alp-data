#!/usr/bin/env bash
#SBATCH --job-name=baringo-build
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Build the Baringo (western Kenya) soundscape dataset (Zenodo 10943500)
# into gs://esp-data-ingestion/baringo-soundscapes/v0.1.0/. Extracts the
# 32 kHz FLACs, re-encodes to clean 16k + 32k WAV mirrors (the source FLACs
# trip a libsndfile decoder bug), builds the single-`all` centerpoint
# manifest, uploads. Runs on Slurm (1-hour files).
#   sbatch scripts/data_preprocessing_scripts/baringo_soundscapes/build_baringo_soundscapes.sh
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${HOME}/esp-data-dev"

GCS="gs://esp-data-ingestion/baringo-soundscapes/v0.1.0"
STAGE="${HOME}/esp-data-staging/baringo"
WORK="${STAGE}/work"
MIRRORS="${STAGE}/mirrors"
SCRIPT="scripts/data_preprocessing_scripts/baringo_soundscapes/build_baringo_soundscapes.py"

export CLOUDSDK_CONFIG="$(mktemp -d)"   # attached SA for gsutil
echo "Node: $(hostname)"
echo "ok" | gsutil cp - "${GCS}/.auth_probe" && gsutil rm "${GCS}/.auth_probe" && echo "auth probe OK"

# 1. Extract FLACs if not already present.
if [ ! -d "${WORK}" ] || [ -z "$(find "${WORK}" -iname '*.flac' 2>/dev/null | head -1)" ]; then
  mkdir -p "${WORK}"; unzip -o -q "${STAGE}/soundscape_data.zip" -d "${WORK}/"
fi
echo "flacs: $(find "${WORK}" -iname '*.flac' | wc -l)"

# 2. Resample -> 16k FLAC mirrors (+copy originals) + durations.csv.
rm -rf "${MIRRORS}"; mkdir -p "${MIRRORS}"
uv run python "${SCRIPT}" resample \
    --audio-root "${WORK}" --out-root "${MIRRORS}" --workers "${SLURM_CPUS_PER_TASK:-16}"

# 3. Build single-all manifest.
uv run python "${SCRIPT}" manifests \
    --anno-csv "${STAGE}/annotations.csv" --species-csv "${STAGE}/species.csv" \
    --durations-csv "${MIRRORS}/durations.csv" --out-dir "${MIRRORS}/manifests"

# 4. Upload audio mirrors + manifests.
gsutil -m -q rsync -r "${MIRRORS}/audio_16k" "${GCS}/audio_16k"
gsutil -m -q rsync -r "${MIRRORS}/audio_32k" "${GCS}/audio_32k"
gsutil -m -q cp "${MIRRORS}/manifests"/*.csv "${GCS}/"

echo "final listing:"; gsutil ls "${GCS}/"
echo "Done."
