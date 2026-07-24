#!/usr/bin/env bash
#SBATCH --job-name=eco-extract
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Extract ECOSoundSet Split-recording clips from the GCS tars and upload
# to gs://esp-data-ingestion/ecosoundset/v0.1.0/audio/. Streams each tar
# (no full-disk staging) and batch-rsyncs; resumable. TAR defaults to
# 'all' (both split tars).
#   sbatch scripts/data_preprocessing_scripts/ecosoundset/build_ecosoundset.sh
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${HOME}/esp-data-dev"
TAR="${TAR:-all}"
WORK="/tmp/eco_extract_${SLURM_JOB_ID:-manual}"; mkdir -p "${WORK}"
echo "Node: $(hostname)  tar=${TAR}  work=${WORK}"

uv run --with pandas python scripts/data_preprocessing_scripts/ecosoundset/build_ecosoundset.py \
    --extract "${TAR}" --workdir "${WORK}" --batch "${BATCH:-2000}"

echo "objects now under audio/:"
gsutil du -s "gs://esp-data-ingestion/ecosoundset/v0.1.0/audio/" 2>/dev/null || true
rm -rf "${WORK}"
echo "Done."
