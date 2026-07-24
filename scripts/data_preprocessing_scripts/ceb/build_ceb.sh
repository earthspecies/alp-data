#!/usr/bin/env bash
#SBATCH --job-name=ceb-extract
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Extract one CEB subset's FLACs from its GCS tar and upload to
# gs://esp-data-ingestion/ceb/v0.1.0/audio/<subset>/.
# Streams the tar (no full-disk staging) and batch-rsyncs; resumable.
# Usage: SUBSET=test_soundscape sbatch scripts/data_preprocessing_scripts/ceb/build_ceb.sh
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${HOME}/esp-data-dev"
SUBSET="${SUBSET:?set SUBSET=train_xenocanto|train_soundscape|test_soundscape}"
WORK="/tmp/ceb_extract_${SLURM_JOB_ID:-manual}"; mkdir -p "${WORK}"
echo "Node: $(hostname)  subset=${SUBSET}  work=${WORK}"

uv run --with pandas python scripts/data_preprocessing_scripts/ceb/build_ceb.py \
    --extract "${SUBSET}" --workdir "${WORK}" --batch "${BATCH:-1500}"

echo "objects now under audio/${SUBSET}:"
gsutil ls -l "gs://esp-data-ingestion/ceb/v0.1.0/audio/${SUBSET}/**" 2>/dev/null | tail -1
rm -rf "${WORK}"
echo "Done."
