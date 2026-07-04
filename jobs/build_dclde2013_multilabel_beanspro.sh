#!/bin/bash
#SBATCH --job-name=build-dclde2013-ml
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output="/home/%u/logs/build_dclde2013_ml_%j.log"
#SBATCH --error="/home/%u/logs/build_dclde2013_ml_%j.err"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Build the BEANS-Pro `dclde2013-multilabel-species` evaluation split.
#
# Windows the held-out DCLDE 2013 NEFSC Stellwagen all-baleen recordings
# into fixed 10 s multilabel species clips (+ true negatives), cuts 32 kHz
# WAVs, emits a BEANS-Pro JSONL, and uploads everything to GCS.
#
# Pure CPU job (I/O bound: reads ~76 GB of source audio from GCS).
#
# USAGE
#   ssh slurm-login
#   cd /home/${USER}/esp-data-dev
#   mkdir -p ~/logs
#   sbatch jobs/build_dclde2013_multilabel_beanspro.sh
#
# Smoke test (10 recordings, no upload):
#   sbatch jobs/build_dclde2013_multilabel_beanspro.sh --limit-clips 10 --no-upload
# ───────────────────────────────────────────────────────────────────

set -euo pipefail
cd /home/david_earthspecies_org/esp-data-dev

OUTPUT_DIR="${OUTPUT_DIR:-data/beans_pro_dclde2013_multilabel_species}"
GCS_DEST="gs://esp-data-ingestion/beans-pro/v0.1.0/raw/dclde2013_multilabel_species"

UPLOAD=1
BUILD_ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == "--no-upload" ]]; then
        UPLOAD=0
    else
        BUILD_ARGS+=("${arg}")
    fi
done

echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Build args: ${BUILD_ARGS[*]:-<none>}"

uv run python scripts/build_beans_pro_dclde2013_multilabel_species.py \
    --output-dir "${OUTPUT_DIR}" \
    "${BUILD_ARGS[@]}"

if [[ "${UPLOAD}" -eq 1 ]]; then
    echo "Uploading audio + manifest to ${GCS_DEST}"
    gsutil -m cp -r "${OUTPUT_DIR}/audio" "${GCS_DEST}/"
    gsutil cp "${OUTPUT_DIR}/test.jsonl" "${GCS_DEST}/test.jsonl"
    gsutil cp "${OUTPUT_DIR}/stats.json" "${GCS_DEST}/stats.json"
    echo "Upload complete."
fi

echo "Done."
