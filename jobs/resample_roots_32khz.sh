#!/usr/bin/env bash
#SBATCH --job-name=resample-roots-32k
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/%u/logs/resample_roots_32k_%A_%a.log
#SBATCH --error=/home/%u/logs/resample_roots_32k_%A_%a.err

# ───────────────────────────────────────────────────────────────────
# Resample ROOTS audio sources to 32kHz mono WAV (librosa kaiser_best).
#
# Array index → dataset:
#   0  animalspeak_pseudovox            (~22.05 kHz → 32 kHz)
#   1  synthetic_sed_scenes_16k         (~16 kHz   → 32 kHz)
#   2  synthetic_sed_diarization_16k    (~16 kHz   → 32 kHz)
#   3  cropped/wabad/audio              (~16 kHz   → 32 kHz)
#
# Listing happens on the compute node (do NOT run from the login VM).
#
# USAGE:
#   mkdir -p ~/logs && sbatch jobs/resample_roots_32khz.sh
#   # Re-run specific indices only:
#   sbatch --array=1,3 jobs/resample_roots_32khz.sh
#
# To resume after a partial run, --skip-existing in the call below handles it.
# ───────────────────────────────────────────────────────────────────

set -euo pipefail
cd /home/david_earthspecies_org/esp-data-dev

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo)}"
echo "GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"

# Use the system CA bundle to avoid relying on certifi's data file living in
# a per-node /scratch uv env — workers occasionally lost track of it on the
# GPU partitions and ~94% of uploads failed with "no suitable TLS CA bundle".
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

SOURCES=(
    "gs://fewshot/data_large_clean/animalspeak_pseudovox/"
    "gs://foundation-model-data/synthetic/synthetic_sed_scenes_16k/audio/"
    "gs://foundation-model-data/synthetic/synthetic_sed_diarization_16k/audio/"
    "gs://foundation-model-data/synthetic/cropped/wabad/audio/"
)
DESTS=(
    "gs://foundation-model-data/synthetic/animalspeak_pseudovox_32k/"
    "gs://foundation-model-data/synthetic/synthetic_sed_scenes_32k/audio/"
    "gs://foundation-model-data/synthetic/synthetic_sed_diarization_32k/audio/"
    "gs://foundation-model-data/synthetic/cropped/wabad/audio_32k/"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
SRC=${SOURCES[$IDX]}
DST=${DESTS[$IDX]}

echo "=== Starting array task $IDX at $(date) ==="
echo "Source: $SRC"
echo "Dest:   $DST"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"

uv run --script scripts/resample_to_32khz.py \
    --source-prefix "$SRC" \
    --dest-prefix "$DST" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --skip-existing

echo "=== Finished array task $IDX at $(date) ==="
