#!/usr/bin/env bash
#SBATCH --job-name=dori-crop
#SBATCH --partition=cpu,t4,a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/logs/dori_crop_%j.log
#SBATCH --error=/home/%u/logs/dori_crop_%j.err

# ───────────────────────────────────────────────────────────────────
# Crop-on-download DORI Phase-1 audio windows and upload to GCS. Streams each
# HF source recording (no disk), reads only the labeled ~15 s window, writes
# original-FLAC + 16k + 32k clips to gs://esp-data-ingestion/dori/v0.1.0/.
# Needs HF internet (compute nodes have it) + GCS. Submit:
#   ssh slurm-login 'mkdir -p ~/logs && cd ~/esp-data-dev && sbatch jobs/dori_crop_download.sh'
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$HOME/esp-data-dev"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo)}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Use the stable system CA bundle: the uv-env certifi on /scratch can be
# evicted mid-run, which previously crashed GCS uploads with a missing cacert.
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

echo "=== DORI crop-download at $(date) (cpus=${SLURM_CPUS_PER_TASK:-?}) ==="
# manifest lives in GCS; pull to this node's (NFS) home where the script expects it
mkdir -p "$HOME/dori_staging"
gsutil -q cp gs://esp-data-ingestion/dori/v0.1.0/metadata/dori_phase1_manifest.csv \
    "$HOME/dori_staging/dori_phase1_manifest.csv"

uv run --script scripts/data_preprocessing_scripts/dori_crop_download.py \
    --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-48}}" \
    --skip-existing
echo "=== finished at $(date) ==="
