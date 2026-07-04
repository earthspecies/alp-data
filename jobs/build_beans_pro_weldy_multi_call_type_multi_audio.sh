#!/usr/bin/env bash
#SBATCH --job-name=build-weldy-mct-ma
#SBATCH --partition=cpu,t4,a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/%u/logs/build_weldy_mct_ma_%A.log
#SBATCH --error=/home/%u/logs/build_weldy_mct_ma_%A.err

# ───────────────────────────────────────────────────────────────────
# Build a BEANS-Pro Weldy multi-call-type few-shot multi-audio split for
# a specified num-shots (default 1). Outputs to a num-shots-suffixed dir
# and uploads to GCS.
#
# Usage (from slurm-login):
#   sbatch jobs/build_beans_pro_weldy_multi_call_type_multi_audio.sh
#   NUM_SHOTS=2 sbatch jobs/build_beans_pro_weldy_multi_call_type_multi_audio.sh
#   NUM_SHOTS=3 sbatch jobs/build_beans_pro_weldy_multi_call_type_multi_audio.sh
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$HOME/esp-data-dev"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo)}"
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

NUM_SHOTS="${NUM_SHOTS:-1}"
MIN_PER_CLASS="${MIN_PER_CLASS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/beans_pro_weldy_multi_call_type_fewshot_${NUM_SHOTS}shot}"
GCS_DEST="${GCS_DEST:-gs://esp-data-ingestion/beans-pro/v0.1.0/raw/weldy_multi_call_type_fewshot_${NUM_SHOTS}shot/}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "Building Weldy multi-call-type few-shot multi-audio split"
echo "  num-shots:     ${NUM_SHOTS}"
echo "  min-per-class: ${MIN_PER_CLASS}"
echo "  output_dir:    ${OUTPUT_DIR}"
echo "  gcs_dest:      ${GCS_DEST}"

srun uv run python scripts/build_beans_pro_weldy_multi_call_type_multi_audio.py \
    --output-dir "${OUTPUT_DIR}" \
    --min-per-class "${MIN_PER_CLASS}" \
    --num-shots "${NUM_SHOTS}"

echo "Uploading to GCS..."
gsutil -m cp "${OUTPUT_DIR}/test.jsonl" "${GCS_DEST}"
gsutil -m cp -r "${OUTPUT_DIR}/audio" "${GCS_DEST}"

echo "Done."
gsutil ls "${GCS_DEST}" | head
