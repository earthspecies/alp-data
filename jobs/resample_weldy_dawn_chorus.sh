#!/usr/bin/env bash
#SBATCH --job-name=resample-weldy
#SBATCH --partition=cpu,t4,a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/%u/logs/resample_weldy_%A_%a.log
#SBATCH --error=/home/%u/logs/resample_weldy_%A_%a.err

# ───────────────────────────────────────────────────────────────────
# Resample Weldy NW dawn-chorus recordings (32 kHz stereo WAV) to mono 16-bit
# PCM WAV at 16 kHz and 32 kHz. Audio is downmixed to mono by resample_to_sr.py.
#
# Array index → target rate:
#   0  -> audio_16k/
#   1  -> audio_32k/
#
# Submit from the login node:
#   ssh slurm-login 'mkdir -p ~/logs && cd ~/esp-data-dev && sbatch jobs/resample_weldy_dawn_chorus.sh'
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$HOME/esp-data-dev"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo)}"
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

ROOT="gs://esp-data-ingestion/weldy_dawn_chorus/v0.1.0"
SRC="$ROOT/recordings/"
RATES=(16000 32000)
DESTS=("$ROOT/audio_16k/" "$ROOT/audio_32k/")

IDX=${SLURM_ARRAY_TASK_ID:-0}
SR=${RATES[$IDX]}
DST=${DESTS[$IDX]}

echo "=== Weldy resample array task $IDX ($SR Hz) at $(date) ==="
uv run --script scripts/resample_to_sr.py \
    --source-prefix "$SRC" \
    --dest-prefix "$DST" \
    --target-sr "$SR" \
    --workers "${SLURM_CPUS_PER_TASK:-48}" \
    --skip-existing
echo "=== Finished array task $IDX at $(date) ==="
