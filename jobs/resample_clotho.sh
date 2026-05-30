#!/usr/bin/env bash
#SBATCH --job-name=resample-clotho
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/%u/logs/resample_clotho_%A_%a.log
#SBATCH --error=/home/%u/logs/resample_clotho_%A_%a.err

set -euo pipefail
cd "$HOME/esp-data-dev"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo)}"

ROOT="gs://esp-data-ingestion/clotho/v0.1.0/raw"
RATES=(16000 32000)
DESTS=("$ROOT/audio_16k/" "$ROOT/audio_32k/")
IDX=${SLURM_ARRAY_TASK_ID:-0}
SR=${RATES[$IDX]}
DST=${DESTS[$IDX]}

echo "=== Clotho resample array task $IDX ($SR Hz) at $(date) ==="
for split in development validation evaluation; do
    echo ""
    echo "--- $split @ $SR Hz ---"
    uv run --script scripts/resample_to_sr.py \
        --source-prefix "$ROOT/audio/$split/" \
        --dest-prefix "${DST}$split/" \
        --target-sr "$SR" \
        --workers "${SLURM_CPUS_PER_TASK:-16}" \
        --skip-existing
done
echo "=== Finished array task $IDX at $(date) ==="
