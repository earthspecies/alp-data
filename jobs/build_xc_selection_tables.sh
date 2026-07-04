#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --job-name=xc-sel-tables
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/%u/logs/xc_selection_tables_%A_%a.log
#SBATCH --array=0-1

set -euo pipefail

THRESHOLDS=(0.5 0.2)
THRESHOLD=${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}
SUFFIX=$(echo "$THRESHOLD" | tr '.' '')

RESULTS_JSONL="gs://foundation-model-data/curation/xc_inference_effnet/results.jsonl"
XC_ALL_CSV="gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/all_20260203.csv"
INCLUDE_CSV="gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/train_20260203.csv"
OUTPUT_GCS="gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/train_strong_labels_t${SUFFIX}.csv"

WORKDIR=~/esp-data-dev
LOCAL_OUT="/tmp/train_strong_labels_t${SUFFIX}.csv"

echo "=== Starting job at $(date) ==="
echo "Threshold: $THRESHOLD"
echo "Output:    $OUTPUT_GCS"

cd "$WORKDIR"
uv sync

uv run python scripts/build_xc_selection_tables.py \
    --results-jsonl "$RESULTS_JSONL" \
    --xc-csv "$XC_ALL_CSV" \
    --include-csv "$INCLUDE_CSV" \
    --output "$LOCAL_OUT" \
    --threshold "$THRESHOLD" \
    --min-bg-windows 1

echo "Uploading to $OUTPUT_GCS ..."
gsutil -o 'GSUtil:parallel_composite_upload_threshold=150M' cp "$LOCAL_OUT" "$OUTPUT_GCS"

echo "=== Finished at $(date) ==="
echo "Rows: $(wc -l < "$LOCAL_OUT")"
ls -lh "$LOCAL_OUT"
rm -f "$LOCAL_OUT"
