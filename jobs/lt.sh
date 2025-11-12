#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --job-name=loading_time_benchmark
#SBATCH --output=/home/%u/outputs/loading_time/%A.log
#SBATCH --mail-type=FAIL

cd ~/esp-data
uv sync

uv run python scripts/benchmarks/generate_list.py

# Read dataset list from text file line per line (one dataset name per line)
if [[ -f scripts/benchmarks/dataset_list.txt ]]; then
    mapfile -t DATASETS < scripts/benchmarks/dataset_list.txt
else
    echo "Dataset list file not found: scripts/benchmarks/dataset_list.txt" >&2
    exit 1
fi

# Log DATASETS for debugging
echo "Datasets to benchmark: ${DATASETS[*]}"

echo "=== Starting job at $(date) ==="

for DATASET in "${DATASETS[@]}"; do
    # # trim whitespace and ignore empty lines or comments
    # DATASET="$(echo "$DATASET" | xargs)"
    # [[ -z "$DATASET" || "${DATASET:0:1}" == "#" ]] && continue

    echo "Running benchmark for dataset: $DATASET"
    sbatch jobs/loading_time.sh --dataset "$DATASET"
    sleep 10 # wait 10 seconds between submissions
done

echo "=== Finished job at $(date) ==="
