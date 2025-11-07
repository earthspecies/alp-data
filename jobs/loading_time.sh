#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --job-name=loading_time_benchmark
#SBATCH --output=/home/%u/outputs/loading_time/%A.log
#SBATCH --mail-type=FAIL


while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "Error: --dataset argument is required"
    exit 1
fi

WORKDIR=~/esp-data
SCRIPT=scripts/benchmarks/loading_time.py

echo "=== Starting job at $(date) ==="

cd "$WORKDIR"
uv sync
srun uv run --with torch python $SCRIPT \
        --data-location 'bucket' \
        --dataset-name "$DATASET"

echo "=== Finished job at $(date) ==="
