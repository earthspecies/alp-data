#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --job-name=loading_time_benchmark_default
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
uv sync --group benchmark
srun uv run python $SCRIPT \
        --dataset-name "$DATASET" \
        --save \
        --plot \

echo "=== Finished job at $(date) ==="
