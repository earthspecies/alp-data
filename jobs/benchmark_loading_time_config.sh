#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --job-name=loading_time_benchmark_config
#SBATCH --output=/home/%u/outputs/loading_time/%A.log
#SBATCH --mail-type=FAIL


while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data-location)
            DATA_LOCATION="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Error: --config argument is required"
    exit 1
fi

if [ -z "$DATA_LOCATION" ]; then
    echo "Error: --data-location argument is required"
    exit 1
fi

WORKDIR=~/esp-data
SCRIPT=scripts/benchmarks/loading_time.py

echo "=== Starting job at $(date) ==="

cd "$WORKDIR"
uv sync --group benchmark
srun uv run python $SCRIPT \
        --data-location "$DATA_LOCATION" \
        --config-path "$CONFIG" \
        --save \
        --plot \

echo "=== Finished job at $(date) ==="
