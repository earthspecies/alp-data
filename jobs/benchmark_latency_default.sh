#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/%u/outputs/latency/%A.log
#SBATCH --job-name="benchmark_latency_bucket"
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

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

cd ~/esp-data
uv sync --group benchmark

nw_values=(0 2 4 8)

for nw in "${nw_values[@]}"; do
    echo "Running benchmark_latency with num_workers=$nw and batch_size=128"
    srun uv run python scripts/benchmarks/benchmark_latency.py \
        --log-interval 5 \
        --max-iterations 1000 \
        --num-workers $nw \
        --batch-size 128 \
        --prefetch-factor 0 \
        --sleep 0.5 \
        --dataset-name "$DATASET" \
        --save
done

num_values=${#nw_values[@]}
uv run python scripts/benchmarks/plot_latency_bench.py --number-of-samples $num_values --parameter "num_workers"

pf_values=(0 1 2 4)

for pf in "${pf_values[@]}"; do
    echo "Running benchmark_latency with prefetch_factor=$pf and batch_size=128"
    srun uv run python scripts/benchmarks/benchmark_latency.py \
        --log-interval 5 \
        --max-iterations 1000 \
        --num-workers 4 \
        --batch-size 128 \
        --prefetch-factor $pf \
        --sleep 0.5 \
        --dataset-name "$DATASET" \
        --save
done

num_values=${#pf_values[@]}
uv run python scripts/benchmarks/plot_latency_bench.py --number-of-samples $num_values --parameter "prefetch_factor"

echo "=== Finished job at $(date) ==="