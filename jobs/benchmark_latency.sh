#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --output="/home/paul_earthspecies_org/outputs/%x_%j.log"
#SBATCH --job-name="benchmark_latency"
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/benchmarking
cd ~/esp-data
uv sync

nw_values=(0 2 4 8 16)

for nw in "${nw_values[@]}"; do
    echo "Running benchmark_latency with num_workers=$nw and batch_size=64"
    srun uv run --with torch python scripts/benchmarks/benchmark_latency.py \
        -c scripts/benchmarks/benchmark_config.yaml \
        --data-location 'bucket' \
        --log-interval 5 \
        --max-iterations 1000 \
        --num-workers $nw \
        --batch-size 64
done