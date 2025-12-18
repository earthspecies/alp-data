#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/%u/outputs/latency/%A.log
#SBATCH --job-name="save_benchmark_latency_array"
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

cd ~/esp-data
uv sync --group benchmark

srun uv run python scripts/benchmarks/merge_array_results.py 1
srun uv run python scripts/benchmarks/merge_array_results.py 2
srun uv run python scripts/benchmarks/merge_array_results.py 4
srun uv run python scripts/benchmarks/merge_array_results.py 8

srun uv run python scripts/benchmarks/plot_latency_bench.py \
    --number-of-samples 30 \
    --parameter nb_array \
    --array-exp
