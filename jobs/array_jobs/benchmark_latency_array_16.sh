#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --array=1-16
#SBATCH --output=/home/%u/outputs/latency/%A_%a.log
#SBATCH --job-name="benchmark_latency_array_16"
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

cd ~/esp-data
uv sync --group benchmark

srun uv run python scripts/benchmarks/benchmark_latency.py \
   --log-interval 5 \
   --max-iterations 1000 \
   --num-workers 2 \
   --batch-size 64 \
   --prefetch-factor 0 \
   --sleep 0.5 \
   --config-path scripts/benchmarks/job_array_exp_no_resampling_config.yaml \
   --data-location bucket \
   --save "local" \
   --nb_array 16


srun uv run python scripts/benchmarks/benchmark_latency.py \
    --log-interval 5 \
    --max-iterations 1000 \
    --num-workers 2 \
    --batch-size 64 \
    --prefetch-factor 0 \
    --sleep 0.5 \
    --config-path scripts/benchmarks/job_array_exp_no_resampling_config.yaml \
    --data-location nfs \
    --save "local" \
    --nb_array 16
