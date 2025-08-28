#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --output="/home/gagan_earthspecies_org/logs/%x_%j.log"
#SBATCH --job-name="beans-raw-audio"

cd ~/esp-data
uv sync
srun uv run --with torch python scripts/benchmarks/benchmark_dataset.py \
    -c scripts/benchmarks/benchmark_config.yaml \
    --data-location bucket \
    --num-workers 8 \
    --batch-size 256 \
    --max-iterations 10 \
    --log-interval 5
srun uv run --with torch python scripts/benchmarks/benchmark_dataset.py \
    -c scripts/benchmarks/benchmark_config.yaml \
    --data-location bucket \
    --raw-dataset \
    --max-iterations 100 \
    --log-interval 5
