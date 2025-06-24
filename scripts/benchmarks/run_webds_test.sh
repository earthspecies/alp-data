#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH -p cpu
#SBATCH --output="/home/gagan_earthspecies_org/logs/%A.log"
#SBATCH --job-name="benchmark-webds"
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/esp-data
cd ~/esp-data
uv sync
uv run --with torch --with webdataset scripts/benchmarks/webds_test.py \
  --use_beans \
  --num_workers 8 \
  --batch_size 256 \
  --max_iters 100