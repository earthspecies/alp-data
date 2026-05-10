#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/home/%u/logs/iteration_check/%j_%x.log
#SBATCH --job-name="iteration_check"
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

cd ~/esp-data
uv sync
export PYTHONUNBUFFERED=1
srun uv run --group benchmark python -u scripts/iteration_check.py \
    --config-in "$1" \
    --batch-size 256 \
    --num-workers 8
