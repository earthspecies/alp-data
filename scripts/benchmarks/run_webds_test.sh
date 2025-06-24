#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH -p cpu
#SBATCH --cpus-per-task=12
#SBATCH --output="/home/gagan_earthspecies_org/logs/%x_%A.log"
#SBATCH --job-name="benchmark-webds"
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/esp-data-benchmarks
cd ~/esp-data
uv sync
srun uv run --with torch --with torchaudio --with tqdm \
 --with webdataset python scripts/benchmarks/webds_test.py \
  --use_beans \
  --num_workers 8 \
  --batch_size 256 \
  --max_iters 100