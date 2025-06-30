#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH -p cpu
#SBATCH --cpus-per-task=12
#SBATCH --array=1-4
#SBATCH --output="/home/gagan_earthspecies_org/logs/%x_%A_%a.log"
#SBATCH --job-name="benchmark-webds-enhanced"

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================="

export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/esp-data-benchmarks
cd ~/esp-data
uv sync

# Add timestamp to identify concurrent access patterns
echo "Starting benchmark at: $(date '+%Y-%m-%d %H:%M:%S.%3N')"

srun uv run --with torch --with torchaudio --with tqdm \
 --with webdataset python scripts/benchmarks/webds_test.py \
  --use_beans \
  --path_to_beans "/home/gagan_earthspecies_org/esp-ml-datasets/beans_zero_tar_v0.1.0/tar" \
  --num_workers 8 \
  --batch_size 256 \
  --shard_shuffle \
  --shard_shuffle_size 100 \
  --shuffle_size 2000 \
  --max_iters 100

echo "Completed benchmark at: $(date '+%Y-%m-%d %H:%M:%S.%3N')"
echo "========================================="
