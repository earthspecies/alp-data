#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH -p cpu
#SBATCH --cpus-per-task=12
#SBATCH --array=1-2
#SBATCH --output="/home/gagan_earthspecies_org/logs/%x_%A_%a.log"
#SBATCH --job-name="benchmark-parquet-enhanced"

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
 --with pyarrow python scripts/benchmarks/parquetds_test.py \
  --num_workers 10 \
  --batch_size 256 \
  --max_iters 500 --shard_strategy "distributed"

echo "Completed benchmark at: $(date '+%Y-%m-%d %H:%M:%S.%3N')"
echo "========================================="