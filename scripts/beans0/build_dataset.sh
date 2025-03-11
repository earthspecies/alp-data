#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output="/home/gagan_earthspecies_org/logs/%x_%j_%A.log"
#SBATCH --job-name="build_parquet"
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/esp-data
cd ~/esp-data
uv sync
srun uv run python create_sharded_arrow_dataset.py \
--metadata_path metadata_v6.jsonl \
--original_paths_file original_paths_v6.jsonl \
--dataset_path gs://esp-ml-datasets/beans0/raw/data \
--shard_type parquet \
--output_shard_pattern "**/*.parquet" \
--num_samples_per_shard 3000 \
--version 0.1.0 \
--changelog "Parquet version for HF upload"
