#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -c 5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gagan@earthspecies.org
#SBATCH --mem=64G
#SBATCH --output="/home/gagan_earthspecies_org/logs/%x_%j.log"
#SBATCH --job-name="beans_parquet"
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/esp-data
cd ~/esp-data
uv sync
cd scripts/beans0
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
srun uv run python create_sharded_arrow_dataset.py \
--metadata_path metadata_v6.jsonl \
--original_paths_file original_paths_v7.jsonl \
--dataset_path gs://esp-ml-datasets/beans0/processed/ \
--shard_type parquet \
--num_workers 4 \
--output_shard_pattern "**/*.parquet" \
--num_samples_per_shard 3000 \
--version "0.1.0" \
--changelog "Parquet version for HF upload"
