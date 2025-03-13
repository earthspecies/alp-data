#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -c 10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gagan@earthspecies.org
#SBATCH --mem=64G
#SBATCH --output="/home/gagan_earthspecies_org/logs/%x_%j.log"
#SBATCH --job-name="beans_webds"
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/esp-data
cd ~/esp-data
uv sync
cd scripts/beans0
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
srun uv run python create_sharded_webdataset.py \
--metadata_path metadata_v6.jsonl \
--original_paths_file original_paths_v7.jsonl \
--dataset_path gs://esp-ml-datasets/beans0/processed/ \
--num_workers 8 \
--num_samples_per_shard 3000 \
--version "0.1.1" \
--changelog "Parquet version for HF upload" \
--log_every 300
