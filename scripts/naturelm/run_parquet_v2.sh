#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -c 42
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gagan@earthspecies.org
#SBATCH --mem=256G
#SBATCH --output="/home/gagan_earthspecies_org/logs/%x_%j.log"
#SBATCH --job-name="build_natlm_parqv2"
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/esp-data
cd ~/esp-data
uv sync
cd scripts/naturelm
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
srun uv run --with fastparquet --with torchaudio --with mlflow --with psutil python create_sharded_dataset_metadata.py \
--path_to_jsonl_files "gs://esp-ml-datasets/naturelm/raw/v0.1.1/train/train_chunks" \
--output_path "gs://esp-ml-datasets/naturelm/processed/v0.1.0/parquet/train/" \
--version "0.1.0" \
--num_samples_per_shard 2500 \
--log_every 500 \
--shard_type "parquet" \
--changelog "Parquet file version for training split" \
--num_workers 40 \
--error_handling raise
