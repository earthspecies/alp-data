#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gagan@earthspecies.org
#SBATCH --mem=64G
#SBATCH --output="/home/gagan_earthspecies_org/logs/%x_%j.log"
#SBATCH --job-name="voxbox_parquet"
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/esp-data
cd ~/esp-data
uv sync
cd scripts/voxaboxen
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
srun uv run --with torchaudio python create_sharded_dataset.py \
--output_path gs://esp-ml-datasets/voxaboxen/processed/v1.0.0/parquet \
--num_samples_per_shard 1000 \
--num_workers 1 \
--log_every 10 \
--changelog "Parquet version for upload"
