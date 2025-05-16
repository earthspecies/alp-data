#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -c 20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gagan@earthspecies.org
#SBATCH --mem=180G
#SBATCH --output="/home/gagan_earthspecies_org/logs/%x_%j.log"
#SBATCH --job-name="natlm_reduce_size"
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/esp-data
cd ~/esp-data
uv sync
cd scripts/naturelm
srun uv run --with "dataset[audio]" --with "dask" python reduce_size_audio_cast_v3.py
