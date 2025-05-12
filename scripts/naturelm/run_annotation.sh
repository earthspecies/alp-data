#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -c 10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gagan@earthspecies.org
#SBATCH --mem=256G
#SBATCH --output="/home/gagan_earthspecies_org/logs/%x_%j.log"
#SBATCH --job-name="build_natlm_annotation"
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/esp-data
cd ~/esp-data
uv sync
cd scripts/naturelm
srun uv run python create_train_annotations.py
