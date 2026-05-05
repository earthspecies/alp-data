#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/%u/logs/inat_m4a_to_wav_%j.log
#SBATCH --job-name="inat_m4a_to_wav"
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

cd ~/esp-data
uv sync
srun uv run --with tqdm python scripts/data_preprocessing_scripts/inat_m4a_to_wav.py \
    --n-workers 8 \
    --output-suffix _1 \
    --dry-run \
    --report-path "inat_m4a_to_wav_report.json"
