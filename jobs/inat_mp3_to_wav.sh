#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/home/%u/logs/inat_mp3_to_wav_%j.log
#SBATCH --job-name="inat_mp3_to_wav"
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

cd ~/esp-data
uv sync
srun uv run --with tqdm python scripts/data_preprocessing_scripts/inat_mp3_to_wav.py \
    --n-workers 12 \
    --output-suffix _v3 \
    --report-path "inat_mp3_to_wav_report.json"
