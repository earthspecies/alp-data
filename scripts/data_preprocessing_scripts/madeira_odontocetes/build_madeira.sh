#!/usr/bin/env bash
#SBATCH --job-name=madeira-build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/%u/logs/%A_%x.log
#SBATCH --qos=naturelm
set -euo pipefail
cd ~/esp-data-dev
uv run --with pandas --with openpyxl --with soundfile --with librosa \
  python scripts/data_preprocessing_scripts/madeira_odontocetes/build_madeira.py
