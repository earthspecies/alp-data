#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/%u/logs/migrate_datasets/%j.log
#SBATCH --job-name="migrate_datasets_birdset"
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

cd ~/esp-data
uv sync
# get dataset name from command line argument
dsname=$1
srun uv run python scripts/migrate_datasets.py \
    --datasets "$dsname" \
    --new-bucket esp-data-274503 \
    --new-protocol gs \
    --report-path "migrate_datasets_report_$dsname.json" \