#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --output="/home/%u/logs/birdset_resample_%A.log"
#SBATCH --job-name="birdset-resample"
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --mem=32G

# Install required tools and set up authentication
uv tool install keyring --with keyrings.google-artifactregistry-auth
export GOOGLE_APPLICATION_CREDENTIALS=/home/marius_miron_earthspecies_org/.config/gcloud/application_default_credentials.json
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# Change to esp-data directory and sync dependencies
cd ~/code/esp-data
uv sync

# Set default parameters (can be overridden via environment variables)
TARGET_DIR=${TARGET_DIR:-"/home/marius_miron_earthspecies_org/data/16000"}
TARGET_SR=${TARGET_SR:-16000}
WORKERS=${WORKERS:-10}
SPLITS=${SPLITS:-"all"}
USE_DATALOADER=${USE_DATALOADER:-""}
BATCH_SIZE=${BATCH_SIZE:-1}

# Construct the command
CMD="uv run python scripts/data_preprocessing_scripts/birdset_resample.py --target_dir $TARGET_DIR --target_sr $TARGET_SR --workers $WORKERS"

# Add splits if specified
if [ "$SPLITS" != "all" ]; then
    CMD="$CMD --splits $SPLITS"
else
    CMD="$CMD --splits all"
fi

# Add dataloader options if enabled
if [ -n "$USE_DATALOADER" ]; then
    CMD="$CMD --use-dataloader --batch-size $BATCH_SIZE"
fi

echo "Starting BirdSet resampling with command:"
echo "$CMD"
echo "Target directory: $TARGET_DIR"
echo "Target sample rate: $TARGET_SR"
echo "Workers: $WORKERS"
echo "Splits: $SPLITS"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Run the resampling script
srun $CMD 