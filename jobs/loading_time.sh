#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --job-name=loading_time_benchmark
#SBATCH --output=/home/%u/outputs/loading_time/%A.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN,END

echo "=== Starting job at $(date) ==="

# your actual work here
cd ~/esp-data
uv sync
uv run --with torch python scripts/benchmarks/loading_time.py --data-location 'nfs'
uv run --with torch python scripts/benchmarks/loading_time.py --data-location 'bucket'



echo "=== Finished job at $(date) ==="

# resubmit this same script for 1 hours later
sbatch --begin=now+1hours $0