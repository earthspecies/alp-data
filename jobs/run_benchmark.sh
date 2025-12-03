#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --job-name=benchmark
#SBATCH --output=/home/%u/outputs/benchmark/%A.log
#SBATCH --mail-type=FAIL

cd ~/esp-data
uv sync --group benchmark

uv run python scripts/benchmarks/generate_list.py

# Read dataset list from text file line per line (one dataset name per line)
if [[ -f scripts/benchmarks/dataset_list.txt ]]; then
    mapfile -t DATASETS < scripts/benchmarks/dataset_list.txt
else
    echo "Dataset list file not found: scripts/benchmarks/dataset_list.txt" >&2
    exit 1
fi

# Log DATASETS for debugging
echo "Datasets to benchmark: ${DATASETS[*]}"

echo "=== Starting benchmark at $(date) ==="
JOB2=""
for DATASET in "${DATASETS[@]}"; do
    echo "Running benchmark for dataset: $DATASET"
    echo "Submitting loading time benchmark job..."
    if [ -n "$JOB2" ]; then
        echo "Previous latency benchmark job ID: $JOB2"
        JOB1=$(sbatch --parsable --dependency=afterok:$JOB2 jobs/benchmark_loading_time_default.sh --dataset "$DATASET")
    else
        JOB1=$(sbatch --parsable jobs/benchmark_loading_time_default.sh --dataset "$DATASET")
    fi

    echo "Loading time benchmark job submitted with Job ID: $JOB1"
    echo "Submitting latency benchmark job dependent on loading time benchmark completion..."
    JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/benchmark_latency_default.sh --dataset "$DATASET")
    echo "Latency benchmark job submitted with Job ID: $JOB2"
done

echo "=== Finished benchmark at $(date) ==="