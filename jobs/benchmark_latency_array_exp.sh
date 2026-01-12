#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/%u/outputs/latency/%A.log
#SBATCH --job-name="benchmark_latency_array"
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

cd ~/esp-data
rm -rf scripts/benchmarks/array_*
uv sync --group benchmark

echo "=== Starting job at $(date) ==="

JOB_1=$(sbatch --parsable jobs/array_jobs/benchmark_latency_array_1.sh)
# wait for previous jobs to finish
JOB_2=$(sbatch --parsable --dependency=afterok:$JOB_1 jobs/array_jobs/benchmark_latency_array_2.sh)
JOB_3=$(sbatch --parsable --dependency=afterok:$JOB_2 jobs/array_jobs/benchmark_latency_array_4.sh)
JOB_4=$(sbatch --parsable --dependency=afterok:$JOB_3 jobs/array_jobs/benchmark_latency_array_8.sh)
#JOB_5=$(sbatch --parsable --dependency=afterok:$JOB_4 jobs/array_jobs/benchmark_latency_array_16.sh)

sbatch --dependency=afterok:$JOB_4 jobs/array_jobs/save_and_plot.sh
echo "=== Finished job at $(date) ==="
