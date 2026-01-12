"""
Merge local results from multiple array experiments and upload to cloud
"""

import glob
from sys import argv

import pandas as pd
from benchmark_utils import save_results


def merge_and_upload_array_results(
    nb_array: int,
    local_dir: str = "scripts/benchmarks",
    cloud_path: str = "gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_latency_array.csv",
) -> None:
    """
    Merge all local array experiment results into one CSV and upload to cloud.

    Parameters
    ----------
    local_dir : str
        Directory containing array experiment result folders.
    cloud_path : str
        GCS path to save the merged results.
    """
    # Find all result CSVs in array_* subfolders
    pattern = f"{local_dir}/array_{nb_array}/benchmark_latency_results*.csv"
    csv_files = glob.glob(pattern)
    if not csv_files:
        print("No array experiment result files found.")
        return

    # Read and concatenate all results
    dfs = [pd.read_csv(f) for f in csv_files]
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save to cloud using your existing utility
    save_results(merged_df, cloud_path)
    print(f"Merged {len(csv_files)} files and uploaded to {cloud_path}")


if __name__ == "__main__":
    merge_and_upload_array_results(int(argv[1]))  # Pass nb_array as command line argument
