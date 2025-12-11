import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from benchmark_utils import set_logging_config

set_logging_config()


def get_saved_results_from_cloud() -> pd.DataFrame:
    """Retrieve all benchmark loading time results from a CSV file saved in a GCS bucket.
    Returns
    -------
    A pandas DataFrame containing the benchmark loading time results.
    """
    df = pd.read_csv(
        "gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_loading_time.csv"
    )
    return df


def plot_last_experiment_results_against_all(results: pd.DataFrame) -> None:
    """Plots last experiment results against all saved benchmark results with the same configuration
    (dataset_name, split_name, sample_rate).
    Boxplots are created for each measured metric, with the last experiment results overlaid as red
    dots.
    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame containing benchmark results for a specific dataset.
    """
    logger = logging.getLogger("last_results_plotter")

    df = get_saved_results_from_cloud()
    name = results["dataset_name"].iloc[0]
    split = results["split_name"].iloc[0]
    df = df[df["dataset_name"] == name]
    df = df[df["split_name"] == split]
    df.fillna({"sample_rate": "default"}, inplace=True)
    sr = results["sample_rate"].iloc[0]
    if sr is None:
        sr = "default"
    df = df[df["sample_rate"] == sr]

    # Separate the dataset by data_location
    data_locations = df["data_location"].unique()
    for location in data_locations:
        subset = df[df["data_location"] == location]
        results_subset = results[results["data_location"] == location]
        bucket_location = subset["bucket_location"].iloc[-1]
        machine_location = subset["machine_location"].iloc[-1]
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        for i, col in enumerate(
            subset[
                [
                    "loading_time",
                    "time_for_first_sample",
                    "time_for_10_samples",
                    "nominal_speed",
                    "data_memory_usage",
                ]
            ].columns
        ):
            ax = axs[i // 3, i % 3]
            subset.boxplot(column=col, vert=True, patch_artist=True, ax=ax)
            # Plot the new results as red dots
            ax.scatter(
                [1] * len(results_subset),
                results_subset[col],
                color="red",
                label="New Results",
                marker="o",
                s=100,
                zorder=5,
            )
            ax.set_title(f"Boxplot of {col}")
            ax.set_ylabel(
                [
                    "Loading Time (s)",
                    "Time for First Sample (s)",
                    "Time for 10 Samples (s)",
                    "Nominal Speed (samples/s)",
                    "Data Memory Usage (MiB)",
                ][i]
            )
            ax.grid(axis="y")
        plt.suptitle(
            f"Boxplots of {name} - Data Location: {location} "
            f"-- Bucket: {bucket_location} -- Machine: {machine_location}"
            f"\n Sample Rate: {sr}, Split: {split}"
        )
        plt.tight_layout()
        # Ensure the fig directory exists
        os.makedirs("scripts/benchmarks/fig/loading_time/", exist_ok=True)

        plt.savefig(f"scripts/benchmarks/fig/loading_time/boxplot_{name}_{location}.png")
        logger.info(
            f"Saved plot: scripts/benchmarks/fig/loading_time/boxplot_{name}_{location}.png"
        )
        plt.close()


def plot_datasets_results_comparison(
    df: pd.DataFrame,
) -> None:
    """Plots every measured metric for all available datasets.
    Boxplots are created for each measured metric, grouped by dataset_name.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing benchmark results.
    """
    logger = logging.getLogger("datasets_comparison_plotter")
    datalocations = df["data_location"].unique()
    for location in datalocations:
        subset = df[df["data_location"] == location]
        subset = subset[subset["sample_rate"] == "default"]
        subset = subset[subset["split"] == "default"]
        fig, axs = plt.subplots(6, 1, figsize=(10, 35))
        for i, col in enumerate(
            subset[
                [
                    "loading_time",
                    "time_for_first_sample",
                    "time_for_10_samples",
                    "nominal_speed",
                    "data_memory_usage",
                    "dataset_length",
                ]
            ].columns
        ):
            ax = axs[i]
            subset.boxplot(column=col, by="dataset_name", vert=True, patch_artist=True, ax=ax)
            ax.set_title(f"{col}")
            ax.set_xlabel(None)
            ax.set_ylabel(
                [
                    "Loading Time (s)",
                    "Time for First Sample (s)",
                    "Time for 10 Samples (s)",
                    "Nominal Speed (samples/s)",
                    "Data Memory Usage (MiB)",
                    "Dataset Length (samples)",
                ][i]
            )
            ax.grid(axis="y")
            ax.tick_params(axis="x", labelsize=8, rotation=90)
        plt.suptitle(
            f"Boxplots of all features - Data Location: {location} -- Sample Rate: default "
            f"-- Split: default"
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        # Ensure the fig directory exists
        os.makedirs("scripts/benchmarks/fig/loading_time/", exist_ok=True)
        plt.savefig(f"scripts/benchmarks/fig/loading_time/boxplot_all_features_{location}.png")
        logger.info(
            f"Saved plot: scripts/benchmarks/fig/loading_time/boxplot_all_features_{location}.png"
        )
        plt.close()


def main() -> None:
    results = get_saved_results_from_cloud()
    # Replace rows where sample_rate is NaN
    results["sample_rate"].fillna("default", inplace=True)

    logger.info(f"\n{results.info()}")

    logger.info(f"\n{results.describe()}")

    plot_datasets_results_comparison(results)


if __name__ == "__main__":
    logger = logging.getLogger("loading_benchmark_plotter")
    main()
