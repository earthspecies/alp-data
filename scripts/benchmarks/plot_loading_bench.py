import logging

import matplotlib.pyplot as plt
import pandas as pd
from benchmark_utils import set_logging_config

set_logging_config()


def get_results() -> pd.DataFrame:
    """Retrieve benchmark loading time results from a CSV file in a GCS bucket.
    Returns
    -------
    A pandas DataFrame containing the benchmark loading time results.
    """
    df = pd.read_csv(
        "gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_loading_time.csv"
    )
    return df


def plot_results(results: pd.DataFrame) -> None:
    """Create a complete visualization for a specific dataset.
    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame containing benchmark results for a specific dataset.
    """
    logger = logging.getLogger("results_plotter")

    df = get_results()
    name = results["dataset_name"].iloc[0]
    df = df[df["dataset_name"] == name]
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
                    "loading_time_seconds",
                    "time_to_first_sample_seconds",
                    "time_for_10_samples_seconds",
                    "nominal_speed (samples/second)",
                    "peak_memory_usage_mib",
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
            ax.set_ylabel("Values")
            ax.grid(axis="y")
            col_name = col.replace(" ", "_").replace("/", "_")
        plt.suptitle(
            f"Boxplots of {name} - Data Location: {location} "
            f"-- Bucket: {bucket_location} -- Machine: {machine_location}"
        )
        plt.tight_layout()
        plt.savefig(f"scripts/benchmarks/fig/boxplot_{name}_{location}_{col_name}.png")
        logger.info(f"Saved plot: scripts/benchmarks/fig/boxplot_{name}_{location}_{col_name}.png")
        plt.close()


def plot_feature(
    df: pd.DataFrame,
) -> None:
    """Plot a specific feature for different datasets.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing benchmark results.
    """
    logger = logging.getLogger("feature_plotter")
    datalocations = df["data_location"].unique()
    for location in datalocations:
        subset = df[df["data_location"] == location]
        subset = subset[subset["sample_rate"] == "default"]
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        for i, col in enumerate(
            subset[
                [
                    "loading_time_seconds",
                    "time_to_first_sample_seconds",
                    "time_for_10_samples_seconds",
                    "nominal_speed (samples/second)",
                    "peak_memory_usage_mib",
                    "dataset_length",
                ]
            ].columns
        ):
            ax = axs[i // 3, i % 3]
            subset.boxplot(column=col, by="dataset_name", vert=True, patch_artist=True, ax=ax)
            ax.set_title(f"{col}")
            ax.set_xlabel("Dataset Name")
            ax.set_ylabel(col)
            ax.grid(axis="y")
            ax.tick_params(axis="x", rotation=45)
        plt.suptitle(f"Boxplots of all features - Data Location: {location}")
        plt.tight_layout()
        plt.savefig(f"scripts/benchmarks/fig/boxplot_all_features_{location}.png")
        logger.info(f"Saved plot: scripts/benchmarks/fig/boxplot_all_features_{location}.png")
        plt.close()


def main() -> None:
    results = get_results()
    # Replace rows where sample_rate is NaN
    results["sample_rate"].fillna("default", inplace=True)

    logger.info(results.info())
    logger.info(results.describe())

    plot_feature(results)


if __name__ == "__main__":
    logger = logging.getLogger("loading_benchmark_plotter")
    main()
