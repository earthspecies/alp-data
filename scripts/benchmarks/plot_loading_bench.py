import logging

import matplotlib.pyplot as plt
import pandas as pd
from benchmark_utils import filter_cloud_warnings, save_and_log, set_logging_config

plt.rcParams.update({"font.size": 16})

filter_cloud_warnings()

set_logging_config()


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

    df = pd.read_csv(
        "gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_loading_time.csv"
    )
    name = results["dataset_name"].iloc[0]
    split = results["split_name"].iloc[0]
    df = df[df["dataset_name"] == name]
    df = df[df["split_name"] == split]
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
        fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey="row")
        boxplot_cols = [
            "loading_time",
            "time_for_first_sample",
            "time_for_10_samples",
            "nominal_speed",
        ]
        units = [
            "seconds",
            "seconds",
            "seconds",
            "samples/second",
        ]
        for i, col in enumerate(subset[boxplot_cols].columns):
            unit = units[i]
            ax = axs[i // 3, i % 3]
            filtered_subset = pd.DataFrame()
            # Remove outliers using the IQR method
            Q1 = subset[col].quantile(0.25)
            Q3 = subset[col].quantile(0.75)
            IQR = Q3 - Q1
            filtered_subset = subset[
                (subset[col] >= Q1 - 1.5 * IQR) & (subset[col] <= Q3 + 1.5 * IQR)
            ]
            filtered_subset.boxplot(column=col, vert=True, patch_artist=True, ax=ax)
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
            ax.set_title(f"{col}")
            ax.set_xlabel(None)
            ax.set_ylabel(f"{unit}")
            ax.grid(axis="y")
            # hide non used subplots
        if len(boxplot_cols) < 6:
            for j in range(len(boxplot_cols), 6):
                axs[j // 3, j % 3].axis("off")

        plt.suptitle(
            f"{name} - Data Location: {location} "
            f"-- Bucket: {bucket_location} -- Machine: {machine_location}"
            f"\n Sample Rate: {sr}, Split: {split}"
        )
        plt.tight_layout()
        save_and_log(f"scripts/benchmarks/fig/loading_time/boxplot_{name}_{location}.png")

        metrics = [
            ["nominal_speed", "samples/second"],
            ["loading_time", "seconds"],
        ]
        # Plot a evolution of metrics over time
        fig2, ax2 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for i, metric in enumerate(metrics):
            unit = metric[1]
            metric = metric[0]
            ax = ax2[i]
            # Convert timestamp to datetime (round to day for better visualization)
            subset["timestamp"] = pd.to_datetime(subset["timestamp"]).dt.floor("d")
            results_subset["timestamp"] = pd.to_datetime(results_subset["timestamp"]).dt.floor("d")
            subset_sorted = subset.sort_values(by="timestamp")
            # if same date -> take the mean of the nominal_speed
            subset_metric = subset_sorted.groupby("timestamp")[metric].mean().reset_index()

            results_metric = results_subset.groupby("timestamp")[metric].mean().reset_index()

            ax.plot(
                subset_metric["timestamp"],
                subset_metric[metric],
                marker="o",
                label="Previous Results",
            )
            ax.plot(
                results_metric["timestamp"],
                results_metric[metric],
                marker="o",
                color="red",
                label="New Results",
            )
            ax.set_title(f"{metric} over time")
            ax.set_xlabel(None)
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylabel(f"{unit}")
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        save_and_log(
            f"scripts/benchmarks/fig/loading_time/{metric}_evolution_{name}_{location}.png"
        )


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
    datalocations = df["data_location"].unique()
    for location in datalocations:
        subset = df[df["data_location"] == location]
        subset = subset[subset["sample_rate"] == "default"]
        subset = subset[subset["split_config"] == "default"]
        fig, axs = plt.subplots(2, 3, figsize=(30, 12), sharey="row")
        boxplot_cols = [
            "loading_time",
            "time_for_first_sample",
            "time_for_10_samples",
            "nominal_speed",
        ]
        units = [
            "seconds",
            "seconds",
            "seconds",
            "samples/second",
        ]
        dataset_names = subset["dataset_name"].unique()
        dataset_number_map = {name: idx + 1 for idx, name in enumerate(dataset_names)}

        for i, col in enumerate(subset[boxplot_cols].columns):
            ax_id = i // 3, i % 3
            ax = axs[ax_id]
            # Remove outliers per dataset using the IQR method
            filtered_subset = pd.DataFrame()
            for dataset in dataset_names:
                ds = subset[subset["dataset_name"] == dataset]
                Q1 = ds[col].quantile(0.25)
                Q3 = ds[col].quantile(0.75)
                IQR = Q3 - Q1
                ds_filtered = ds[(ds[col] >= Q1 - 1.5 * IQR) & (ds[col] <= Q3 + 1.5 * IQR)]
                ds_filtered = ds_filtered.copy()
                ds_filtered["dataset_number"] = dataset_number_map[dataset]
                filtered_subset = pd.concat([filtered_subset, ds_filtered], ignore_index=True)
            filtered_subset.boxplot(
                column=col, by="dataset_number", vert=True, patch_artist=True, ax=ax
            )
            ax.set_title(f"{col}")
            ax.set_xlabel("Dataset Number")
            ax.set_ylabel(units[i])
            ax.grid(axis="y")
            ax.tick_params(axis="x", labelsize=12, rotation=0)
            # Set x-ticks to dataset numbers
            ax.set_xticks(list(dataset_number_map.values()))
            ax.set_xticklabels(list(dataset_number_map.values()))

        if len(boxplot_cols) < 6:
            for j in range(len(boxplot_cols), 6):
                axs[j // 3, j % 3].axis("off")
        plt.suptitle(
            f"Boxplots of all features - Data Location: {location} -- Sample Rate: default "
            f"-- Split: default"
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        save_and_log(f"scripts/benchmarks/fig/loading_time/boxplot_all_features_{location}.png")


def main() -> None:
    results = pd.read_csv(
        "gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_loading_time.csv"
    )

    logger.info(f"\n{results.info()}")

    logger.info(f"\n{results.describe()}")

    plot_datasets_results_comparison(results)


if __name__ == "__main__":
    logger = logging.getLogger("loading_benchmark_plotter")
    main()
