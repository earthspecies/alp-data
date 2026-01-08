import logging

import click
import matplotlib.pyplot as plt
import pandas as pd
from benchmark_utils import filter_cloud_warnings, save_and_log, set_logging_config

plt.rcParams.update({"font.size": 16})

filter_cloud_warnings()

set_logging_config()


@click.command()
@click.option(
    "--number-of-samples",
    "-n",
    type=int,
    default=None,
    show_default=True,
    help="Number of samples to plot latency results for",
)
@click.option(
    "--parameter",
    "-p",
    type=str,
    default="num_workers",
    show_default=True,
    help="Parameter to plot against latency results",
)
@click.option(
    "--array-exp",
    is_flag=True,
    help="Indicates this is part of a job array experiment",
)
def main(number_of_samples: int, parameter: str, array_exp: bool) -> None:
    """Retrieve and plot benchmark latency results from a CSV file in a GCS bucket."""

    metrics = [  # title, column name, y-axis label
        ("Nominal Speed", "nominal_speed", "Nominal Speed (samples/second)"),
        ("Total Time", "total_time", "Total Time (seconds)"),
        ("First batch download time", "first_download_time", "First Batch Download Time (seconds)"),
    ]

    if array_exp:
        df = pd.read_csv(
            "gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_latency_array.csv"
        )
        if number_of_samples is not None:  # only keep the last n samples i.e. last experiment
            df = df.tail(number_of_samples)

        name = df["dataset_name"].iloc[0]
        num_workers = df["num_workers"].iloc[0]
        batch_size = df["batch_size"].iloc[0]
        prefetch_factor = df["prefetch_factor"].iloc[0]
        num_locations = len(df["data_location"].unique())
        fig, axs = plt.subplots(num_locations * 3, figsize=(8, 6 * num_locations))
        fig.suptitle(
            f"Latency {parameter.capitalize()} Results - "
            f"Num Workers = {num_workers}, Batch Size = {batch_size},"
            f"Prefetch Factor = {prefetch_factor}"
        )

        unique_locations = df["data_location"].unique()
        for i, data_location in enumerate(unique_locations):
            df_location = df[df["data_location"] == data_location]
            for j, (title, col, ylabel) in enumerate(metrics):
                df_metric = df_location.groupby(parameter)[col].mean().reset_index()
                ax_idx = i * len(metrics) + j
                axs[ax_idx].plot(
                    df_metric[parameter],
                    df_metric[col],
                    marker="o",
                    label=data_location,
                )
                axs[ax_idx].set_xlabel(parameter)
                axs[ax_idx].set_xticks(df_metric[parameter])
                axs[ax_idx].set_ylabel(ylabel)
                axs[ax_idx].grid(True)
                axs[ax_idx].set_title(f"{title} - Data Location: {data_location}")
        fig.tight_layout()
        save_and_log(f"scripts/benchmarks/fig/latency/{name}_latency_{parameter}_array_exp.png")

    else:
        df = pd.read_csv(
            "gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_latency.csv"
        )
        if number_of_samples is not None:
            df = df.tail(number_of_samples)

        name = df["dataset_name"].iloc[0]
        parameters = ["num_workers", "batch_size", "prefetch_factor"]

        other_params = [p for p in parameters if p != parameter]
        param1 = other_params[0]
        value1 = df[param1].iloc[0]
        param2 = other_params[1]
        value2 = df[param2].iloc[0]

        # replace Nan with -1 to be able to sort, but label should show "None"
        df.replace({float("nan"): -1}, inplace=True)
        print(df[parameter])
        labels = [x if x != -1 else "None" for x in df.sort_values(parameter)[parameter]]
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))

        sorted_df = df.sort_values(parameter)
        x = sorted_df[parameter]
        for i, (title, col, ylabel) in enumerate(metrics):
            axs[i].set_title(title)
            axs[i].set_xlabel(parameter)
            axs[i].set_xticks(x)
            axs[i].set_xticklabels(labels)
            axs[i].set_ylabel(ylabel)
            axs[i].plot(x, sorted_df[col], marker="o")
            axs[i].grid(True)
        plt.suptitle(
            f"Latency {parameter.capitalize()} Results - {param1.capitalize()} = {value1}, "
            f"{param2.capitalize()} = {value2}"
        )
        plt.tight_layout()
        save_and_log(
            f"scripts/benchmarks/fig/latency/{name}_latency_{parameter}_{value1}_{value2}.png"
        )


if __name__ == "__main__":
    logger = logging.getLogger("latency_results_plotter")
    main()
