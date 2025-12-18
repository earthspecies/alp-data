import logging
import os

import click
import matplotlib.pyplot as plt
import pandas as pd
from benchmark_utils import set_logging_config

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
    if array_exp:
        df = pd.read_csv(
            "gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_latency_array.csv"
        )
        if number_of_samples is not None:
            df = df.tail(number_of_samples)

        name = df["dataset_name"].iloc[0]
        num_workers = df["num_workers"].iloc[0]
        batch_size = df["batch_size"].iloc[0]
        prefetch_factor = df["prefetch_factor"].iloc[0]
        num_locations = len(df["data_location"].unique())
        fig, axs = plt.subplots(num_locations * 2, figsize=(8, 6 * num_locations))
        fig.suptitle(
            f"Latency {parameter.capitalize()} Results - "
            f"Num Workers = {num_workers}, Batch Size = {batch_size},"
            f"Prefetch Factor = {prefetch_factor}"
        )
        for i, data_location in enumerate(df["data_location"].unique()):
            df_location = df[df["data_location"] == data_location]
            df_speed = (
                df_location["nominal_speed"].groupby(df_location[parameter]).mean().reset_index()
            )
            df_time = df_location["total_time"].groupby(df_location[parameter]).mean().reset_index()
            axs[0 + i * 2].plot(
                df_speed[parameter],
                df_speed["nominal_speed"],
                marker="o",
                label=data_location,
            )
            axs[0 + i * 2].set_xlabel(parameter)
            axs[0 + i * 2].set_xticks(df_speed[parameter])
            axs[0 + i * 2].set_ylabel("Nominal Speed (samples/second)")
            # set grid
            axs[0 + i * 2].grid(True)
            axs[0 + i * 2].set_title(f"Data Location: {data_location}")
            axs[1 + i * 2].plot(
                df_time[parameter],
                df_time["total_time"],
                marker="o",
                label=data_location,
            )
            axs[1 + i * 2].set_xlabel(parameter)
            axs[1 + i * 2].set_xticks(df_time[parameter])
            axs[1 + i * 2].set_ylabel("Total Time (seconds)")
            axs[1 + i * 2].grid(True)
            axs[1 + i * 2].set_title(f"Data Location: {data_location}")
        fig.tight_layout()
        # Ensure the fig directory exists
        os.makedirs("scripts/benchmarks/fig/latency", exist_ok=True)
        plt.savefig(f"scripts/benchmarks/fig/latency/{name}_latency_{parameter}_array_exp.png")

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
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        axs[0].set_title("Nominal Speed")
        axs[0].set_xlabel(parameter)
        axs[0].set_xticks(df.sort_values(parameter)[parameter])
        axs[0].set_xticklabels(labels)
        axs[0].set_ylabel("Nominal Speed (samples/second)")
        axs[0].plot(
            df.sort_values(parameter)[parameter],
            df.sort_values(parameter)["nominal_speed"],
            marker="o",
        )
        axs[0].grid(True)

        axs[1].set_title("Total Time")
        axs[1].set_xlabel(parameter)
        axs[1].set_xticks(df.sort_values(parameter)[parameter])
        axs[1].set_xticklabels(labels)
        axs[1].set_ylabel("Total Time (seconds)")
        axs[1].plot(
            df.sort_values(parameter)[parameter],
            df.sort_values(parameter)["total_time"],
            marker="o",
        )
        axs[1].grid(True)
        plt.suptitle(
            f"Latency {parameter.capitalize()} Results - {param1.capitalize()} = {value1}, "
            f"{param2.capitalize()} = {value2}"
        )
        plt.tight_layout()
        # Ensure the fig directory exists
        os.makedirs("scripts/benchmarks/fig/latency", exist_ok=True)
        plt.savefig(
            f"scripts/benchmarks/fig/latency/{name}_latency_{parameter}_{value1}_{value2}.png"
        )


if __name__ == "__main__":
    logger = logging.getLogger("latency_results_plotter")
    main()
