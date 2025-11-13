import logging

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
def main(number_of_samples: int, parameter: str) -> None:
    """Retrieve and plot benchmark latency results from a CSV file in a GCS bucket."""
    df = pd.read_csv("gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_latency.csv")
    if number_of_samples is not None:
        df = df.tail(number_of_samples)

    name = df["dataset_name"].iloc[0]
    parameters = ["num_workers", "batch_size", "prefetch_factor"]

    other_params = [p for p in parameters if p != parameter]
    param1 = other_params[0]
    value1 = df[param1].iloc[0]
    param2 = other_params[1]
    value2 = df[param2].iloc[0]

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    axs[0].set_title("Nominal Speed")
    axs[0].set_xlabel(parameter)
    axs[0].set_ylabel("Nominal Speed (samples/second)")
    axs[0].plot(
        df.sort_values(parameter)[parameter], df.sort_values(parameter)["nominal_speed"], marker="o"
    )
    axs[0].grid(True)

    axs[1].set_title("Total Time")
    axs[1].set_xlabel(parameter)
    axs[1].set_ylabel("Total Time (seconds)")
    axs[1].plot(
        df.sort_values(parameter)[parameter], df.sort_values(parameter)["total_time"], marker="o"
    )
    axs[1].grid(True)
    plt.suptitle(
        f"Latency {parameter.capitalize()} Results - {param1.capitalize()} = {value1}, "
        f"{param2.capitalize()} = {value2}"
    )
    plt.tight_layout()
    plt.savefig(f"scripts/benchmarks/fig/{name}_latency_{parameter}_{value1}_{value2}.png")


if __name__ == "__main__":
    logger = logging.getLogger("latency_results_plotter")
    main()
