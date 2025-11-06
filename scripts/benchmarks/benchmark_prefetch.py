"""Benchmark DataLoader Prefetch Parameter.

This script benchmarks the performance of PyTorch DataLoaders with different
prefetch settings.

"""

import logging
import time
import warnings
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas
import torch
import torch.multiprocessing as mp
from benchmark_utils import (
    Timer,
    build_dataloader,
    build_raw_dataset,
    save_results,
    set_logging_config,
)

from esp_data.io.filesystem import filesystem_from_path

warnings.filterwarnings(
    "ignore",
    message=r".*end user credentials from Google Cloud SDK without a quota project.*",
    category=UserWarning,
    module=r"google\.auth\._default",
)

set_logging_config()


def benchmark_prefetch(
    loader: torch.utils.data.DataLoader,
    name: str,
    sleep: float = 0.0,
    log_interval: int = 10,
    max_iterations: int = 1000,
    df: pandas.DataFrame = None,
    config_info: dict = None,
    prefetch_factor: int = None,
) -> pandas.DataFrame:
    """Benchmark a DataLoader by iterating through it and logging stats.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        The DataLoader to benchmark.
    name : str
        Name of the DataLoader for logging purposes.
    sleep : float, optional
        Optional sleep time (in seconds) per batch to simulate work, by default 0.0
    log_interval : int, optional
        Log stats every N batches, by default 10
    max_iterations : int, optional
        Maximum number of iterations to run, by default 1000
    df : pandas.DataFrame, optional
        DataFrame to store results, by default None
    config_info : dict, optional
        Configuration information to store with results

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the benchmark results.
    """
    time_stored = {}
    logger = logging.getLogger("benchmark_prefetch")
    logger.info(f"Benchmarking prefetch_factors on {name} dataloader: {len(loader)} batches.")

    n_batches = 0
    n_samples = 0

    # List to store values
    n_batches_list = []
    n_samples_list = []
    elapsed_time_list = []
    samples_per_sec_list = []

    with Timer("total_time", store=time_stored) as t0:
        last_log_time = t0.start
        for batch in loader:
            n_batches += 1
            batch_size = batch.shape[0]
            n_samples += batch_size

            if n_batches == 1:
                shard_download_time = t0.elapsed()
                t0.start = time.perf_counter()  # Reset start time after first batch

            if sleep > 0:
                time.sleep(sleep)  # Simulate work by sleeping

            with Timer("batch_time", store=None) as t1:
                # Log stats every log_interval batches
                if n_batches % log_interval == 0:
                    elapsed_since_start = t0.elapsed()
                    elapsed_since_last = t1.start - last_log_time

                    samples_per_sec = n_samples / elapsed_since_start
                    batches_per_sec = n_batches / elapsed_since_start
                    recent_samples_per_sec = (
                        (log_interval * batch_size) / elapsed_since_last
                        if elapsed_since_last > 0
                        else 0
                    )
                    # Store values in lists
                    n_batches_list.append(n_batches)
                    n_samples_list.append(n_samples)
                    elapsed_time_list.append(elapsed_since_start)
                    samples_per_sec_list.append(samples_per_sec)

                    logger.info(
                        f"{name} batch {n_batches}/{len(loader)}: "
                        f"{n_samples} samples in {elapsed_since_start:.2f}s "
                        f"({samples_per_sec:.2f} samples/s, {batches_per_sec:.2f} batches/s) "
                        f"[recent: {recent_samples_per_sec:.2f} samples/s]"
                    )
                    last_log_time = t1.start

            if max_iterations > 0 and n_batches >= max_iterations:
                logger.info(f"Reached max_iterations={max_iterations}, stopping early.")
                break

    # Final stats
    elapsed = time_stored["total_time"]
    samples_per_sec = n_samples / elapsed if elapsed > 0 else 0
    batches_per_sec = n_batches / elapsed if elapsed > 0 else 0

    # Store final values in lists
    n_batches_list.append(n_batches)
    n_samples_list.append(n_samples)
    elapsed_time_list.append(elapsed)
    samples_per_sec_list.append(samples_per_sec)

    logger.info(
        f"{name} FINAL: {n_batches} batches, {n_samples} samples in {elapsed:.2f}s "
        f"({samples_per_sec:.2f} samples/s, {batches_per_sec:.2f} batches/s)"
        f" (shard download time for first batch: {shard_download_time:.2f}s)"
    )

    # Create or append to DataFrame with results
    if df is None:
        df = pandas.DataFrame()

    # Add the benchmark results to the DataFrame
    new_row = {
        **(config_info or {}),  # Add configuration info if provided
        "date_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "dataset_name": name,
        "max_iterations": max_iterations,
        "n_samples": n_samples,
        "elapsed_time": elapsed,
        "samples_per_sec": samples_per_sec,
    }

    # Use pandas.concat
    df = pandas.concat([df, pandas.DataFrame([new_row])], ignore_index=True)

    return df, n_samples_list, samples_per_sec_list, elapsed_time_list, n_batches_list


@click.command()
@click.option(
    "--config_path",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="scripts/benchmarks/benchmark_config.yaml",
    show_default=True,
    help="Path to the config file",
)
@click.option(
    "--data-location",
    type=str,
    default="bucket",
    help="Data location to benchmark (e.g., 'nfs' or 'bucket')",
    show_default=True,
)
@click.option(
    "--sleep",
    type=float,
    default=0.0,
    show_default=True,
    help="Optional sleep (in seconds) per batch to simulate work (default: 0)",
)
@click.option(
    "--log-interval",
    type=int,
    default=10,
    show_default=True,
    help="Log stats every N batches (default: 10)",
)
@click.option(
    "--max-iterations",
    type=int,
    default=1000,
    show_default=True,
    help="Maximum number of iterations through samples to run (default: 1000)",
)
@click.option(
    "--num-workers-list",
    type=str,
    default="2,4,8",
    show_default=True,
    help="Number of worker threads for DataLoader (default: [2, 4, 8])",
)
@click.option(
    "--batch-size",
    type=int,
    default=128,
    show_default=True,
    help="Batch sizes for DataLoader (default: 128)",
)
@click.option(
    "--plots",
    is_flag=True,
    help="Whether to generate and save plots of the results (default: False)",
)
def main(
    config_path: Path,
    data_location: str,
    sleep: float,
    log_interval: int,
    max_iterations: int,
    num_workers_list: list[int],
    batch_size: int,
    plots: bool = False,
) -> None:
    time_stored = {}

    with Timer("loading_time", store=time_stored):
        train_ds, raw_config = build_raw_dataset(config_path, data_location)
    logger.info(f"Loaded dataset in {time_stored['loading_time']:.2f} seconds")

    # Convert string inputs to lists of integers
    if isinstance(num_workers_list, str):
        num_workers_list = [int(x) for x in num_workers_list.split(",")]

    # Initialize empty DataFrame to collect all results
    all_results = pandas.DataFrame()

    # Define the CSV file path in the bucket
    csv_path = "gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/prefetch_test.csv"

    prefetch_factors = [None, 1, 2, 4]  # Example prefetch factors

    nw = 4
    plot_dict = {}
    for pf in prefetch_factors:
        logger.info(
            f"Running benchmark with num_workers={nw},"
            f" batch_size={batch_size}, prefetch_factor={pf}"
        )

        config_info = {
            "data_location": data_location,
            **raw_config,
            "loading_time": time_stored["loading_time"],
            "raw_dataset": False,
            "num_workers": nw,
            "batch_size": batch_size,
            "prefetch_factor": pf,
        }

        train_dl = build_dataloader(
            train_ds,
            num_workers=nw,
            batch_size=batch_size,
            prefetch_factor=pf,
        )

        # Run benchmark and collect results
        result_df, n_samples_list, samples_per_sec_list, elapsed_time_list, n_batches_list = (
            benchmark_prefetch(
                train_dl,
                name=train_ds.info.name,
                sleep=sleep,
                log_interval=log_interval,
                max_iterations=max_iterations / batch_size,
                config_info=config_info,
                prefetch_factor=pf,
            )
        )

        # Add to our collection of all results
        all_results = pandas.concat([all_results, result_df], ignore_index=True)

        plot_dict[f"prefetch_{pf}"] = {
            "n_samples_list": n_samples_list,
            "samples_per_sec_list": samples_per_sec_list,
            "elapsed_time_list": elapsed_time_list,
            "n_batches_list": n_batches_list,
        }

    # Save all results to the single CSV file in the bucket

    save_results(all_results, csv_path)

    if plots:
        for prefetch_factor, data in plot_dict.items():
            # Plot some graphs of the results lists
            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(data["n_batches_list"], data["samples_per_sec_list"], marker="o")
            plt.title("Samples per Second vs Batches")
            plt.xlabel("Number of Batches")
            plt.ylabel("Samples per Second")
            plt.grid(True)
            plt.subplot(2, 1, 2)
            plt.plot(data["elapsed_time_list"], data["n_batches_list"], marker="o")
            plt.title("Batches vs Elapsed Time")
            plt.xlabel("Elapsed Time (s)")
            plt.ylabel("Batches")
            plt.grid(True)
            plt.tight_layout()
            with filesystem_from_path(
                f"gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_{train_ds.info.name}_prefetch_{prefetch_factor}.png"
            ) as f:
                plt.savefig(f)
            plt.close()

        # Plot comparison of prefetch factors
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        for pf, data in plot_dict.items():
            plt.plot(data["n_batches_list"], data["samples_per_sec_list"], marker="o", label=pf)
        plt.title("Samples per Second vs Batches")
        plt.xlabel("Number of Batches")
        plt.ylabel("Samples per Second")
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        for pf, data in plot_dict.items():
            plt.plot(data["elapsed_time_list"], data["n_batches_list"], marker="o", label=pf)
        plt.title("Batches vs Elapsed Time")
        plt.xlabel("Elapsed Time (s)")
        plt.ylabel("Batches")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        with filesystem_from_path(
            f"gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_{train_ds.info.name}_prefetch_comparison.png"
        ) as f:
            plt.savefig(f)
        plt.close()


if __name__ == "__main__":
    # Use 'spawn' to be compatible with DataLoader multiprocessing.
    # When running the benchmark with data_location set on 'bucket'
    # if we don't use the following line we get a freeze for num_workers > 0.
    mp.set_start_method("spawn", force=True)
    logger = logging.getLogger("main")
    main()
