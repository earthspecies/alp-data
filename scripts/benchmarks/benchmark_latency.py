"""Benchmark DataLoader and raw dataset iteration.

This will load individual samples from bucket or disk.


"""

import logging
import time
import warnings
from pathlib import Path

import click
import pandas
import torch
import torch.multiprocessing as mp
from benchmark_utils import (
    Timer,
    build_dataloader,
    build_raw_dataset,
    get_bucket_location,
    get_GCP_instance_location,
    save_results,
    set_logging_config,
)

from esp_data import Dataset

warnings.filterwarnings(
    "ignore",
    message=r".*end user credentials from Google Cloud SDK without a quota project.*",
    category=UserWarning,
    module=r"google\.auth\._default",
)

set_logging_config()


def benchmark_loader(
    loader: torch.utils.data.DataLoader,
    name: str,
    sleep: float = 0.0,
    log_interval: int = 10,
    max_iterations: int = 1000,
    df: pandas.DataFrame = None,
    config_info: dict = None,
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
    logger = logging.getLogger("benchmark_loader")
    logger.info(f"Benchmarking {name} dataloader: {len(loader)} batches")
    n_batches = 0
    n_samples = 0

    with Timer("total_time", store=time_stored) as t0:
        last_log_time = t0.start[0]
        for batch in loader:
            n_batches += 1
            batch_size = batch.shape[0]
            n_samples += batch_size

            if sleep > 0:
                time.sleep(sleep)

            if n_batches == 1:
                first_batch_time = t0.elapsed()[0]

            with Timer("batch_time", store=None) as t1:
                # Log stats every log_interval batches
                if n_batches > 1 and n_batches % log_interval == 0:
                    elapsed_since_start = t0.elapsed()[0]
                    elapsed_since_last = t1.start[0] - last_log_time

                    samples_per_sec = n_samples / elapsed_since_start
                    batches_per_sec = n_batches / elapsed_since_start
                    recent_samples_per_sec = (
                        (log_interval * batch_size) / elapsed_since_last
                        if elapsed_since_last > 0
                        else 0
                    )

                    logger.info(
                        f"{name} batch {n_batches}/{len(loader)}: "
                        f"{n_samples} samples in {elapsed_since_start:.2f}s "
                        f"({samples_per_sec:.2f} samples/s, {batches_per_sec:.2f} batches/s) "
                        f"[recent: {recent_samples_per_sec:.2f} samples/s]"
                    )
                    last_log_time = t1.start[0]

            if max_iterations > 0 and n_batches >= max_iterations:
                logger.info(f"Reached max_iterations={max_iterations}, stopping early.")
                break

    # Final stats
    elapsed = time_stored["total_time"][0]
    samples_per_sec = n_samples / elapsed if elapsed > 0 else 0
    batches_per_sec = n_batches / elapsed if elapsed > 0 else 0

    logger.info(
        f"{name} FINAL: {n_batches} batches, {n_samples} samples in {elapsed:.2f}s "
        f"({samples_per_sec:.2f} samples/s, {batches_per_sec:.2f} batches/s)"
        f" (Time for first batch: {first_batch_time:.2f}s)"
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
        "first_download_time": first_batch_time,
        "samples_per_sec": samples_per_sec,
    }

    # Use pandas.concat
    df = pandas.concat([df, pandas.DataFrame([new_row])], ignore_index=True)

    return df


def benchmark_raw_dataset(
    dataset: Dataset,
    name: str,
    sleep: float = 0.0,
    log_interval: int = 100,
    max_iterations: int = 1000,
    df: pandas.DataFrame = None,
    config_info: dict = None,
) -> None:
    """Benchmark direct iteration over a raw dataset."""
    time_stored = {}
    logger = logging.getLogger("benchmark_raw_dataset")
    try:
        L = len(dataset)
    except TypeError:
        L = -1

    logger.info(f"Benchmarking {name} raw dataset with length {L}")

    n_samples = 0

    with Timer("total_time", store=time_stored) as t0:
        last_log_time = t0.start
        for _ in dataset:
            n_samples += 1

            # Get the sample (this is what we're benchmarking)
            # sample = dataset[idx]  # Only works for indexed or "map-style" datasets

            if sleep > 0:
                time.sleep(sleep)

            if n_samples == 1:
                first_download_time = t0.elapsed()
                t0.start = time.perf_counter()  # Reset start time after first sample

            with Timer("sample_time", store=None) as t1:
                # Log stats every log_interval samples
                if n_samples > 1 and (n_samples % log_interval == 0):
                    elapsed_since_start = t0.elapsed()
                    elapsed_since_last = t1.start - last_log_time

                    samples_per_sec = n_samples / elapsed_since_start
                    recent_samples_per_sec = (
                        log_interval / elapsed_since_last if elapsed_since_last > 0 else 0
                    )

                    logger.info(
                        f"{name} sample {n_samples}/{L}: "
                        f"{n_samples} samples in {elapsed_since_start:.2f}s "
                        f"({samples_per_sec:.2f} samples/s) "
                        f"[recent: {recent_samples_per_sec:.2f} samples/s]"
                    )
                    last_log_time = t1.start

            if max_iterations > 0 and n_samples >= max_iterations:
                logger.info(f"Reached max_iterations={max_iterations}, stopping early.")
                break

    # Final stats
    elapsed = time_stored["total_time"]
    logger.info(
        f"{name} FINAL: {n_samples} samples in {elapsed:.2f}s "
        f"({n_samples / elapsed:.2f} samples/s)\n"
        f"(shard (or first sample) download time: {first_download_time:.2f}s)"
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
        "first_download_time": first_download_time,
    }

    # Use pandas.concat
    df = pandas.concat([df, pandas.DataFrame([new_row])], ignore_index=True)

    return df


@click.command()
@click.option(
    "--config_path",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
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
    "--raw-dataset",
    is_flag=True,
    help="Benchmark raw dataset iteration instead of DataLoader",
)
@click.option(
    "--max-iterations",
    type=int,
    default=1000,
    show_default=True,
    help="Maximum number of iterations through samples to run (default: 1000)",
)
@click.option(
    "--num-workers",
    type=int,
    default=8,
    show_default=True,
    help="Number of worker threads for DataLoader (default: 8)",
)
@click.option(
    "--batch-size",
    type=int,
    default=128,
    show_default=True,
    help="Batch size for DataLoader (default: 128)",
)
@click.option(
    "--dataset-name",
    type=str,
    default=None,
    help="Name of the dataset to benchmark (used if --from-config is not set)",
    show_default=True,
)
@click.option(
    "--prefetch-factor",
    type=int,
    default=None,
    show_default=True,
    help="Prefetch factor for DataLoader (default: None)",
)
@click.option(
    "--save",
    is_flag=True,
    help="Save results to CSV in GCS bucket after benchmarking",
)
def main(
    config_path: Path,
    data_location: str,
    sleep: float,
    log_interval: int,
    raw_dataset: bool,
    max_iterations: int,
    num_workers: int,
    batch_size: int,
    dataset_name: str,
    prefetch_factor: int,
    save: bool,
) -> None:
    time_stored = {}
    with Timer("loading_time", store=time_stored):
        train_ds, raw_config = build_raw_dataset(
            config_path, data_location, dataset_name=dataset_name
        )
    logger.info(f"Loaded dataset in {time_stored['loading_time'][0]:.2f} seconds")

    # Initialize empty DataFrame to collect all results
    all_results = pandas.DataFrame()

    # Get bucket location (zone and region) and machine location
    if data_location == "bucket":
        bucket_location = get_bucket_location(raw_config["data_root"])
        logger.info(f"Bucket location for {raw_config['data_root']}: zone{bucket_location}")
        machine_location = get_GCP_instance_location()

    if raw_dataset:
        config_info = {
            "data_location": data_location,
            **raw_config,
            "loading_time": time_stored["loading_time"][0],
            "raw_dataset": True,
            "machine_location": machine_location if data_location == "bucket" else None,
            "bucket_location": bucket_location if data_location == "bucket" else None,
        }
        result_df = benchmark_raw_dataset(
            train_ds,
            train_ds.info.name,
            sleep,
            log_interval,
            max_iterations,
            config_info=config_info,
        )

        # Add to our collection of all results
        all_results = pandas.concat([all_results, result_df], ignore_index=True)

    else:
        prefetch_factor = None if num_workers == 0 else prefetch_factor
        if prefetch_factor <= 0:
            logger.warning(f"Invalid prefetch_factor={prefetch_factor}, setting to None")
            prefetch_factor = None
        logger.info(
            f"Running benchmark with num_workers={num_workers}, batch_size={batch_size}, "
            f"prefetch_factor={prefetch_factor}"
        )
        config_info = {
            "data_location": data_location,
            **raw_config,
            "loading_time": time_stored["loading_time"],
            "raw_dataset": False,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "prefetch_factor": prefetch_factor,
            "machine_location": machine_location if data_location == "bucket" else None,
            "bucket_location": bucket_location if data_location == "bucket" else None,
        }

        train_dl = build_dataloader(
            train_ds,
            num_workers=num_workers,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
        )

        # Run benchmark and collect results
        result_df = benchmark_loader(
            train_dl,
            name=train_ds.info.name,
            sleep=sleep,
            log_interval=log_interval,
            max_iterations=max_iterations / batch_size,
            config_info=config_info,
        )

        # Add to our collection of all results
        all_results = pandas.concat([all_results, result_df], ignore_index=True)

    if save:
        # Define the CSV file path in the bucket
        csv_path = "gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_dataset.csv"

        # Save all results to the single CSV file in the bucket
        save_results(all_results, csv_path)


if __name__ == "__main__":
    # Use 'spawn' to be compatible with DataLoader multiprocessing.
    # When running the benchmark with data_location set on 'bucket'
    # if we don't use the following line we get a freeze for num_workers > 0.
    # Here is more info about 'spawn' vs 'fork' methods: https://docs.pytorch.org/docs/stable/notes/multiprocessing.html
    mp.set_start_method("spawn", force=True)

    logger = logging.getLogger("main")
    main()
