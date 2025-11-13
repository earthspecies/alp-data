"""Benchmark loading time of datasets. Script to be run regularly to track performance over time."""

import logging
import warnings
from pathlib import Path

import click
import numpy as np
import pandas
from benchmark_utils import (
    Timer,
    build_raw_dataset,
    get_bucket_location,
    get_GCP_instance_location,
    save_results,
    set_logging_config,
)
from memory_profiler import memory_usage

warnings.filterwarnings(
    "ignore",
    message=r".*end user credentials from Google Cloud SDK without a quota project.*",
    category=UserWarning,
    module=r"google\.auth\._default",
)

set_logging_config()


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
    "--dataset-name",
    type=str,
    default=None,
    help="Name of the dataset to benchmark",
    show_default=True,
)
@click.option(
    "--save",
    is_flag=True,
    default=False,
    help="Whether to save the benchmark results to a CSV file.",
    show_default=True,
)
def main(
    data_location: str,
    config_path: Path,
    dataset_name: str,
    save: bool,
) -> None:
    times_stored = {}
    with Timer("loading_time", store=times_stored):
        mem_profile, (ds, raw_config) = memory_usage(
            (
                build_raw_dataset,
                (config_path, data_location, dataset_name),
            ),
            max_iterations=1,
            retval=True,
            interval=0.1,
        )

    logger.info(
        f"Loaded dataset in {times_stored['loading_time'][0]:.2f} seconds from {data_location}\n"
        f"CPU time: {times_stored['loading_time'][1]:.2f} seconds"
    )
    mem_profile = np.array(mem_profile)
    peak_mem = max(mem_profile) - min(mem_profile)
    logger.info(f"Peak memory usage during loading (dataset memory usage): {peak_mem:.2f} MiB")
    logger.info(f"Imports memory usage: {mem_profile[0]:.2f} MiB")

    # Dataset length
    data_length = len(ds)
    logger.info(f"Dataset length: {data_length} samples")

    # Measure time to first sample
    with Timer("time_to_first_sample", store=times_stored):
        next(iter(ds))
    logger.info(
        f"Time to first sample: {times_stored['time_to_first_sample'][0]:.2f} seconds\n"
        f" CPU time: {times_stored['time_to_first_sample'][1]:.2f} seconds"
    )

    # Measure time for 10 following samples
    with Timer("time_for_10_samples", store=times_stored):
        for _ in range(10):
            next(iter(ds))
    logger.info(
        f"Time for 10 following samples: {times_stored['time_for_10_samples'][0]:.2f} seconds "
        f"{times_stored['time_for_10_samples'][0] / 10:.2f} seconds/sample\n"
        f" CPU time: {times_stored['time_for_10_samples'][1]:.2f} seconds "
        f"{times_stored['time_for_10_samples'][1] / 10:.2f} seconds/sample"
    )

    if data_location == "bucket":
        bucket_location = get_bucket_location(raw_config["data_root"])
        logger.info(f"Bucket location for {raw_config['data_root']}: zone{bucket_location}")
        machine_location = get_GCP_instance_location()

    if save:
        csv_path = (
            "gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_loading_time.csv"
        )

        # Create a DataFrame for the new results it should contain config info
        all_results = pandas.DataFrame(
            [
                {
                    "data_location": data_location,
                    "bucket_location": bucket_location if data_location == "bucket" else None,
                    "machine_location": machine_location if data_location == "bucket" else None,
                    "dataset_length": data_length,
                    "loading_time_seconds": times_stored["loading_time"][0],
                    "time_to_first_sample_seconds": times_stored["time_to_first_sample"][0],
                    "time_for_10_samples_seconds": times_stored["time_for_10_samples"][0],
                    "nominal_speed (samples/second)": 10.0
                    / (times_stored["time_for_10_samples"][0]),
                    "peak_memory_usage_mib": peak_mem,
                    "imports_memory_usage_mib": mem_profile[0],
                    "timestamp": pandas.Timestamp.now(),
                    **raw_config,
                }
            ]
        )

        save_results(all_results, csv_path)


if __name__ == "__main__":
    logger = logging.getLogger("main")
    main()
