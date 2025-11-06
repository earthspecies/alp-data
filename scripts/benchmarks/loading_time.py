"""Benchmark loading time of datasets. Script to be run regularly to track performance over time."""

import logging
import warnings
from pathlib import Path

import click
import pandas
from benchmark_utils import Timer, build_raw_dataset, save_results, set_logging_config

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
def main(
    data_location: str,
    config_path: Path,
) -> None:
    times_stored = {}
    with Timer("loading_time", store=times_stored):
        ds, raw_config = build_raw_dataset(config_path, data_location)
    logger.info(
        f"Loaded dataset in {times_stored['loading_time']:.2f} seconds from {data_location}"
    )

    # Measure time to first sample
    with Timer("time_to_first_sample", store=times_stored):
        next(iter(ds))
    logger.info(f"Time to first sample: {times_stored['time_to_first_sample']:.2f} seconds")

    # Measure time for 10 following samples
    with Timer("time_for_10_samples", store=times_stored):
        for _ in range(10):
            next(iter(ds))
    logger.info(
        f"Time for 10 following samples: {times_stored['time_for_10_samples']:.2f} seconds"
        f"{times_stored['time_for_10_samples'] / 10:.2f} seconds/sample"
    )

    csv_path = "gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/benchmark_loading_time.csv"

    # Create a DataFrame for the new results it should contain config info
    all_results = pandas.DataFrame(
        [
            {
                "data_location": data_location,
                "loading_time_seconds": times_stored["loading_time"],
                "time_to_first_sample_seconds": times_stored["time_to_first_sample"],
                "time_for_10_samples_seconds": times_stored["time_for_10_samples"],
                "nominal_speed (samples/second)": 10.0 / (times_stored["time_for_10_samples"]),
                "timestamp": pandas.Timestamp.now(),
                **raw_config,
            }
        ]
    )

    save_results(all_results, csv_path)


if __name__ == "__main__":
    logger = logging.getLogger("main")
    main()
