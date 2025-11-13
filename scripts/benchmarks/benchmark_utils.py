"""Common utilities and functions for benchmarks.

This module provides shared functions and classes used across various benchmark scripts.
"""

import logging
import time
from pathlib import Path
from types import TracebackType

import pandas
import torch
import yaml

from esp_data import Dataset, dataset_from_config, get_class_from_name
from esp_data.io.paths import PureGSPath, anypath


def set_logging_config() -> None:
    """Set up logging configuration for benchmark scripts."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class Timer:
    def __init__(self, name: str = None, store: dict = None) -> None:
        self.name = name
        self.store = store
        self.start = None

    def __enter__(self) -> "Timer":
        self.start = (time.perf_counter(), time.process_time())
        return self

    def __exit__(self, exc_type: type, exc_value: Exception, traceback: TracebackType) -> None:
        duration = (time.perf_counter() - self.start[0], time.process_time() - self.start[1])
        if self.store is not None and self.name:
            self.store[self.name] = duration

    def elapsed(self) -> float:
        """Get the elapsed time since the timer was started.
        Returns
        -------
        float
            The elapsed time in seconds.
        Raises
        -------
        ValueError
            If the timer has not been started.
        """
        if self.start is None:
            raise ValueError("Timer has not been started. Use 'with Timer(...) as t:' to start it.")
        return (time.perf_counter() - self.start[0], time.process_time() - self.start[1])


def build_raw_dataset(config_path: Path, data_location: str, dataset_name: str) -> Dataset:
    """Build raw datasets without DataLoaders for direct iteration.

    Parameters
    ----------
    config_path : Path
        The run configuration containing dataset and model specifications.
    data_location : str
        The data location key in the config file (e.g., 'nfs' or 'bucket').
    dataset_name : str
        The name of the dataset to build if no config path is provided.

    Returns
    -------
    Dataset
        The raw dataset.

    Raises
    -------
    ValueError
        If both config_path and dataset_name are provided.

    """
    logger = logging.getLogger("dataset_builder")

    if config_path is not None and dataset_name is not None:
        raise ValueError("Cannot provide both config_path and dataset_name.")

    if config_path is not None:
        logger.info(f"Building dataset config from {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        logger.info(
            f"Building dataset '{raw[data_location]['dataset']['dataset_name']}' from config'"
        )
        dataset, _ = dataset_from_config(config_path, key=data_location)
        raw = raw[data_location]["dataset"]
    else:
        logger.info(f"Building dataset '{dataset_name}' with default parameters")
        dataset = get_class_from_name(dataset_name)()
        # Get corresponding raw information with following format:
        # dataset_name:
        # split:
        # sample_rate:
        # data_root:
        raw = {}
        raw["dataset_name"] = dataset_name
        if hasattr(dataset, "split"):
            raw["split"] = dataset.split
        if hasattr(dataset, "sample_rate"):
            raw["sample_rate"] = dataset.sample_rate
        if hasattr(dataset, "data_root"):
            raw["data_root"] = dataset.data_root

    return dataset, raw


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function to handle variable-length audio sequences.

    Parameters
    ----------
    batch : list[dict]
        List of samples in the batch.

    Returns
    -------
    dict
        A dictionary containing the collated batch.
    """
    # Assuming each sample has a 'audio' key and a 'label' keys
    # 'audio' needs to be padded or truncated to maximum length
    max_length = 10 * 16000  # Example: 10 seconds at 16kHz
    audios = [sample["audio"][:max_length] for sample in batch if "audio" in sample]
    audios = torch.stack(
        [
            torch.nn.functional.pad(
                torch.from_numpy(sample["audio"]),
                (0, max_length - sample["audio"].shape[0]),
            )
            for sample in batch
        ]
    )

    return audios


def build_dataloader(
    ds: Dataset, num_workers: int = 0, batch_size: int = 256, prefetch_factor: int = None
) -> torch.utils.data.DataLoader:
    """
    Build DataLoader for the given dataset.

    Parameters
    ----------
    ds : Dataset
        The dataset to build the DataLoader for.
    num_workers : int, optional
        Number of worker threads for data loading, by default 0.
    batch_size : int, optional
        Batch size for the DataLoader, by default 256.

    Returns
    -------
    tuple[torch.utils.data.DataLoader]
        A tuple containing the DataLoader.
    """

    # Create DataLoader
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
    )

    return loader


def save_results(
    results: pandas.DataFrame,
    output_path: Path,
) -> None:
    """Add benchmark results to a CSV file.

    Parameters
    ----------
    results : list[dict]
        List of benchmark results.
    output_path : Path
        Path to save the CSV file.
    """
    logger = logging.getLogger("results_saver")
    try:
        existing = pandas.read_csv(output_path)
        results = pandas.concat([existing, results], ignore_index=True)
    except (FileNotFoundError, Exception) as e:
        logger.info(f"Could not read existing CSV (maybe it doesn't exist): {e}")

    results.to_csv(output_path, index=False)
    logger.info(f"Saved {len(results)} new benchmark results to {output_path}")
    logger.info(f"Total records in file: {len(results)}")


def get_bucket_location(gcs_path: str) -> str:
    """Get the location (zone and region) of a GCS bucket.

    Parameters
    ----------
    gcs_path : str
        The GCS path (e.g., "gs://my-bucket/path/to/file").

    Returns
    -------
    str
        The location of the bucket (e.g., "US", "EUROPE-WEST1").

    Raises
    -------
    ValueError
        If the provided path is not a valid GCS path.
    """
    from google.cloud import storage

    p = anypath(str(gcs_path))
    if not isinstance(p, PureGSPath):
        raise ValueError("GCS path must start with 'gs://'")

    # Extract bucket name using PureGSPath.bucket
    bucket_name = p.bucket

    # Initialize GCS client and get bucket metadata
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.reload()  # Ensure we have the latest metadata

    return bucket.location


def get_GCP_instance_location() -> tuple[str, str]:
    """
    Get the zone and region of the current GCP VM instance.

    Returns
    -------
    (region, zone): tuple of str
        Example: ("europe-west1", "europe-west1-b")
    """
    logger = logging.getLogger("machine_location")
    metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/"
    headers = {"Metadata-Flavor": "Google"}

    try:
        import requests

        zone_path = requests.get(metadata_url + "zone", headers=headers, timeout=2).text.strip()
        zone = zone_path.split("/")[-1]  # → europe-west1-b
        region = "-".join(zone.split("-")[:-1])  # → europe-west1

        logger.info(f"Machine region: {region}, zone: {zone}")
        return region, zone

    except Exception as e:
        logger.warning(f"Could not determine machine location: {e}")
        return "unknown", "unknown"
