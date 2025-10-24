"""Benchmark DataLoader and raw dataset iteration.

This will load individual samples from bucket or disk.


"""

import logging
import time
from pathlib import Path

import click
import torch
import yaml

from esp_data import Dataset, DatasetConfig, dataset_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("benchmark_dataloader")


def build_raw_dataset(config_path: Path, data_location: str) -> Dataset:
    """Build raw datasets without DataLoaders for direct iteration.

    Parameters
    ----------
    config_path : Path
        The run configuration containing dataset and model specifications.

    Returns
    -------
    Dataset
        The raw dataset.
    """
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    dataset_config = DatasetConfig.model_validate(raw[data_location])
    dataset, _ = dataset_from_config(dataset_config)

    return dataset


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
    ds: Dataset, num_workers: int = 0, batch_size: int = 256
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
    )

    return loader


def benchmark_loader(
    loader: torch.utils.data.DataLoader,
    name: str,
    sleep: float = 0.0,
    log_interval: int = 10,
    max_iterations: int = 1000,
) -> None:
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
    """
    logger.info(f"Benchmarking {name} dataloader: {len(loader)} batches")
    start = time.time()
    n_batches = 0
    n_samples = 0
    last_log_time = start

    for batch in loader:
        n_batches += 1
        batch_size = batch.shape[0]
        n_samples += batch_size

        if sleep > 0:
            time.sleep(sleep)

        # Log stats every log_interval batches
        if n_batches % log_interval == 0:
            current_time = time.time()
            elapsed_since_start = current_time - start
            elapsed_since_last = current_time - last_log_time

            samples_per_sec = n_samples / elapsed_since_start
            batches_per_sec = n_batches / elapsed_since_start
            recent_samples_per_sec = (
                (log_interval * batch_size) / elapsed_since_last if elapsed_since_last > 0 else 0
            )

            logger.info(
                f"{name} batch {n_batches}/{len(loader)}: "
                f"{n_samples} samples in {elapsed_since_start:.2f}s "
                f"({samples_per_sec:.2f} samples/s, {batches_per_sec:.2f} batches/s) "
                f"[recent: {recent_samples_per_sec:.2f} samples/s]"
            )
            last_log_time = current_time

        if max_iterations > 0 and n_batches >= max_iterations:
            logger.info(f"Reached max_iterations={max_iterations}, stopping early.")
            break

    # Final stats
    elapsed = time.time() - start
    logger.info(
        f"{name} FINAL: {n_batches} batches, {n_samples} samples in {elapsed:.2f}s "
        f"({n_samples / elapsed:.2f} samples/s, {n_batches / elapsed:.2f} batches/s)"
    )


def benchmark_raw_dataset(
    dataset: Dataset,
    name: str,
    sleep: float = 0.0,
    log_interval: int = 100,
    max_iterations: int = 1000,
) -> None:
    """Benchmark direct iteration over a raw dataset."""
    try:
        L = len(dataset)
    except TypeError:
        L = -1

    logger.info(f"Benchmarking {name} raw dataset with length {L}")
    start = time.time()
    n_samples = 0
    last_log_time = start
    shard_download_time = 0.0
    samples_per_sec = 0.0
    running_avg_samples_per_sec = 0.0

    for _ in dataset:
        n_samples += 1

        # Get the sample (this is what we're benchmarking)
        # sample = dataset[idx]  # Only works for indexed or "map-style" datasets

        if sleep > 0:
            time.sleep(sleep)

        if n_samples == 1:
            shard_download_time = time.time() - start
            start = time.time()  # Reset start time after first sample

        # Log stats every log_interval samples
        if n_samples > 1 and (n_samples % log_interval == 0):
            current_time = time.time()
            elapsed_since_start = current_time - start
            elapsed_since_last = current_time - last_log_time

            running_avg_samples_per_sec = 0.99 * running_avg_samples_per_sec + 0.01 * (
                n_samples / elapsed_since_start
            )
            samples_per_sec = n_samples / elapsed_since_start
            recent_samples_per_sec = (
                log_interval / elapsed_since_last if elapsed_since_last > 0 else 0
            )

            logger.info(
                f"{name} sample {n_samples}/{L}: "
                f"{n_samples} samples in {elapsed_since_start:.2f}s "
                f"({samples_per_sec:.2f} samples/s) "
                f"(running avg: {running_avg_samples_per_sec:.2f} samples/s, "
                f"[recent: {recent_samples_per_sec:.2f} samples/s]"
            )
            last_log_time = current_time

        if max_iterations > 0 and n_samples >= max_iterations:
            logger.info(f"Reached max_iterations={max_iterations}, stopping early.")
            break

    # Final stats
    elapsed = time.time() - start
    logger.info(
        f"{name} FINAL: {n_samples} samples in {elapsed:.2f}s "
        f"({n_samples / elapsed:.2f} samples/s)\n"
        f"(shard (or first sample) download time: {shard_download_time:.2f}s)"
    )


@click.command()
@click.option(
    "--config_path",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="benchmark_config.yaml",
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
    help="Maximum number of iterations to run (default: 1000)",
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
    default=256,
    show_default=True,
    help="Batch size for DataLoader (default: 256)",
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
) -> None:
    t0 = time.time()
    train_ds = build_raw_dataset(config_path, data_location)
    time_taken = time.time() - t0
    logger.info(f"Loaded dataset in {time_taken:.2f} seconds")

    if raw_dataset:
        benchmark_raw_dataset(train_ds, train_ds.info.name, sleep, log_interval, max_iterations)
    else:
        train_dl = build_dataloader(
            train_ds,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        benchmark_loader(
            train_dl,
            name=train_ds.info.name,
            sleep=sleep,
            log_interval=log_interval,
            max_iterations=max_iterations,
        )


if __name__ == "__main__":
    main()
