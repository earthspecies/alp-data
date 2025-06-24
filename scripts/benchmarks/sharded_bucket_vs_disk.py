"""Compare some sharded data formats like arrow, parquet and tar against each
other on bucketed data and disk data."""

import io
import json
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import webdataset as wds

from esp_data.io import filesystem_from_path

fs = filesystem_from_path("gs://")


def iterate_parquet_shard(shard_path: str, batch_size: int = 2) -> None:
    """Iterate over a Parquet shard file and process audio data in batches."""
    parquet_file = pq.ParquetFile(shard_path)
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
        batch = batch.to_pandas()
        # Do something with the batch
        _ = sf.read(io.BytesIO(batch["audio"].iloc[0]["bytes"]))
        if (i + 1) % 50 == 0:
            print(f"Done with batch {i}")
    parquet_file.close()


def read_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as audio_buffer:
        data, samplerate = sf.read(audio_buffer)

    return data, samplerate


def data_processor(data: dict) -> dict[str, Any]:
    """Process the data from the dataset.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the audio data and metadata.
    """
    audio, _ = read_audio_bytes(data["audio.wav"])
    metadata = json.loads(data["metadata.json"])
    return {"audio": audio, **metadata}


def load_webdataset(
    shard_files: str | list[str],
    data_processor: Callable,
    shuffle_size: int | None = None,
    batch_size: int | None = None,
    shard_shuffle: bool = False,
    shard_shuffle_size: int = 1000,
    split_by_worker: bool = False,
    batch_collate_fn: Callable = None,
    seed: int | bool | None = 0,
) -> wds.WebDataset:
    """Create a pipeline for loading the dataset

    Parameters
    ----------
    shard_files: str | list[str]
        Path to the shard files or a list of shard file paths.
    data_processor: Callable
        Function to process the data.
    shuffle_size: int, optional
        Size of the shuffle buffer for the dataset.
    batch_size: int, optional
        Size of the batches to be created from the dataset.
    shard_shuffle: bool, optional
        Whether to shuffle the shards.
    shard_shuffle_size: int, optional
        Size of the shuffle buffer for the shards.
    split_by_worker: bool, optional
        Whether to split the dataset by worker.
    batch_collate_fn: Callable, optional
        Function to collate the batches.
    seed: int | bool | None, optional
        Seed for random operations. If False, no shuffling is applied.

    Returns
    -------
    wds.WebDataset
        A WebDataset object that can be iterated over to access the data.
    """

    webds = wds.WebDataset(
        shard_files,
        shardshuffle=shard_shuffle_size if shard_shuffle else False,
        seed=seed,
        workersplitter=split_by_worker,
    )

    if shuffle_size:
        webds = webds.shuffle(shuffle_size)
    if data_processor:
        webds = webds.map(data_processor)
    if batch_size is not None:
        webds = webds.batched(batch_size, collation_fn=batch_collate_fn)

    return webds


# use naturelm data for it
naturelm_tar_bucket = "gs://esp-ml-datasets/naturelm/processed/v0.1.0/tar/train/shard_000000.tar"
naturelm_tar_disk = "/home/gagan_earthspecies_org/scratch_code/shard_000000.tar"
naturelm_parquet_bucket = (
    "gs://esp-ml-datasets/naturelm/processed/v0.1.1/parquet/train/shard_000000.parquet"
)
naturelm_parquet_disk = "/home/gagan_earthspecies_org/scratch_code/shard_000000.parquet"


def benchmark_sharded_bucket_vs_disk() -> None:
    """Benchmark the performance of sharded data formats on bucketed and disk data."""
    # Load WebDataset from tar files
    webds_bucket = load_webdataset(
        path=naturelm_tar_bucket,
        file_pattern="shard*tar",
        data_processor=data_processor,
        shuffle_size=1000,
        batch_size=32,
        shard_shuffle=False,
        shard_shuffle_size=1000,
        seed=42,
    )

    webds_disk = load_webdataset(
        path=naturelm_tar_disk,
        file_pattern="shard*tar",
        data_processor=data_processor,
        shuffle_size=1000,
        batch_size=32,
        shard_shuffle=False,
        shard_shuffle_size=1000,
        seed=42,
    )

    # Iterate through the datasets
    print("Iterating through bucketed WebDataset...")
    for _ in webds_bucket:
        pass  # Process the sample as needed

    print("Iterating through disk WebDataset...")
    for _ in webds_disk:
        pass  # Process the sample as needed

    print("Benchmark completed.")
