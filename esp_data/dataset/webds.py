import json
from typing import Any, Callable

import pandas as pd
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

import esp_data.file_io.functional as F
from esp_data.config import DatasetConfig
from esp_data.config.project_config import WEBDS_DEFAULT_CFG
from esp_data.paths import AnyPath
from esp_data.utils import make_simple_logger

from .utils import _make_file_opener

logger = make_simple_logger("web_dataset")


def load_dataset(
    path: str | AnyPath,
    pattern: str = "shard_*.tar",
    data_processor: Callable = None,
    shuffle_size: int = 1000,
    **webdataset_kwargs,
):
    """Create a pipeline for loading the dataset

    Args:
        path (str | AnyPath): Path to the dataset
        pattern (str, optional): Pattern to match the shard files. Defaults to "shard_*.tar".
        data_processor (Callable, optional): Function to process the data. Defaults to None.
        shuffle_size (int, optional): Size of the shuffle buffer. Defaults to 1000.
        **webdataset_kwargs: Additional arguments for webdataset
    """
    path = AnyPath(path)
    shard_files = F.list_files(path, pattern=pattern)

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {path}")

    # Log what we found for debugging
    logger.info(f"Found {len(shard_files)} shard files in {path}")

    return wds.WebDataset(shard_files, **webdataset_kwargs).shuffle(shuffle_size).map(data_processor)


def get_item_from_dataset(
    idx: int,
    dataset_path: str | AnyPath,
    data_processor: Callable,
    metadata_df: pd.DataFrame = None,
) -> dict[str, dict[str, Any]]:
    """Get a single item from the dataset using metadata lookup.

    Args:
        idx: Integer index to get from metadata DataFrame

    Returns:
        Dictionary containing the sample data (usually raw data and metadata)
    """
    if metadata_df is None:
        metadata_df = pd.read_parquet(AnyPath(dataset_path) / "metadata.parquet")

    # Get row from metadata
    row = metadata_df.iloc[idx]
    shard_path = row["shard_path"]
    sample_id = str(row["id"])

    # Load the specific shard
    ds = wds.WebDataset(str(AnyPath(dataset_path) / shard_path))

    # Find the specific sample by id
    for sample in ds:
        if sample["__key__"] == sample_id:
            return data_processor(sample)

    raise ValueError(f"Sample {sample_id} not found in shard {shard_path}")


def get_batch(
    indices: list[int],
    dataset_path: str | AnyPath,
    data_processor: Callable,
    metadata_df: pd.DataFrame = None,
) -> list[dict[str, dict[str, Any]]]:
    """Get a batch of items using metadata lookup.

    Args:
        indices: List of indices to get from metadata DataFrame

    Returns:
        list[dict[str, dict[str, Any]]]: List of dictionaries containing:
    """
    batch = []
    for idx in indices:
        item = get_item_from_dataset(idx, dataset_path, data_processor, metadata_df)
        batch.append(item)

    return batch


def apply_and_save(
    ds: wds.WebDataset,
    output_path: str | AnyPath,
    apply_fn: Callable,
    num_samples_per_shard: int = 1000,
    num_workers: int = 4,
):
    """Apply a function to each sample in the dataset and save the results to new shards.

    Args:
        ds (wds.WebDataset): WebDataset object
        output_path (str | AnyPath): Path to save the sharded dataset
        apply_fn (Callable): Function to apply to each sample
        num_samples_per_shard (int, optional): Number of samples per output shard. Defaults to 1000.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 4.
    """
    output_path = AnyPath(output_path)

    # Create DataLoader for efficient processing
    dataloader = DataLoader(ds, batch_size=None, num_workers=num_workers)

    # Prepare pattern for output shards
    pattern = output_path / "shard_%06d.tar"

    # Initialize sink for writing shards
    sink = wds.TarWriter(pattern, maxcount=num_samples_per_shard, opener=_make_file_opener(pattern))

    # Process each sample
    for sample in tqdm(dataloader, desc="Processing samples", total=len(dataloader)):
        # Get the key for this sample
        key = sample.get("__key__", None)
        if key is None:
            raise ValueError("Sample missing __key__ field")

        # Apply the function to the sample
        modified_sample = apply_fn(sample)

        # Write to the sink with the original key
        sink.write({"__key__": key, **modified_sample})

    # Close the sink to ensure all data is written
    sink.close()


def apply_and_save_v2(
    ds: wds.WebDataset,
    output_path: str | AnyPath,
    apply_fn: Callable,
    num_samples_per_shard: int = 1000,
    shuffle_buffer_size: int = 100,
):
    """Apply a function to each sample in the dataset and save the results to new shards.

    Args:
        ds (wds.WebDataset): WebDataset object
        output_path (str | AnyPath): Path to save the sharded dataset
        apply_fn (Callable): Function to apply to each sample
        samples_per_shard (int, optional): Number of samples per output shard. Defaults to 1000.
        shuffle_buffer_size (int, optional): Size of the shuffle buffer. Defaults to 100.
    """
    output_path = AnyPath(output_path)
    pattern = output_path / "shard_%06d.tar"

    processed_ds = ds.map(apply_fn, handler=wds.handlers.warn_and_continue)

    # Write the modified dataset to disk
    # The DataPipeline will run with multiple workers in parallel
    (
        processed_ds.compose(wds.filters.detshuffle(shuffle_buffer_size))
        .to_tuple("__key__", "*")
        .pipe(lambda data: ({"__key__": key, **rest} for key, *values in data for rest in [dict(values)]))
        .compose(wds.writers.TarWriter(pattern, maxcount=num_samples_per_shard, opener=_make_file_opener(pattern)))
    )


class WebDataset:
    """Class for loading and accessing a tar file based dataset.

    Args:
        web_dataset_path (str): Path to the directory where the sharded dataset will be stored or is already stored.
        metadata_df (pd.DataFrame, optional): Optional metadata DataFrame, if not provided will be read from disk. Defaults to None.
        shard_size (int, optional): Number of samples per shard. Defaults to 1000.
        num_workers (int, optional): Number of workers for parallel processing. Defaults to 4.
        batch_size (int, optional): Batch size for processing audio files. Defaults to 100.
        metadata_path (str): Path to the metadata file, if different from web_dataset_path. Defaults to None.
        sample_prep_function (Callable, optional): Function to prepare a sample for sharding. Defaults to None.
        shuffle_size (int, optional): Size of the shuffle buffer. Defaults to 1000.
        storage_options (dict, optional): Storage options for reading and writing files from buckets. Defaults to None.

    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        ds: wds.WebDataset = None,
        path: str | AnyPath | None = None,
        load_metadata: bool = WEBDS_DEFAULT_CFG["load_metadata"],
        metadata_df: pd.DataFrame = WEBDS_DEFAULT_CFG["metadata_df"],
        file_pattern: str = WEBDS_DEFAULT_CFG["file_pattern"],
        storage_options: dict = WEBDS_DEFAULT_CFG["storage_options"],
        metadata_path: str | None = WEBDS_DEFAULT_CFG["metadata_path"],
        data_processor: Callable = WEBDS_DEFAULT_CFG["data_processor"],
        shuffle_size: int = WEBDS_DEFAULT_CFG["shuffle_size"],
    ):
        assert path is None and ds is None, "Only one of path or ds should be provided"
        self.path = AnyPath(path)
        self.config = dataset_config
        self.metadata_path = AnyPath(metadata_path if metadata_path is not None else self.web_dataset_path)
        self.metadata_df = metadata_df
        self.storage_options = storage_options
        self.shuffle_size = shuffle_size

        # Read metadata if missing
        if self.metadata_df is None and load_metadata:
            if AnyPath(self.metadata_path / "metadata.parquet").exists():
                self.metadata_df = pd.read_parquet(
                    self.metadata_path / "metadata.parquet", storage_options=storage_options
                )
            elif AnyPath(self.metadata_path / "metadata.csv").exists():
                self.metadata_df = pd.read_csv(self.metadata_path / "metadata.csv", storage_options=storage_options)
            elif AnyPath(self.metadata_path / "metadata.json").exists():
                self.metadata_df = pd.read_json(self.metadata_path / "metadata.json", storage_options=storage_options)
            else:
                logger.warning(
                    "No metadata found. Won't be able to create a sharded dataset or index directly into the data"
                )

        self._data_processor = data_processor

        # load dataset if available
        self.ds = ds
        shard_files = F.list_files(self.web_dataset_path, pattern=file_pattern)
        if len(shard_files) > 0:
            self._load_dataset(shuffle_size=self.shuffle_size)

    @property
    def columns(self):
        return list(self.metadata_df.columns) if self.metadata_df is not None else None

    @property
    def version(self):
        return self.config.version

    def _set_config(self, dataset_config) -> None:
        if isinstance(dataset_config, dict):
            return DatasetConfig(**dataset_config)
        elif isinstance(dataset_config, DatasetConfig):
            return dataset_config

    def _load_dataset(self, shuffle_size: int = 1000, **webdataset_kwargs):
        self.ds = load_dataset(
            path=self.path,
            data_processor=self._data_processor,
            shuffle_size=shuffle_size,
            **webdataset_kwargs,
        )

    @classmethod
    def from_path(cls, path: str | AnyPath, **kwargs):
        path = AnyPath(path)

        # load config from json
        config_file = AnyPath(path / "dataset_config.json")
        if not config_file.exists():
            raise FileNotFoundError("No dataset config found")

        with config_file.open("r") as fp:
            config = json.load(fp)

        return cls(config, path=path, **kwargs)

    def __getitem__(self, idx: int) -> Any:
        if self.metadata_df is None:
            raise ValueError("No metadata found. Cannot access individual samples.")

        if isinstance(idx, slice):
            return get_batch(
                indices=list(range(idx.start, idx.stop, idx.step)),
                dataset_path=self.web_dataset_path,
                data_processor=self._data_processor,
                metadata_df=self.metadata_df,
            )

        return get_item_from_dataset(
            idx=idx,
            dataset_path=self.web_dataset_path,
            data_processor=self._data_processor,
            metadata_df=self.metadata_df,
        )

    def __len__(self):
        return len(self.metadata_df) or None

    def __iter__(self):
        if self.ds is None:
            self._load_dataset(shuffle_size=self.shuffle_size)

        return iter(self.ds)
