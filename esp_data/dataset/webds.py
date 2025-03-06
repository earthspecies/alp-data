import json
from functools import partial
from typing import Any, Callable, Generator, Literal

import pandas as pd
import webdataset as wds

import esp_data.file_io.functional as F
from esp_data.config import DatasetConfig
from esp_data.config.project_config import WEBDS_DEFAULT_CFG
from esp_data.paths import AnyPath
from esp_data.utils import make_simple_logger

from .base import BaseIterableDataset, BaseMapDataset
from .shard_creator import write_webdataset_shard

logger = make_simple_logger("web_dataset")


def load_dataset(
    path: str | AnyPath,
    file_pattern: str = "shard_*.tar",
    data_processor: Callable = None,
    shuffle_size: int | None = None,
    batch_size: int | None = None,
    shard_shuffle: bool = False,
    shard_shuffle_size: bool = 1000,
    split_by_worker: bool = False,
    batch_collate_fn: Callable = None,
    seed: int | bool | None = True,
):
    """Create a pipeline for loading the dataset

    Args:
        path (str | AnyPath): Path to the dataset
        file_pattern (str, optional): Pattern to match the shard files. Defaults to "shard_*.tar".
        data_processor (Callable, optional): Function to process the data. Defaults to None.
        shuffle_size (int, optional): Size of the shuffle buffer. Defaults to 1000.
        batch_size (int, optional): Batch size for processing audio files. Defaults to None.
        shard_shuffle (bool, optional): Whether to shuffle the shards. Defaults to False.
        shard_shuffle_size (int, optional): Size of the shuffle buffer for shards. Defaults to 1000.
        split_by_worker (bool, optional): Whether to split the dataset by worker. Defaults to False.
        batch_collate_fn (Callable, optional): Function to collate the batch. Defaults to None.
        seed (int, optional): Seed for shuffling. Defaults to True, random seed. If None, means no shuffling!
    """
    path = AnyPath(path)
    shard_files = F.list_files(path, pattern=file_pattern)

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {path}")

    # Log what we found for debugging
    logger.debug(f"Found {len(shard_files)} shard files in {path}")

    if batch_size is not None:
        return (
            wds.WebDataset(
                shard_files,
                shardshuffle=shard_shuffle_size if shard_shuffle else False,
                seed=seed,
                workersplitter=split_by_worker,
            )
            .shuffle(shuffle_size)
            .map(data_processor)
            .batched(batch_size, collation_fn=batch_collate_fn)
        )
    return (
        wds.WebDataset(
            shard_files,
            shardshuffle=shard_shuffle_size if shard_shuffle else False,
            seed=seed,
            workersplitter=split_by_worker,
        )
        .shuffle(shuffle_size)
        .map(data_processor)
    )

    # operations = [wds.SimpleShardList(shard_files, seed)]
    # if shard_shuffle:
    #     operations.append(wds.shuffle(shard_shuffle_size))
    # if split_by_worker:
    #     operations.append(wds.split_by_worker)
    # operations.append(wds.tarfile_to_samples())
    # if shuffle_size:
    #     operations.append(wds.shuffle(shuffle_size))
    # if data_processor:
    #     operations.append(wds.map(data_processor))
    # if batch_size:
    #     batched_kwargs = {"batchsize": batch_size}
    #     if batch_collate_fn:
    #         batched_kwargs["collate_fn"] = batch_collate_fn
    #     operations.append(wds.batched(batch_size))

    # return wds.DataPipeline(*operations)


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


class WebDataset(BaseMapDataset, BaseIterableDataset):
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
        path: str | AnyPath | None = None,
        dataset_config: DatasetConfig | None = None,
        ds: wds.WebDataset = None,
        load_metadata: bool = WEBDS_DEFAULT_CFG["load_metadata"],
        metadata_df: pd.DataFrame = WEBDS_DEFAULT_CFG["metadata_df"],
        file_pattern: str = WEBDS_DEFAULT_CFG["file_pattern"],
        storage_options: dict = WEBDS_DEFAULT_CFG["storage_options"],
        metadata_path: str | None = WEBDS_DEFAULT_CFG["metadata_path"],
        data_processor: Callable = WEBDS_DEFAULT_CFG["data_processor"],
        shuffle_size: int = WEBDS_DEFAULT_CFG["shuffle_size"],
        shard_shuffle: bool = WEBDS_DEFAULT_CFG["shard_shuffle"],
        shard_shuffle_size: int = WEBDS_DEFAULT_CFG["shard_shuffle_size"],
        batch_size: int | None = WEBDS_DEFAULT_CFG["batch_size"],
        batch_collate_fn: Callable = WEBDS_DEFAULT_CFG["batch_collate_fn"],
        split_by_worker: bool = WEBDS_DEFAULT_CFG["split_by_worker"],
        seed: int | bool | None = WEBDS_DEFAULT_CFG["seed"],
    ):
        assert not (path is None and ds is None), "One of path or ds should be provided"
        self.path = AnyPath(path)
        self.config = dataset_config
        self.metadata_path = AnyPath(metadata_path if metadata_path is not None else self.path)
        self.metadata_df = metadata_df
        self.storage_options = storage_options

        # Read metadata if missing
        if self.metadata_df is None and load_metadata:
            if (self.metadata_path / "metadata.parquet").exists():
                self.metadata_df = pd.read_parquet(
                    str(self.metadata_path / "metadata.parquet"), storage_options=storage_options
                )
            elif (self.metadata_path / "metadata.csv").exists():
                self.metadata_df = pd.read_csv(
                    str(self.metadata_path / "metadata.csv"), storage_options=storage_options
                )
            elif (self.metadata_path / "metadata.json").exists():
                self.metadata_df = pd.read_json(
                    str(self.metadata_path / "metadata.json"), storage_options=storage_options
                )
            else:
                logger.warning(
                    "No metadata found. Won't be able to create a sharded dataset or index directly into the data"
                )

        # try and load the dataset_config
        self.config = self._set_config(dataset_config)

        self._data_processor = data_processor

        # load dataset if available
        self.ds = ds or load_dataset(
            path=self.path,
            data_processor=data_processor,
            shuffle_size=shuffle_size,
            shard_shuffle=shard_shuffle,
            shard_shuffle_size=shard_shuffle_size,
            batch_size=batch_size,
            batch_collate_fn=batch_collate_fn,
            seed=seed,
            file_pattern=file_pattern,
            split_by_worker=split_by_worker,
        )

    @property
    def columns(self):
        return list(self.metadata_df.columns) if self.metadata_df is not None else None

    @property
    def version(self):
        return self.config.version

    def set_ds(self, ds: wds.WebDataset):
        self.ds = ds

    def get_ds(self):
        return self.ds

    def _set_config(self, dataset_config) -> None:
        if isinstance(dataset_config, dict):
            return DatasetConfig(**dataset_config)
        elif isinstance(dataset_config, DatasetConfig):
            return dataset_config
        else:
            return DatasetConfig.from_skeleton()

    @classmethod
    def from_path(cls, path: str | AnyPath, **kwargs):
        path = AnyPath(path)

        # load config from json
        config_file = AnyPath(path / "dataset_config.json")
        if not config_file.exists():
            logger.warning("No dataset config found, making skeleton config")
            config = DatasetConfig.from_skeleton()
        else:
            with config_file.open("r") as fp:
                config = json.load(fp)

        return cls(path=path, dataset_config=config, **kwargs)

    def __getitem__(self, idx: int) -> Any:
        if self.metadata_df is None:
            raise ValueError("No metadata found. Cannot access individual samples.")

        if isinstance(idx, slice):
            return get_batch(
                indices=list(range(idx.start, idx.stop, idx.step)),
                dataset_path=self.path,
                data_processor=self._data_processor,
                metadata_df=self.metadata_df,
            )

        return get_item_from_dataset(
            idx=idx,
            dataset_path=self.path,
            data_processor=self._data_processor,
            metadata_df=self.metadata_df,
        )

    def __len__(self):
        return len(self.metadata_df) or None

    def __iter__(self):
        return iter(self.ds)

    def map(self, function: Callable, **kwargs):
        self.ds = self.ds.map(function, **kwargs)
        return self

    def save_to_path(
        self, path: str | AnyPath, num_samples_per_shard: int = 1000, sample_prep_function: Callable = None
    ):
        """Save the dataset to a new path."""
        path = AnyPath(path)

        if self.metadata_df is not None:
            # get num_shards from metadata
            logger.info("Using metadata to determine number of shards")
            num_shards = len(self.metadata_df["shard_path"].unique())
            num_samples_per_shard = len(self.metadata_df) // num_shards

        batch = []
        shard_id = 0
        for i, sample in enumerate(self):
            batch.append(sample)

            if len(batch) == num_samples_per_shard:
                write_webdataset_shard(batch, path, shard_id=shard_id, sample_prep_function=sample_prep_function)
                batch = []
                shard_id += 1

        if batch:
            write_webdataset_shard(batch, path, shard_id=i, sample_prep_function=sample_prep_function)

        # save config
        self.config.write_json(path / "dataset_config.json")
        self.config.generate_readme(path / "README.md")
        # save metadata if available
        if self.metadata_df is not None:
            self.metadata_df.to_parquet(path / "metadata.parquet")


def apply_fn(
    ds: WebDataset,
    function: Callable,
    fn_kwargs: dict = {},
    output_path: str | AnyPath | None = None,
    num_samples_per_shard: int = 1000,
    changelog: str | None = None,
    version_update_mode: Literal["major", "minor", "patch"] = None,
) -> Generator[dict, None, None] | WebDataset:
    """Apply a function to each sample in the dataset and save the results to new shards.

    Args:
        ds (wds.WebDataset): WebDataset object
        function (Callable): Function to apply to each sample
        fn_kwargs (dict, optional): Additional keyword arguments for the function. Defaults to {}.
        batched (bool, optional): Whether to process the dataset in batches. Defaults to False.
        batch_size (int, optional): Batch size for processing audio files. Defaults to 1000.
        output_path (str | AnyPath, optional): Path to the output directory. Defaults to None.
        num_samples_per_shard (int, optional): Number of samples per output shard. Defaults to 1000.
        changelog (str, optional): Changelog for the dataset. Defaults to None.
        version_update_mode (Literal["major", "minor", "patch"], optional): Mode for updating the version number. Defaults to None.

    Yields:
        Generator[dict, None, None]: Generator for the processed samples if output_path is None

    Returns:
        WebDataset: WebDataset object if output_path is provided
    """
    if fn_kwargs is not None:
        function = partial(function, **fn_kwargs)

    ds = ds.map(function)

    if changelog:
        ds.config.update_changelog(changelog)

    if version_update_mode:
        ds.config.increment_version(version_update_mode)

    if output_path is None:
        for sample in ds:
            yield sample

    # Apply and save to new shards
    ds.save_to_path(output_path, num_samples_per_shard=num_samples_per_shard)

    return ds
