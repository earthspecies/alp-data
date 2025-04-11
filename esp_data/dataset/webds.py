import json
import logging
from functools import lru_cache, partial
from typing import Any, Callable, Generator, Literal

import pandas as pd
import webdataset as wds

from esp_data.config import DatasetConfig
from esp_data.config.project_config import default_webds_loader_cfg
from esp_data.io import AnyPathT, anypath
from esp_data.io.filesystem import filesystem_from_path

from .base import BaseIterableDataset, BaseMapDataset
from .shard_creator import write_webdataset_shard

logger = logging.getLogger("esp_data")


def load_dataset(
    path: str | AnyPathT,
    file_pattern: str = default_webds_loader_cfg.file_pattern,
    data_processor: Callable = default_webds_loader_cfg.data_processor,
    shuffle_size: int | None = default_webds_loader_cfg.shuffle_size,
    batch_size: int | None = default_webds_loader_cfg.batch_size,
    shard_shuffle: bool = default_webds_loader_cfg.shard_shuffle,
    shard_shuffle_size: int = default_webds_loader_cfg.shard_shuffle_size,
    split_by_worker: bool = default_webds_loader_cfg.split_by_worker,
    batch_collate_fn: Callable = default_webds_loader_cfg.batch_collate_fn,
    seed: int | bool | None = default_webds_loader_cfg.seed,
) -> wds.WebDataset:
    """Create a pipeline for loading the dataset

    Arguments
    ---------
    path: str | AnyPathT
            Path to the directory where the sharded dataset will be stored or is already stored.
    file_pattern: str, optional
        Pattern to match the shard files.
    data_processor: Callable, optional
        Function to process the data.
    shuffle_size: int, optional
        Size of the shuffle buffer.
    batch_size: int, optional
        Batch size for processing audio files.
    shard_shuffle: bool, optional
        Whether to shuffle the shards.
    shard_shuffle_size: int, optional
        Size of the shuffle buffer for shards.
    split_by_worker: bool, optional
        Whether to split the dataset by worker.
    batch_collate_fn: Callable, optional
        Function to collate the batch.
    seed Union[int, bool, None]:
        Seed for shuffling. Defaults to True, random seed. If None, means no shuffling!

    Returns
    -------
        wds.WebDataset: WebDataset object

    Examples
    --------
    >>> from esp_data.dataset.webds import load_dataset
    >>> from pathlib import Path
    >>> path = Path("/tmp/tmp_webdss/shard_0.tar").make_dir(parents=True, exist_ok=True).write_bytes(b"hello")
    >>> ds = load_dataset(
    ...     path="/tmp/tmp_webdss",
    ...     data_processor=lambda x: x,
    ...     shuffle_size=1000,
    ...     batch_size=32,
    ...     shard_shuffle=True,
    ...     shard_shuffle_size=1000,
    ...     split_by_worker=True,
    ...     seed=42,
    ... )

    """
    path = anypath(path)
    shard_files = filesystem_from_path(path).glob(str(path / file_pattern))

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {path}")

    # .glob() removes the prefix from the path, so we need to add it back
    # TODO (milad) Gagan used to deal with this in yield_files(). Decide if it's worth
    #              adding it to a wrapper
    if path.is_cloud:
        shard_files = [path.cloud_prefix + str(p) for p in shard_files]

    # Log what we found for debugging
    logger.debug(f"Found {len(shard_files)} shard files in {path}")

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

    # DATA PIPELINE APPROACH
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
    dataset_path: str | AnyPathT,
    data_processor: Callable,
    metadata_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Get a single item from the dataset using metadata lookup.

    Arguments
    ---------
    idx: int
        Integer index to get from metadata DataFrame
    data_processor: Callable
        Function to parse the binary data into a dictionary
    metadata_df: pd.DataFrame
        metadata DataFrame, containing the shard path and sample id
    dataset_path: str | AnyPathT
        Path to the dataset, if different from metadata path

    Returns
    -------
        Dictionary containing the sample data (usually raw data and metadata)
    """
    # Get row from metadata
    row = metadata_df.iloc[idx]
    shard_path = row["shard_path"]
    sample_id = str(row["id"])

    # Load the specific shard
    ds = wds.WebDataset(str(anypath(dataset_path) / shard_path))

    # Find the specific sample by id
    for sample in ds:
        if sample["__key__"] == sample_id:
            return data_processor(sample)

    raise ValueError(f"Sample {sample_id} not found in shard {shard_path}")


def get_batch(
    indices: list[int],
    dataset_path: str | AnyPathT,
    data_processor: Callable,
    metadata_df: pd.DataFrame,
) -> list[dict[str, dict[str, Any]]]:
    """Get a batch of items using metadata lookup.

    Args:
        indices: List of indices to get from metadata DataFrame
        data_processor: Function to process the data
        metadata_df: Optional metadata DataFrame, if not provided will be read from disk
        dataset_path: Path to the dataset, if different from metadata path

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

    Arguments
    ---------
    path: str | AnyPathT:
            Path to the directory where the sharded dataset will be stored or is already stored.
    dataset_config: DatasetConfig, optional
        DatasetConfig object. Defaults to None.
            If not provided, will try to load from disk. If not found, will create a skeleton config.
    ds: wds.WebDataset, optional
        WebDataset object. Defaults to None. If provided, will be used instead of loading from disk.
    load_metadata: bool, optional
        Whether to load metadata from disk.
    metadata_df: pd.DataFrame, optional
        Optional metadata DataFrame, if not provided will be read from disk.
    data_processor: Callable, optional
        Function to process the data. Otherwise, the data returned will be a dict with bytes as values.
    file_pattern: str, optional
        Pattern to match the shard files.
    num_workers: int, optional
        Number of workers for parallel processing.
    metadata_path: str | AnyPathT, optional
        Path to the metadata file, if different from web_dataset_path.
    shuffle_size: int, optional
        Size of the shuffle buffer.
    storage_options: dict, optional
        Storage options for reading and writing files from buckets.
    shard_shuffle: bool, optional
        Whether to shuffle the shards. Defaults to False.
    shard_shuffle_size (int, optional):
        Size of the shuffle buffer for shards.
    batch_size (int, optional):
        Batch size for processing audio files.
    batch_collate_fn (Callable, optional):
        Function to collate the batch.
    split_by_worker (bool, optional):
        Whether to split the dataset by worker.
    seed (int, optional): Seed for shuffling. Defaults to 0.

    """

    def __init__(
        self,
        path: str | AnyPathT | None = None,
        dataset_config: DatasetConfig | None = None,
        ds: wds.WebDataset = None,
        load_metadata: bool = default_webds_loader_cfg.load_metadata,
        metadata_df: pd.DataFrame | None = default_webds_loader_cfg.metadata_df,
        file_pattern: str = default_webds_loader_cfg.file_pattern,
        storage_options: dict | None = default_webds_loader_cfg.storage_options,
        metadata_path: str | None = default_webds_loader_cfg.metadata_path,
        data_processor: Callable = default_webds_loader_cfg.data_processor,
        shuffle_size: int = default_webds_loader_cfg.shuffle_size,
        shard_shuffle: bool = default_webds_loader_cfg.shard_shuffle,
        shard_shuffle_size: int = default_webds_loader_cfg.shard_shuffle_size,
        batch_size: int | None = default_webds_loader_cfg.batch_size,
        batch_collate_fn: Callable = default_webds_loader_cfg.batch_collate_fn,
        split_by_worker: bool = default_webds_loader_cfg.split_by_worker,
        seed: int | bool | None = default_webds_loader_cfg.seed,
    ):
        assert not (path is None and ds is None), "One of path or ds should be provided"
        self.path = anypath(path)
        self.config = dataset_config
        self.metadata_path = anypath(metadata_path if metadata_path is not None else self.path)
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
            elif (self.metadata_path / "metadata.jsonl").exists():
                self.metadata_df = pd.read_json(
                    str(self.metadata_path / "metadata.jsonl"),
                    storage_options=storage_options,
                    lines=True,
                    orient="records",
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
    def from_path(cls, path: str | AnyPathT, **kwargs):
        path = anypath(path)

        # load config from json
        config_file = anypath(path / "dataset_config.json")
        if not config_file.exists():
            logger.warning("No dataset config found, making skeleton config")
            config = DatasetConfig.from_skeleton()
        else:
            with config_file.open("r") as fp:
                config = json.load(fp)

        return cls(path=path, dataset_config=config, **kwargs)

    @lru_cache(maxsize=None)  # TODO check if this is necessary
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
        self, path: str | AnyPathT, num_samples_per_shard: int = 1000, sample_prep_function: Callable = None
    ):
        """Save the dataset to a new path."""
        path = anypath(path)

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
    output_path: str | AnyPathT | None = None,
    num_samples_per_shard: int = 1000,
    changelog: str | None = None,
    version_update_mode: Literal["major", "minor", "patch"] = None,
) -> Generator[dict, None, None] | WebDataset:
    """Apply a function to each sample in the dataset and save the results to new shards.

    Arguments
    ---------
        ds (WebDataset): WebDataset object
        function (Callable): Function to apply to each sample
        fn_kwargs (dict, optional): Additional keyword arguments for the function. Defaults to {}.
        batched (bool, optional): Whether to process the dataset in batches. Defaults to False.
        batch_size (int, optional): Batch size for processing audio files. Defaults to 1000.
        output_path (str | AnyPathT, optional): Path to the output directory. Defaults to None.
        num_samples_per_shard (int, optional): Number of samples per output shard. Defaults to 1000.
        changelog (str, optional): Changelog for the dataset. Defaults to None.
        version_update_mode (Literal["major", "minor", "patch"], optional): Mode for updating the version number. Defaults to None.

    Yields
    ------
        Generator[dict, None, None]: Generator for the processed samples if output_path is None

    Returns
    -------
        WebDataset: WebDataset object if output_path is provided, which is returned after saving the shards.
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
        return

    # Apply and save to new shards
    ds.save_to_path(output_path, num_samples_per_shard=num_samples_per_shard)

    return ds
