import os
from typing import Any, Callable, Generator, Iterable, Literal, Optional

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

from esp_data.config.db_config import DataSample, DatasetConfig
from esp_data.paths import AnyPath, make_storage_options
from esp_data.utils import make_simple_logger

from .base import BaseMapDataset
from .shard_creator import write_huggingface_shard
from .utils import generate_random_indices

logger = make_simple_logger("esp_data")


HF_DATASET_TYPES = [
    "arrow",
    "csv",
    "tsv",
    "json",
    "parquet",
    "audiofolder",
    "imagefolder",
    "local_hf",
    "bucket_hf",
    "hf_hub",
]


def load_hf_dataset(hf_dataset_type: str, path: str | AnyPath, **hf_ds_kwargs) -> "HFDataset":
    if hf_dataset_type == "hf_hub":
        ds = load_dataset(path, **hf_ds_kwargs)

        cfg = DatasetConfig(
            name=ds.info.dataset_name,
            description=str(ds.info),
            version="0.0.0",
            sources="see description",
            license="see description",
            creator="see description",
        )
        return HFDataset(cfg, ds=ds, streaming_dataset=hf_ds_kwargs.get("streaming", False))

    elif hf_dataset_type in [
        "csv",
        "tsv",
        "json",
        "parquet",
        "arrow",
        "audiofolder",
        "imagefolder",
        "local_hf",
        "bucket_hf",
    ]:
        return HFDataset.from_path(path, hf_dataset_type, **hf_ds_kwargs)


class HFDataset(BaseMapDataset):
    def __init__(
        self, dataset_config: DatasetConfig | dict | None, ds: Optional[Dataset] = None, streaming_dataset: bool = False
    ):
        self.config = self._set_config(dataset_config)
        self.ds: Dataset = ds
        self._streaming = streaming_dataset

    @property
    def columns(self):
        return list(self.ds.column_names)

    @property
    def features(self):
        return self.ds.features

    @property
    def version(self):
        return self.config.version

    def _set_config(self, dataset_config: DatasetConfig | dict | None) -> None:
        if isinstance(dataset_config, dict):
            return DatasetConfig(**dataset_config)
        elif isinstance(dataset_config, DatasetConfig):
            return dataset_config
        return DatasetConfig.from_skeleton()

    def set_ds(self, ds: Dataset) -> None:
        self.ds = ds

    def get_ds(self) -> Dataset:
        return self.ds

    def __getitem__(self, idx: int) -> dict:
        if self._streaming:
            raise TypeError("Cannot access samples in a streaming dataset")

        return self.ds[idx]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if self._streaming:
            raise TypeError("Cannot get length of a streaming dataset")

        return len(self.ds)

    def __iter__(self):
        return iter(self.ds)

    @classmethod
    def from_dict(cls, data: dict, dataset_config: DatasetConfig | dict) -> "HFDataset":
        """Create a dataset from a dictionary.
        The dictionary keys are the columns, and the values are lists of data.
        All lists must be the same length.
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

        if not all(len(data[key]) == len(data[list(data.keys())[0]]) for key in data.keys()):
            raise ValueError("All columns must have the same length")

        return cls(dataset_config, ds=Dataset.from_dict(data))

    @classmethod
    def from_samples(cls, samples: Iterable[DataSample | dict], dataset_config: DatasetConfig | dict) -> "HFDataset":
        """Create a dataset from a Iterable of DataSample records."""
        if len(samples) == 0:
            raise ValueError("Data must have at least one sample")

        new_ds = cls(dataset_config)

        if not isinstance(samples[0], dict):
            samples = [s.to_dict() for s in samples]

        new_ds.set_ds(Dataset.from_list(samples))

        return new_ds

    @classmethod
    def from_path(
        cls,
        path: str | os.PathLike,
        hf_dataset_type: str | None = None,
        storage_options: dict | None = None,
        streaming: bool = False,
        file_pattern: str = "shard*arrow",
        split: str = "train",
    ) -> "HFDataset":
        """Create a dataset from a path. This uses pyarrow so the dataset is memory-mapped.

        Args:
            path: The path to the dataset.
            storage_options: The storage options for cloud paths.
            streaming: Whether to stream the dataset.
            file_pattern: The file pattern to load.
            split: The split to load.

        Returns:
            The loaded HFDataset.
        """
        path = AnyPath(path)

        # load config from json
        config_file = AnyPath(path / "dataset_config.json")
        if not config_file.exists():
            logger.warning(f"Config file not found at {config_file}, creating an empty new one")
            config = DatasetConfig.from_skeleton()
        else:
            config = DatasetConfig.from_json(config_file)

        if hf_dataset_type == "local_hf":
            ds = load_from_disk(str(path), storage_options=storage_options)
            if streaming:
                ds = ds.to_iterable_dataset()

        elif hf_dataset_type in ["arrow", "csv", "tsv", "json", "parquet", "bucket_hf"]:
            ds_type = hf_dataset_type if hf_dataset_type != "bucket_hf" else "arrow"
            ds = load_dataset(
                ds_type,
                data_files=str(path / file_pattern),
                storage_options=storage_options,
                split=split,
                streaming=streaming,
            )
        elif hf_dataset_type in ["audiofolder", "imagefolder"]:
            ds = load_dataset(hf_dataset_type, data_dir=str(path), split=split, streaming=streaming)
        else:
            raise ValueError(f"Unsupported dataset type: {hf_dataset_type}")

        return cls(config, ds, streaming_dataset=streaming)

    def to_dict(self) -> dict:
        """Convert the dataset to a dictionary."""
        d = self.ds.to_dict()
        d["config"] = self.config.to_dict()
        return d

    def subset(self, indices: Iterable[int]) -> "HFDataset":
        """Return a subset of the dataset."""
        return HFDataset(self.config.copy(), ds=self.ds.select(indices), streaming_dataset=self._streaming)

    def sample(
        self,
        n: int,
        with_replacement: bool = False,
        probs: Optional[Iterable[float]] = None,
        normalize_probs: bool = True,
        seed: int = None,
    ) -> "HFDataset":
        """Return a random sample of the dataset, optionally with probabilities for each sample.

        Args:
            n: The number of samples to return.
            with_replacement: Whether to sample with replacement.
            probs: The probability of sampling each sample.
            normalize_probs: Whether to normalize the probabilities if unnormalized.
            seed: The random seed to use.

        Returns:
            A new HFDataset with the sampled data.

        Raises:
            ValueError: If n is greater than the length of the dataset.
            ValueError: If the probability vector is not the same length as the dataset.
            ValueError: If the probability vector is not normalized and normalize_probs is False.
        """
        ids = generate_random_indices(
            n,
            len(self),
            probs=probs,
            normalize_probs=normalize_probs,
            with_replacement=with_replacement,
            seed=seed,
        )
        return self.subset(indices=ids)

    def concatenate(
        self,
        other: "HFDataset",
        new_dataset_config: Optional[DatasetConfig | dict] = None,
        version_update_mode: str | None = None,
        change_log: str | None = None,
    ) -> "HFDataset":
        """Concatenate two datasets.

        Args:
            other: The other dataset to concatenate.
            new_dataset_config: The config for the new dataset. If None, it will be the same as the first dataset.
            version_update_mode: The version update mode to use, one of "major", "minor", "patch". Default is None, which
                will not update the version number.
            change_log: A change log to add to the dataset description.

        Returns:
            The new concatenated HFDataset.
        """
        # check the two datasets have the same features
        if list(set(self.columns)) != list(set(other.columns)):
            raise ValueError("Both datasets need to have the same columns")

        if new_dataset_config:
            new_ds = HFDataset(new_dataset_config, streaming_dataset=self._streaming)
        else:
            new_ds = HFDataset(self.config.copy(), streaming_dataset=self._streaming)
            if version_update_mode:
                new_ds.config.increment_version(mode=version_update_mode)

        new_ds.set_ds(concatenate_datasets([self.ds, other.ds]))

        if change_log:
            new_ds.config.update_changelog(change_log)
        else:
            s = f"""-> Concatenated two datasets. Dataset 1: {self.config.name}, Dataset 2: {other.config.name}
            Dataset 1 version: {self.config.version}, Dataset 2 version: {other.config.version}
            """
            new_ds.config.update_changelog(s)

        return new_ds

    def filter(
        self, condition: Callable, version_update_mode: str | None = None, change_log: str | None = None
    ) -> "HFDataset":
        """Filter the dataset for samples that meet a condition.

        Args:
            condition: The condition to filter by. This should be a function that takes a sample and returns a boolean.
            version_update_mode: The version update mode to use. Default is None.
            change_log: A change log to add to the dataset description.

        Returns:
            A new HFDataset with the filtered data.
        """
        new_ds = HFDataset(self.config.copy(), streaming_dataset=self._streaming)
        new_ds.set_ds(self.ds.filter(condition))

        if version_update_mode:
            new_ds.config.increment_version(mode=version_update_mode)

        if change_log:
            new_ds.config.update_changelog(change_log)
        else:
            new_ds.config.update_changelog(f"-> Filtered the dataset with condition {condition.__name__}")

        return new_ds

    def add_column(
        self,
        column_name: str,
        column_data: list[Any],
        version_update_mode: str | None = None,
        change_log: str | None = None,
    ) -> "HFDataset":
        """Add a column to the dataset.

        Args:
            column_name: The name of the column to add.
            column_data: The data to add to the column.
            version_update_mode: The version update mode to use. Default is None.
            change_log: A change log to add to the dataset description.
        """
        # TODO: create a decorated function, so that each sample has a
        # new DataSample config, with new id, and copy other stuff etc.
        new_ds = HFDataset(self.config.copy(), streaming_dataset=self._streaming)
        new_ds.set_ds(self.ds.add_column(column_name, column_data))

        if version_update_mode:
            new_ds.config.increment_version(mode=version_update_mode)

        if change_log:
            new_ds.config.update_changelog(change_log)
        else:
            new_ds.config.update_changelog(f"-> Added column {column_name}")

        return new_ds

    def append(self, samples: Iterable[DataSample]) -> "HFDataset":
        """Append samples to the dataset. Same as concatenating
        with a new dataset created from the samples."""
        return self.concatentate(HFDataset.from_samples(samples, self.config))

    def map(
        self,
        function: Callable,
        **map_kwargs,
    ) -> "HFDataset":
        """Apply a function over each sample in the dataset, creates a new dataset."""
        new_ds = HFDataset(self.config.copy(), streaming_dataset=self._streaming)
        new_ds.set_ds(self.ds.map(function, **map_kwargs))

        return new_ds

    def save_config(self, path: str | os.PathLike | AnyPath) -> None:
        """Save the dataset config to a local or cloud path."""
        g = AnyPath(path) / "dataset_config.json"
        self.config.write_json(g)

    def _save_streaming(
        self, path: AnyPath, num_samples_per_shard: int = 1000, storage_options: dict | None = None
    ) -> None:
        """Save a streaming dataset to a local or cloud path."""
        batch = []
        shard_id = 0
        for sample in self.ds:
            batch.append(sample)

            if len(batch) == num_samples_per_shard:
                write_huggingface_shard(
                    self.ds,
                    path,
                    shard_id=shard_id,
                    num_samples_per_shard=num_samples_per_shard,
                    sample_prep_function=None,  # None because the samples are already prepared in iterator
                    storage_options=storage_options,
                )
                shard_id += 1
                batch = []

        if batch:
            write_huggingface_shard(
                self.ds,
                path,
                shard_id=shard_id,
                num_samples_per_shard=num_samples_per_shard,
                sample_prep_function=None,
                storage_options=storage_options,
            )

    def save_to_path(
        self,
        path: str | os.PathLike | AnyPath,
        changelog: str | None = None,
        version_update_mode: Literal["major", "minor", "patch"] = None,
        num_samples_per_shard: int = 1000,
        max_shard_size: int | str | None = None,
        num_shards: int | None = None,
        num_proc: int | None = None,
        storage_options: dict | None = None,
    ) -> None:
        """Save the dataset to a local or cloud path.

        Args:
            path: The path to save the dataset to.
            save_config: Whether to save the dataset config.
            max_shard_size: The maximum size of a shard in bytes.
            num_shards: The number of shards to make. DO NOT provide both max_shard_size and num_shards.
            num_proc: The number of processes to use for saving.
            storage_options: The storage options for saving to cloud.
                see https://huggingface.co/docs/datasets/v3.2.0/en/filesystems#google-cloud-storage

        Raises:
            ValueError: If both max_shard_size and num_shards are provided.
        """
        path = AnyPath(path)

        # set storage options if not provided
        if not storage_options:
            storage_options = make_storage_options(storage_options)

        if changelog:
            self.config.update_changelog(changelog)

        if version_update_mode:
            self.config.increment_version(version_update_mode)

        if self._streaming:
            self._save_streaming(path, storage_options=storage_options, num_samples_per_shard=num_samples_per_shard)

        if max_shard_size and num_shards:
            raise ValueError("Provide either max_shard_size or num_shards, not both")

        self.ds.save_to_disk(
            str(path),
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            num_proc=num_proc,
            storage_options=storage_options,
        )

        self.save_config(path)
        self.config.generate_readme(path / "README.md")

    def __str__(self):
        if self._streaming:
            return f"HFDataset: {self.config.name}, version: {self.config.version}, streaming: True, columns: {self.columns}"
        return f"HFDataset: {self.config.name}, version: {self.config.version}, num samples: {len(self)}, columns: {self.columns}"

    def __repr__(self):
        return self.__str__()


def apply_fn(
    ds: HFDataset,
    function: Callable,
    changelog: str | None = None,
    version_update_mode: Literal["major", "minor", "patch"] = None,
    function_kwargs: dict = None,
    **map_kwargs,
) -> Generator[dict, None, None] | HFDataset:
    """Apply a function to a dataset, creates a new dataset.

    Args:
        ds: The dataset to apply the function to.
        function: The function to apply to each sample.
        changelog: A change log to add to the dataset description.
        version_update_mode: The version update mode to use. Default is None.
        function_kwargs: Any kwargs to pass to the function during mapping.
        map_kwargs: Any kwargs to pass to the map function.
    """
    ds: HFDataset = ds.map(function, fn_kwargs=function_kwargs, **map_kwargs)

    if changelog:
        ds.config.update_changelog(changelog)

    if version_update_mode:
        ds.config.increment_version(version_update_mode)

    if ds._streaming:
        for sample in ds:
            yield sample

    return ds


def concatenate_hf_datasets(
    datasets: Iterable[HFDataset],
    new_dataset_config: DatasetConfig | dict = None,
    version_update_mode: str | None = "patch",
) -> HFDataset:
    """Concatenate multiple HFDatasets and create a new dataset.

    Args:
        datasets: The datasets to concatenate.
        new_dataset_config: The config for the new dataset. If None, it will be the same as the first dataset.
        version_update_mode: The version update mode to use, one of "major", "minor", "patch". Default is "patch".
            If None, it will not update the version number.

    Returns:
        The new concatenated HFDataset.
    """
    if new_dataset_config:
        new_ds = HFDataset(new_dataset_config)
    else:
        new_ds = HFDataset(datasets[0].config.copy())
        if version_update_mode:
            new_ds.config.increment_version(mode=version_update_mode)

    new_ds.set_ds(concatenate_datasets([d.ds for d in datasets]))

    s = f"""-> Concatenated multiple datasets. Datasets: {[d.config.name for d in datasets]}
    """
    new_ds.config.update_changelog(s)

    return new_ds


def build_concatenated_dataset(
    dataset_paths: str | os.PathLike | AnyPath,
    storage_options: dict,
    sharded: bool = False,
    new_dataset_config: DatasetConfig | dict = None,
    version_update_mode: str | None = "patch",
) -> HFDataset:
    """Build a concatenated dataset from a list of dataset paths."""

    # load all datasets
    datasets = [HFDataset.from_path(p, storage_options=storage_options, sharded=sharded) for p in dataset_paths]
    new_ds = concatenate_hf_datasets(
        datasets, new_dataset_config=new_dataset_config, version_update_mode=version_update_mode
    )

    return new_ds
