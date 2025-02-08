import json
import os
import warnings
from typing import Any, Callable, Iterable, Optional

from cloudpathlib import AnyPath
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

from esp_data.config.db_config import DataSample, DatasetConfig
from esp_data.config.project_config import REQUIRED_DATASAMPLE_FIELDS
from esp_data.file_io import File
from esp_data.utils import is_cloud_path, is_local_path, make_id, utc_now

from .base import BaseIterableDataset, BaseMapDataset
from .ds_utils import generate_random_indices

warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")


def wrap_sample_function(
    function: Callable, sample_config=DataSample, version_update_mode: str = "patch", **function_kwargs
) -> dict:
    """Creates a wrapper around a function that transforms each sample in the dataset
      to handle DataSample metadata updates.

      ASSUMPTION: This assumes that your function either implements:
        - A single sample transformation, where the function takes a sample as a dict and returns a transformed dict
        - A batch transformation, where the function takes a dict with keys = sample_config fields, and values = list of
            values for each field, length = batch_size, and returns a dict with the same structure.

    Args:
        function: The user's function to wrap. It should take a sample as a dict and return a transformed dict
        sample_config: The config class to use for samples
        static_kwargs: Any static kwargs that should always be passed to function
    """

    def update_sample(transformed_sample: dict) -> dict:
        # remove the id and created_at fields, as they will be updated
        sample_id = transformed_sample.pop("id", None)
        transformed_sample.pop("created_at", None)

        transformed_sample = sample_config(
            **transformed_sample,  # generates id and created_at
        )

        transformed_sample.derived_from = sample_id
        if transformed_sample.version is not None:
            transformed_sample.increment_version(mode=version_update_mode)

        return transformed_sample.to_dict()

    def update_batch(transformed_batch: dict) -> dict:
        """When batched=True and batch_size > 1 is part of map_kwargs,
        the function will return a dict with keys = sample_config fields,
        and values = list of values for each field, length = batch_size.
        """
        some_key = list(transformed_batch.keys())[0]
        batch_size = len(transformed_batch[some_key])  # batch size
        new_dict = {k: [] for k in transformed_batch.keys()}

        for i in range(batch_size):
            sample = {k: transformed_batch[k][i] for k in transformed_batch.keys()}
            transformed_sample: dict = update_sample(sample)
            for k, v in transformed_sample.items():
                new_dict[k].append(v)

        return new_dict

    def wrapped_function(sample: dict) -> dict:
        # First apply the user's function to get transformed data
        transformed_sample: dict = function(sample, **function_kwargs)

        # Now update the sample metadata, namely id and created_at
        some_key = list(transformed_sample.keys())[0]
        if isinstance(transformed_sample[some_key], list):
            # If the function returns a batch of samples, apply the update_batch function
            transformed_sample = update_batch(transformed_sample)
        else:
            # If the function returns a single sample, apply the update function
            transformed_sample = update_sample(transformed_sample)

        return transformed_sample

    return wrapped_function


class HFDataset(BaseMapDataset):
    def __init__(
        self, dataset_config: DatasetConfig | dict, ds: Optional[Dataset] = None, streaming_dataset: bool = False
    ):
        self.config = self._set_config(dataset_config)
        self.ds: Dataset = ds
        self._streaming = streaming_dataset

    @property
    def columns(self):
        return list(self.ds.column_names)

    @property
    def version(self):
        return self.config.version

    def _set_config(self, dataset_config) -> None:
        if isinstance(dataset_config, dict):
            return DatasetConfig(**dataset_config)
        elif isinstance(dataset_config, DatasetConfig):
            return dataset_config

    def set_ds(self, ds: Dataset) -> None:
        self.ds = ds

    def __getitem__(self, idx: int, as_dict: bool = True) -> DataSample | dict:
        if self._streaming:
            raise TypeError("Cannot access samples in a streaming dataset")

        d = self.ds[idx]
        if not as_dict:
            return DataSample(**d)

        return d

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if self._streaming:
            raise TypeError("Cannot get length of a streaming dataset")

        return len(self.ds)

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

        if not all([i in data for i in REQUIRED_DATASAMPLE_FIELDS]):
            raise ValueError("Data must have id, source_dataset, and metadata fields")

        return cls(dataset_config, ds=Dataset.from_dict(data))

    @classmethod
    def from_samples(cls, samples: Iterable[DataSample], dataset_config: DatasetConfig | dict) -> "HFDataset":
        """Create a dataset from a Iterable of DataSample records."""
        if len(samples) == 0:
            raise ValueError("Data must have at least one sample")

        new_ds = cls(dataset_config)

        if not isinstance(samples[0], dict):
            samples = [s.to_dict() for s in samples]

        if not all([i in samples[0] for i in REQUIRED_DATASAMPLE_FIELDS]):
            raise ValueError("Data must have id, source_dataset, and metadata fields")

        new_ds.set_ds(Dataset.from_list(samples))

        return new_ds

    @classmethod
    def from_path(
        cls, path: str | os.PathLike, storage_options: dict = None, sharded: bool = False, keep_in_memory: bool = None
    ) -> "HFDataset":
        """Create a dataset from a path. This uses pyarrow so the dataset is memory-mapped.

        Args:
            path: The path to the dataset.
            storage_options: The storage options for cloud paths.
            sharded: Whether the dataset is sharded, i.e. made of sub-directories called 'shard_1', 'shard_2' etc.
            If True, the dataset is concatenated from all shards.

            keep_in_memory: Whether to keep the dataset in memory. Default is None, the dataset will be kept in memory,
            upto datasets.config.IN_MEMORY_MAX_SIZE (which is default = 0 bytes). If True, the entire dataset will be
            kept in memory. If False, the dataset will be memory-mapped.

        Returns:
            The loaded HFDataset.
        """
        if not is_local_path(path) and not is_cloud_path(path):
            raise ValueError(f"Path {path} must be a local or cloud path")

        path = AnyPath(path)

        if is_cloud_path(path) and not storage_options:
            raise ValueError("""You need to provide a dict here,
            e.g. for google it is storage_options={"project": "my-google-project"}
            see https://huggingface.co/docs/datasets/v3.2.0/en/filesystems#google-cloud-storage for details
            """)

        # load config from json
        config_file = File(path / "dataset_config.json")
        if not config_file.exists:
            raise FileNotFoundError("No dataset config found")

        with config_file.open("r") as fp:
            config = json.load(fp)

        # load dataset using load_dataset / load_from_disk
        if sharded:
            # find all *shard_* directories
            b = AnyPath(path)
            shard_paths = list(b.rglob("*shard_*"))
            if len(shard_paths) == 0:
                raise FileNotFoundError("No shards found")

            ds = concatenate_datasets(
                [
                    load_from_disk(str(p), storage_options=storage_options, keep_in_memory=keep_in_memory)
                    for p in shard_paths
                ]
            )
        else:
            ds = load_from_disk(str(path), storage_options=storage_options, keep_in_memory=keep_in_memory)

        return cls(config, ds)

    def to_dict(self) -> dict:
        """Convert the dataset to a dictionary."""
        d = self.ds.to_dict()
        d["config"] = self.config.to_dict()
        return d

    def subset(self, indices: Iterable[int]) -> "HFDataset":
        """Return a subset of the dataset."""
        return HFDataset(self.config.copy(), ds=self.ds.select(indices))

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
            new_ds = HFDataset(new_dataset_config)
        else:
            new_ds = HFDataset(self.config.copy())
            if version_update_mode:
                new_ds.config.increment_version(mode=version_update_mode)

        new_ds.set_ds(concatenate_datasets([self.ds, other.ds]))

        if change_log:
            new_ds.config.description += "\n" + change_log
        else:
            s = f"""-> Concatenated two datasets. Dataset 1: {self.config.name}, Dataset 2: {other.config.name}
            Dataset 1 version: {self.config.version}, Dataset 2 version: {other.config.version}
            """
            new_ds.config.description += "\n" + s

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
        new_ds = HFDataset(self.config.copy())
        new_ds.set_ds(self.ds.filter(condition))

        if version_update_mode:
            new_ds.config.increment_version(mode=version_update_mode)

        if change_log:
            new_ds.config.description += "\n" + change_log
        else:
            new_ds.config.description += f"-> Filtered the dataset with condition {condition.__name__}"

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
        new_ds = HFDataset(self.config.copy())
        new_ds.set_ds(self.ds.add_column(column_name, column_data))

        if version_update_mode:
            new_ds.config.increment_version(mode=version_update_mode)

        if change_log:
            new_ds.config.description += "\n" + change_log
        else:
            new_ds.config.description += f"-> Added column {column_name}"

        return new_ds

    def append(self, samples: Iterable[DataSample]) -> "HFDataset":
        """Append samples to the dataset. Same as concatenating
        with a new dataset created from the samples."""
        return self.concatentate(HFDataset.from_samples(samples, self.config))

    def map(
        self,
        function: Callable,
        sample_config: "DataSample" = DataSample,
        version_update_mode: str | None = None,
        function_kwargs: dict = None,
        change_log: str | None = None,
        **map_kwargs,
    ) -> "HFDataset":
        """Apply a function over each sample in the dataset, creates a new dataset.

        Args:
            function: The function to apply to each sample
            sample_config: The config class to use for samples. It could be a subclass of DataSample.
            version_update_mode: The version update mode to use. Default is "patch".
            function_kwargs: Any kwargs to pass to the function during mapping
            change_log: A change log to add to the dataset description.
            map_kwargs: Any kwargs to pass to the map function

        Returns:
            A new HFDataset with the function applied to each sample
        """
        function_kwargs = function_kwargs or {}
        wrapped_function = wrap_sample_function(
            function, sample_config=sample_config, version_update_mode=version_update_mode, **function_kwargs
        )

        new_ds = HFDataset(self.config)
        new_ds.set_ds(self.ds.map(wrapped_function, **map_kwargs))
        if version_update_mode:
            # now update the version number of the dataset
            new_ds.config.increment_version(mode=version_update_mode)

        # TODO: this is a very rudimentary solution, there are much more elegant ways to do this
        if change_log:
            new_ds.config.description += "\n" + change_log
        else:
            new_ds.config.description += f"-> Applied function {function.__name__} to each sample"

        return new_ds

    def _validate_path(self, path: str | os.PathLike | AnyPath, storage_options: dict = None) -> AnyPath:
        if not is_local_path(path) and not is_cloud_path(path):
            raise ValueError(f"Path {path} must be a local or cloud path")

        path = AnyPath(path)

        if is_cloud_path(path) and not storage_options:
            raise ValueError("""You need to provide a dict here,
            e.g. for google it is storage_options={"project": "my-google-project"}
            see https://huggingface.co/docs/datasets/v3.2.0/en/filesystems#google-cloud-storage for details
            """)

        return path

    def save_config(self, path: str | os.PathLike | AnyPath) -> None:
        """Save the dataset config to a local or cloud path."""
        path = AnyPath(path)
        g = File(path / "dataset_config.json")
        g.create(exist_ok=True)

        with g.open("w") as fp:
            json.dump(self.config.to_dict(make_serializable=True), fp)

    def save_shard(
        self,
        path: str | os.PathLike | AnyPath,
        shard_idx: int,
        num_shards: int,
        storage_options: dict = None,
        num_proc: int = 8,
        save_config: bool = True,
    ) -> None:
        """Save a shard of the dataset to a local or cloud path.

        Args:
        """
        path = self._validate_path(path, storage_options)

        shard = self.ds.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
        shard.save_to_disk(
            str(path / f"shard_{shard_idx}"),
            num_proc=num_proc,
            storage_options=storage_options,
        )

        if save_config:
            self.save_config(path)

    def save_to_path(
        self,
        path: str | os.PathLike | AnyPath,
        save_config: bool = True,
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
        if max_shard_size and num_shards:
            raise ValueError("Provide either max_shard_size or num_shards, not both")

        path = self._validate_path(path, storage_options)

        self.ds.save_to_disk(
            str(path),
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            num_proc=num_proc,
            storage_options=storage_options,
        )

        if save_config:
            self.save_config(path)

    def __str__(self):
        return f"HFDataset: {self.config.name}, version: {self.config.version}, num samples: {len(self)}, columns: {self.columns}"

    def __repr__(self):
        return self.__str__()


class HFStreamingDataset(BaseIterableDataset):
    def __init__(self, dataset_config: DatasetConfig | dict, ds: Optional[Dataset] = None):
        self.config = self._set_config(dataset_config)
        self.ds: Dataset = ds

    def _set_config(self, dataset_config) -> None:
        if isinstance(dataset_config, dict):
            return DatasetConfig(**dataset_config)
        elif isinstance(dataset_config, DatasetConfig):
            return dataset_config

    def __getitem__(self, idx: int, as_dict: bool = True) -> DataSample | dict:
        raise TypeError("Cannot access samples in a streaming dataset")

    def __len__(self) -> int:
        raise TypeError("Cannot get length of a streaming dataset")

    def __iter__(self):
        return iter(self.ds)

    # TODO:
    def map(self, function: Callable, **map_kwargs) -> "HFStreamingDataset":
        pass

    # TODO:
    def filter(self, condition: Callable) -> "HFStreamingDataset":
        pass

    # TODO:
    def save_to_path(self, path: str | os.PathLike | AnyPath) -> None:
        pass

    @classmethod
    def from_path(
        cls, path: str | os.PathLike, storage_options: dict = None, file_format: str = "arrow"
    ) -> "HFStreamingDataset":
        """Create a dataset from a path. This uses pyarrow so the dataset is memory-mapped.
        Uses load_dataset technique
        Args:

        """
        if not is_local_path(path) and not is_cloud_path(path):
            raise ValueError(f"Path {path} must be a local or cloud path")

        path = AnyPath(path)

        if is_cloud_path(path) and not storage_options:
            raise ValueError("""You need to provide a dict here,
            e.g. for google it is storage_options={"project": "my-google-project"}
            see https://huggingface.co/docs/datasets/v3.2.0/en/filesystems#google-cloud-storage for details
            """)

        # load config from json
        config_file = File(path / "dataset_config.json")
        if not config_file.exists:
            raise FileNotFoundError("No dataset config found")

        with config_file.open("r") as fp:
            config = json.load(fp)

        # find all files in the directory with the file_format
        files = [str(s) for s in path.rglob(f"*.{file_format}")]
        if len(files) == 0:
            raise FileNotFoundError("No files found")

        ds = load_dataset(path=file_format, data_files=files, storage_options=storage_options, streaming=True)

        return cls(config, ds=ds)


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
            new_ds.config.increment_version(mode="patch")

    new_ds.set_ds(concatenate_datasets([d.ds for d in datasets]))

    s = f"""-> Concatenated multiple datasets. Datasets: {[d.config.name for d in datasets]}
    """
    new_ds.config.description += "\n" + s

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
