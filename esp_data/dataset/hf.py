import json
import os
import warnings
from typing import Any, Callable, Iterable, Optional

from cloudpathlib import AnyPath
from datasets import Dataset, concatenate_datasets, load_from_disk

from esp_data.config.db_config import DataSample, DatasetConfig
from esp_data.config.project_config import REQUIRED_DATASAMPLE_FIELDS
from esp_data.dataset.base import BaseMapDataset
from esp_data.file_io import File
from esp_data.utils import is_cloud_path, is_local_path

from .ds_utils import generate_random_indices

warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")


def wrap_sample_function(
    function: Callable, sample_config=DataSample, version_update_mode: str = "patch", **function_kwargs
) -> dict:
    """Creates a wrapper around a function that transforms each sample in the dataset
      to handle DataSample metadata updates.

      ASSUMPTION: This assumes that your function is a one-to-one map not a one-to-many map, i.e.
      doesn't create more than one new sample from one sample.

    Args:
        function: The user's function to wrap. It should take a sample as a dict and return a transformed dict
        sample_config: The config class to use for samples
        static_kwargs: Any static kwargs that should always be passed to function
    """

    def wrapped_function(sample: dict) -> dict:
        # First apply the user's function to get transformed data
        transformed_sample: dict = function(sample, **function_kwargs)
        # remove the id and created_at fields, as they will be updated
        sample_id = transformed_sample.pop("id", None)
        transformed_sample.pop("created_at", None)

        # Create new sample config with updated metadata
        transformed_sample = sample_config(
            **transformed_sample,
        )
        # update the derived_from field
        transformed_sample.derived_from = sample_id
        if transformed_sample.version is not None:
            # update the version number of the sample if it was provided
            transformed_sample.increment_version(mode=version_update_mode)

        return transformed_sample.to_dict()

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
    def from_path(cls, path: str | os.PathLike, storage_options: dict = None, sharded: bool = False) -> "HFDataset":
        """Create a dataset from a path. This uses pyarrow so the dataset is memory-mapped.

        Args:
            path: The path to the dataset.
            storage_options: The storage options for cloud paths.
            sharded: Whether the dataset is sharded. If True, the dataset is concatenated from all shards.

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
            ds = concatenate_datasets([load_from_disk(str(p), storage_options=storage_options) for p in shard_paths])
        else:
            ds = load_from_disk(str(path), storage_options=storage_options)

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
        version_update_mode: str = "patch",
        change_log: str | None = None,
    ) -> "HFDataset":
        """Concatenate two datasets."""
        # check the two datasets have the same features
        if list(set(self.columns)) != list(set(other.columns)):
            raise ValueError("Both datasets need to have the same columns")

        if new_dataset_config:
            new_ds = HFDataset(new_dataset_config)
        else:
            new_ds = HFDataset(self.config.copy())
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
        self, condition: Callable, version_update_mode: str = "patch", change_log: str | None = None
    ) -> "HFDataset":
        """Filter the dataset for samples that meet a condition."""
        # FIXME: should this lead to a new version ?
        new_ds = HFDataset(self.config.copy())
        new_ds.set_ds(self.ds.filter(condition))
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
        version_update_mode: str = "patch",
        change_log: str | None = None,
    ) -> "HFDataset":
        """Add a column to the dataset."""
        # TODO: create a decorated function, so that each sample has a
        # new DataSample config, with new id, and copy other stuff etc.
        new_ds = HFDataset(self.config.copy())
        new_ds.set_ds(self.ds.add_column(column_name, column_data))
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
        version_update_mode: str = "patch",
        function_kwargs: dict = None,
        change_log: str | None = None,
        **map_kwargs,
    ) -> "HFDataset":
        """Apply a function over each sample in the dataset, creates a new dataset.

        Args:
            function: The function to apply to each sample
            sample_config: The config class to use for samples. It could be a subclass of DataSample.
            version_update_mode: The version update mode to use
            function_kwargs: Any kwargs to pass to the function during mapping
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
