"""Base classes for paths that require validation."""

import os
from typing import Iterable

from esp_data.config.db_config import DataSample, DatasetConfig


class BaseMapDataset:
    """A base class for all map-style datasets."""

    config: DatasetConfig

    def __getitem__(self, idx: int, as_dict: bool = False) -> DataSample | dict:
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the length of the dataset."""
        raise NotImplementedError

    def from_samples(self, samples: Iterable[DataSample]) -> "BaseMapDataset":
        """Create a dataset from a Iterable of DataSample records."""
        raise NotImplementedError

    def from_path(self, path: str | os.PathLike) -> "BaseMapDataset":
        """Create a dataset from a path."""
        raise NotImplementedError

    def save_to_path(self, path: str | os.PathLike) -> None:
        """Save the dataset to a path."""
        raise NotImplementedError


class BaseIterableDataset:
    """A base class for all iterable-style datasets."""

    config: DatasetConfig

    def __iter__(self) -> Iterable[DataSample]:
        """Return an iterator over the dataset."""
        raise NotImplementedError

    def from_samples(self, samples: Iterable[DataSample]) -> "BaseIterableDataset":
        """Create a dataset from a Iterable of DataSample records."""
        raise NotImplementedError

    def from_path(self, path: str | os.PathLike) -> "BaseIterableDataset":
        """Create a dataset from a path."""
        raise NotImplementedError

    def save_to_path(self, path: str | os.PathLike) -> None:
        """Save the dataset to a path."""
        raise NotImplementedError
