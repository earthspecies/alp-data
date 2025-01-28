"""Base classes for paths that require validation."""

import os
from typing import Callable, Iterable, Optional
from esp_data.config.db_config import DataSample, DatasetConfig


class BaseDataset:
    """A base class for all datasets."""

    config: DatasetConfig

    def __getitem__(self, idx: int, as_dict: bool = False) -> DataSample | dict:
        raise NotImplementedError

    @property
    def __len__(self) -> int:
        """Return the length of the dataset."""
        raise NotImplementedError

    def subset(self, indices: Iterable[int]) -> "BaseDataset":
        """Return a subset of the dataset."""
        raise NotImplementedError

    def sample(self, n: int, probabilities: Optional[Iterable[float]] = None) -> "BaseDataset":
        """Return a random sample of the dataset, optionally with probabilities for each sample."""
        raise NotImplementedError

    def from_dict(self, data: dict) -> "BaseDataset":
        """Create a dataset from a dictionary."""
        raise NotImplementedError

    def from_samples(self, samples: Iterable[DataSample]) -> "BaseDataset":
        """Create a dataset from a Iterable of DataSample records."""
        raise NotImplementedError

    def from_path(self, path: str | os.PathLike) -> "BaseDataset":
        """Create a dataset from a path."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Convert the dataset to a dictionary."""
        raise NotImplementedError

    def concatentate(self, other: "BaseDataset") -> "BaseDataset":
        """Concatenate two datasets."""
        raise NotImplementedError

    def filter(self, condition: Callable) -> "BaseDataset":
        """Filter the dataset for samples that meet a condition."""
        raise NotImplementedError

    def map(self, function: Callable) -> "BaseDataset":
        """Map a function over the dataset."""
        raise NotImplementedError

    def save_to_path(self, path: str | os.PathLike) -> None:
        """Save the dataset to a path."""
        raise NotImplementedError
