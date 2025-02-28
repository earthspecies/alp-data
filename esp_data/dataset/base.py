"""Base classes for paths that require validation."""

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


class BaseIterableDataset:
    """A base class for all iterable-style datasets."""

    config: DatasetConfig

    def __iter__(self) -> Iterable[DataSample]:
        """Return an iterator over the dataset."""
        raise NotImplementedError
