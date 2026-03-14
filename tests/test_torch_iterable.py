"""Integration tests for Dataset.as_torch_iterable().

These tests require torch, which is provided by the 'benchmark' dependency
group. The entire module is skipped when torch is not installed.
"""

import pytest
from typing import Any, Dict, Iterator

# Skip this module if torch is not installed (i.e. benchmark group not synced)
torch = pytest.importorskip("torch")

import pandas as pd

from esp_data import Dataset, DatasetInfo
from esp_data.backends import PandasBackend


class _TorchTestDataset(Dataset):
    """Minimal in-memory dataset used only for these tests."""

    info = DatasetInfo(
        name="_torch_test_dataset",
        owner="test",
        split_paths={"train": "dummy"},
        version="0.1.0",
        description="Minimal in-memory dataset for torch iterable tests",
        sources=["test"],
    )

    def __init__(self) -> None:
        super().__init__()
        df = pd.DataFrame({"value": list(range(5))})
        self._data = PandasBackend(df, streaming=False)

    @property
    def available_splits(self) -> list[str]:
        return ["train"]

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns)

    def _load(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._data[idx]

    def __str__(self) -> str:
        return "_TorchTestDataset"

    @classmethod
    def from_config(cls, dataset_config: Any) -> tuple["_TorchTestDataset", dict]:
        return cls(), {}


def test_as_torch_iterable_returns_self():
    """as_torch_iterable() should return the same dataset instance."""
    ds = _TorchTestDataset()
    with pytest.warns(UserWarning, match="worker_init_fn"):
        result = ds.as_torch_iterable()
    assert result is ds


def test_as_torch_iterable_isinstance_iterable_dataset():
    """After as_torch_iterable(), the dataset must be an IterableDataset."""
    from torch.utils.data import IterableDataset

    ds = _TorchTestDataset()
    with pytest.warns(UserWarning, match="worker_init_fn"):
        ds.as_torch_iterable()
    assert isinstance(ds, IterableDataset)


def test_as_torch_iterable_dataloader():
    """A converted dataset must iterate correctly through a DataLoader with num_workers=0."""
    from torch.utils.data import DataLoader

    ds = _TorchTestDataset()
    with pytest.warns(UserWarning, match="worker_init_fn"):
        ds.as_torch_iterable()

    loader = DataLoader(ds, batch_size=2, num_workers=0)
    batches = list(loader)
    total_samples = sum(len(b["value"]) for b in batches)
    assert total_samples == 5


def test_as_torch_iterable_dataloader_multiworker():
    """With num_workers > 0 and no worker_init_fn, each worker iterates the full dataset.

    This test documents the known duplication behaviour. Users must supply a
    worker_init_fn to the DataLoader to distribute indices across workers.
    """
    from torch.utils.data import DataLoader

    ds = _TorchTestDataset()
    with pytest.warns(UserWarning, match="worker_init_fn"):
        ds.as_torch_iterable()

    loader = DataLoader(ds, batch_size=2, num_workers=2)
    batches = list(loader)
    total_samples = sum(len(b["value"]) for b in batches)
    # Each of the 2 workers iterates all 5 items → 10 total
    assert total_samples == 10


def test_as_torch_iterable_idempotent():
    """Calling as_torch_iterable() twice should not raise or break iteration."""
    from torch.utils.data import IterableDataset

    ds = _TorchTestDataset()
    with pytest.warns(UserWarning, match="worker_init_fn"):
        ds.as_torch_iterable()
    with pytest.warns(UserWarning, match="worker_init_fn"):
        ds.as_torch_iterable()
    assert isinstance(ds, IterableDataset)
    assert list(ds) == [{"value": i} for i in range(5)]
