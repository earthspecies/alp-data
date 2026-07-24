"""Unit tests for the NeWT image dataset.

These tests load the manifests from GCS, so they require network access and
the ``gs://esp-data-ingestion/newt/v0.1.0/`` upload to be present (run
``jobs/build_newt.sh`` first).

Run with:
    pytest -q tests/test_newt.py
"""

from __future__ import annotations

import numpy as np
import pytest

from esp_data.datasets import NeWT

# NeWT is a benchmark of 164 binary tasks (visipedia/newt).
EXPECTED_NUM_TASKS = 164


@pytest.fixture(scope="module")
def test_ds() -> NeWT:
    """Load the test split (pandas backend).

    Returns
    -------
    NeWT
        The loaded test split.
    """
    return NeWT(split="test", backend="pandas")


def test_available_splits(test_ds: NeWT) -> None:
    """available_splits should expose all/train/test."""
    for split in ["all", "train", "test"]:
        assert split in test_ds.available_splits


def test_required_columns(test_ds: NeWT) -> None:
    """Manifest columns should be present."""
    expected = [
        "asset_id", "modality", "label", "text_label", "task",
        "task_cluster", "task_subcluster", "split", "image_path",
    ]
    for col in expected:
        assert col in test_ds.columns, f"missing column {col}"


def test_labels_are_binary(test_ds: NeWT) -> None:
    """Every task label should be binary (0 / 1)."""
    rows = list(test_ds._data)
    assert {int(r["label"]) for r in rows} <= {0, 1}


def test_task_count() -> None:
    """The full dataset should contain all 164 NeWT tasks."""
    ds = NeWT(split="all", backend="pandas")
    tasks = {str(r["task"]) for r in ds._data}
    assert len(tasks) == EXPECTED_NUM_TASKS


def test_image_item(test_ds: NeWT) -> None:
    """An image item should decode to an HWC uint8 array."""
    assert len(test_ds) > 0
    item = test_ds[0]
    assert item["modality"] == "image"
    assert isinstance(item["image"], np.ndarray)
    assert item["image"].dtype == np.uint8
    assert item["image"].ndim == 3
