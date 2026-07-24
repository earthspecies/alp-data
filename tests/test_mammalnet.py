"""Unit tests for the MammalNet behavior video dataset.

These tests load the manifests from GCS, so they require network access and
the ``gs://esp-data-ingestion/mammalnet/v0.1.0/`` upload to be present (run
``jobs/build_mammalnet.sh`` first). Video-decode tests require the optional
``video`` extra (PyAV); they are skipped when ``av`` is unavailable.

Run with:
    pytest -q tests/test_mammalnet.py
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from esp_data.datasets import MammalNet

EXPECTED_NUM_BEHAVIORS = 12
_HAS_AV = importlib.util.find_spec("av") is not None


@pytest.fixture(scope="module")
def test_ds() -> MammalNet:
    """Load the test split (pandas backend, 8 frames).

    Returns
    -------
    MammalNet
        The loaded test split.
    """
    return MammalNet(split="test", backend="pandas", max_frames=8)


def test_available_splits(test_ds: MammalNet) -> None:
    """available_splits should expose all/train/val/test."""
    for split in ["all", "train", "val", "test"]:
        assert split in test_ds.available_splits


def test_required_columns(test_ds: MammalNet) -> None:
    """Manifest columns should be present."""
    for col in ["asset_id", "modality", "label", "behavior", "split", "video_path"]:
        assert col in test_ds.columns, f"missing column {col}"


def test_behavior_count(test_ds: MammalNet) -> None:
    """The full behavior label space should be the 12 MammalNet behaviors."""
    behaviors = {str(r["behavior"]) for r in test_ds._data}
    assert len(behaviors) == EXPECTED_NUM_BEHAVIORS


@pytest.mark.skipif(not _HAS_AV, reason="PyAV (video extra) not installed")
def test_video_item(test_ds: MammalNet) -> None:
    """A clip should decode to frames (T,H,W,C uint8)."""
    assert len(test_ds) > 0
    item = test_ds[0]
    assert item["modality"] == "video"
    assert isinstance(item["video_frames"], np.ndarray)
    assert item["video_frames"].ndim == 4  # (T, H, W, C)
    assert item["video_frames"].shape[0] <= 8
