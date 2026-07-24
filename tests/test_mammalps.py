"""Unit tests for the MammAlps Benchmark I audiovisual dataset.

These tests load the manifests from GCS, so they require network access and
the ``gs://esp-data-ingestion/mammalps/v0.1.0/`` upload to be present (run
``jobs/build_mammalps.sh`` first). Video-decode tests require the optional
``video`` extra (PyAV); they are skipped when ``av`` is unavailable.

Run with:
    pytest -q tests/test_mammalps.py
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from esp_data.datasets import MammAlps

EXPECTED_NUM_ACTIVITIES = 11
_HAS_AV = importlib.util.find_spec("av") is not None


@pytest.fixture(scope="module")
def test_ds() -> MammAlps:
    """Load the test split (pandas backend, 8 frames, 16 kHz audio).

    Returns
    -------
    MammAlps
        The loaded test split.
    """
    return MammAlps(split="test", backend="pandas", max_frames=8, sample_rate=16000)


def test_available_splits(test_ds: MammAlps) -> None:
    """available_splits should expose all/train/val/test."""
    for split in ["all", "train", "val", "test"]:
        assert split in test_ds.available_splits


def test_required_columns(test_ds: MammAlps) -> None:
    """Manifest columns should be present."""
    for col in ["asset_id", "modality", "activity", "species", "split", "video_path"]:
        assert col in test_ds.columns, f"missing column {col}"


def test_activity_label_space(test_ds: MammAlps) -> None:
    """Activities present should be within the 11-activity label space."""
    activities = {str(r["activity"]) for r in test_ds._data}
    assert 0 < len(activities) <= EXPECTED_NUM_ACTIVITIES


@pytest.mark.skipif(not _HAS_AV, reason="PyAV (video extra) not installed")
def test_video_and_audio_item(test_ds: MammAlps) -> None:
    """A clip should decode to frames plus an aligned audio track."""
    assert len(test_ds) > 0
    item = test_ds[0]
    assert item["modality"] == "video"
    assert isinstance(item["video_frames"], np.ndarray)
    assert item["video_frames"].ndim == 4  # (T, H, W, C)
    assert item["video_frames"].shape[0] <= 8
    # Aligned audio (best-effort): present for most clips.
    if item["audio"] is not None:
        assert item["audio"].ndim == 1
        assert item["sample_rate"] == 16000
