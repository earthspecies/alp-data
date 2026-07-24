"""Unit tests for the Animal Kingdom (AR) action video dataset.

These tests load the manifests from GCS, so they require network access and
the ``gs://esp-data-ingestion/animal_kingdom/v0.1.0/`` upload to be present
(run ``jobs/build_animal_kingdom.sh`` first). Video-decode tests require the
optional ``video`` extra (PyAV); they are skipped when ``av`` is unavailable.

Run with:
    pytest -q tests/test_animal_kingdom.py
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from esp_data.datasets import AnimalKingdom

# Animal Kingdom AR has 140 action classes; a split may not contain all of them.
MIN_ACTIONS = 100
MAX_ACTIONS = 140
_HAS_AV = importlib.util.find_spec("av") is not None


@pytest.fixture(scope="module")
def test_ds() -> AnimalKingdom:
    """Load the test split (pandas backend, 8 frames).

    Returns
    -------
    AnimalKingdom
        The loaded test split.
    """
    return AnimalKingdom(split="test", backend="pandas", max_frames=8)


def test_available_splits(test_ds: AnimalKingdom) -> None:
    """available_splits should expose all/train/test."""
    for split in ["all", "train", "test"]:
        assert split in test_ds.available_splits


def test_required_columns(test_ds: AnimalKingdom) -> None:
    """Manifest columns should be present."""
    for col in ["asset_id", "modality", "labels", "split", "video_path"]:
        assert col in test_ds.columns, f"missing column {col}"


def test_action_vocabulary(test_ds: AnimalKingdom) -> None:
    """The union of per-clip action sets should span most of the 140 actions."""
    actions = set()
    for r in test_ds._data:
        for tok in str(r["labels"]).split(","):
            t = tok.strip()
            if t and t.lower() != "none":
                actions.add(t)
    assert MIN_ACTIONS <= len(actions) <= MAX_ACTIONS


@pytest.mark.skipif(not _HAS_AV, reason="PyAV (video extra) not installed")
def test_video_item(test_ds: AnimalKingdom) -> None:
    """A clip should decode to frames (T,H,W,C uint8)."""
    assert len(test_ds) > 0
    item = test_ds[0]
    assert item["modality"] == "video"
    assert isinstance(item["video_frames"], np.ndarray)
    assert item["video_frames"].ndim == 4  # (T, H, W, C)
    assert item["video_frames"].shape[0] <= 8
