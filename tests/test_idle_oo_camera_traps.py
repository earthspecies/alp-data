"""Unit tests for the IDLE-OO Camera Traps image dataset.

These tests load the manifests from GCS, so they require network access and
the ``gs://esp-data-ingestion/idle_oo_camera_traps/v0.1.0/`` upload to be
present (run ``jobs/build_idle_oo.sh`` first).

Run with:
    pytest -q tests/test_idle_oo_camera_traps.py
"""

from __future__ import annotations

import numpy as np
import pytest

from esp_data.datasets import IDLEOOCameraTraps

# BioCLIP 2's IDLE-OO Camera Traps benchmark: ~2,590 images, 119 species.
EXPECTED_NUM_SPECIES = 119
MIN_IMAGES = 2000


@pytest.fixture(scope="module")
def test_ds() -> IDLEOOCameraTraps:
    """Load the test split (pandas backend).

    Returns
    -------
    IDLEOOCameraTraps
        The loaded test split.
    """
    return IDLEOOCameraTraps(split="test", backend="pandas")


def test_available_splits(test_ds: IDLEOOCameraTraps) -> None:
    """available_splits should expose all/test."""
    for split in ["all", "test"]:
        assert split in test_ds.available_splits


def test_required_columns(test_ds: IDLEOOCameraTraps) -> None:
    """Manifest columns should be present."""
    expected = [
        "asset_id", "modality", "label", "canonical_name", "species_common",
        "original_label", "kingdom", "phylum", "class", "order", "family",
        "genus", "species", "split", "image_path",
    ]
    for col in expected:
        assert col in test_ds.columns, f"missing column {col}"


def test_species_count(test_ds: IDLEOOCameraTraps) -> None:
    """There should be exactly 119 distinct species."""
    rows = list(test_ds._data)
    assert len({str(r["canonical_name"]) for r in rows}) == EXPECTED_NUM_SPECIES
    assert len(rows) >= MIN_IMAGES


def test_image_item(test_ds: IDLEOOCameraTraps) -> None:
    """An image item should decode to an HWC uint8 array."""
    assert len(test_ds) > 0
    item = test_ds[0]
    assert item["modality"] == "image"
    assert isinstance(item["image"], np.ndarray)
    assert item["image"].dtype == np.uint8
    assert item["image"].ndim == 3
