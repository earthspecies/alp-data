"""Unit tests for the NABirds image dataset.

These tests load the manifests from GCS, so they require network access and
the ``gs://esp-data-ingestion/nabirds/v0.1.0/`` upload to be present (run
``jobs/build_nabirds.sh`` first).

Run with:
    pytest -q tests/test_nabirds.py
"""

from __future__ import annotations

import numpy as np
import pytest

from esp_data.datasets import NABirds

# NABirds has ~400 species; the exact count depends on the category->species
# roll-up, so we assert a sane range rather than a brittle exact value.
MIN_SPECIES = 350
MAX_SPECIES = 600


@pytest.fixture(scope="module")
def test_ds() -> NABirds:
    """Load the test split (pandas backend).

    Returns
    -------
    NABirds
        The loaded test split.
    """
    return NABirds(split="test", backend="pandas")


def test_available_splits(test_ds: NABirds) -> None:
    """available_splits should expose all/train/test."""
    for split in ["all", "train", "test"]:
        assert split in test_ds.available_splits


def test_required_columns(test_ds: NABirds) -> None:
    """Manifest columns should be present."""
    expected = [
        "asset_id", "modality", "label", "split", "species_code",
        "canonical_name", "species_common", "family", "order",
        "kingdom", "phylum", "class", "genus", "image_path",
    ]
    for col in expected:
        assert col in test_ds.columns, f"missing column {col}"


def test_gbif_linked(test_ds: NABirds) -> None:
    """Every row should carry a non-empty GBIF canonical name."""
    rows = list(test_ds._data)
    assert all(str(r.get("canonical_name", "")).strip() for r in rows)


def test_species_count(test_ds: NABirds) -> None:
    """The number of distinct species should be in the expected range."""
    rows = list(test_ds._data)
    n_species = len({str(r["canonical_name"]) for r in rows})
    assert MIN_SPECIES <= n_species <= MAX_SPECIES


def test_image_item(test_ds: NABirds) -> None:
    """An image item should decode to an HWC uint8 array."""
    assert len(test_ds) > 0
    item = test_ds[0]
    assert item["modality"] == "image"
    assert isinstance(item["image"], np.ndarray)
    assert item["image"].dtype == np.uint8
    assert item["image"].ndim == 3
