"""
Unit tests for WABAD dataset.

Run with:
    pytest -q test_wabad.py
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import pandas as pd
import pytest

from esp_data.datasets import WABAD


@pytest.fixture(scope="module")
def ds() -> WABAD:
    """Load WABAD dataset for testing."""
    return WABAD(
        split="all",
        sample_rate=16000,
    )


@pytest.fixture(scope="module")
def sample_indices(ds: WABAD) -> List[int]:
    """Deterministically choose up to 5 random indices for quick spot checks."""
    n = len(ds)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: WABAD):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_check_audio(ds: WABAD, sample_indices: List[int]):
    """Basic audio integrity checks on a few random items."""
    for idx in sample_indices:
        item = ds[idx]
        audio = item.get("audio", None)

        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert (
            audio.dtype == np.float32
        ), f"[{idx}] audio dtype is {audio.dtype}, expected float32"
        assert audio.size >= 10, f"[{idx}] audio too short (size={audio.size})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


def test_get_available_labels(ds: WABAD):
    """Test get_available_labels for bird ID column."""
    labels = ds.get_available_labels(anno_column="Species")
    assert isinstance(labels, list), "get_available_labels should return a list"
    assert len(labels) > 0, "Should have at least one bird ID"
    # Check that all labels can be converted to strings
    for label in labels:
        assert isinstance(label, str), f"Species label for {label} should be string"


def test_check_selection_table(ds: WABAD, sample_indices: List[int]):
    """Selection table should be a DataFrame with required columns and sane times."""
    required = {"Begin Time (s)", "End Time (s)", "Species"}

    for idx in sample_indices:
        item = ds[idx]
        st = item.get("selection_table", None)

        assert isinstance(
            st, pd.DataFrame
        ), f"[{idx}] selection_table is not a DataFrame"
        missing = required - set(st.columns)
        assert (
            not missing
        ), f"[{idx}] selection_table missing columns: {sorted(missing)}"

        if len(st) > 0:
            assert not (
                st["Begin Time (s)"] < 0
            ).any(), f"[{idx}] negative begin times present"


def test_label_mapping_loaded(ds: WABAD):
    """Sanity check that label mappings for expected taxonomy fields are present."""
    expected = {"Genus", "Family", "Order", "Common", "Species"}
    assert set(ds.annotation_columns) == expected
    for k in ds.annotation_columns:
        assert isinstance(
            ds.label_mappings.get(k, {}), dict
        ), f"label_mappings[{k}] is not a dict"
