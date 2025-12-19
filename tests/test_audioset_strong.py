"""
Unit tests for audioset_strong dataset.

Run with:
    pytest -q test_audioset_strong.py
"""

from __future__ import annotations

import hashlib
import random
from typing import List

import numpy as np
import pandas as pd
import polars as pl
import pytest

from esp_data.datasets import AudioSetStrong

EXPECTED_LEN_TRAIN = 8115
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "47b0c29f8f2b206bfceceb9923d7a63670b596d6014f06208795ea79f8db9af8"
)
ANNOTATIONS_SHA256 = "3d5be67b9fff56146d67523b258cc1a165cd3fbb198aacb6a7e7c2f3ab7fc797"


@pytest.fixture(scope="module")
def ds() -> AudioSetStrong:
    """Load AudioSetStrong dataset for testing.

    Uses 32kHz sample rate (native pre-resampled rate) and pandas backend
    for hash comparison compatibility.
    """
    return AudioSetStrong(split="train", sample_rate=32000, backend="pandas")


@pytest.fixture(scope="module")
def sample_indices(ds: AudioSetStrong) -> List[int]:
    """Deterministically choose up to 5 random indices for quick spot checks."""
    n = len(ds)
    rng = random.Random(42)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: AudioSetStrong):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_dataset_length_matches_expected(ds: AudioSetStrong):
    """The dataset length should match the known, version-controlled expectation."""
    assert len(ds) == EXPECTED_LEN_TRAIN, (
        f"Dataset length mismatch: got {len(ds)}, expected {EXPECTED_LEN_TRAIN}. "
        "If this change is intentional (new data / new filtering), update EXPECTED_LEN_TRAIN."
    )


def test_check_audio(ds: AudioSetStrong, sample_indices: List[int]):
    """Basic audio integrity checks on a few random items."""
    for idx in sample_indices:
        item = ds[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert audio.dtype == np.float32, f"[{idx}] audio dtype is {audio.dtype}, expected float32"
        assert audio.size >= 32000, f"[{idx}] audio too short (size={audio.size})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


def test_reference_item_stability(ds: AudioSetStrong):
    """Check that a canonical item (index 0) is bitwise-stable.

    We hash the raw float32 audio buffer and annotations CSV.
    """
    idx = 0
    item = ds[idx]

    assert "audio" in item, "[0] missing 'audio' key"
    audio = item["audio"]
    assert isinstance(audio, np.ndarray), "[0] audio is not a numpy array"
    assert audio.dtype == np.float32, f"[0] audio dtype is {audio.dtype}, expected float32"

    h = hashlib.sha256(audio.tobytes()).hexdigest()
    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )

    csv_bytes = (
        ds._data.unwrap.sort_index(axis=0)
        .sort_index(axis=1)
        .to_csv(index=True)
        .encode("utf-8")
    )
    h = hashlib.sha256(csv_bytes).hexdigest()
    assert h == ANNOTATIONS_SHA256, (
        "Annotation's hash changed.\n"
        f"Got    {h}\n"
        f"Expect {ANNOTATIONS_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace ANNOTATIONS_SHA256 with the new hash."
    )


def test_check_selection_table(ds: AudioSetStrong, sample_indices: List[int]):
    """Selection table should be a polars DataFrame with required columns and sane times."""
    required = {"Begin Time (s)", "End Time (s)", "Label"}

    for idx in sample_indices:
        item = ds[idx]
        assert "selection_table" in item, f"[{idx}] missing 'selection_table' key"
        st = item["selection_table"]

        assert isinstance(st, pd.DataFrame), f"[{idx}] selection_table is not a DataFrame"
        missing = required - set(st.columns)
        assert not missing, f"[{idx}] selection_table missing columns: {sorted(missing)}"

        if len(st) > 0:
            assert not (st["Begin Time (s)"] < 0).any(), f"[{idx}] negative begin times present"


def test_get_available_labels(ds: AudioSetStrong, sample_indices: List[int]):
    """Test that labels in selection tables are valid strings."""
    labels_sample = set()
    for idx in sample_indices:
        item = ds[idx]
        st = item["selection_table"]
        if "Label" in st.columns and len(st) > 0:
            labels_sample.update(st["Label"].to_list())

    assert len(labels_sample) > 0, "Should have at least one label"
    for label in labels_sample:
        assert isinstance(label, str), f"Label {label} should be string"


def test_annotation_columns_attribute(ds: AudioSetStrong):
    """Check that annotation_columns is set correctly."""
    assert hasattr(ds, "annotation_columns"), "Dataset should have annotation_columns attribute"
    assert ds.annotation_columns == ["Label"], f"Expected ['Label'], got {ds.annotation_columns}"


def test_str_representation(ds: AudioSetStrong):
    """Test if string representation works correctly."""
    str_repr = str(ds)
    assert "audioset_strong" in str_repr
    assert "0.1.0" in str_repr
    assert "train" in str_repr


def test_available_sample_rates(ds: AudioSetStrong):
    """Test if available_sample_rates property works correctly."""
    sample_rates = ds.available_sample_rates
    assert isinstance(sample_rates, list)
    assert "32khz_path" in ds.columns, "32khz_path column should exist"
    assert 32000 in sample_rates, "32kHz should be available"


def test_invalid_split():
    """Test if initializing with invalid split raises error."""
    with pytest.raises(LookupError):
        AudioSetStrong(split="invalid_split")
