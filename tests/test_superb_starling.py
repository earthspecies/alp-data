"""
Unit tests for superb_starling dataset.

Run with:
    pytest -q test_superb_starling.py
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import pandas as pd
import pytest
import hashlib

#from superb_starling_dataset import SuperbStarling
from esp_data.datasets.superb_starling_dataset import SuperbStarling

# --- Dataset snapshot ---

# # Code to generate snapshot:
# import hashlib
# from esp_data.datasets import SuperbStarling
# ds = SuperbStarling(split="all", sample_rate=16000)

# print("len(ds) =", len(ds))

# audio0 = ds[0]["audio"]
# print("dtype:", audio0.dtype, "shape:", audio0.shape)

# h = hashlib.sha256(audio0.tobytes()).hexdigest()
# print("sha256:", h)

# csv_bytes = ds._data.sort_index(axis=0).sort_index(axis=1).to_csv(index=True).encode("utf-8")
# h = hashlib.sha256(csv_bytes).hexdigest()

# print("annotations sha256:", h)

# quit()
# # #

# TODO: Run the code above once you have your dataset working to generate these values
EXPECTED_LEN_ALL = 2179  # Update this after running snapshot generation
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "REPLACE_WITH_ACTUAL_HASH"  # Update this after running snapshot generation
)
ANNOTATIONS_SHA256 = (
    "REPLACE_WITH_ACTUAL_HASH"  # Update this after running snapshot generation
)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds() -> SuperbStarling:
    """Load SuperbStarling dataset for testing."""
    return SuperbStarling(split="all", sample_rate=16000)


@pytest.fixture(scope="module")
def sample_indices(ds: SuperbStarling) -> List[int]:
    """Deterministically choose up to 5 random indices for quick spot checks."""
    n = len(ds)
    rng = random.Random(42)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: SuperbStarling):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_expected_length(ds: SuperbStarling):
    """Check that dataset has expected number of vocalizations."""
    actual_len = len(ds)
    assert actual_len == EXPECTED_LEN_ALL, (
        f"Dataset length changed: expected {EXPECTED_LEN_ALL}, got {actual_len}. "
        "If this is intentional, update EXPECTED_LEN_ALL."
    )


def test_check_audio(ds: SuperbStarling, sample_indices: List[int]):
    """Basic audio integrity checks on a few random items."""
    for idx in sample_indices:
        item = ds[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert audio.dtype == np.float32, f"[{idx}] audio dtype is {audio.dtype}, expected float32"
        assert audio.size >= 10, f"[{idx}] audio too short (size={audio.size})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


def test_check_sample_rate(ds: SuperbStarling, sample_indices: List[int]):
    """Verify that sample_rate field is present and correct."""
    for idx in sample_indices:
        item = ds[idx]
        assert "sample_rate" in item, f"[{idx}] missing 'sample_rate' key"
        sr = item["sample_rate"]
        assert sr == 16000, f"[{idx}] expected sample_rate=16000, got {sr}"


def test_check_duration(ds: SuperbStarling, sample_indices: List[int]):
    """Check that duration_s field is present and matches audio length."""
    for idx in sample_indices:
        item = ds[idx]
        assert "duration_s" in item, f"[{idx}] missing 'duration_s' key"
        duration = item["duration_s"]
        
        # Verify duration matches audio length
        audio_len = len(item["audio"])
        sr = item["sample_rate"]
        expected_duration = audio_len / sr
        
        assert abs(duration - expected_duration) < 1e-6, (
            f"[{idx}] duration mismatch: duration_s={duration}, "
            f"audio_len/sr={expected_duration}"
        )


def test_reference_item_stability(ds: SuperbStarling):
    """
    Check that a canonical item (index 0) is bitwise-stable.

    We hash the raw float32 audio buffer. This catches:
    - sample rate changes (resampling -> different samples)
    - channel handling changes (stereo->mono logic changed)
    - dtype changes
    - ordering changes in the split (if a different recording moved to idx 0)

    If this fails for a legitimate/intentional reason, recompute the hash
    and update EXPECTED_FIRST_ITEM_AUDIO_SHA256.

    We do the same for the annotations dataframe.
    """
    # choose deterministic index
    idx = 0
    item = ds[idx]

    # audio presence/type checks (defensive, so the hash failure message is clearer)
    assert "audio" in item, "[0] missing 'audio' key"
    audio = item["audio"]
    assert isinstance(audio, np.ndarray), "[0] audio is not a numpy array"
    assert audio.dtype == np.float32, (
        f"[0] audio dtype is {audio.dtype}, expected float32"
    )

    # compute sha256 over raw bytes of the float32 array
    h = hashlib.sha256(audio.tobytes()).hexdigest()

    if EXPECTED_FIRST_ITEM_AUDIO_SHA256 != "REPLACE_WITH_ACTUAL_HASH":
        assert (
            h == EXPECTED_FIRST_ITEM_AUDIO_SHA256
        ), (
            "First item's audio hash changed.\n"
            f"Got    {h}\n"
            f"Expect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
            "If this is an intentional dataset/content update, "
            "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
        )

    # compute sha256 over annotations dataframe
    csv_bytes = ds._data.sort_index(axis=0).sort_index(axis=1).to_csv(index=True).encode("utf-8")
    h = hashlib.sha256(csv_bytes).hexdigest()

    if ANNOTATIONS_SHA256 != "REPLACE_WITH_ACTUAL_HASH":
        assert (
            h == ANNOTATIONS_SHA256
        ), (
            "Annotation's hash changed.\n"
            f"Got    {h}\n"
            f"Expect {ANNOTATIONS_SHA256}\n\n"
            "If this is an intentional dataset/content update, "
            "replace ANNOTATIONS_SHA256 with the new hash."
        )


def test_check_metadata_fields(ds: SuperbStarling, sample_indices: List[int]):
    """Verify that required metadata fields are present."""
    required_fields = {
        "Selection",
        "Begin Time (s)",
        "End Time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Begin File",
        "Species",
        "bird",
        "group",
        "sex",
    }

    for idx in sample_indices:
        item = ds[idx]
        missing = required_fields - set(item.keys())
        assert not missing, f"[{idx}] missing required fields: {sorted(missing)}"



def test_get_available_labels_bird(ds: SuperbStarling):
    """Test get_available_labels for bird ID column."""
    labels = ds.get_available_labels("bird")
    assert isinstance(labels, list), "get_available_labels should return a list"
    assert len(labels) > 0, "Should have at least one bird ID"
    # Check that all labels can be converted to strings
    for label in labels:
        assert isinstance(label, str), f"Bird ID {label} should be string"


def test_get_available_labels_invalid_column(ds: SuperbStarling):
    """Test that get_available_labels raises error for invalid column."""
    with pytest.raises(ValueError, match="Column.*not found"):
        ds.get_available_labels("nonexistent_column")


def test_get_individual_stats(ds: SuperbStarling):
    """Test get_individual_stats method."""
    stats = ds.get_individual_stats()
    
    assert isinstance(stats, pd.DataFrame), "Should return a DataFrame"
    assert len(stats) > 0, "Should have stats for at least one individual"
    
    required_cols = {"bird", "group", "sex", "num_vocalizations"}
    assert required_cols.issubset(stats.columns), (
        f"Missing columns: {required_cols - set(stats.columns)}"
    )
    
    # Check that num_vocalizations is positive
    assert (stats["num_vocalizations"] > 0).all(), (
        "All individuals should have at least one vocalization"
    )
    
    # Check that results are sorted by num_vocalizations descending
    assert stats["num_vocalizations"].is_monotonic_decreasing, (
        "Results should be sorted by num_vocalizations descending"
    )


def test_get_group_stats(ds: SuperbStarling):
    """Test get_group_stats method."""
    stats = ds.get_group_stats()
    
    assert isinstance(stats, pd.DataFrame), "Should return a DataFrame"
    assert len(stats) > 0, "Should have stats for at least one group"
    
    required_cols = {"group", "num_individuals", "num_vocalizations"}
    assert required_cols.issubset(stats.columns), (
        f"Missing columns: {required_cols - set(stats.columns)}"
    )
    
    # Check that counts are positive
    assert (stats["num_individuals"] > 0).all(), (
        "All groups should have at least one individual"
    )
    assert (stats["num_vocalizations"] > 0).all(), (
        "All groups should have at least one vocalization"
    )


def test_output_take_and_give(ds: SuperbStarling):
    """Test that output_take_and_give filtering works correctly."""
    # Create dataset with filtered output
    ds_filtered = SuperbStarling(
        split="all",
        sample_rate=16000,
        output_take_and_give={
            "audio": "audio",
            "Species": "species",
            "bird": "bird_id",
        }
    )
    
    item = ds_filtered[0]
    
    # Should only have the mapped keys
    assert set(item.keys()) == {"audio", "species", "bird_id"}
    assert "Begin Time (s)" not in item
    assert "group" not in item


def test_iteration(ds: SuperbStarling):
    """Test that dataset iteration works correctly."""
    items = list(ds)
    assert len(items) == len(ds), "Iteration should yield all items"
    
    # Check a few items
    for i in range(min(3, len(items))):
        assert "audio" in items[i]
        assert "Species" in items[i]


def test_str_representation(ds: SuperbStarling):
    """Test string representation of dataset."""
    s = str(ds)
    assert "superb_starling" in s.lower()
    assert "v0.1.0" in s
    assert str(len(ds)) in s