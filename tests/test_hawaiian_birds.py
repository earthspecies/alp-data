"""
Unit tests for hawaiian_birds dataset.

Run with:
    pytest -q test_hawaiian_birds.py
"""

from __future__ import annotations

import hashlib
import random
from typing import List

import numpy as np
import pandas as pd
import pytest

from esp_data.datasets import HawaiianBirds


# # --- Dataset snapshot ---

# # Code to generate snapshot:
# import hashlib
# from esp_data.datasets import HawaiianBirds
# ds = HawaiianBirds(split="all", sample_rate=16000, backend="pandas")

# print("len(ds) =", len(ds))

# audio0 = ds[0]["audio"]
# print("dtype:", audio0.dtype, "shape:", audio0.shape)

# h = hashlib.sha256(audio0.tobytes()).hexdigest()
# print("sha256:", h)

# csv_bytes = (
#         ds._data.unwrap.sort_index(axis=0)
#         .sort_index(axis=1)
#         .to_csv(index=True)
#         .encode("utf-8")
#     )
# h = hashlib.sha256(csv_bytes).hexdigest()

# print("annotations sha256:", h)

# quit()
# # #

EXPECTED_LEN_ALL = 635  #
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "43c7ab6d988a5d329c82b24fe6cfc8642159e3e9809d38db88eff24632d539a3"
)
ANNOTATIONS_SHA256 = "f84030805f8c3987110397d0a82174fdbd88ef5d93a17962dd1c817d65284f3e"
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> HawaiianBirds:
    """Load HawaiianBirds dataset for testing."""
    return HawaiianBirds(split="all", sample_rate=16000)


@pytest.fixture(scope="module")
def ds_pandas() -> HawaiianBirds:
    """Load HawaiianBirds dataset for testing with pandas backend."""
    return HawaiianBirds(split="all", sample_rate=16000, backend="pandas")


@pytest.fixture(scope="module")
def sample_indices(ds: HawaiianBirds) -> List[int]:
    """Deterministically choose up to 5 random indices for quick spot checks."""
    n = len(ds)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: HawaiianBirds):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_get_available_labels(ds: HawaiianBirds):
    """Dataset should provide available labels"""
    labels = ds.get_available_labels()
    assert len(labels) > 0, "Dataset has no labels"

def test_check_audio(ds: HawaiianBirds, sample_indices: List[int]):
    """Basic audio integrity checks on a few random items."""
    for idx in sample_indices:
        item = ds[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert (
            audio.dtype == np.float32
        ), f"[{idx}] audio dtype is {audio.dtype}, expected float32"
        assert audio.size >= 10, f"[{idx}] audio too short (size={audio.size})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


def test_dataset_length_matches_expected(ds: HawaiianBirds):
    """
    The dataset length should match the known, version-controlled expectation.

    This will fail loudly if:
    - the CSV split changed
    - files went missing
    - we accidentally filtered/augmented items differently

    If this fails intentionally (e.g. dataset grew), update EXPECTED_LEN_ALL.
    """
    assert len(ds) == EXPECTED_LEN_ALL, (
        f"Dataset length mismatch: got {len(ds)}, expected {EXPECTED_LEN_ALL}. "
        "If this change is intentional (new data / new filtering), update EXPECTED_LEN_ALL "
        "in the test."
    )


def test_get_available_labels(ds: HawaiianBirds):
    """Test get_available_labels for bird ID column."""
    labels = ds.get_available_labels(anno_column="Species")
    assert isinstance(labels, list), "get_available_labels should return a list"
    assert len(labels) > 0, "Should have at least one bird ID"
    # Check that all labels can be converted to strings
    for label in labels:
        assert isinstance(label, str), f"Species label for {label} should be string"


def test_reference_item_stability(ds_pandas: HawaiianBirds):
    """
    Check that a canonical item (index 0) is bitwise-stable.

    We hash the raw float32 audio buffer. This catches:
    - sample rate changes (resampling -> different samples)
    - channel handling changes (stereo->mono logic changed)
    - dtype changes
    - ordering changes in the split (if a different recording moved to idx 0)

    If this fails for a legitimate/intentional reason, recompute the hash below
    and update EXPECTED_FIRST_ITEM_AUDIO_SHA256.
    """
    # choose deterministic index
    idx = 0
    item = ds_pandas[idx]

    # audio presence/type checks (defensive, so the hash failure message is clearer)
    assert "audio" in item, "[0] missing 'audio' key"
    audio = item["audio"]
    assert isinstance(audio, np.ndarray), "[0] audio is not a numpy array"
    assert (
        audio.dtype == np.float32
    ), f"[0] audio dtype is {audio.dtype}, expected float32"

    # compute sha256 over raw bytes of the float32 array
    h = hashlib.sha256(audio.tobytes()).hexdigest()

    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )

    # compute sha256 over raw bytes of the float32 array of annotations
    csv_bytes = (
        ds_pandas._data.unwrap.sort_index(axis=0)
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
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )


def test_check_selection_table(ds: HawaiianBirds, sample_indices: List[int]):
    """Selection table should be a DataFrame with required columns and sane times."""
    required = {
        "Begin Time (s)",
        "End Time (s)",
        "Species",
    }

    for idx in sample_indices:
        item = ds[idx]
        assert "selection_table" in item, f"[{idx}] missing 'selection_table' key"
        st = item["selection_table"]

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
