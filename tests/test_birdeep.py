"""
Unit tests for birdeep dataset.

Run with:
    pytest -q test_birdeep.py
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import pandas as pd
import pytest
import hashlib

from esp_data.datasets import Birdeep


# # --- Dataset snapshot ---

# # Code to generate snapshot:
# from esp_data.datasets import Birdeep
# ds = Birdeep(split="all", sample_rate=16000, backend="pandas")

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
# # # #

EXPECTED_LEN_ALL = 291  #
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "dd81518532b80282b5d9c27b371398a3eb6469bb50da1717ed4113a51023396d"
)
ANNOTATIONS_SHA256 = "667a29118d20db83e672a46e0a2bdb7fd44da8b6fc53ff968ae1201d77472404"
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> Birdeep:
    """Load Birdeep dataset for testing."""
    return Birdeep(split="all", sample_rate=16000)


@pytest.fixture(scope="module")
def ds_pandas() -> Birdeep:
    """Load Birdeep dataset for testing with pandas backend."""
    return Birdeep(split="all", sample_rate=16000, backend="pandas")


@pytest.fixture(scope="module")
def sample_indices(ds: Birdeep) -> List[int]:
    """Deterministically choose up to 5 random indices for quick spot checks."""
    n = len(ds)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: Birdeep):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"

def test_check_audio(ds: Birdeep, sample_indices: List[int]):
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


def test_available_splits(ds: Birdeep) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should contain these
    expected_splits = ["train", "val", "test", "all"]
    assert all(split in ds.available_splits for split in expected_splits)


def test_get_available_labels(ds: Birdeep):
    """Test get_available_labels for bird ID column."""
    labels = ds.get_available_labels(anno_column="Species")
    assert isinstance(labels, list), "get_available_labels should return a list"
    assert len(labels) > 0, "Should have at least one bird ID"
    # Check that all labels can be converted to strings
    for label in labels:
        assert isinstance(label, str), f"Species label for {label} should be string"


def test_reference_item_stability(ds_pandas: Birdeep):
    """
    Check that a canonical item (index 0) is bitwise-stable.

    We hash the raw float32 audio buffer. This catches:
    - sample rate changes (resampling -> different samples)
    - channel handling changes (stereo->mono logic changed)
    - dtype changes
    - ordering changes in the split (if a different recording moved to idx 0)

    If this fails for a legitimate/intentional reason, recompute the hash below
    and update EXPECTED_FIRST_ITEM_AUDIO_SHA256.

    We do the same for the annotations csv.
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


def test_check_selection_table(ds: Birdeep, sample_indices: List[int]):
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
