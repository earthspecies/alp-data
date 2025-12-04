"""
Unit tests for subsegmentation dataset.

Run with:
    pytest -q test_subsegmentation.py
"""

from __future__ import annotations

import hashlib
import random
from typing import List

import numpy as np
import pandas as pd
import pytest

from esp_data.datasets import Subsegmentation


# # --- Dataset snapshot ---

# # Code to generate snapshot:
# import hashlib
# from esp_data.datasets import Subsegmentation
# ds = Subsegmentation(split="all", sample_rate=16000)

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

EXPECTED_LEN_ALL = 11617  #
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "4cef10c4acacce969ca48fd1e698bf9ad1971bdd44102bc4030b6193979f6977"
)
ANNOTATIONS_SHA256 = "92ab0d4b3c83788f60bd0734359dd9041a16c933545295bd824526d1ebeb252f"
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> Subsegmentation:
    """Load Subsegmentation dataset for testing."""
    return Subsegmentation(split="all", sample_rate=16000)


@pytest.fixture(scope="module")
def ds_pandas() -> Subsegmentation:
    """Load Subsegmentation dataset for testing with pandas backend."""
    return Subsegmentation(split="all", sample_rate=16000, backend="pandas")


@pytest.fixture(scope="module")
def sample_indices(ds: Subsegmentation) -> List[int]:
    """Deterministically choose up to 5 random indices for quick spot checks."""
    n = len(ds)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: Subsegmentation):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_check_audio(ds: Subsegmentation, sample_indices: List[int]):
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


def test_reference_item_stability(ds_pandas: Subsegmentation):
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


def test_check_selection_table(ds: Subsegmentation, sample_indices: List[int]):
    """Selection table should be a DataFrame with required columns and sane times."""
    required = {
        "Begin Time (s)",
        "End Time (s)",
        "Species",
        "Annotation",
        "Genus",
        "Family",
        "Order",
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


def test_qc_flag_consistent(ds: Subsegmentation, sample_indices: List[int]):
    """QC flag should reflect whether there are any rows in the selection table."""
    for idx in sample_indices:
        item = ds[idx]
        assert "pass_qc" in item, f"[{idx}] missing 'pass_qc' key"
        st = item["selection_table"]
        expected_pass_qc = len(st) > 0
        assert (
            item["pass_qc"] == expected_pass_qc
        ), f"[{idx}] qc inconsistent: pass_qc={item['pass_qc']} but len(st)={len(st)}"
