"""
Unit tests for the PteroSet dataset.

Run with:
    pytest -q tests/test_pteroset.py
"""

from __future__ import annotations

import hashlib
import random
from typing import List

import numpy as np
import pandas as pd
import pytest

from esp_data.datasets import PteroSet

# # --- Dataset snapshot ---
# # Code to regenerate the snapshot constants below:
# from esp_data.datasets import PteroSet
# ds = PteroSet(split="all", sample_rate=16000, backend="pandas")
# print("len(ds) =", len(ds))
# audio0 = ds[0]["audio"]
# print("sha256:", hashlib.sha256(audio0.tobytes()).hexdigest())
# csv_bytes = (
#     ds._data.unwrap.sort_index(axis=0).sort_index(axis=1).to_csv(index=True).encode("utf-8")
# )
# print("annotations sha256:", hashlib.sha256(csv_bytes).hexdigest())

EXPECTED_LEN_ALL = 563
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "681cab53137f12da8e28020802d0ce0aa965693d58a5da5dd38f9e5fae1f94c7"
)
ANNOTATIONS_SHA256 = "7f9d34804760d26835a84ce33a15951e3963107fc6e2067f56e6f7f64393ddc0"
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> PteroSet:
    """Load the dataset (pandas backend) for testing."""
    return PteroSet(split="all", sample_rate=16000, backend="pandas")


@pytest.fixture(scope="module", autouse=True)
def ds_polars() -> PteroSet:
    """Load the dataset (polars backend) for testing."""
    return PteroSet(split="all", sample_rate=16000, backend="polars")


@pytest.fixture(scope="module")
def sample_indices(ds_polars: PteroSet) -> List[int]:
    """Deterministically choose up to 5 random indices for quick spot checks."""
    n = len(ds_polars)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: PteroSet):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_expected_length(ds: PteroSet):
    """The 'all' split should contain all 563 recordings."""
    assert len(ds) == EXPECTED_LEN_ALL


def test_available_splits(ds: PteroSet) -> None:
    """available_splits should expose all six split names."""
    for split in ["all", "map1", "ppa1", "ppa2", "ppa3", "ppa4"]:
        assert split in ds.available_splits


def test_check_audio(ds_polars: PteroSet, sample_indices: List[int]):
    """Basic audio integrity checks on a few random items."""
    for idx in sample_indices:
        item = ds_polars[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]
        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert audio.dtype == np.float32, f"[{idx}] audio dtype is {audio.dtype}, expected float32"
        assert audio.size >= 10, f"[{idx}] audio too short (size={audio.size})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


def test_check_selection_table(ds: PteroSet, sample_indices: List[int]):
    """Selection table should be a DataFrame with required columns and sane times."""
    required = {"Begin Time (s)", "End Time (s)", "Species", "Species Code"}

    for idx in sample_indices:
        item = ds[idx]
        assert "selection_table" in item, f"[{idx}] missing 'selection_table' key"
        st = item["selection_table"]
        assert isinstance(st, pd.DataFrame), f"[{idx}] selection_table is not a DataFrame"
        missing = required - set(st.columns)
        assert not missing, f"[{idx}] selection_table missing columns: {sorted(missing)}"

        if len(st) > 0:
            assert not (st["Begin Time (s)"] < 0).any(), f"[{idx}] negative begin times present"
            durs = st["End Time (s)"] - st["Begin Time (s)"]
            assert not durs.min() <= 0, f"[{idx}] events of dur <= 0"


@pytest.mark.skipif(
    EXPECTED_FIRST_ITEM_AUDIO_SHA256.startswith("__FILL"),
    reason="snapshot hashes not yet populated",
)
def test_reference_item_stability(ds: PteroSet):
    """Check that the canonical item (index 0) and annotations are bitwise-stable."""
    idx = 0
    item = ds[idx]
    audio = item["audio"]
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32

    h = hashlib.sha256(audio.tobytes()).hexdigest()
    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\nExpect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, replace "
        "EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )

    csv_bytes = (
        ds._data.unwrap.sort_index(axis=0).sort_index(axis=1).to_csv(index=True).encode("utf-8")
    )
    h = hashlib.sha256(csv_bytes).hexdigest()
    assert h == ANNOTATIONS_SHA256, (
        "Annotation's hash changed.\n"
        f"Got    {h}\nExpect {ANNOTATIONS_SHA256}\n\n"
        "If this is an intentional dataset/content update, replace "
        "ANNOTATIONS_SHA256 with the new hash."
    )
