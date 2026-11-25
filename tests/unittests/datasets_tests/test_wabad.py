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

from alp_data.datasets import WABAD
from alp_data.utils import create_hash


EXPECTED_LEN_ALL = 4297  #
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "c0febadbf616be8c083a22e128e82e25be9be8953bf3dec33c2b8ee1f88ad330"
)
ANNOTATIONS_SHA256 = "c3523d6e6aad348813d05579f410e30d48f445e801947361142fb0c36c7d6a85"
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> WABAD:
    """Load WABAD dataset for testing."""
    return WABAD(split="all", sample_rate=16000, backend="pandas")


@pytest.fixture(scope="module")
def ds_sub() -> WABAD:
    """Load WABAD subdataset for testing."""
    return WABAD(split="CAT", sample_rate=16000, backend="pandas")


@pytest.fixture(scope="module")
def sample_indices(ds: WABAD) -> List[int]:
    """Deterministically choose up to 3 random indices for quick spot checks."""
    n = len(ds)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(3, n))]


def test_ds_not_empty(ds: WABAD):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_get_available_labels(ds: WABAD):
    """Test get_available_labels for bird ID column."""
    labels = ds.get_available_labels()
    assert isinstance(labels, list), "get_available_labels should return a list"
    assert len(labels) > 0, "Should have at least one bird ID"
    # Check that all labels can be converted to strings
    for label in labels:
        assert isinstance(label, str), f"Species label for {label} should be string"


def test_check_audio(ds: WABAD, sample_indices: List[int]):
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


def test_available_splits(ds: WABAD) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should contain these
    expected_splits = ["all"]
    assert all(split in ds.available_splits for split in expected_splits)


def test_get_available_labels(ds: WABAD, ds_sub: WABAD):
    """Test get_available_labels for bird ID column."""
    labels = ds.get_available_labels(anno_column="Species")
    assert isinstance(labels, list), "get_available_labels should return a list"
    assert len(labels) > 0, "Should have at least one bird ID"
    # Check that all labels can be converted to strings
    for label in labels:
        assert isinstance(label, str), f"Species label for {label} should be string"


    labels_sub = ds_sub.get_available_labels(anno_column="Species")
    assert isinstance(labels_sub, list), "get_available_labels should return a list"
    assert len(labels_sub) > 0, "Should have at least one bird ID"
    # Check that all labels can be converted to strings
    for label in labels_sub:
        assert isinstance(label, str), f"Species label for {label} should be string"
    assert len(labels_sub) < len(labels), "Subdataset should have fewer classes than all"


def test_reference_item_stability(ds: WABAD):
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
    item = ds[idx]

    # audio presence/type checks (defensive, so the hash failure message is clearer)
    assert "audio" in item, "[0] missing 'audio' key"
    audio = item["audio"]
    assert isinstance(audio, np.ndarray), "[0] audio is not a numpy array"
    assert (
        audio.dtype == np.float32
    ), f"[0] audio dtype is {audio.dtype}, expected float32"

    # compute sha256 over raw bytes of the float32 array
    h = create_hash(audio.tobytes())

    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )

    # compute sha256 over raw bytes of the float32 array of annotations
    csv_bytes = (
        ds._data.unwrap.sort_index(axis=0)
        .sort_index(axis=1)
        .to_csv(index=True)
        .encode("utf-8")
    )
    h = create_hash(csv_bytes)

    assert h == ANNOTATIONS_SHA256, (
        "Annotation's hash changed.\n"
        f"Got    {h}\n"
        f"Expect {ANNOTATIONS_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )


def test_presampled_columns_exist(ds: WABAD):
    """Pre-resampled path columns should be present in the loaded data."""
    assert "16khz_path" in ds.columns
    assert "32khz_path" in ds.columns


def test_load_presampled_32khz():
    """Loading with sample_rate=32000 should use pre-resampled 32kHz audio."""
    ds = WABAD(split="all", sample_rate=32000, streaming=True)
    item = next(iter(ds))
    audio = item["audio"]
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.size >= 10


def test_check_selection_table(ds: WABAD, sample_indices: List[int]):
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
            durs = st["End Time (s)"] - st["Begin Time (s)"]
            assert not durs.min() <= 0, f"[{idx}] events of dur <= 0"


if __name__ == "__main__":
    # Code to generate snapshot for WABAD:
    from alp_data.datasets import WABAD
    ds = WABAD(split="all", sample_rate=16000, backend="pandas")

    print("len(ds) =", len(ds))

    audio0 = ds[0]["audio"]
    print("dtype:", audio0.dtype, "shape:", audio0.shape)

    h = create_hash(audio0.tobytes())
    print("sha256:", h)

    csv_bytes = (
            ds._data.unwrap.sort_index(axis=0)
            .sort_index(axis=1)
            .to_csv(index=True)
            .encode("utf-8")
        )
    h = create_hash(csv_bytes)

    print("annotations sha256:", h)
