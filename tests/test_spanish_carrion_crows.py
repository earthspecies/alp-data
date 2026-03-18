"""
Unit tests for SpanishCarrionCrows dataset.

Run with:
    pytest -q test_spanish_carrion_crows.py
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import pandas as pd
import pytest
import hashlib

from esp_data.datasets import SpanishCarrionCrows


# # --- Dataset snapshot ---

# # Code to generate snapshot:
# from esp_data.datasets import SpanishCarrionCrows
# import hashlib
# ds = SpanishCarrionCrows(split="all", sample_rate=16000, backend="polars")
# print("len(ds) =", len(ds))
# audio0 = ds[0]["audio"]
# print("dtype:", audio0.dtype, "shape:", audio0.shape)
# h = hashlib.sha256(audio0.tobytes()).hexdigest()
# print("sha256:", h)
# df = ds._data.unwrap
# csv_bytes = df.sort(df.columns).write_csv().encode("utf-8")
# h = hashlib.sha256(csv_bytes).hexdigest()
# print("annotations sha256:", h)
# quit()
# # # # #

EXPECTED_LEN_ALL = 953  #
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "08b28fcc371250e424e2de0ac613e56d296659e0f3ccd55365e61a2629e988b7"
)
ANNOTATIONS_SHA256 = "ed692970908e8413ef9003130af29d083e4ebfa3d217315edad9d665aaf07009"
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ds_polars() -> SpanishCarrionCrows:
    """Load SpanishCarrionCrows dataset for testing."""
    return SpanishCarrionCrows(split="all", sample_rate=16000, backend="polars", streaming=False)

@pytest.fixture(scope="module", autouse=True)
def ds_streaming() -> SpanishCarrionCrows:
    """Load SpanishCarrionCrows dataset for testing."""
    return SpanishCarrionCrows(split="all", sample_rate=16000, backend="polars", streaming=True)


@pytest.fixture(scope="module")
def sample_indices(ds_polars: SpanishCarrionCrows) -> List[int]:
    """Deterministically choose up to 5 random indices for quick spot checks."""
    n = len(ds_polars)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds_polars: SpanishCarrionCrows):
    """Dataset should have at least one example."""
    assert len(ds_polars) > 0, "Dataset appears empty"

def test_get_available_labels(ds_streaming: SpanishCarrionCrows):
    """Test get_available_labels for caller ID column."""
    labels = ds_streaming.get_available_labels(anno_column="Annotation")
    assert isinstance(labels, list), "get_available_labels should return a list"
    assert len(labels) > 0, "Should have at least one caller ID"
    # Check that all labels can be converted to strings
    for label in labels:
        assert isinstance(label, str), f"Caller label for {label} should be string"

def test_check_audio(ds_polars: SpanishCarrionCrows, sample_indices: List[int]):
    """Basic audio integrity checks on a few random items."""
    for idx in sample_indices:
        item = ds_polars[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert (
            audio.dtype == np.float32
        ), f"[{idx}] audio dtype is {audio.dtype}, expected float32"
        assert audio.size >= 10, f"[{idx}] audio too short (size={audio.size})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"
        assert len(audio.shape) == 1, f"[{idx}] is not 1D, expected (N_samples,) but got shape={audio.shape}"

def test_available_splits(ds_streaming: SpanishCarrionCrows) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should contain these
    expected_splits = ["all"]
    assert all(split in ds_streaming.available_splits for split in expected_splits)


def test_reference_item_stability(ds_polars: SpanishCarrionCrows):
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
    item = ds_polars[idx]

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
    df = ds_polars._data.unwrap
    csv_bytes = df.sort(df.columns).write_csv().encode("utf-8")
    h = hashlib.sha256(csv_bytes).hexdigest()

    assert h == ANNOTATIONS_SHA256, (
        "Annotation's hash changed.\n"
        f"Got    {h}\n"
        f"Expect {ANNOTATIONS_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )


def test_check_selection_table(ds_polars: SpanishCarrionCrows, sample_indices: List[int]):
    """Selection table should be a DataFrame with required columns and sane times."""
    required = {
        "Begin Time (s)",
        "End Time (s)",
        "Annotation",
        "Detection Prob",
        "Class Prob",
        "RMS Amplitude"
    }

    for idx in sample_indices:
        item = ds_polars[idx]
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
