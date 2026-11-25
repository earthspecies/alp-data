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

from alp_data.datasets import Subsegmentation


# # --- Dataset snapshot ---

# # Code to generate snapshot:
# import hashlib
# from alp_data.datasets import Subsegmentation
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


def test_get_available_labels(ds: Subsegmentation):
    """Test get_available_labels for bird ID column."""
    labels = ds.get_available_labels(anno_column="Species")
    assert isinstance(labels, list), "get_available_labels should return a list"
    assert len(labels) > 0, "Should have at least one bird ID"
    # Check that all labels can be converted to strings
    for label in labels:
        assert isinstance(label, str), f"Species label for {label} should be string"


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


# ---------------------------------------------------------------------------
# Single-song split tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds_single_song() -> Subsegmentation:
    """Load single-song-all split for testing."""
    return Subsegmentation(split="single_song_all", sample_rate=16000)


@pytest.fixture(scope="module")
def sample_indices_single_song(ds_single_song: Subsegmentation) -> List[int]:
    """Deterministically choose up to 5 random indices for single-song spot checks."""
    n = len(ds_single_song)
    rng = random.Random(42)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_single_song_not_empty(ds_single_song: Subsegmentation):
    """Single-song split should have at least one example."""
    assert len(ds_single_song) > 0, "Single-song split appears empty"


def test_single_song_check_audio(
    ds_single_song: Subsegmentation, sample_indices_single_song: List[int]
):
    """Basic audio integrity checks on a few random single-song items."""
    for idx in sample_indices_single_song:
        item = ds_single_song[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert (
            audio.dtype == np.float32
        ), f"[{idx}] audio dtype is {audio.dtype}, expected float32"
        assert audio.size >= 10, f"[{idx}] audio too short (size={audio.size})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


def test_single_song_check_selection_table(
    ds_single_song: Subsegmentation, sample_indices_single_song: List[int]
):
    """Selection table should have required columns and single-song structural invariants."""
    required = {
        "Begin Time (s)",
        "End Time (s)",
        "Species",
        "Annotation",
        "Genus",
        "Family",
        "Order",
    }

    for idx in sample_indices_single_song:
        item = ds_single_song[idx]
        assert "selection_table" in item, f"[{idx}] missing 'selection_table' key"
        st = item["selection_table"]

        assert isinstance(st, pd.DataFrame), f"[{idx}] selection_table is not a DataFrame"
        missing = required - set(st.columns)
        assert not missing, f"[{idx}] selection_table missing columns: {sorted(missing)}"

        # Every single-song item is a complete song: at minimum 'a' and 'z'
        assert len(st) >= 2, f"[{idx}] expected at least 2 syllables (a + z), got {len(st)}"
        assert (
            st.iloc[0]["Annotation"] == "a"
        ), f"[{idx}] first annotation is {st.iloc[0]['Annotation']!r}, expected 'a'"
        assert (
            st.iloc[-1]["Annotation"] == "z"
        ), f"[{idx}] last annotation is {st.iloc[-1]['Annotation']!r}, expected 'z'"

        valid_annos = {"a", "s", "z"}
        bad = set(st["Annotation"].unique()) - valid_annos
        assert not bad, f"[{idx}] unexpected annotations in single-song table: {bad}"

        # Times should be re-zeroed (no negative begin times)
        assert not (
            st["Begin Time (s)"] < 0
        ).any(), f"[{idx}] negative begin times present after re-zeroing"


def test_single_song_pass_qc_always_true(
    ds_single_song: Subsegmentation, sample_indices_single_song: List[int]
):
    """Every single-song item is a complete song, so pass_qc must always be True."""
    for idx in sample_indices_single_song:
        item = ds_single_song[idx]
        assert "pass_qc" in item, f"[{idx}] missing 'pass_qc' key"
        assert item["pass_qc"] is True, f"[{idx}] pass_qc is False for a single-song item"


def test_single_song_audio_covers_annotations(
    ds_single_song: Subsegmentation, sample_indices_single_song: List[int]
):
    """Audio duration should cover the full selection table (with 1-sample floor-rounding tolerance)."""
    for idx in sample_indices_single_song:
        item = ds_single_song[idx]
        audio = item["audio"]
        st = item["selection_table"]
        sr = item["sample_rate"]

        audio_dur = len(audio) / float(sr)
        max_end = float(st["End Time (s)"].max())
        tolerance = 1.0 / sr  # one sample worth of tolerance for int() floor rounding
        assert max_end <= audio_dur + tolerance, (
            f"[{idx}] selection table end time {max_end:.6f}s exceeds "
            f"audio duration {audio_dur:.6f}s (tolerance={tolerance:.6f}s)"
        )
