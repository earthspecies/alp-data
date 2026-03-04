"""
Unit tests for BirdSet dataset (v0.1.0).

Run with:
    pytest -q test_birdset.py
"""

from __future__ import annotations

import hashlib
import random
from typing import List

import numpy as np
import pytest

from esp_data.datasets import BirdSet


# # --- Dataset snapshot ---

# # Code to generate snapshot:
# import hashlib
# from esp_data.datasets import BirdSet
# ds = BirdSet(split="PER-test_5s", sample_rate=16000, backend="pandas")

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

EXPECTED_LEN = 15120
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = "c6c84647649f958f1ab9eef45276bc590d629ebfb94999d241f9e045b94acde8"
ANNOTATIONS_SHA256 = "1dca08123cfc08671be754f028cdfb028523885f757924c52e5574c2033ca4f7"
# ---------------------------------------------------------------------------

SPLIT = "PER-test_5s"


@pytest.fixture(scope="module")
def ds() -> BirdSet:
    """Load BirdSet dataset for testing."""
    return BirdSet(split=SPLIT, sample_rate=16000)


@pytest.fixture(scope="module")
def ds_pandas() -> BirdSet:
    """Load BirdSet dataset for testing with pandas backend."""
    return BirdSet(split=SPLIT, sample_rate=16000, backend="pandas")


@pytest.fixture(scope="module")
def sample_indices(ds: BirdSet) -> List[int]:
    """Deterministically choose up to 5 random indices for quick spot checks."""
    n = len(ds)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: BirdSet):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_check_audio(ds: BirdSet, sample_indices: List[int]):
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


def test_available_splits(ds: BirdSet) -> None:
    """Test if available_splits returns correct split names."""
    expected_splits = [
        "HSN-test", "HSN-test_5s",
        "PER-test", "PER-test_5s",
        "POW-test", "POW-test_5s",
        "all",
    ]
    assert all(split in ds.available_splits for split in expected_splits)


def test_expected_columns(ds: BirdSet) -> None:
    """Key columns from v0.1.0 schema should be present."""
    expected = ["audio_path", "species", "ebird_code", "16khz_path", "32khz_path"]
    for col in expected:
        assert col in ds.columns, f"Missing expected column: {col}"


def test_presampled_columns_exist(ds: BirdSet):
    """Pre-resampled path columns should be present in the loaded data."""
    assert "16khz_path" in ds.columns
    assert "32khz_path" in ds.columns


def test_available_sample_rates(ds: BirdSet):
    """available_sample_rates should report both pre-resampled rates."""
    rates = ds.available_sample_rates
    assert 16000 in rates
    assert 32000 in rates


def test_load_presampled_32khz():
    """Loading with sample_rate=32000 should use pre-resampled 32kHz audio."""
    ds = BirdSet(split=SPLIT, sample_rate=32000)
    item = ds[0]
    audio = item["audio"]
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.size >= 10


def test_reference_item_stability(ds_pandas: BirdSet):
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
    if EXPECTED_FIRST_ITEM_AUDIO_SHA256 is None:
        pytest.skip("Snapshot hashes not yet computed")

    idx = 0
    item = ds_pandas[idx]

    assert "audio" in item, "[0] missing 'audio' key"
    audio = item["audio"]
    assert isinstance(audio, np.ndarray), "[0] audio is not a numpy array"
    assert (
        audio.dtype == np.float32
    ), f"[0] audio dtype is {audio.dtype}, expected float32"

    h = hashlib.sha256(audio.tobytes()).hexdigest()

    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )

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
        "replace ANNOTATIONS_SHA256 with the new hash."
    )


def test_sample_consistency(ds: BirdSet) -> None:
    """Samples accessed by index vs iteration should match."""
    direct = ds[0]
    via_iter = next(iter(ds))
    assert direct["audio_path"] == via_iter["audio_path"]


def test_invalid_split() -> None:
    """Initialising with an unknown split should raise LookupError."""
    with pytest.raises(LookupError):
        BirdSet(split="invalid_split")


def test_output_take_and_give() -> None:
    """output_take_and_give should filter and rename columns."""
    ds = BirdSet(
        split=SPLIT,
        sample_rate=16000,
        output_take_and_give={"species": "label"},
    )
    sample = ds[0]
    assert "label" in sample
    assert "species" not in sample


def test_from_config() -> None:
    """from_config round-trip should produce a usable dataset."""
    from esp_data import DatasetConfig

    cfg = DatasetConfig(dataset_name="birdset", split=SPLIT)
    ds, meta = BirdSet.from_config(cfg)
    assert ds.info.name == "birdset"
    assert len(ds) > 0


def test_str_representation(ds: BirdSet) -> None:
    """String representation should contain key info."""
    s = str(ds)
    assert "birdset" in s
    assert "0.1.0" in s
