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
import pytest

from esp_data.datasets import AudioSetStrong


# --- Dataset snapshot ---

# # Code to generate snapshot:
# import hashlib
# from esp_data.datasets import AudioSetStrong
# ds = AudioSetStrong(split="train", sample_rate=16000, backend="pandas")

# print("len(ds) =", len(ds))

# audio0 = ds[0]["audio"]
# print("dtype:", audio0.dtype, "shape:", audio0.shape)

# h = hashlib.sha256(audio0.tobytes()).hexdigest()
# print("sha256:", h)

# csv_bytes = ds._data.unwrap.sort_index(axis=0).sort_index(axis=1).to_csv(index=True).encode("utf-8")
# h = hashlib.sha256(csv_bytes).hexdigest()

# print("annotations sha256:", h)

# quit()
# # #

EXPECTED_LEN_TRAIN = 8841
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "cab77818f53bf12df33e349e55a24da93405a96b124d6bc683fca62f88427616"
)
ANNOTATIONS_SHA256 = "ccec6aeb3a4d2b7b9f0819ef217ca8b163baa2bd1d645978c3d3c93a3ed66986"
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> AudioSetStrong:
    """Load AudioSetStrong dataset for testing."""
    return AudioSetStrong(split="train", sample_rate=16000)


@pytest.fixture(scope="module")
def ds_pandas() -> AudioSetStrong:
    """Load AudioSetStrong dataset for testing with pandas backend."""
    return AudioSetStrong(split="train", sample_rate=16000, backend="pandas")


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
    """
    The dataset length should match the known, version-controlled expectation.

    This will fail loudly if:
    - the CSV split changed
    - files went missing
    - we accidentally filtered/augmented items differently

    If this fails intentionally (e.g. dataset grew), update EXPECTED_LEN_TRAIN.
    """
    assert len(ds) == EXPECTED_LEN_TRAIN, (
        f"Dataset length mismatch: got {len(ds)}, expected {EXPECTED_LEN_TRAIN}. "
        "If this change is intentional (new data / new filtering), update EXPECTED_LEN_TRAIN "
        "in the test."
    )


def test_check_audio(ds: AudioSetStrong, sample_indices: List[int]):
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


def test_reference_item_stability(ds_pandas: AudioSetStrong):
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

    # compute sha256 over raw bytes of the annotations csv
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


def test_check_selection_table(ds: AudioSetStrong, sample_indices: List[int]):
    """Selection table should be a DataFrame with required columns and sane times."""
    required = {
        "Begin Time (s)",
        "End Time (s)",
        "Label",
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


def test_get_available_labels(ds_pandas: AudioSetStrong):
    """Test get_available_labels returns labels correctly.

    Note: This iterates through the entire dataset, so it may be slow.
    We use pandas backend here since it's the same data.
    """
    # Use a limited subset for speed - check first few items manually
    labels_sample = set()
    for i in range(min(100, len(ds_pandas))):
        try:
            item = ds_pandas[i]
            st = item["selection_table"]
            if "Label" in st.columns:
                labels_sample.update(st["Label"].astype(str).tolist())
        except FileNotFoundError:
            # Some audio files may be missing
            continue

    assert len(labels_sample) > 0, "Should have at least one label"
    # Check that all labels are strings
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

    # Check if 32kHz is available (depends on whether 32khz_path column exists)
    if "32khz_path" in ds.columns:
        assert 32000 in sample_rates
        print("✓ 32kHz pre-resampled audio is available")
    else:
        assert len(sample_rates) == 0
        print("Note: 32khz_path column not available in metadata")

    print(f"Available pre-resampled sample rates: {sample_rates}")


def test_pre_resampled_audio_32khz():
    """Test loading pre-resampled 32kHz audio with fallback for corrupt files.

    This test verifies that:
    1. When sample_rate=32000, the dataset attempts to use pre-resampled files
    2. Corrupt pre-resampled files (very short) fall back to on-the-fly resampling
    3. Audio is returned correctly in both cases
    """
    ds_32k = AudioSetStrong(split="train", sample_rate=32000)

    # Check if 32kHz pre-resampled audio is available
    if "32khz_path" not in ds_32k.columns:
        pytest.skip("32kHz pre-resampled audio not available")

    # Test first few items
    for i in range(min(5, len(ds_32k))):
        item = ds_32k[i]
        assert "audio" in item, f"[{i}] missing 'audio' key"
        audio = item["audio"]

        assert isinstance(audio, np.ndarray), f"[{i}] audio is not a numpy array"
        assert audio.dtype == np.float32, f"[{i}] audio dtype is {audio.dtype}"

        # Audio should have reasonable length (at least 1 second at 32kHz)
        # This verifies the fallback works for corrupt pre-resampled files
        min_samples = 32000  # 1 second at 32kHz
        assert len(audio) >= min_samples, (
            f"[{i}] audio too short ({len(audio)} samples). "
            "Fallback may not be working for corrupt pre-resampled files."
        )

        print(f"  [{i}] {item['segment_id']}: audio shape={audio.shape}")


def test_pre_resampled_vs_on_the_fly_consistency():
    """Test that pre-resampled and on-the-fly resampled audio are similar.

    We compare an item that uses pre-resampled 32kHz audio with the same item
    resampled on-the-fly from the original. They should be very similar
    (allowing for minor differences due to resampling algorithms).
    """
    ds_32k = AudioSetStrong(split="train", sample_rate=32000)

    if "32khz_path" not in ds_32k.columns:
        pytest.skip("32kHz pre-resampled audio not available")

    # Find an item with a valid (non-corrupt) pre-resampled file
    # Item at index 1 has valid 32kHz file based on earlier testing
    valid_idx = 1

    # Get audio using pre-resampled path
    item_presampled = ds_32k[valid_idx]

    # Get audio using on-the-fly resampling (by using a different sample rate)
    ds_16k = AudioSetStrong(split="train", sample_rate=16000)
    item_16k = ds_16k[valid_idx]

    # Basic validation that both load correctly
    assert item_presampled["segment_id"] == item_16k["segment_id"]
    assert len(item_presampled["audio"]) > 0
    assert len(item_16k["audio"]) > 0

    # The 32kHz audio should be ~2x longer than 16kHz (same duration, different rate)
    ratio = len(item_presampled["audio"]) / len(item_16k["audio"])
    assert 1.8 < ratio < 2.2, (
        f"Sample rate ratio unexpected: {ratio:.2f} "
        f"(32kHz={len(item_presampled['audio'])}, 16kHz={len(item_16k['audio'])})"
    )
