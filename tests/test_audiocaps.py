"""
Unit tests for the AudioCaps dataset.

Run with:
    pytest -q tests/test_audiocaps.py

The ``test`` split is the smallest with 5 captions per clip, so we use it as
the smoke-test fixture; the ``all`` / ``train`` / ``val`` splits are checked
for size only.
"""

from __future__ import annotations

import hashlib
import random
from typing import List

import numpy as np
import pytest

from esp_data.datasets import AudioCaps

# --- Dataset snapshot ----------------------------------------------------
# Regenerate with:
#   import hashlib
#   from esp_data.datasets import AudioCaps
#   ds = AudioCaps(split="test", sample_rate=32000, backend="pandas")
#   audio0 = ds[0]["audio"]
#   print("audio sha256:", hashlib.sha256(audio0.tobytes()).hexdigest())
#   csv_bytes = (
#       ds._data.unwrap.sort_index(axis=0).sort_index(axis=1).to_csv(index=True).encode("utf-8")
#   )
#   print("annotations sha256:", hashlib.sha256(csv_bytes).hexdigest())

EXPECTED_LEN_TEST = 4445
EXPECTED_LEN_VAL = 2245
EXPECTED_LEN_TRAIN = 45493
EXPECTED_LEN_ALL = 52183

EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "f804200925ad5a65256a334f8a4dff657a99893aee55a44e1e3ce2c0a418d8f9"
)
ANNOTATIONS_SHA256 = "9b2aaace7c7af0da6cdb3b3ae50dbeae4cc2d8cb49257dae3fda1e4dc9711b09"


@pytest.fixture(scope="module")
def ds() -> AudioCaps:
    """Load the ``test`` split (4,445 captions, 889 unique clips) for testing."""
    return AudioCaps(split="test", sample_rate=32000, backend="pandas")


@pytest.fixture(scope="module")
def sample_indices(ds: AudioCaps) -> List[int]:
    """Deterministically choose up to 5 random indices for spot checks."""
    rng = random.Random(23)
    n = len(ds)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: AudioCaps):
    """Dataset should have at least one example."""
    assert len(ds) > 0


def test_available_splits(ds: AudioCaps):
    """available_splits should expose all four split names."""
    for split in ["all", "train", "val", "test"]:
        assert split in ds.available_splits


def test_expected_length(ds: AudioCaps):
    """The 'test' split length should be stable."""
    assert len(ds) == EXPECTED_LEN_TEST


def test_other_split_lengths():
    """val / train / all split lengths should be stable."""
    assert len(AudioCaps(split="val", sample_rate=32000, backend="pandas")) == EXPECTED_LEN_VAL
    assert len(AudioCaps(split="train", sample_rate=32000, backend="pandas")) == EXPECTED_LEN_TRAIN
    assert len(AudioCaps(split="all", sample_rate=32000, backend="pandas")) == EXPECTED_LEN_ALL


def test_check_audio(ds: AudioCaps, sample_indices: List[int]):
    """Basic audio integrity checks on a few items."""
    for idx in sample_indices:
        item = ds[idx]
        audio = item["audio"]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        # AudioCaps clips are 10s at 32 kHz = 320,000 samples (or close to it)
        assert audio.size >= 100_000
        assert not np.any(np.isnan(audio))
        assert not np.all(audio == 0)
        assert item["sample_rate"] == 32000


def test_check_caption_schema(ds: AudioCaps, sample_indices: List[int]):
    """Each row should carry a non-empty caption + youtube_id."""
    for idx in sample_indices:
        item = ds[idx]
        assert isinstance(item.get("caption"), str) and len(item["caption"]) > 0
        assert isinstance(item.get("youtube_id"), str) and len(item["youtube_id"]) > 0
        assert item.get("source_dataset") == "audiocaps"


@pytest.mark.skipif(
    EXPECTED_FIRST_ITEM_AUDIO_SHA256.startswith("__FILL"),
    reason="snapshot hashes not yet populated",
)
def test_reference_item_stability(ds: AudioCaps):
    """Index-0 audio + annotations should be bitwise-stable."""
    item = ds[0]
    audio = item["audio"]
    h = hashlib.sha256(audio.tobytes()).hexdigest()
    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        f"First item audio hash changed.\nGot {h}\nExpect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}"
    )
    csv_bytes = (
        ds._data.unwrap.sort_index(axis=0).sort_index(axis=1).to_csv(index=True).encode("utf-8")
    )
    h = hashlib.sha256(csv_bytes).hexdigest()
    assert h == ANNOTATIONS_SHA256, (
        f"Annotations hash changed.\nGot {h}\nExpect {ANNOTATIONS_SHA256}"
    )
