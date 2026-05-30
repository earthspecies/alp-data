"""
Unit tests for the Clotho v2.1 dataset.

Run with:
    pytest -q tests/test_clotho.py
"""

from __future__ import annotations

import hashlib
import random
from typing import List

import numpy as np
import pytest

from esp_data.datasets import Clotho

# --- Dataset snapshot ----------------------------------------------------
EXPECTED_LEN_TEST = 1045
EXPECTED_LEN_VAL = 1045
EXPECTED_LEN_TRAIN = 3839
EXPECTED_LEN_ALL = 5929

# Generated post-resample (replace if snapshot intentionally changes):
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "5facb55780d4ad9b32f9fc810817c4484900c7b197caf0721cd0c86b2d5f08f1"
)
ANNOTATIONS_SHA256 = "9d0680810c6390555063a38b3d1db935ded8f2312447b506581c21c741b1bb95"


@pytest.fixture(scope="module")
def ds() -> Clotho:
    """Load the ``test`` (evaluation) split (1,045 clips × 5 captions) for testing."""
    return Clotho(split="test", sample_rate=32000, backend="pandas")


@pytest.fixture(scope="module")
def sample_indices(ds: Clotho) -> List[int]:
    """Deterministically choose up to 5 random indices for spot checks."""
    rng = random.Random(23)
    n = len(ds)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: Clotho):
    """Dataset should have at least one example."""
    assert len(ds) > 0


def test_available_splits(ds: Clotho):
    """available_splits should expose all four split names."""
    for split in ["all", "train", "val", "test"]:
        assert split in ds.available_splits


def test_expected_length(ds: Clotho):
    """The 'test' split length should be stable."""
    assert len(ds) == EXPECTED_LEN_TEST


def test_other_split_lengths():
    """val / train / all split lengths should be stable."""
    assert len(Clotho(split="val", sample_rate=32000, backend="pandas")) == EXPECTED_LEN_VAL
    assert len(Clotho(split="train", sample_rate=32000, backend="pandas")) == EXPECTED_LEN_TRAIN
    assert len(Clotho(split="all", sample_rate=32000, backend="pandas")) == EXPECTED_LEN_ALL


def test_check_audio(ds: Clotho, sample_indices: List[int]):
    """Basic audio integrity checks on a few items."""
    for idx in sample_indices:
        item = ds[idx]
        audio = item["audio"]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        # Clotho clips are 15-30s at 32 kHz = 480,000-960,000 samples
        assert audio.size >= 50_000
        assert not np.any(np.isnan(audio))
        assert not np.all(audio == 0)
        assert item["sample_rate"] == 32000


def test_check_caption_schema(ds: Clotho, sample_indices: List[int]):
    """Each row should carry 5 non-empty captions + a file_name."""
    for idx in sample_indices:
        item = ds[idx]
        assert isinstance(item.get("file_name"), str) and len(item["file_name"]) > 0
        for k in ("caption_1", "caption_2", "caption_3", "caption_4", "caption_5"):
            assert isinstance(item.get(k), str), f"[{idx}] {k} is not a string"
            assert len(item[k]) > 0, f"[{idx}] {k} is empty"
        assert item.get("source_dataset") == "clotho_v2_1"


@pytest.mark.skipif(
    EXPECTED_FIRST_ITEM_AUDIO_SHA256.startswith("__FILL"),
    reason="snapshot hashes not yet populated",
)
def test_reference_item_stability(ds: Clotho):
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
