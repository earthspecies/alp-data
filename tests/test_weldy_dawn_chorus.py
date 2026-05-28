"""
Unit tests for the Weldy NW Dawn Chorus dataset.

Run with:
    pytest -q tests/test_weldy_dawn_chorus.py
"""

from __future__ import annotations

import hashlib
import random
from typing import List

import numpy as np
import pandas as pd
import pytest

from esp_data.datasets import WeldyDawnChorus

EXPECTED_LEN_ALL = 1575
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "3a2f07b689ad6f0f6c5adf57d1f39d015f425edcad8b6739de2313ca41e4247b"
)
ANNOTATIONS_SHA256 = "c133cc85887575e6cd853ffaa3ce3d543c1dbfe0bf06652e3374ce97a9b3c120"


@pytest.fixture(scope="module")
def ds() -> WeldyDawnChorus:
    """Load the 'complete' split (the rigorous label set) for testing."""
    return WeldyDawnChorus(split="complete", sample_rate=16000, backend="pandas")


@pytest.fixture(scope="module")
def sample_indices(ds: WeldyDawnChorus) -> List[int]:
    """Deterministically choose up to 5 random indices for spot checks."""
    rng = random.Random(23)
    n = len(ds)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: WeldyDawnChorus):
    """Dataset should have at least one example."""
    assert len(ds) > 0


def test_available_splits(ds: WeldyDawnChorus) -> None:
    """available_splits should expose all five split names."""
    for split in ["all", "complete", "partial", "labeled", "unlabeled"]:
        assert split in ds.available_splits


def test_check_audio(ds: WeldyDawnChorus, sample_indices: List[int]):
    """Basic audio integrity checks on a few items."""
    for idx in sample_indices:
        item = ds[idx]
        audio = item["audio"]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.size >= 10
        assert not np.any(np.isnan(audio))
        assert not np.all(audio == 0)
        assert item["sample_rate"] == 16000


def test_check_selection_table(ds: WeldyDawnChorus, sample_indices: List[int]):
    """Selection table should be a multi-label DataFrame with sonotype + GBIF columns."""
    required = {"Begin Time (s)", "End Time (s)", "Species", "Species Code",
                "Common Name", "Sonotype", "Category", "Label"}
    for idx in sample_indices:
        item = ds[idx]
        st = item["selection_table"]
        assert isinstance(st, pd.DataFrame)
        missing = required - set(st.columns)
        assert not missing, f"[{idx}] missing columns: {sorted(missing)}"
        if len(st) > 0:
            assert not (st["Begin Time (s)"] < 0).any()
            assert not (st["End Time (s)"] - st["Begin Time (s)"] <= 0).any()


@pytest.mark.skipif(
    str(EXPECTED_LEN_ALL).startswith("__FILL"), reason="expected length not yet populated"
)
def test_expected_length():
    """The 'all' split length should be stable."""
    ds_all = WeldyDawnChorus(split="all", sample_rate=None, backend="pandas")
    assert len(ds_all) == int(EXPECTED_LEN_ALL)


@pytest.mark.skipif(
    str(EXPECTED_FIRST_ITEM_AUDIO_SHA256).startswith("__FILL"),
    reason="snapshot hashes not yet populated",
)
def test_reference_item_stability(ds: WeldyDawnChorus):
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
