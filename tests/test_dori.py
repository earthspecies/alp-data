"""
Unit tests for the DORI dataset (Phase 1).

Run with:
    pytest -q tests/test_dori.py
"""

from __future__ import annotations

import hashlib
import random
from typing import List

import numpy as np
import pytest

from esp_data.datasets import DORI

# --- snapshot (fill after the split CSVs + audio are built) ---
EXPECTED_LEN_ALL = 27255
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "9bf7a8bfd8e10d58b4714cb85b1908211dad85da5db0cd46718766c738ca0d4a"
)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> DORI:
    """Load the 'all' split at 16 kHz (pandas backend)."""
    return DORI(split="all", sample_rate=16000, backend="pandas")


def test_ds_not_empty(ds: DORI):
    """Dataset should have clips."""
    assert len(ds) > 0


@pytest.mark.skipif(
    str(EXPECTED_LEN_ALL).startswith("__FILL"), reason="expected length not yet populated"
)
def test_expected_length(ds: DORI):
    """The 'all' split length should be stable."""
    assert len(ds) == int(EXPECTED_LEN_ALL)


def test_available_splits(ds: DORI) -> None:
    """available_splits should expose all split names."""
    for split in ["all", "train", "test", "onc", "orcasound", "ooi", "onc_benchmark"]:
        assert split in ds.available_splits


def test_label_columns(ds: DORI) -> None:
    """Clip-level label columns should be present."""
    for col in ["species", "species_common", "ecotype", "call_type", "presence", "source"]:
        assert col in ds.columns, f"missing column {col}"


def test_check_audio(ds: DORI):
    """Audio integrity on a few deterministic indices."""
    rng = random.Random(23)
    for idx in [rng.randrange(len(ds)) for _ in range(min(5, len(ds)))]:
        item = ds[idx]
        audio = item["audio"]
        assert isinstance(audio, np.ndarray), f"[{idx}] audio not ndarray"
        assert audio.dtype == np.float32, f"[{idx}] dtype {audio.dtype}"
        assert audio.size >= 10, f"[{idx}] too short"
        assert not np.any(np.isnan(audio)), f"[{idx}] NaN"
        assert not np.all(audio == 0), f"[{idx}] all zero"
        assert item["sample_rate"] == 16000, f"[{idx}] sr {item['sample_rate']}"


@pytest.mark.skipif(
    str(EXPECTED_FIRST_ITEM_AUDIO_SHA256).startswith("__FILL"),
    reason="snapshot hash not yet populated",
)
def test_reference_item_stability(ds: DORI):
    """Index-0 audio should be bitwise-stable."""
    audio = ds[0]["audio"]
    h = hashlib.sha256(audio.tobytes()).hexdigest()
    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        f"First item audio hash changed.\nGot {h}\nExpect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}"
    )
