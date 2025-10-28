"""
Unit tests for xeno_canto_annotated_jeantet_23 dataset.

Run with:
    pytest -q test_xeno_canto_annotated_jeantet_23.py
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import pandas as pd
import pytest

from esp_data.datasets import XenoCantoAnnotatedJeantet23


@pytest.fixture(scope="module")
def ds() -> XenoCantoAnnotatedJeantet23:
    """Load XenoCantoAnnotatedJeantet23 dataset for testing."""
    return XenoCantoAnnotatedJeantet23(split="all", sample_rate=16000)


@pytest.fixture(scope="module")
def sample_indices(ds: XenoCantoAnnotatedJeantet23) -> List[int]:
    """Deterministically choose up to 5 random indices for quick spot checks."""
    n = len(ds)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: XenoCantoAnnotatedJeantet23):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_check_audio(ds: XenoCantoAnnotatedJeantet23, sample_indices: List[int]):
    """Basic audio integrity checks on a few random items."""
    for idx in sample_indices:
        item = ds[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert audio.dtype == np.float32, f"[{idx}] audio dtype is {audio.dtype}, expected float32"
        assert audio.size >= 10, f"[{idx}] audio too short (size={audio.size})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


def test_check_selection_table(ds: XenoCantoAnnotatedJeantet23, sample_indices: List[int]):
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

        assert isinstance(st, pd.DataFrame), f"[{idx}] selection_table is not a DataFrame"
        missing = required - set(st.columns)
        assert not missing, f"[{idx}] selection_table missing columns: {sorted(missing)}"

        if len(st) > 0:
            assert not (st["Begin Time (s)"] < 0).any(), f"[{idx}] negative begin times present"
