"""
Unit tests for subsegmentation dataset.

Run with:
    pytest -q test_subsegmentation.py
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import pandas as pd
import pytest

from esp_data.datasets import Subsegmentation


@pytest.fixture(scope="module")
def ds() -> Subsegmentation:
    """Load Subsegmentation dataset for testing."""
    return Subsegmentation(split="all", sample_rate=16000)


@pytest.fixture(scope="module")
def sample_indices(ds: Subsegmentation) -> List[int]:
    """Deterministically choose up to 5 random indices for quick spot checks."""
    n = len(ds)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: Subsegmentation):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_check_audio(ds: Subsegmentation, sample_indices: List[int]):
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

        assert isinstance(st, pd.DataFrame), f"[{idx}] selection_table is not a DataFrame"
        missing = required - set(st.columns)
        assert not missing, f"[{idx}] selection_table missing columns: {sorted(missing)}"

        if len(st) > 0:
            assert not (st["Begin Time (s)"] < 0).any(), f"[{idx}] negative begin times present"


def test_qc_flag_consistent(ds: Subsegmentation, sample_indices: List[int]):
    """QC flag should reflect whether there are any rows in the selection table."""
    for idx in sample_indices:
        item = ds[idx]
        assert "pass_qc" in item, f"[{idx}] missing 'pass_qc' key"
        st = item["selection_table"]
        expected_pass_qc = len(st) > 0
        assert item["pass_qc"] == expected_pass_qc, (
            f"[{idx}] qc inconsistent: pass_qc={item['pass_qc']} but len(st)={len(st)}"
        )
