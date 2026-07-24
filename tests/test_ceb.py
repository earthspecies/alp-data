"""Unit tests for the CEB dataset.

Run with:
    pytest -q tests/test_ceb.py
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import pandas as pd
import pytest

from esp_data.datasets import CEB

EXPECTED_LENS = {
    "train_xenocanto": 2210,
    "train_soundscape": 18469,
    "test_soundscape": 147,
}
ST_REQUIRED = {
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
    "sound_type",
    "ebird_code",
}


@pytest.fixture(scope="module")
def ds() -> CEB:
    """Load the strong test split (pandas backend).

    Returns
    -------
    CEB
        The ``test_soundscape`` split loaded at 32 kHz.
    """
    return CEB(split="test_soundscape", sample_rate=32000, backend="pandas")


@pytest.fixture(scope="module")
def sample_indices(ds: CEB) -> List[int]:
    """Deterministically choose up to 5 indices for spot checks.

    Returns
    -------
    list[int]
        Up to five reproducible random row indices.
    """
    n = len(ds)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_available_splits(ds: CEB) -> None:
    """All three subset splits should be exposed."""
    for split in EXPECTED_LENS:
        assert split in ds.available_splits


@pytest.mark.parametrize("split,expected", list(EXPECTED_LENS.items()))
def test_expected_lengths(split: str, expected: int) -> None:
    """Each split manifest should have the expected file count."""
    assert len(CEB(split=split, backend="polars")) == expected


def test_label_quality(ds: CEB) -> None:
    """test_soundscape is a strongly-labeled subset."""
    assert ds._data[0]["label_quality"] == "strong"


def test_selection_table(ds: CEB, sample_indices: List[int]) -> None:
    """Selection table parses to a DataFrame with required columns and sane times."""
    for idx in sample_indices:
        item = ds[idx]
        st = item["selection_table"]
        assert isinstance(st, pd.DataFrame), f"[{idx}] selection_table not a DataFrame"
        missing = ST_REQUIRED - set(st.columns)
        assert not missing, f"[{idx}] selection_table missing columns: {sorted(missing)}"
        if len(st) > 0:
            assert not (st["Begin Time (s)"] < 0).any(), f"[{idx}] negative begin times"
            durs = st["End Time (s)"] - st["Begin Time (s)"]
            assert durs.min() > 0, f"[{idx}] non-positive event durations"
            # strong subset carries frequency boxes
            assert (st["High Freq (Hz)"] >= st["Low Freq (Hz)"]).all(), f"[{idx}] high<low freq"


def test_check_audio(ds: CEB, sample_indices: List[int]) -> None:
    """Sampled rows decode to non-trivial float32 audio.

    Skips if the FLACs have not been extracted to GCS yet (build in progress).
    """
    try:
        ds[sample_indices[0]]
    except Exception as exc:  # noqa: BLE001 - audio not staged yet
        pytest.skip(f"CEB audio not available yet: {exc}")
    for idx in sample_indices:
        audio = ds[idx]["audio"]
        assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
        assert audio.size >= 10 and not np.any(np.isnan(audio))
        assert not np.all(audio == 0), f"[{idx}] audio all zeros"
