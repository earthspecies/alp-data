"""Unit tests for the ECOSoundSet dataset.

Run with:
    pytest -q tests/test_ecosoundset.py
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import pandas as pd
import pytest

from esp_data.datasets import EcoSoundSet

EXPECTED_LENS = {"train": 25351, "val": 9653, "test": 6879}
ST_REQUIRED = {
    "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)",
    "Species", "subspecies", "label_category",
}


@pytest.fixture(scope="module")
def ds() -> EcoSoundSet:
    """Load the test split (pandas backend).

    Returns
    -------
    EcoSoundSet
        The ``test`` split at 32 kHz.
    """
    return EcoSoundSet(split="test", sample_rate=32000, backend="pandas")


@pytest.fixture(scope="module")
def sample_indices(ds: EcoSoundSet) -> List[int]:
    """Deterministically choose up to 5 indices for spot checks.

    Returns
    -------
    list[int]
        Up to five reproducible random row indices.
    """
    n = len(ds)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_available_splits(ds: EcoSoundSet) -> None:
    """All three subset splits should be exposed."""
    for split in EXPECTED_LENS:
        assert split in ds.available_splits


@pytest.mark.parametrize("split,expected", list(EXPECTED_LENS.items()))
def test_expected_lengths(split: str, expected: int) -> None:
    """Each split manifest should have the expected clip count."""
    assert len(EcoSoundSet(split=split, backend="polars")) == expected


def test_species_lists(ds: EcoSoundSet, sample_indices: List[int]) -> None:
    """Every clip has a non-empty target list, contained in the all-species list."""
    for idx in sample_indices:
        row = ds._data[idx]
        target = str(row["target_species_list"])
        allsp = str(row["all_species_list"])
        assert target, f"[{idx}] empty target_species_list"
        assert set(target.split(", ")) <= set(allsp.split(", ")), f"[{idx}] target not ⊆ all"


def test_selection_table(ds: EcoSoundSet, sample_indices: List[int]) -> None:
    """Selection table parses with required columns and clip-relative times."""
    for idx in sample_indices:
        st = ds[idx]["selection_table"]
        assert isinstance(st, pd.DataFrame), f"[{idx}] selection_table not a DataFrame"
        miss = ST_REQUIRED - set(st.columns)
        assert not miss, f"[{idx}] selection_table missing columns: {sorted(miss)}"
        if len(st) > 0:
            assert st["Begin Time (s)"].min() >= 0, f"[{idx}] negative begin time"
            assert st["End Time (s)"].max() <= 4.0 + 1e-6, f"[{idx}] end beyond clip"
            assert (st["High Freq (Hz)"] >= st["Low Freq (Hz)"]).all(), f"[{idx}] high<low freq"


def test_check_audio(ds: EcoSoundSet, sample_indices: List[int]) -> None:
    """Sampled clips decode to non-trivial float32 audio.

    Skips if clips have not been extracted to GCS yet (build in progress).
    """
    try:
        ds[sample_indices[0]]
    except Exception as exc:  # noqa: BLE001 - audio not staged yet
        pytest.skip(f"ECOSoundSet audio not available yet: {exc}")
    for idx in sample_indices:
        audio = ds[idx]["audio"]
        assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
        assert audio.size >= 10 and not np.any(np.isnan(audio))
