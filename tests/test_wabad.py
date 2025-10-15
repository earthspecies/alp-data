"""
Unit tests for WABAD dataset.

Run with:
    pytest -q test_wabad.py
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esp_data.datasets import WABAD

@pytest.fixture(scope="module")
def ds() -> WABAD:
    """
    Requires:
      - split CSV available at WABAD.info.split_paths["all"]
      - label_mapping.json produced by preprocess_wabad.py and placed next to wabad_dataset.py
      - audio files accessible via data_root (implicitly parent of split unless overridden)
    """
    return WABAD(
        split="all",
        sample_rate=16000,
    )

def _basic_item_qc(item: dict) -> list[str]:
    issues = []

    # Audio checks
    audio = item["audio"]
    if not isinstance(audio, np.ndarray):
        issues.append("audio_not_ndarray")
    elif audio.dtype != np.float32:
        issues.append(f"audio_dtype_not_float32:{audio.dtype}")
    if audio.size < 10:
        issues.append("audio_too_short")
    if np.any(np.isnan(audio)):
        issues.append("audio_has_nan")
    if np.all(audio == 0):
        issues.append("audio_all_zeros")

    # Selection table checks
    st = item["selection_table"]
    if not isinstance(st, pd.DataFrame):
        issues.append("selection_table_not_df")
    else:
        for col in ["Begin Time (s)", "End Time (s)", "Species"]:
            if col not in st.columns:
                issues.append(f"missing_col_{col}")

        # ensure no begin >= audio duration
        # we do not know SR here; dataset trimmed by duration already
        if "Begin Time (s)" in st.columns and len(st) > 0:
            if (st["Begin Time (s)"] < 0).any():
                issues.append("negative_begin_time")

    return issues


def test_random_five_items_quality(ds: WABAD):
    n = len(ds)
    assert n > 0, "Dataset appears empty"

    rng = random.Random(23)
    indices = [rng.randrange(n) for _ in range(min(5, n))]

    all_issues = []
    for idx in indices:
        item = ds[idx]
        issues = _basic_item_qc(item)
        all_issues.extend([(idx, i) for i in issues])

    # If you want this to be strictly clean, assert no issues:
    # assert not all_issues, f"Issues found: {all_issues}"
    # Instead: allow visibility without hard failing CI on minor glitches
    if all_issues:
        pytest.fail(f"Quality issues in random items: {all_issues}")


def test_label_mapping_loaded(ds: WABAD):
    # sanity: label mappings exist for all expected columns
    assert set(ds.annotation_columns) == {"Genus", "Family", "Order", "Common", "Species"}
    for k in ds.annotation_columns:
        assert isinstance(ds.label_mappings.get(k, {}), dict)
