"""
Unit tests for DCLDE2026 dataset.

Run with:
    pytest -q test_dclde2026.py
"""

from __future__ import annotations

import random
from io import StringIO
from typing import List

import numpy as np
import pandas as pd
import pytest

from alp_data.datasets.dclde2026 import (
    DCLDE2026,
    ECOTYPE_LABELS,
    PROVENANCE_COLUMNS,
    SPECIES_LABELS,
)
from alp_data.io.filesystem import filesystem_from_path
from alp_data.utils import create_hash

EXPECTED_LEN_ALL = 9883
EXPECTED_LEN_VFPA_SRKW_STANDARD = 1336
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "87cbf23a8a86233ca55c6263bed7c25bdf4cd61ec69c91b602bc90f8b74fad92"
)
ANNOTATIONS_SHA256 = "715975d12bf739e576239c06f267251f6936b1a2c3b08165d23efbd9fbc7b1ec"
VFPA_SRKW_STANDARD_SHA256 = (
    "97de2b0b18a661b08261fc758c5f12265c82dbbc120e76844732e21657166317"
)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds_pandas() -> DCLDE2026:
    """Load DCLDE2026 dataset with pandas backend for testing.

    Returns
    -------
    DCLDE2026
        Dataset instance with pandas backend.
    """
    return DCLDE2026(split="all", sample_rate=16000, backend="pandas")


@pytest.fixture(scope="module")
def ds_vfpa_srkw_standard() -> DCLDE2026:
    """Load DCLDE2026 VFPA SRKW standard split with pandas backend.

    Returns
    -------
    DCLDE2026
        Dataset instance for the VFPA SRKW standard call-type split.
    """
    return DCLDE2026(split="vfpa_srkw_standard", sample_rate=16000, backend="pandas")


@pytest.fixture(scope="module")
def sample_indices(ds_pandas: DCLDE2026) -> List[int]:
    """Deterministically choose up to 3 random indices for quick spot checks.

    Returns
    -------
    List[int]
        Up to 5 randomly selected indices.
    """
    n = len(ds_pandas)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(3, n))]


def test_ds_not_empty(ds_pandas: DCLDE2026) -> None:
    """Dataset should have at least one example."""
    assert len(ds_pandas) > 0, "Dataset appears empty"


def test_dataset_length_matches_expected(ds_pandas: DCLDE2026) -> None:
    """
    The dataset length should match the known, version-controlled expectation.

    This will fail loudly if:
    - the CSV split changed
    - files went missing
    - we accidentally filtered/augmented items differently

    If this fails intentionally (e.g. dataset grew), update EXPECTED_LEN_ALL.
    """
    assert len(ds_pandas) == EXPECTED_LEN_ALL, (
        f"Dataset length mismatch: got {len(ds_pandas)}, expected {EXPECTED_LEN_ALL}. "
        "If this change is intentional (new data / new filtering), update EXPECTED_LEN_ALL "
        "in the test."
    )


def test_available_splits(ds_pandas: DCLDE2026) -> None:
    """Test if available_splits returns correct split names."""
    expected_splits = ["all", "vfpa_srkw_standard"]
    assert all(split in ds_pandas.available_splits for split in expected_splits)


def test_vfpa_srkw_standard_split(ds_vfpa_srkw_standard: DCLDE2026) -> None:
    """VFPA SRKW standard split should match the traced custom dataset."""
    assert len(ds_vfpa_srkw_standard) == EXPECTED_LEN_VFPA_SRKW_STANDARD

    split_path = ds_vfpa_srkw_standard.info.split_paths["vfpa_srkw_standard"]
    h = create_hash(filesystem_from_path(split_path).cat(split_path))
    assert h == VFPA_SRKW_STANDARD_SHA256

    df = ds_vfpa_srkw_standard._data.unwrap
    assert set(df["provider"]) == {"JASCO_VFPA", "JASCO_VFPA_ONC"}
    assert {"window_start_sec", "window_end_sec"}.issubset(df.columns)

    selection_tables = [
        pd.read_csv(StringIO(selection_table), sep="\t", keep_default_na=False)
        for selection_table in df["selection_table"]
    ]
    call_types = pd.concat(selection_tables, ignore_index=True)["call_type"]
    assert call_types.nunique() == 27
    assert len(call_types) == EXPECTED_LEN_VFPA_SRKW_STANDARD

    first_selection = selection_tables[0]
    assert len(first_selection) == 1
    assert first_selection["Begin Time (s)"].iloc[0] == 0
    assert first_selection["End Time (s)"].iloc[0] == pytest.approx(
        df["window_end_sec"].iloc[0] - df["window_start_sec"].iloc[0]
    )


def test_check_audio(ds_pandas: DCLDE2026, sample_indices: List[int]) -> None:
    """Basic audio integrity checks on a few random items."""
    for idx in sample_indices:
        item = ds_pandas[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert (
            audio.dtype == np.float32
        ), f"[{idx}] audio dtype is {audio.dtype}, expected float32"
        assert audio.size >= 10, f"[{idx}] audio too short (size={audio.size})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


def test_reference_item_stability(ds_pandas: DCLDE2026) -> None:
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
    h = create_hash(audio.tobytes())

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
    h = create_hash(csv_bytes)

    assert h == ANNOTATIONS_SHA256, (
        "Annotation's hash changed.\n"
        f"Got    {h}\n"
        f"Expect {ANNOTATIONS_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace ANNOTATIONS_SHA256 with the new hash."
    )


def test_check_selection_table(ds_pandas: DCLDE2026, sample_indices: List[int]) -> None:
    """Selection table should be a DataFrame with required columns and sane times."""
    required = {
        "Begin Time (s)",
        "End Time (s)",
        "species",
    }

    for idx in sample_indices:
        item = ds_pandas[idx]
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
            durs = st["End Time (s)"] - st["Begin Time (s)"]
            assert not durs.min() <= 0, f"[{idx}] events of dur <= 0"


def test_annotation_columns(ds_pandas: DCLDE2026) -> None:
    """annotation_columns should list the expected annotation fields."""
    expected = {"species", "ecotype", "call_type", "acoustic_behavior", "pod", "clan"}
    assert set(ds_pandas.annotation_columns) == expected


def test_get_available_labels_species(ds_pandas: DCLDE2026) -> None:
    """get_available_labels('species') should return SPECIES_LABELS."""
    labels = ds_pandas.get_available_labels("species")
    assert labels == SPECIES_LABELS


def test_get_available_labels_ecotype(ds_pandas: DCLDE2026) -> None:
    """get_available_labels('ecotype') should return ECOTYPE_LABELS."""
    labels = ds_pandas.get_available_labels("ecotype")
    assert labels == ECOTYPE_LABELS


def test_get_available_labels_unknown_raises(ds_pandas: DCLDE2026) -> None:
    """get_available_labels with an unknown column should raise ValueError."""
    with pytest.raises(ValueError, match="No predefined label set"):
        ds_pandas.get_available_labels("call_type")


def test_item_keys(ds_pandas: DCLDE2026) -> None:
    """Each item should have the expected top-level keys (incl. provenance)."""
    item = ds_pandas[0]
    expected_keys = {
        "audio_path", "audio", "sample_rate", "selection_table",
        *PROVENANCE_COLUMNS,
    }
    assert expected_keys.issubset(
        set(item.keys())
    ), f"Missing keys: {expected_keys - set(item.keys())}"


def test_str_representation(ds_pandas: DCLDE2026) -> None:
    """__str__ should include the dataset name and version."""
    s = str(ds_pandas)
    assert "dclde2026" in s
    assert "0.1.0" in s
    assert "CC-BY-4.0" in s


# if __name__ == "__main__":
#     # Code to generate snapshot:
#     from alp_data.datasets.dclde2026 import DCLDE2026
#     ds = DCLDE2026(split="all", sample_rate=16000, backend="pandas")

#     audio0 = ds[0]["audio"]
#     print("dtype:", audio0.dtype, "shape:", audio0.shape)

#     h = create_hash(audio0.tobytes())
#     print("audio sha256:", h)

#     csv_bytes = (
#             ds._data.unwrap.sort_index(axis=0)
#             .sort_index(axis=1)
#             .to_csv(index=True)
#             .encode("utf-8")
#         )
#     h = create_hash(csv_bytes)

#     print("annotations sha256:", h)
