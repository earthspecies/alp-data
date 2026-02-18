"""
Unit tests for DCLDE2026 dataset.

Run with:
    pytest -q test_dclde2026.py
"""

from __future__ import annotations

import hashlib
import random
from typing import List

import numpy as np
import pandas as pd
import pytest

from esp_data.datasets.dclde2026 import (
    DCLDE2026,
    ECOTYPE_LABELS,
    PROVENANCE_COLUMNS,
    PROVIDERS,
    SPECIES_LABELS,
    _selection_table_has_events,
)

# # --- Dataset snapshot ---

# # Code to generate snapshot:
# import hashlib
# from esp_data.datasets.dclde2026 import DCLDE2026
# ds = DCLDE2026(split="all", sample_rate=16000, backend="pandas")

# print("len(ds) =", len(ds))

# audio0 = ds[0]["audio"]
# print("dtype:", audio0.dtype, "shape:", audio0.shape)

# h = hashlib.sha256(audio0.tobytes()).hexdigest()
# print("audio sha256:", h)

# csv_bytes = (
#         ds._data.unwrap.sort_index(axis=0)
#         .sort_index(axis=1)
#         .to_csv(index=True)
#         .encode("utf-8")
#     )
# h = hashlib.sha256(csv_bytes).hexdigest()

# print("annotations sha256:", h)

# quit()
# # # # #

EXPECTED_LEN_ALL = 10061
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "87cbf23a8a86233ca55c6263bed7c25bdf4cd61ec69c91b602bc90f8b74fad92"
)
ANNOTATIONS_SHA256 = "44f038b229201df4a426c2ad8fb21662f347bc19e8d2efd35e0ddb87ee74ce9c"
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> DCLDE2026:
    """Load DCLDE2026 dataset with polars backend (default) for testing.

    Returns
    -------
    DCLDE2026
        Dataset instance with default polars backend.
    """
    return DCLDE2026(split="all", sample_rate=16000)


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
def sample_indices(ds: DCLDE2026) -> List[int]:
    """Deterministically choose up to 5 random indices for quick spot checks.

    Returns
    -------
    List[int]
        Up to 5 randomly selected indices.
    """
    n = len(ds)
    rng = random.Random(23)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: DCLDE2026) -> None:
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_dataset_length_matches_expected(ds: DCLDE2026) -> None:
    """
    The dataset length should match the known, version-controlled expectation.

    This will fail loudly if:
    - the CSV split changed
    - files went missing
    - we accidentally filtered/augmented items differently

    If this fails intentionally (e.g. dataset grew), update EXPECTED_LEN_ALL.
    """
    assert len(ds) == EXPECTED_LEN_ALL, (
        f"Dataset length mismatch: got {len(ds)}, expected {EXPECTED_LEN_ALL}. "
        "If this change is intentional (new data / new filtering), update EXPECTED_LEN_ALL "
        "in the test."
    )


def test_available_splits(ds: DCLDE2026) -> None:
    """Test if available_splits returns correct split names."""
    expected_splits = ["all"]
    assert all(split in ds.available_splits for split in expected_splits)


def test_check_audio(ds: DCLDE2026, sample_indices: List[int]) -> None:
    """Basic audio integrity checks on a few random items."""
    for idx in sample_indices:
        item = ds[idx]
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
    h = hashlib.sha256(audio.tobytes()).hexdigest()

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
    h = hashlib.sha256(csv_bytes).hexdigest()

    assert h == ANNOTATIONS_SHA256, (
        "Annotation's hash changed.\n"
        f"Got    {h}\n"
        f"Expect {ANNOTATIONS_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace ANNOTATIONS_SHA256 with the new hash."
    )


def test_check_selection_table(ds: DCLDE2026, sample_indices: List[int]) -> None:
    """Selection table should be a DataFrame with required columns and sane times."""
    required = {
        "Begin Time (s)",
        "End Time (s)",
        "species",
    }

    for idx in sample_indices:
        item = ds[idx]
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


def test_annotation_columns(ds: DCLDE2026) -> None:
    """annotation_columns should list the expected annotation fields."""
    expected = {"species", "ecotype", "call_type", "acoustic_behavior", "pod", "clan"}
    assert set(ds.annotation_columns) == expected


def test_get_available_labels_species(ds: DCLDE2026) -> None:
    """get_available_labels('species') should return SPECIES_LABELS."""
    labels = ds.get_available_labels("species")
    assert labels == SPECIES_LABELS


def test_get_available_labels_ecotype(ds: DCLDE2026) -> None:
    """get_available_labels('ecotype') should return ECOTYPE_LABELS."""
    labels = ds.get_available_labels("ecotype")
    assert labels == ECOTYPE_LABELS


def test_get_available_labels_unknown_raises(ds: DCLDE2026) -> None:
    """get_available_labels with an unknown column should raise ValueError."""
    with pytest.raises(ValueError, match="No predefined label set"):
        ds.get_available_labels("call_type")


def test_item_keys(ds: DCLDE2026) -> None:
    """Each item should have the expected top-level keys (incl. provenance)."""
    item = ds[0]
    expected_keys = {
        "audio_path", "audio", "sample_rate", "selection_table",
        *PROVENANCE_COLUMNS,
    }
    assert expected_keys.issubset(
        set(item.keys())
    ), f"Missing keys: {expected_keys - set(item.keys())}"


def test_str_representation(ds: DCLDE2026) -> None:
    """__str__ should include the dataset name and version."""
    s = str(ds)
    assert "dclde2026" in s
    assert "0.1.0" in s
    assert "CC-BY-4.0" in s


# ---------------------------------------------------------------------------
# Provider / sub-dataset tests
# ---------------------------------------------------------------------------


def test_available_providers_returns_all(ds: DCLDE2026) -> None:
    """Without filtering, all known providers should be present."""
    providers = ds.available_providers
    assert isinstance(providers, list)
    assert len(providers) > 0
    # Every provider that appears must be in the canonical list
    for p in providers:
        assert p in PROVIDERS, f"Unknown provider in data: {p}"


def test_provider_filtering() -> None:
    """Loading with a provider subset should reduce dataset length."""
    subset = ["SIMRES"]
    ds_filtered = DCLDE2026(split="all", sample_rate=16000, providers=subset)
    assert len(ds_filtered) > 0
    assert len(ds_filtered) < EXPECTED_LEN_ALL
    assert ds_filtered.available_providers == subset


def test_provider_filtering_invalid() -> None:
    """Passing an unknown provider name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown providers"):
        DCLDE2026(split="all", sample_rate=16000, providers=["NONEXISTENT"])


def test_str_includes_providers(ds: DCLDE2026) -> None:
    """__str__ should now include provider information."""
    s = str(ds)
    assert "Providers:" in s


# ---------------------------------------------------------------------------
# Negative-clip control tests
# ---------------------------------------------------------------------------


def test_selection_table_has_events_positive() -> None:
    """A TSV with header + data row should be detected as having events."""
    tsv = "Begin Time (s)\tEnd Time (s)\tspecies\n0.5\t1.2\tKiller whale"
    assert _selection_table_has_events(tsv) is True


def test_selection_table_has_events_negative() -> None:
    """A TSV with only a header (no event rows) should be negative."""
    tsv = "Begin Time (s)\tEnd Time (s)\tspecies"
    assert _selection_table_has_events(tsv) is False


def test_selection_table_has_events_empty() -> None:
    """An empty string should be negative."""
    assert _selection_table_has_events("") is False


def test_positives_only_empty_dict_filters() -> None:
    """positives_only={} should drop rows with no events (default True for all)."""
    ds_all = DCLDE2026(split="all", sample_rate=16000)
    ds_pos = DCLDE2026(split="all", sample_rate=16000, positives_only={})
    # Positive-only should have ≤ the full count
    assert len(ds_pos) <= len(ds_all)


def test_positives_only_none_returns_all() -> None:
    """positives_only=None (default) should return every row."""
    ds = DCLDE2026(split="all", sample_rate=16000)
    assert len(ds) == EXPECTED_LEN_ALL


def test_positives_only_per_provider() -> None:
    """Allowing negatives from one provider should give ≥ the positives-only count."""
    # Pick a single provider
    subset = ["SIMRES"]
    ds_pos = DCLDE2026(split="all", sample_rate=16000, providers=subset, positives_only={})
    ds_neg = DCLDE2026(
        split="all", sample_rate=16000, providers=subset,
        positives_only={"SIMRES": False},
    )
    # With negatives allowed, we should have at least as many rows
    assert len(ds_neg) >= len(ds_pos)
