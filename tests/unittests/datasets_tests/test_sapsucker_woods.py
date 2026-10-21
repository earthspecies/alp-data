"""
Unit tests for sapsucker_woods dataset.

Run with:
    pytest -q test_sapsucker_woods.py

Recordings are hour-long, and resampling one to 16 kHz with `kaiser_best`
takes ~2 minutes, so every test that reads audio does so at the default (native)
32 kHz.
"""

from __future__ import annotations

import random
from io import StringIO
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from alp_data.datasets import SapsuckerWoods
from alp_data.utils import create_hash

# # --- Dataset snapshot ---

# # Code to generate snapshot:
# from io import StringIO
# import pandas as pd
# from alp_data.datasets import SapsuckerWoods
# from alp_data.utils import create_hash
# for split in ["all", "all_with_unknowns"]:
#     ds = SapsuckerWoods(split=split, backend="pandas")
#     df = ds._data.unwrap
#     print(split, "len(ds) =", len(ds))
#     st = pd.concat([pd.read_csv(StringIO(x), sep="\t") for x in df["selection_table"]])
#     print("events:", len(st), "labels:", len(ds.get_available_labels()))
#     csv_bytes = df.sort_index(axis=0).sort_index(axis=1).to_csv(index=True).encode("utf-8")
#     print("annotations sha256:", create_hash(csv_bytes))

# ds = SapsuckerWoods(split="all", backend="pandas")
# audio0 = ds[0]["audio"]
# print("dtype:", audio0.dtype, "shape:", audio0.shape)
# print("sha256:", create_hash(audio0.tobytes()))
# # #

SPLITS = ["all", "all_with_unknowns"]
DEFAULT_SAMPLE_RATE = 32000
RECORDING_DURATION_S = 3600.0
UNKNOWN_LABEL = "Unknown"

EXPECTED_LEN = {"all": 285, "all_with_unknowns": 285}
EXPECTED_NUM_EVENTS = {"all": 50756, "all_with_unknowns": 53494}
EXPECTED_NUM_LABELS = {"all": 81, "all_with_unknowns": 82}
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "7fd40e29b33bc838a6fa38bdeaf24c54211de95d1f6661cd7b2404686783fa39"
)
ANNOTATIONS_SHA256 = {
    "all": "6d2c89fe9ce87bad45007defbeca81f22e70740221663c6e95abd213b7f49d13",
    "all_with_unknowns": "de7cfe3fcebb1c782bd30e104183d4c304de2b134e0623072b0e4960f3e5ec48",
}
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> SapsuckerWoods:
    """Load SapsuckerWoods dataset with default settings for testing."""
    return SapsuckerWoods(split="all")


@pytest.fixture(scope="module", params=SPLITS)
def ds_pandas(request: pytest.FixtureRequest) -> SapsuckerWoods:
    """Load each split with the pandas backend, without reading audio."""
    return SapsuckerWoods(split=request.param, backend="pandas")


@pytest.fixture(scope="module")
def sample_items(ds: SapsuckerWoods) -> Dict[int, Dict[str, Any]]:
    """Read up to 3 deterministic random items once, shared across tests."""
    n = len(ds)
    rng = random.Random(23)
    indices = rng.sample(range(n), min(3, n))
    return {idx: ds[idx] for idx in indices}


def test_ds_not_empty(ds: SapsuckerWoods):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_invalid_split_raises():
    """Unknown split names should raise a LookupError."""
    with pytest.raises(LookupError):
        SapsuckerWoods(split="train")


def test_check_audio(sample_items: Dict[int, Dict[str, Any]]):
    """Basic audio integrity checks on a few random items."""
    for idx, item in sample_items.items():
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert audio.dtype == np.float32, f"[{idx}] audio dtype is {audio.dtype}, expected float32"
        assert audio.ndim == 1, f"[{idx}] audio is not mono (shape={audio.shape})"
        assert audio.size >= 10, f"[{idx}] audio too short (size={audio.size})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"

        assert item["sample_rate"] == DEFAULT_SAMPLE_RATE, (
            f"[{idx}] sample_rate is {item['sample_rate']}, expected {DEFAULT_SAMPLE_RATE}"
        )
        duration = audio.size / item["sample_rate"]
        assert duration == pytest.approx(RECORDING_DURATION_S), (
            f"[{idx}] recording is {duration:.1f}s, expected ~{RECORDING_DURATION_S}s"
        )


def test_dataset_length_matches_expected(ds_pandas: SapsuckerWoods):
    """
    The dataset length should match the known, version-controlled expectation.

    This will fail loudly if:
    - the CSV split changed
    - files went missing
    - we accidentally filtered/augmented items differently

    If this fails intentionally (e.g. dataset grew), update EXPECTED_LEN.
    """
    split = ds_pandas.split
    assert len(ds_pandas) == EXPECTED_LEN[split], (
        f"[{split}] Dataset length mismatch: got {len(ds_pandas)}, expected "
        f"{EXPECTED_LEN[split]}. If this change is intentional (new data / new filtering), "
        "update EXPECTED_LEN in the test."
    )


def test_num_events_matches_expected(ds_pandas: SapsuckerWoods):
    """Total number of annotated events per split should match the snapshot."""
    split = ds_pandas.split
    n_events = sum(
        len(pd.read_csv(StringIO(st), sep="\t")) for st in ds_pandas._data.unwrap["selection_table"]
    )
    assert n_events == EXPECTED_NUM_EVENTS[split], (
        f"[{split}] got {n_events} events, expected {EXPECTED_NUM_EVENTS[split]}"
    )


def test_get_available_labels(ds_pandas: SapsuckerWoods):
    """Test get_available_labels for the Species column."""
    split = ds_pandas.split
    labels = ds_pandas.get_available_labels(anno_column="Species")
    assert isinstance(labels, list), "get_available_labels should return a list"
    assert len(labels) == EXPECTED_NUM_LABELS[split], (
        f"[{split}] got {len(labels)} labels, expected {EXPECTED_NUM_LABELS[split]}"
    )
    for label in labels:
        assert isinstance(label, str), f"Species label for {label} should be string"
    assert "????" not in labels, f"[{split}] raw eBird unknown code '????' was not mapped"


def test_unknown_label_only_in_with_unknowns_split(ds_pandas: SapsuckerWoods):
    """`Unknown` events must be removed from `all` and kept in `all_with_unknowns`."""
    labels = ds_pandas.get_available_labels(anno_column="Species")
    if ds_pandas.split == "all":
        assert UNKNOWN_LABEL not in labels, "'all' split should not contain Unknown events"
    else:
        assert UNKNOWN_LABEL in labels, "'all_with_unknowns' split should contain Unknown events"


def test_splits_share_recordings():
    """Both splits should list the same recordings in the same order."""
    paths = [
        list(SapsuckerWoods(split=s, backend="pandas")._data.unwrap["audio_path"]) for s in SPLITS
    ]
    assert paths[0] == paths[1], "Splits list different recordings"


def test_reference_item_stability(ds_pandas: SapsuckerWoods):
    """
    Check that the annotations of each split, and a canonical item (index 0),
    are bitwise-stable.

    We hash the raw float32 audio buffer (default 32 kHz). This catches:
    - channel handling changes (stereo->mono logic changed)
    - dtype changes
    - ordering changes in the split (if a different recording moved to idx 0)

    If this fails for a legitimate/intentional reason, recompute the hashes
    and update EXPECTED_FIRST_ITEM_AUDIO_SHA256 / ANNOTATIONS_SHA256.
    """
    split = ds_pandas.split

    # compute sha256 over the manifest, including the selection tables
    csv_bytes = (
        ds_pandas._data.unwrap.sort_index(axis=0)
        .sort_index(axis=1)
        .to_csv(index=True)
        .encode("utf-8")
    )
    h = create_hash(csv_bytes)

    assert h == ANNOTATIONS_SHA256[split], (
        f"[{split}] Annotation's hash changed.\n"
        f"Got    {h}\n"
        f"Expect {ANNOTATIONS_SHA256[split]}\n\n"
        "If this is an intentional dataset/content update, "
        "replace ANNOTATIONS_SHA256 with the new hash."
    )

    # Both splits share the same recordings, so only read audio once
    if split != "all":
        return

    item = ds_pandas[0]

    # audio presence/type checks (defensive, so the hash failure message is clearer)
    assert "audio" in item, "[0] missing 'audio' key"
    audio = item["audio"]
    assert isinstance(audio, np.ndarray), "[0] audio is not a numpy array"
    assert audio.dtype == np.float32, f"[0] audio dtype is {audio.dtype}, expected float32"

    # compute sha256 over raw bytes of the float32 array
    h = create_hash(audio.tobytes())

    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )


def test_check_selection_table(sample_items: Dict[int, Dict[str, Any]]):
    """Selection table should be a DataFrame with required columns and sane boxes."""
    required = {
        "Begin Time (s)",
        "End Time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Species",
    }

    for idx, item in sample_items.items():
        assert "selection_table" in item, f"[{idx}] missing 'selection_table' key"
        st = item["selection_table"]

        assert isinstance(st, pd.DataFrame), f"[{idx}] selection_table is not a DataFrame"
        missing = required - set(st.columns)
        assert not missing, f"[{idx}] selection_table missing columns: {sorted(missing)}"
        assert len(st) > 0, f"[{idx}] selection_table is empty"

        duration = item["audio"].size / item["sample_rate"]
        assert not (st["Begin Time (s)"] < 0).any(), f"[{idx}] negative begin times present"
        assert (st["End Time (s)"] > st["Begin Time (s)"]).all(), (
            f"[{idx}] zero-length or inverted events present"
        )
        assert (st["Begin Time (s)"] < duration).all(), f"[{idx}] events begin after audio end"
        assert (st["High Freq (Hz)"] > st["Low Freq (Hz)"]).all(), (
            f"[{idx}] inverted frequency bounds present"
        )
        assert (st["High Freq (Hz)"] <= item["sample_rate"] / 2).all(), (
            f"[{idx}] high frequency above Nyquist"
        )
        assert UNKNOWN_LABEL not in set(st["Species"]), (
            f"[{idx}] Unknown events present in 'all' split"
        )
