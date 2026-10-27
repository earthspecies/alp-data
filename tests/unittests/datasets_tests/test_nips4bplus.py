"""
Unit tests for nips4bplus dataset.

Run with:
    pytest -q test_nips4bplus.py
"""

from __future__ import annotations

import random
from io import StringIO
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from alp_data.datasets import NIPS4BPlus
from alp_data.utils import create_hash

# # --- Dataset snapshot ---

# # Code to generate snapshot:
# from io import StringIO
# import pandas as pd
# from alp_data.datasets import NIPS4BPlus
# from alp_data.utils import create_hash
# for split in ["all", "all_with_nonbirds"]:
#     ds = NIPS4BPlus(split=split, backend="pandas")
#     df = ds._data.unwrap
#     print(split, "len(ds) =", len(ds))
#     sts = [pd.read_csv(StringIO(x), sep="\t") for x in df["selection_table"]]
#     print("events:", sum(len(st) for st in sts), "empty:", sum(st.empty for st in sts))
#     print("species:", len(ds.get_available_labels()))
#     print("classes:", len(ds.get_available_labels(anno_column="class_name")))
#     csv_bytes = df.sort_index(axis=0).sort_index(axis=1).to_csv(index=True).encode("utf-8")
#     print("annotations sha256:", create_hash(csv_bytes))

# ds = NIPS4BPlus(split="all", backend="pandas")
# audio0 = ds[0]["audio"]
# print("dtype:", audio0.dtype, "shape:", audio0.shape)
# print("sha256:", create_hash(audio0.tobytes()))
# # #

SPLITS = ["all", "all_with_nonbirds"]
DEFAULT_SAMPLE_RATE = 44100
MIN_DURATION_S = 1.0
MAX_DURATION_S = 5.004
UNKNOWN_LABEL = "Unknown"
# Non-bird classes, which only `all_with_nonbirds` keeps
NONBIRD_CLASSES = {
    "Cicatr_song",
    "Cicorn_song",
    "Lyrple_song",
    "Pelgra_call",
    "Phofem_song",
    "Plaaff_song",
    "Plasab_song",
    "Ptehey_song",
    "Tetpyg_song",
    "Tibtom_song",
    "Unknown",
    "Human",
}
# Training recordings without an annotation file, and with a 0-byte one
MISSING_FILE_IDS = [224, 280, 283, 305, 306, 343, 352, 432, 553, 568, 588, 650, 666]
UNANNOTATED_FILE_IDS = [61, 90, 215, 453, 492]
OMITTED_FILE_IDS = MISSING_FILE_IDS + UNANNOTATED_FILE_IDS

EXPECTED_LEN = {"all": 669, "all_with_nonbirds": 669}
EXPECTED_NUM_EVENTS = {"all": 5279, "all_with_nonbirds": 5775}
EXPECTED_NUM_EMPTY = {"all": 130, "all_with_nonbirds": 100}
EXPECTED_NUM_SPECIES = {"all": 51, "all_with_nonbirds": 63}
EXPECTED_NUM_CLASSES = {"all": 77, "all_with_nonbirds": 89}
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "ae66f750c7ed4b849205b7c956ce510e2d7740317a872a575322df52f49080c1"
)
ANNOTATIONS_SHA256 = {
    "all": "0005a933b73735375a789c3e53a781d257fbe5ad909cb1dc1e225e144d076346",
    "all_with_nonbirds": "5a48336f6c93e040ef57326cd1071e26fedaa8aa80a55040ef5a0c396ed3b60c",
}
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> NIPS4BPlus:
    """Load NIPS4BPlus dataset with default settings for testing."""
    return NIPS4BPlus(split="all")


@pytest.fixture(scope="module", params=SPLITS)
def ds_pandas(request: pytest.FixtureRequest) -> NIPS4BPlus:
    """Load each split with the pandas backend, without reading audio."""
    return NIPS4BPlus(split=request.param, backend="pandas")


@pytest.fixture(scope="module")
def sample_items(ds: NIPS4BPlus) -> Dict[int, Dict[str, Any]]:
    """Read up to 3 deterministic random items once, shared across tests."""
    n = len(ds)
    rng = random.Random(23)
    indices = rng.sample(range(n), min(3, n))
    return {idx: ds[idx] for idx in indices}


def _selection_tables(ds: NIPS4BPlus) -> list[pd.DataFrame]:
    """Parse every selection table of a split without reading audio."""
    return [pd.read_csv(StringIO(st), sep="\t") for st in ds._data.unwrap["selection_table"]]


def test_ds_not_empty(ds: NIPS4BPlus):
    """Dataset should have at least one example."""
    assert len(ds) > 0, "Dataset appears empty"


def test_invalid_split_raises():
    """Unknown split names should raise a LookupError."""
    with pytest.raises(LookupError):
        NIPS4BPlus(split="train")


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
        assert MIN_DURATION_S <= duration <= MAX_DURATION_S + 1e-3, (
            f"[{idx}] recording is {duration:.3f}s, expected {MIN_DURATION_S}-{MAX_DURATION_S}s"
        )


def test_resample():
    """Audio should be resampled to the requested rate."""
    ds = NIPS4BPlus(split="all", sample_rate=16000, backend="pandas")
    item = ds[0]
    native = NIPS4BPlus(split="all", backend="pandas")[0]
    assert item["sample_rate"] == 16000, f"sample_rate is {item['sample_rate']}, expected 16000"
    expected_size = native["audio"].size * 16000 / DEFAULT_SAMPLE_RATE
    assert item["audio"].size == pytest.approx(expected_size, abs=1), (
        f"resampled audio has {item['audio'].size} samples, expected ~{expected_size:.0f}"
    )


def test_dataset_length_matches_expected(ds_pandas: NIPS4BPlus):
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


def test_num_events_matches_expected(ds_pandas: NIPS4BPlus):
    """Total number of annotated events per split should match the snapshot."""
    split = ds_pandas.split
    n_events = sum(len(st) for st in _selection_tables(ds_pandas))
    assert n_events == EXPECTED_NUM_EVENTS[split], (
        f"[{split}] got {n_events} events, expected {EXPECTED_NUM_EVENTS[split]}"
    )


def test_num_empty_recordings_matches_expected(ds_pandas: NIPS4BPlus):
    """
    Recordings without events should be kept.

    In `all_with_nonbirds` these are exactly the 100 background-noise (`Empty`)
    recordings; `all` also has recordings with only non-bird events.
    """
    split = ds_pandas.split
    n_empty = sum(st.empty for st in _selection_tables(ds_pandas))
    assert n_empty == EXPECTED_NUM_EMPTY[split], (
        f"[{split}] got {n_empty} recordings without events, expected {EXPECTED_NUM_EMPTY[split]}"
    )


def test_get_available_labels(ds_pandas: NIPS4BPlus):
    """Test get_available_labels for the Species and class_name columns."""
    split = ds_pandas.split
    species = ds_pandas.get_available_labels(anno_column="Species")
    classes = ds_pandas.get_available_labels(anno_column="class_name")
    assert isinstance(species, list), "get_available_labels should return a list"
    assert len(species) == EXPECTED_NUM_SPECIES[split], (
        f"[{split}] got {len(species)} species, expected {EXPECTED_NUM_SPECIES[split]}"
    )
    assert len(classes) == EXPECTED_NUM_CLASSES[split], (
        f"[{split}] got {len(classes)} classes, expected {EXPECTED_NUM_CLASSES[split]}"
    )
    for label in species + classes:
        assert isinstance(label, str), f"Label {label} should be string"

    # Source spellings GBIF does not resolve should have been corrected
    assert "Lullula arborea" in species and "lullula arborea" not in species, (
        f"[{split}] 'lullula arborea' was not corrected"
    )
    assert "Regulus ignicapilla" in species and "Regulus ignicapillus" not in species, (
        f"[{split}] 'Regulus ignicapillus' was not corrected"
    )


def test_nonbird_labels_only_in_with_nonbirds_split(ds_pandas: NIPS4BPlus):
    """Non-bird events must be removed from `all` and kept in `all_with_nonbirds`."""
    classes = set(ds_pandas.get_available_labels(anno_column="class_name"))
    species = set(ds_pandas.get_available_labels(anno_column="Species"))
    if ds_pandas.split == "all":
        assert not classes & NONBIRD_CLASSES, (
            f"'all' split contains non-bird classes {sorted(classes & NONBIRD_CLASSES)}"
        )
    else:
        assert NONBIRD_CLASSES <= classes, (
            f"'all_with_nonbirds' split is missing {sorted(NONBIRD_CLASSES - classes)}"
        )
        assert {UNKNOWN_LABEL, "Homo sapiens", "Pelophylax kl. grafi"} <= species, (
            "'all_with_nonbirds' Unknown, Human or frog events are not labelled as expected"
        )
    assert "Human" not in species, "'Human' was not mapped to 'Homo sapiens'"
    assert "Empty" not in classes, "noise events should be omitted"


def test_class_name_maps_to_one_species(ds_pandas: NIPS4BPlus):
    """Each class_name should correspond to a single Species."""
    st = pd.concat([st for st in _selection_tables(ds_pandas) if not st.empty])
    n_species = st.groupby("class_name")["Species"].nunique()
    ambiguous = n_species[n_species > 1]
    assert ambiguous.empty, f"class_names with several species: {ambiguous.to_dict()}"


def test_omitted_recordings():
    """Recordings without an annotation file, or with a 0-byte one, should be omitted."""
    fns = set(NIPS4BPlus(split="all", backend="pandas")._data.unwrap["audio_file_name"])
    present = sorted(i for i in OMITTED_FILE_IDS if f"nips4b_birds_trainfile{i:03d}.wav" in fns)
    assert not present, f"unannotated recordings present: {present}"
    assert len(fns) + len(OMITTED_FILE_IDS) == 687, "recordings don't add up to 687 train files"


def test_splits_share_recordings():
    """Both splits should list the same recordings in the same order."""
    paths = [list(NIPS4BPlus(split=s, backend="pandas")._data.unwrap["audio_path"]) for s in SPLITS]
    assert paths[0] == paths[1], "Splits list different recordings"


def test_reference_item_stability(ds_pandas: NIPS4BPlus):
    """
    Check that the annotations of each split, and a canonical item (index 0),
    are bitwise-stable.

    We hash the raw float32 audio buffer (default 44.1 kHz). This catches:
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
    """Selection table should be a DataFrame with required columns and sane events."""
    required = {"Begin Time (s)", "End Time (s)", "Species", "class_name"}

    for idx, item in sample_items.items():
        assert "selection_table" in item, f"[{idx}] missing 'selection_table' key"
        st = item["selection_table"]

        assert isinstance(st, pd.DataFrame), f"[{idx}] selection_table is not a DataFrame"
        missing = required - set(st.columns)
        assert not missing, f"[{idx}] selection_table missing columns: {sorted(missing)}"

        duration = item["audio"].size / item["sample_rate"]
        assert not (st["Begin Time (s)"] < 0).any(), f"[{idx}] negative begin times present"
        assert (st["End Time (s)"] > st["Begin Time (s)"]).all(), (
            f"[{idx}] zero-length or inverted events present"
        )
        assert (st["End Time (s)"] <= duration + 1e-6).all(), f"[{idx}] events end after audio end"
        assert not set(st["class_name"]) & NONBIRD_CLASSES, (
            f"[{idx}] non-bird events present in 'all' split"
        )
