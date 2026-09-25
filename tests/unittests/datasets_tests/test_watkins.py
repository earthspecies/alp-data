"""Test suite for the Watkins Marine Mammal Sound Database dataset."""

import hashlib
import random

import numpy as np
import pytest

from alp_data import DatasetConfig
from alp_data.datasets import Watkins

# --- Dataset snapshot ---
# To regenerate:
#   from alp_data.datasets import Watkins
#   import hashlib
#   ds = Watkins(split="train", sample_rate=16000, backend="polars")
#   print("len:", len(ds))
#   audio0 = ds[0]["audio"]
#   print("audio sha256:", hashlib.sha256(audio0.tobytes()).hexdigest())
#   df = ds._data.unwrap
#   csv_bytes = df.sort(df.columns).write_csv().encode("utf-8")
#   print("annotations sha256:", hashlib.sha256(csv_bytes).hexdigest())

EXPECTED_LEN_TRAIN = 13693
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "3dd0ab979acb66a90d22997efa2ced95a62d60e96e7a8eb9ac5f2ded6f4cc1d6"
)
ANNOTATIONS_SHA256 = "3bbdf3b6a6c0faa4589596b9ccc7093027c40c1fb615d359bb4e9da282844fa4"


@pytest.fixture(scope="module")
def ds() -> Watkins:
    """Streaming polars dataset for cheap metadata tests."""
    return Watkins(split="train", backend="polars", streaming=True)


@pytest.fixture(scope="module")
def ds_eager() -> Watkins:
    """Eager polars dataset for indexing, length, and hash tests."""
    return Watkins(split="train", backend="polars", streaming=False)


@pytest.fixture(scope="module")
def sample_indices(ds_eager: Watkins) -> list[int]:
    """Deterministically choose up to 5 random indices for spot checks."""
    n = len(ds_eager)
    rng = random.Random(42)
    return [rng.randrange(n) for _ in range(min(5, n))]


# -- Metadata tests (streaming fixture, no audio I/O) -------------------------


def test_info_property(ds: Watkins) -> None:
    """Test if the info property returns correct metadata."""
    assert ds.info.name == "watkins"
    assert ds.info.version == "0.1.0"
    assert "train" in ds.info.split_paths
    assert ds.info.split_paths["train"] is not None


def test_columns_property(ds: Watkins) -> None:
    """Test if the columns property returns correct column names."""
    expected = ["audio_path", "species", "canonical_name", "species_common"]
    assert all(col in ds.columns for col in expected)


def test_available_splits(ds: Watkins) -> None:
    """Test if available_splits returns correct split names."""
    assert set(ds.available_splits) == {"train"}


def test_available_sample_rates(ds: Watkins) -> None:
    """Test if available_sample_rates property works correctly."""
    sample_rates = ds.available_sample_rates
    if "16khz_path" in ds.columns:
        assert 16000 in sample_rates
    if "32khz_path" in ds.columns:
        assert 32000 in sample_rates


def test_data_loaded(ds: Watkins) -> None:
    """Test that _data is populated after init."""
    assert ds._data is not None
    assert "audio_path" in ds._data.columns


def test_data_root_handling(ds: Watkins) -> None:
    """Test if data_root is set."""
    assert ds.data_root is not None


def test_iteration(ds: Watkins) -> None:
    """Test that iteration works in streaming mode."""
    for i, sample in enumerate(ds):
        assert isinstance(sample, dict)
        assert "audio" in sample
        assert "sample_rate" in sample
        if i >= 2:
            break


# -- Eager-mode tests (indexing / length) --------------------------------------


def test_length(ds_eager: Watkins) -> None:
    """Test __len__ returns the expected count."""
    assert len(ds_eager) == EXPECTED_LEN_TRAIN


def test_getitem(ds_eager: Watkins) -> None:
    """Test __getitem__ returns correct sample format."""
    sample = ds_eager[0]
    assert isinstance(sample, dict)
    assert "audio" in sample
    assert "sample_rate" in sample
    assert "audio_path" in sample
    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1


def test_check_audio(ds_eager: Watkins, sample_indices: list[int]) -> None:
    """Audio integrity checks on a few random items."""
    for idx in sample_indices:
        item = ds_eager[idx]
        audio = item["audio"]
        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert audio.dtype == np.float32, f"[{idx}] dtype {audio.dtype}, expected float32"
        assert audio.size >= 10, f"[{idx}] audio too short"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


def test_sample_consistency(ds_eager: Watkins) -> None:
    """Samples from indexing and iteration must agree."""
    direct = ds_eager[0]
    iter_sample = next(iter(ds_eager))
    assert direct["audio_path"] == iter_sample["audio_path"]


def test_index_error_handling(ds_eager: Watkins) -> None:
    """Out-of-bounds index must raise IndexError."""
    with pytest.raises(IndexError):
        _ = ds_eager[len(ds_eager)]


def test_str_representation(ds_eager: Watkins) -> None:
    """Test string representation."""
    s = str(ds_eager)
    assert "watkins" in s
    assert "0.1.0" in s
    assert "train" in s


# -- Stability hashes ---------------------------------------------------------


def test_reference_item_stability(ds_eager: Watkins) -> None:
    """First item audio must be bitwise-stable.

    Catches changes in audio processing, sample rate, channel handling,
    dtype, or row ordering.
    """
    item = ds_eager[0]
    assert "audio" in item
    audio = item["audio"]
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32

    h = hashlib.sha256(audio.tobytes()).hexdigest()
    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )


def test_reference_annotation_stability(ds_eager: Watkins) -> None:
    """Annotation CSV must be bitwise-stable.

    Catches changes in annotation content, column additions/removals,
    or row reordering.
    """
    df = ds_eager._data.unwrap
    csv_bytes = df.sort(df.columns).write_csv().encode("utf-8")
    h = hashlib.sha256(csv_bytes).hexdigest()
    assert h == ANNOTATIONS_SHA256, (
        "Annotation hash changed.\n"
        f"Got    {h}\n"
        f"Expect {ANNOTATIONS_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace ANNOTATIONS_SHA256 with the new hash."
    )


# -- Sample-rate variants (streaming, one item each) ---------------------------


def test_load_audio_32k() -> None:
    """Test that pre-resampled 32kHz audio loads correctly."""
    ds = Watkins(split="train", sample_rate=32000, backend="polars", streaming=True)
    sample = next(iter(ds))
    assert sample["audio"].dtype == np.float32
    assert len(sample["audio"]) > 0
    assert sample["sample_rate"] == 32000


def test_load_audio_original_rate() -> None:
    """Test that original-rate audio loads correctly (no resampling)."""
    ds = Watkins(split="train", sample_rate=None, backend="polars", streaming=True)
    sample = next(iter(ds))
    assert sample["audio"].dtype == np.float32
    assert len(sample["audio"]) > 0
    assert isinstance(sample["sample_rate"], int)


# -- Invalid split / config-based loading -------------------------------------


def test_invalid_split() -> None:
    """Invalid split must raise LookupError."""
    with pytest.raises(LookupError):
        Watkins(split="invalid_split", backend="polars", streaming=True)


def test_load_from_config() -> None:
    """Test dataset loading from DatasetConfig."""
    cfg = DatasetConfig(
        dataset_name="watkins",
        split="train",
        backend="polars",
        streaming=True,
    )
    ds, _ = Watkins.from_config(cfg)
    assert isinstance(ds, Watkins)
    assert ds.info.name == "watkins"


def test_transformations_from_config() -> None:
    """Test that transformations from config are applied correctly."""
    cfg = DatasetConfig(
        dataset_name="watkins",
        split="train",
        backend="polars",
        streaming=False,
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "canonical_name",
                "output_feature": "label",
            },
        ],
    )
    ds, metadata = Watkins.from_config(cfg)
    assert "label" in ds._data.columns
    assert "label_from_feature" in metadata
    assert "label_map" in metadata["label_from_feature"]
    assert len(metadata["label_from_feature"]["label_map"]) > 0


def test_output_take_and_give(ds_eager: Watkins) -> None:
    """Test that output_take_and_give correctly maps column names."""
    ds = Watkins(
        split="train",
        output_take_and_give={"canonical_name": "species", "species_common": "common_name"},
        backend="polars",
        streaming=False,
    )
    sample = ds[0]
    assert "species" in sample
    assert "common_name" in sample
    assert "canonical_name" not in sample
    assert "species_common" not in sample
