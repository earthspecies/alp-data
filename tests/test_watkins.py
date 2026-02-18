"""Test suite for the Watkins Marine Mammal Sound Database dataset."""

import hashlib

import numpy as np
import pytest

from esp_data import Dataset, DatasetConfig
from esp_data.datasets import Watkins


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing a Watkins dataset instance."""
    return Watkins(split="train")


@pytest.fixture
def dataset_with_transforms_from_config() -> tuple[Dataset, dict]:
    """Fixture providing a Watkins dataset with transformations applied from config."""
    dataset_config = DatasetConfig(
        dataset_name="watkins",
        split="train",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "canonical_name",
                "output_feature": "label",
            },
        ],
    )
    return Watkins.from_config(dataset_config)


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing a Watkins dataset with output mapping."""
    return Watkins(
        split="train",
        output_take_and_give={"canonical_name": "species", "species_common": "common_name"},
    )


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "watkins"
    assert dataset.info.version == "0.1.0"
    assert "train" in dataset.info.split_paths
    assert dataset.info.split_paths["train"] is not None


def test_data_property(dataset: Dataset) -> None:
    """Test if the data is loaded in __init__."""
    assert dataset._data is not None
    assert "audio_path" in dataset._data.columns


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    expected_columns = ["audio_path", "species", "canonical_name", "species_common"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    expected_splits = ["train"]
    assert set(dataset.available_splits) == set(expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    expected_len = dataset._data.unwrap.shape[0]
    assert len(dataset) == expected_len
    assert len(dataset) > 0


def test_getitem(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "audio" in sample
    assert "sample_rate" in sample
    assert "audio_path" in sample
    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1


def test_iteration(dataset: Dataset) -> None:
    """Test if iteration works correctly."""
    for i, sample in enumerate(dataset):
        assert isinstance(sample, dict)
        assert "audio" in sample
        assert "sample_rate" in sample
        if i >= 2:
            break


def test_invalid_split() -> None:
    """Test if initializing with invalid split raises error."""
    with pytest.raises(LookupError):
        Watkins(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))
    assert direct_sample["audio_path"] == iter_sample["audio_path"]


def test_load_from_config() -> None:
    """Test if dataset can be loaded from configuration."""
    dataset_config = DatasetConfig(
        dataset_name="watkins",
        split="train",
    )
    ds, _ = Watkins.from_config(dataset_config)
    assert isinstance(ds, Watkins)
    assert ds.info.name == "watkins"
    assert len(ds) > 0


def test_transformations_from_config(
    dataset_with_transforms_from_config: tuple[Dataset, dict],
) -> None:
    """Test if transformations from config are applied correctly."""
    ds, metadata = dataset_with_transforms_from_config
    assert "label" in ds._data.columns
    assert "label_from_feature" in metadata
    assert "label_map" in metadata["label_from_feature"]
    assert len(metadata["label_from_feature"]["label_map"]) > 0


def test_output_take_and_give(dataset_with_output_mapping: Dataset) -> None:
    """Test if output_take_and_give correctly maps column names."""
    sample = dataset_with_output_mapping[0]
    assert "species" in sample
    assert "common_name" in sample
    assert "canonical_name" not in sample
    assert "species_common" not in sample


def test_audio_processing(dataset: Dataset) -> None:
    """Test if audio processing works correctly."""
    sample = dataset[0]
    assert "audio" in sample
    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1
    assert len(sample["audio"]) > 0
    assert not np.any(np.isnan(sample["audio"]))
    assert not np.all(sample["audio"] == 0)


def test_str_representation(dataset: Dataset) -> None:
    """Test if string representation works correctly."""
    str_repr = str(dataset)
    assert "watkins" in str_repr
    assert "0.1.0" in str_repr
    assert "train" in str_repr


def test_index_error_handling(dataset: Dataset) -> None:
    """Test if index error handling works correctly."""
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_available_sample_rates(dataset: Dataset) -> None:
    """Test if available_sample_rates property works correctly."""
    sample_rates = dataset.available_sample_rates
    if "16khz_path" in dataset.columns:
        assert 16000 in sample_rates
    if "32khz_path" in dataset.columns:
        assert 32000 in sample_rates


def test_data_root_handling(dataset: Dataset) -> None:
    """Test if data_root parameter works correctly."""
    assert dataset.data_root is not None


def test_reference_item_stability() -> None:
    """Test that the first item produces a consistent audio hash (bitwise stability).

    This catches changes in audio processing, sample rate, channel handling,
    dtype, or row ordering.  If this fails for a legitimate reason, recompute
    the hash and update EXPECTED_FIRST_ITEM_AUDIO_SHA256.
    """
    EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
        "3dd0ab979acb66a90d22997efa2ced95a62d60e96e7a8eb9ac5f2ded6f4cc1d6"
    )

    ds = Watkins(split="train")
    item = ds[0]

    assert "audio" in item, "[0] missing 'audio' key"
    audio = item["audio"]
    assert isinstance(audio, np.ndarray), "[0] audio is not a numpy array"
    assert audio.dtype == np.float32, f"[0] audio dtype is {audio.dtype}, expected float32"

    h = hashlib.sha256(audio.tobytes()).hexdigest()

    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )
