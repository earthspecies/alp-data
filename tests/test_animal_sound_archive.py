"""Test suite for the Animal Sound Archive dataset."""

import hashlib

import numpy as np
import pytest

from esp_data import Dataset, DatasetConfig
from esp_data.datasets import AnimalSoundArchive


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an Animal Sound Archive dataset instance.

    Returns
    -------
    Dataset
        An instance of the Animal Sound Archive dataset (validation split).
    """
    ds = AnimalSoundArchive(split="validation")
    return ds


@pytest.fixture
def dataset_with_transforms_from_config() -> tuple[Dataset, dict]:
    """Fixture providing an Animal Sound Archive dataset instance with
    transformations applied from config.

    Returns
    -------
    tuple[Dataset, dict]
        Dataset instance and metadata dict from applied transformations.
    """
    dataset_config = DatasetConfig(
        dataset_name="animal-sound-archive",
        split="validation",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "canonical_name",
                "output_feature": "label",
            },
        ],
    )
    ds, metadata = AnimalSoundArchive.from_config(dataset_config)
    return ds, metadata


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing an Animal Sound Archive dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance with output mapping applied.
    """
    ds = AnimalSoundArchive(
        split="validation",
        output_take_and_give={"canonical_name": "species"},
    )
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "animal-sound-archive"
    assert dataset.info.version == "0.1.0"
    assert "train" in dataset.info.split_paths
    assert dataset.info.split_paths["train"] is not None


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    assert dataset._data is not None
    assert "originals_path" in dataset._data.columns


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    assert "originals_path" in dataset.columns
    expected_columns = ["originals_path", "canonical_name", "16khz_path", "32khz_path"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    expected_splits = [
        "train", "validation", "all",
        "train_excl_beanszero", "validation_excl_beanszero", "all_excl_beanszero",
    ]
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
    assert "originals_path" in sample

    # Check audio properties
    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1  # Should be 1D for mono


def test_iteration(dataset: Dataset) -> None:
    """Test if iteration works correctly."""
    for i, sample in enumerate(dataset):
        assert isinstance(sample, dict)
        assert "audio" in sample
        assert "sample_rate" in sample
        assert "originals_path" in sample
        if i >= 2:  # Only test first few samples
            break


def test_invalid_split() -> None:
    """Test if initializing with invalid split raises error."""
    with pytest.raises(LookupError):
        AnimalSoundArchive(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))

    assert direct_sample["originals_path"] == iter_sample["originals_path"]


def test_transformations_from_config(
    dataset_with_transforms_from_config: tuple[Dataset, dict],
) -> None:
    """Test if transformations from config are applied correctly."""
    ds, metadata = dataset_with_transforms_from_config
    assert "label" in ds._data.columns
    assert "label_from_feature" in metadata
    assert "label_map" in metadata["label_from_feature"]


def test_output_take_and_give(dataset_with_output_mapping: Dataset) -> None:
    """Test if output_take_and_give correctly maps column names."""
    sample = dataset_with_output_mapping[0]
    assert "species" in sample
    assert "canonical_name" not in sample


def test_data_root_handling(dataset: Dataset) -> None:
    """Test if data_root parameter works correctly."""
    assert dataset.data_root is not None


def test_audio_processing(dataset: Dataset) -> None:
    """Test if audio processing works correctly."""
    sample = dataset[0]

    assert "audio" in sample
    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1  # Should be 1D for mono
    assert len(sample["audio"]) > 0


def test_str_representation(dataset: Dataset) -> None:
    """Test if string representation works correctly."""
    str_repr = str(dataset)
    assert "animal-sound-archive" in str_repr
    assert "0.1.0" in str_repr
    assert "validation" in str_repr


def test_index_error_handling(dataset: Dataset) -> None:
    """Test if index error handling works correctly."""
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_available_sample_rates(dataset: Dataset) -> None:
    """Test if available_sample_rates property works correctly."""
    sample_rates = dataset.available_sample_rates

    if "32khz_path" in dataset.columns:
        assert 32000 in sample_rates
    if "16khz_path" in dataset.columns:
        assert 16000 in sample_rates


def test_pre_resampled_audio_32khz(dataset: Dataset) -> None:
    """Test loading pre-resampled 32kHz audio."""
    dataset.sample_rate = 32000

    if "32khz_path" in dataset.columns:
        sample = dataset[0]
        assert "audio" in sample
        assert sample["audio"].dtype.name == "float32"
    else:
        sample = dataset[0]
        assert "audio" in sample
        assert sample["audio"].dtype.name == "float32"


def test_pre_resampled_audio_16khz(dataset: Dataset) -> None:
    """Test loading pre-resampled 16kHz audio."""
    dataset.sample_rate = 16000

    if "16khz_path" in dataset.columns:
        sample = dataset[0]
        assert "audio" in sample
        assert sample["audio"].dtype.name == "float32"
    else:
        sample = dataset[0]
        assert "audio" in sample
        assert sample["audio"].dtype.name == "float32"


def test_reference_item_stability() -> None:
    """Test that the first item produces a consistent audio hash (bitwise stability).

    This test ensures that audio loading and preprocessing are deterministic.
    If this test fails, it indicates changes in audio processing that affect
    the output waveform.
    """
    EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
        "18861e8b0e74206ada9c8da2ebd7c8f47c2356ae2f853eee6313477778e6e6ba"
    )

    ds = AnimalSoundArchive(split="train")
    idx = 0
    item = ds[idx]

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
