"""Test suite for the AudioSkillsXL dataset."""

import numpy as np
import pytest

from esp_data import Dataset, DatasetConfig
from esp_data.datasets import AudioSkillsXL


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an AudioSkillsXL dataset instance (fsd50k split)."""
    ds = AudioSkillsXL(split="fsd50k")
    return ds


@pytest.fixture
def dataset_with_transforms_from_config() -> tuple[Dataset, dict]:
    """Fixture providing an AudioSkillsXL dataset with transformations from config."""
    dataset_config = DatasetConfig(
        dataset_name="audio_skills_xl",
        split="fsd50k",
        transformations=[
            {
                "type": "filter",
                "property": "source",
                "values": ["fsd50k"],
                "mode": "include",
            },
        ],
    )
    ds, metadata = AudioSkillsXL.from_config(dataset_config)
    return ds, metadata


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing an AudioSkillsXL dataset with output mapping."""
    ds = AudioSkillsXL(
        split="fsd50k",
        output_take_and_give={"audio_path": "path"},
    )
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "audio_skills_xl"
    assert dataset.info.version == "0.1.0"
    assert "fsd50k" in dataset.info.split_paths
    assert "all" in dataset.info.split_paths


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    assert dataset._data is not None
    assert "audio_path" in dataset._data.columns


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    assert "audio_path" in dataset.columns
    assert "messages" in dataset.columns
    assert "source" in dataset.columns


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    expected_splits = ["counting_qa", "wavcaps", "fsd50k", "clotho_v2", "audioset", "audioset_sl", "all"]
    assert set(dataset.available_splits) == set(expected_splits)


def test_available_sources(dataset: Dataset) -> None:
    """Test if available_sources returns correct sources for a single-source split."""
    sources = dataset.available_sources
    assert "fsd50k" in sources


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
    assert "messages" in sample
    assert "audio_path" in sample

    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1


def test_iteration(dataset: Dataset) -> None:
    """Test if iteration works correctly."""
    for i, sample in enumerate(dataset):
        assert isinstance(sample, dict)
        assert "audio" in sample
        assert "messages" in sample
        if i >= 2:
            break


def test_invalid_split() -> None:
    """Test if initializing with invalid split raises error."""
    with pytest.raises(LookupError):
        AudioSkillsXL(split="invalid_split")


def test_invalid_sources() -> None:
    """Test if initializing with invalid sources raises error."""
    with pytest.raises(ValueError, match="Unknown sources"):
        AudioSkillsXL(split="all", sources=["nonexistent_source"])


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))
    assert direct_sample["audio_path"] == iter_sample["audio_path"]


def test_transformations_from_config(
    dataset_with_transforms_from_config: tuple[Dataset, dict],
) -> None:
    """Test if transformations from config are applied correctly."""
    ds, metadata = dataset_with_transforms_from_config
    assert "filter" in metadata


def test_output_take_and_give(dataset_with_output_mapping: Dataset) -> None:
    """Test if output_take_and_give correctly maps column names."""
    sample = dataset_with_output_mapping[0]
    assert "path" in sample
    assert "audio_path" not in sample


def test_audio_processing(dataset: Dataset) -> None:
    """Test if audio processing works correctly."""
    sample = dataset[0]
    assert "audio" in sample
    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1
    assert len(sample["audio"]) > 0


def test_str_representation(dataset: Dataset) -> None:
    """Test if string representation works correctly."""
    str_repr = str(dataset)
    assert "audio_skills_xl" in str_repr
    assert "0.1.0" in str_repr


def test_index_error_handling(dataset: Dataset) -> None:
    """Test if index error handling works correctly."""
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_messages_parsed(dataset: Dataset) -> None:
    """Test that messages are parsed from JSON into structured form."""
    sample = dataset[0]
    messages = sample["messages"]
    assert isinstance(messages, list)
    if len(messages) > 0:
        assert "role" in messages[0]
        assert "content" in messages[0]
