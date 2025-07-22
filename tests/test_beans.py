"""Test suite for the Beans dataset."""

import pytest

from esp_data import Dataset, DatasetConfig
from esp_data.datasets import Beans
from esp_data.io import anypath


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an Beans dataset instance.

    Returns
    -------
    Dataset
        An instance of the Beans dataset.
    """
    ds = Beans(split="validation")
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing an Beans dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the Beans dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="beans",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "label",
                "output_feature": "Label",
            },
            {
                "type": "filter",
                "mode": "exclude",
                "property": "source_dataset",
                "values": ["speech_commands", "esc50"],
            },
        ],
    )
    ds = Beans(split="validation")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing an Beans dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the Beans dataset with output mapping applied.
    """
    dataset_config = DatasetConfig(
        dataset_name="beans",
        output_take_and_give={"source_dataset": "dataset_name", "label": "answer"},
    )
    ds = Beans(split="test", output_take_and_give=dataset_config.output_take_and_give)
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "beans"
    assert dataset.info.version == "0.1.0"
    assert "train" in dataset.info.split_paths
    assert "validation" in dataset.info.split_paths
    for split in dataset.info.split_paths.values():
        assert anypath(split).exists(), f"Split path {split} does not exist"


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be _loaded in __init__
    assert dataset._data is not None
    assert "local_path" in dataset._data
    assert "label" in dataset._data


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    expected_columns = ["local_path", "label", "source_dataset", "file_name"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should contain these
    expected_splits = ["train", "validation", "test", "cbi_test", "esc50_validation"]
    assert all(split in dataset.available_splits for split in expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = dataset._data.shape[0]
    assert len(dataset) == expected_len
    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 62415  # Example expected length, adjust as necessary


def test_getitem(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "label" in sample
    assert "audio" in sample


def test_iteration(dataset: Dataset) -> None:
    """Test if iteration works correctly."""
    for _, sample in enumerate(dataset):
        assert isinstance(sample, dict)
        # Ensure we can access a known key
        assert "audio" in sample
        break


def test_load_from_config() -> None:
    """Test if dataset can be loaded from configuration."""
    dataset_config = DatasetConfig(
        dataset_name="beans",
        split="test",
    )
    dataset, _ = Beans.from_config(dataset_config)
    assert dataset.info.name == "beans"
    assert dataset.info.split_paths["train"] is not None
    assert len(dataset) > 0, "Dataset should not be empty"


def test_invalid_split() -> None:
    """Test if _loading invalid split raises error."""
    with pytest.raises(LookupError):
        Beans(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    # Get same sample through different methods
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))

    # Compare samples
    assert direct_sample["local_path"] == iter_sample["local_path"]


def test_transformations(dataset_with_transforms: Dataset) -> None:
    """Test if transformations are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The filter transformation excludes specified genera

    """
    # Check that label column was created
    assert "Label" in dataset_with_transforms._data.columns

    # Check that the excluded genus is not present
    excluded_genus = "esc50"
    assert not any(dataset_with_transforms._data["source_dataset"] == excluded_genus), (
        f"Genus '{excluded_genus}' should be excluded from the dataset."
    )


def test_output_take_and_give(dataset_with_output_mapping: Dataset) -> None:
    """Test if output_take_and_give correctly maps column names.

    This test verifies that:
    1. The output dictionary contains only the mapped columns
    2. The original column names are mapped to the new names
    3. The values are preserved correctly
    """
    # Get a sample
    sample = dataset_with_output_mapping[0]

    # Check that only mapped columns are present
    assert set(sample.keys()) == {"dataset_name", "answer"}

    # Get the original row to compare values
    original_row = dataset_with_output_mapping._data.iloc[0]

    # Verify the mapping and values
    assert sample["dataset_name"] == original_row["source_dataset"]
    assert sample["answer"] == original_row["label"]


def test_max_duration_parameter() -> None:
    """Test if max_duration parameter works correctly."""
    # Test with custom max_duration
    dataset = Beans(split="validation", max_duration=5.0)
    assert dataset.max_duration == 5.0

    # Test default max_duration
    dataset_default = Beans(split="validation")
    assert dataset_default.max_duration == 10.0


def test_max_duration_audio_limiting() -> None:
    """Test if audio is actually limited by max_duration."""
    # Use a short max_duration to test limiting
    max_duration = 2.0
    dataset = Beans(split="validation", max_duration=max_duration, sample_rate=16000)

    # Get a sample and check audio duration
    sample = dataset[0]
    audio = sample["audio"]

    # Calculate actual duration (assuming 16kHz sample rate)
    actual_duration = len(audio) / 16000

    # Audio should not exceed max_duration (with small tolerance for rounding)
    assert actual_duration <= max_duration + 0.1, (
        f"Audio duration {actual_duration:.3f}s exceeds max_duration {max_duration}s"
    )


def test_max_duration_from_config() -> None:
    """Test if max_duration works correctly when loaded from config."""
    dataset_config = DatasetConfig(
        dataset_name="beans",
        split="validation",
        max_duration=3.0,
        sample_rate=16000,
    )

    dataset, _ = Beans.from_config(dataset_config)
    assert dataset.max_duration == 3.0

    # Test that audio respects the duration limit
    sample = dataset[0]
    audio = sample["audio"]
    actual_duration = len(audio) / 16000
    assert actual_duration <= 3.1  # Small tolerance for rounding


def test_short_audio_file_handling() -> None:
    """Test handling of audio files shorter than max_duration."""
    # Use a large max_duration to test short file handling
    large_max_duration = 300.0  # 5 minutes - should be longer than most files

    dataset = Beans(
        split="validation",
        max_duration=large_max_duration,
        sample_rate=16000,
    )

    # Get a sample - should not crash even with large max_duration
    sample = dataset[0]
    audio = sample["audio"]
    actual_duration = len(audio) / 16000

    # Audio should be the full file duration, not the max_duration
    # (since file is likely shorter than 300 seconds)
    assert actual_duration < large_max_duration, (
        f"Audio duration {actual_duration:.3f}s should be less than "
        f"max_duration {large_max_duration}s"
    )

    # Test that we can successfully load multiple samples without issues
    sample2 = dataset[1]
    audio2 = sample2["audio"]
    actual_duration2 = len(audio2) / 16000

    assert actual_duration2 < large_max_duration, (
        f"Second audio duration {actual_duration2:.3f}s should be less than "
        f"max_duration {large_max_duration}s"
    )
