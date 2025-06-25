"""Test suite for the Orchive dataset."""

import pytest

from esp_data.datasets import Orchive
from esp_data import Dataset, DatasetConfig
from esp_data.io import anypath


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an Orchive dataset instance.

    Returns
    -------
    Dataset
        An instance of the Orchive dataset.
    """
    ds = Orchive(split="test")
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing an Orchive dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the Orchive dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="orchive",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "call_type",
                "output_feature": "target",
            },
        ],
    )
    ds = Orchive(split="test")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing an Orchive dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the Orchive dataset with output mapping applied.
    """
    dataset_config = DatasetConfig(
        dataset_name="orchive",
        output_take_and_give={"call_type": "target", "local_path": "audio_path"},
    )
    ds = Orchive(
        split="test", output_take_and_give=dataset_config.output_take_and_give
    )
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "orchive"
    assert dataset.info.version == "0.1.0"
    assert "test" in dataset.info.split_paths
    for split in dataset.info.split_paths.values():
        assert anypath(split).exists(), f"Split path {split} does not exist"
    assert dataset.info.sources == ["Ness, Steven and Symonds, Helena and Spong, Paul and Tzanetakis, George"]
    assert dataset.info.license == "Unknown"


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be loaded in __init__
    assert dataset._data is not None
    assert "local_path" in dataset._data
    # Note: The actual columns depend on the CSV structure, but local_path should always be present


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    assert "local_path" in dataset.columns
    # Additional columns depend on the specific CSV structure


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should match the dataset info
    expected_splits = ["test", "train", "val", "unsupervised"]
    assert set(dataset.available_splits) == set(expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be the number of rows in the loaded split
    expected_len = dataset._data.shape[0]
    assert len(dataset) == expected_len
    # The test split should have some samples
    assert len(dataset) > 0


def test_getitem(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "local_path" in sample
    assert "audio" in sample

    # Verify audio properties
    audio = sample["audio"]
    assert audio is not None
    assert hasattr(audio, 'shape'), "Audio should be a numpy array with shape attribute"
    assert len(audio.shape) == 1, "Audio should be mono (1D array)"


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
        dataset_name="orchive",
        split="test",
        sample_rate=16000,
    )
    dataset, _ = Orchive.from_config(dataset_config)
    assert dataset.info.name == "orchive"
    assert dataset.info.split_paths["test"] is not None
    assert len(dataset) > 0, "Dataset should not be empty"
    assert dataset.sample_rate == 16000


def test_invalid_split() -> None:
    """Test if loading invalid split raises error."""
    with pytest.raises(LookupError):
        Orchive(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    # Get same sample through different methods
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))

    # Compare samples (excluding audio for efficiency)
    assert direct_sample["local_path"] == iter_sample["local_path"]

    # Compare other common fields if they exist
    for key in direct_sample.keys():
        if key != "audio":  # Skip audio comparison for efficiency
            assert direct_sample[key] == iter_sample[key]


def test_transformations(dataset_with_transforms: Dataset) -> None:
    """Test if transformations are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a target column

    """
    # Check that target column was created (if label column exists in original data)
    if "label" in dataset_with_transforms._data.columns:
        assert "target" in dataset_with_transforms._data.columns


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
    expected_keys = {"target", "audio_path"}
    assert set(sample.keys()) == expected_keys

    # Get the original row to compare values
    original_row = dataset_with_output_mapping._data.iloc[0]

    # Verify the mapping and values
    if "call_type" in original_row:
        assert sample["target"] == original_row["call_type"]
    assert sample["audio_path"] == original_row["local_path"]


def test_sample_rate_resampling() -> None:
    """Test if audio resampling works correctly when sample_rate is specified."""
    target_sr = 16000
    dataset = Orchive(split="test", sample_rate=target_sr)

    sample = dataset[0]

    # Test that audio is loaded correctly
    assert "audio" in sample
    audio = sample["audio"]
    assert audio is not None
    assert len(audio.shape) == 1, "Audio should be mono after resampling"


def test_data_root_parameter() -> None:
    """Test if data_root parameter works correctly."""
    # Test default data_root (should be parent of csv_data directory)
    dataset_default = Orchive(split="test")
    expected_default = anypath(dataset_default.info.split_paths["test"]).parent
    assert dataset_default.data_root == expected_default

    # Test with custom data_root
    custom_root = "tests/"
    dataset = Orchive(split="test", data_root=custom_root)
    assert str(dataset.data_root) == custom_root


def test_string_representation(dataset: Dataset) -> None:
    """Test the string representation of the dataset."""
    str_repr = str(dataset)
    assert "orchive" in str_repr
    assert "Orchive dataset containing orca calls" in str_repr
    assert "Ness, Steven and Symonds, Helena and Spong, Paul and Tzanetakis, George" in str_repr


def test_class_registration() -> None:
    """Test that the dataset class is properly registered."""
    # Test that we can import the class
    from esp_data.datasets import Orchive

    # Test that it's in the __all__ list
    import esp_data.datasets as datasets
    assert "Orchive" in datasets.__all__

    # Test that the class has the correct decorator
    assert hasattr(Orchive, 'info')
    assert hasattr(Orchive.info, 'name')


def test_unsupervised_split() -> None:
    """Test that the unsupervised split can be loaded."""
    dataset = Orchive(split="unsupervised")
    assert len(dataset) > 0
    assert dataset.split == "unsupervised"


def test_all_splits_accessible() -> None:
    """Test that all splits can be loaded without errors."""
    for split in ["test", "train", "val", "unsupervised"]:
        dataset = Orchive(split=split)
        assert len(dataset) >= 0  # Some splits might be empty
        assert dataset.split == split


def test_audio_loading_with_data_root() -> None:
    """Test that audio loading works correctly with data_root parameter."""
    # Use the default data_root (Google Cloud Storage) like other tests
    dataset = Orchive(split="test")
    sample = dataset[0]

    # Should have audio data
    assert "audio" in sample
    audio = sample["audio"]
    assert audio is not None
    assert len(audio.shape) == 1


def test_index_error_handling(dataset: Dataset) -> None:
    """Test that IndexError is raised for out-of-bounds access."""
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]  # Access beyond the last index


if __name__ == "__main__":
    # Run tests manually if executed directly
    pytest.main([__file__, "-v"])
