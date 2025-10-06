"""Test suite for the Giant Otters dataset."""

import pytest

from esp_data.datasets import GiantOtters
from esp_data import Dataset, DatasetConfig
from esp_data.io import anypath, exists


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing a GiantOtters dataset instance.

    Returns
    -------
    Dataset
        An instance of the GiantOtters dataset.
    """
    ds = GiantOtters(split="test",
    )
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing a GiantOtters dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the GiantOtters dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="giant_otters",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "vocalization",
                "output_feature": "label",
            },
        ],
    )
    ds = GiantOtters(split="test")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing a GiantOtters dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the GiantOtters dataset with output mapping applied.
    """
    dataset_config = DatasetConfig(
        dataset_name="giant_otters",
        output_take_and_give={"vocalization": "label", "path": "audio_path"},
    )
    ds = GiantOtters(
        split="test", output_take_and_give=dataset_config.output_take_and_give
    )
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "giant_otters"
    assert dataset.info.version == "0.1.0"
    assert "test" in dataset.info.split_paths
    for split in dataset.info.split_paths.values():
        assert exists(split), f"Split path {split} does not exist"
    assert dataset.info.sources == ["PLOS ONE"]
    assert dataset.info.license == "CC-BY-4.0, CC0"


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be loaded in __init__
    assert dataset._data is not None
    assert "path" in dataset._data.columns
    assert "vocalization" in dataset._data.columns


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    expected_columns = ["path", "vocalization"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should match the dataset info
    expected_splits = ["test"]
    assert set(dataset.available_splits) == set(expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = len(dataset._data)
    assert len(dataset) == expected_len
    assert len(dataset) == 453


def test_getitem(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "vocalization" in sample
    assert "audio" in sample
    assert "path" in sample

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
        dataset_name="giant_otters",
        split="test",
        sample_rate=16000,
    )
    dataset, _ = GiantOtters.from_config(dataset_config)
    assert dataset.info.name == "giant_otters"
    assert dataset.info.split_paths["test"] is not None
    assert len(dataset) > 0, "Dataset should not be empty"
    assert dataset.sample_rate == 16000


def test_invalid_split() -> None:
    """Test if loading invalid split raises error."""
    with pytest.raises(LookupError):
        GiantOtters(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    # Get same sample through different methods
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))

    # Compare samples (excluding audio for efficiency)
    assert direct_sample["path"] == iter_sample["path"]
    assert direct_sample["vocalization"] == iter_sample["vocalization"]


def test_transformations(dataset_with_transforms: Dataset) -> None:
    """Test if transformations are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column

    """
    # Check that label column was created
    assert "label" in dataset_with_transforms._data.columns


def test_output_take_and_give(dataset_with_output_mapping: Dataset) -> None:
    """Test if output_take_and_give correctly maps column names.

    This test verifies that:
    1. The output dictionary contains only the mapped columns
    2. The original column names are mapped to the new names
    3. The values are preserved correctly
    """
    # Get a sample
    sample = dataset_with_output_mapping[0]

    # Check that only mapped columns are present (audio is not included when using output_take_and_give)
    expected_keys = {"label", "audio_path"}
    assert set(sample.keys()) == expected_keys

    # Get the original row to compare values
    original_row = dataset_with_output_mapping._data[0]

    # Verify the mapping and values
    assert sample["label"] == original_row["vocalization"]
    assert sample["audio_path"] == original_row["path"]


def test_sample_rate_resampling() -> None:
    """Test if audio resampling works correctly when sample_rate is specified."""
    target_sr = 16000
    dataset = GiantOtters(split="test", sample_rate=target_sr)

    sample = dataset[0]

    # Test that audio is loaded correctly
    assert "audio" in sample
    audio = sample["audio"]
    assert audio is not None
    assert len(audio.shape) == 1, "Audio should be mono after resampling"


def test_data_root_parameter() -> None:
    """Test if data_root parameter works correctly."""
    # Test default data_root (should be parent of split path)
    dataset_default = GiantOtters(split="test")
    expected_default = anypath(dataset_default.info.split_paths["test"]).parent
    assert dataset_default.data_root == expected_default

    # Test with custom data_root
    custom_root = "tests/"
    dataset = GiantOtters(split="test", data_root=custom_root)
    assert str(dataset.data_root) == custom_root


def test_string_representation(dataset: Dataset) -> None:
    """Test the string representation of the dataset."""
    str_repr = str(dataset)
    assert "Giant Otters" in str_repr


def test_class_registration() -> None:
    """Test that the dataset class is properly registered."""
    # Test that we can import the class
    from esp_data.datasets import GiantOtters

    # Test that it's in the __all__ list
    import esp_data.datasets as datasets
    assert "GiantOtters" in datasets.__all__

    # Test that the class has the correct decorator
    assert hasattr(GiantOtters, 'info')
    assert hasattr(GiantOtters.info, 'name')


if __name__ == "__main__":
    # Run tests manually if executed directly
    pytest.main([__file__, "-v"])
