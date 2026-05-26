"""Test suite for the Zebra Finch Julie Elie dataset."""

import pytest

from esp_data.datasets import ZebraFinchJulieElie
from esp_data import Dataset, DatasetConfig
from esp_data.io import anypath, exists


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing a Zebra Finch Julie Elie dataset instance.

    Returns
    -------
    Dataset
        An instance of the Zebra Finch Julie Elie dataset.
    """
    ds = ZebraFinchJulieElie(split="test",
    )
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing a Zebra Finch Julie Elie dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the Zebra Finch Julie Elie dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="zebra_finch_julie_elie",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "call_type_1",
                "output_feature": "label",
            },
        ],
    )
    ds = ZebraFinchJulieElie(split="test")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing a Zebra Finch Julie Elie dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the Zebra Finch Julie Elie dataset with output mapping applied.
    """
    dataset_config = DatasetConfig(
        dataset_name="zebra_finch_julie_elie",
        output_take_and_give={"call_type_1": "label", "local_path": "audio_path"},
    )
    ds = ZebraFinchJulieElie(
        split="test", output_take_and_give=dataset_config.output_take_and_give
    )
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "zebra_finch_julie_elie"
    assert dataset.info.version == "0.1.0"
    assert "test" in dataset.info.split_paths
    for split in dataset.info.split_paths.values():
        assert exists(split), f"Split path {split} does not exist"
    assert dataset.info.sources == ["Julie Elie"]
    assert dataset.info.license == "CC-BY-4.0, CC0"


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be loaded in __init__
    assert dataset._data is not None
    assert "local_path" in dataset._data.columns
    assert "call_type_1" in dataset._data.columns
    assert "call_type_2" in dataset._data.columns
    assert "call_type_id" in dataset._data.columns
    assert "age" in dataset._data.columns
    assert "id" in dataset._data.columns


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    expected_columns = ["local_path", "call_type_1", "call_type_2", "age", "id", "call_type_id"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should match the dataset info
    expected_splits = ["test", "train", "val", "full_dataset"]
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
    assert "call_type_1" in sample
    assert "call_type_2" in sample
    assert "age" in sample
    assert "id" in sample
    assert "call_type_id" in sample
    assert "audio" in sample
    assert "local_path" in sample

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
        dataset_name="zebra_finch_julie_elie",
        split="test",
        sample_rate=16000,
    )
    dataset, _ = ZebraFinchJulieElie.from_config(dataset_config)
    assert dataset.info.name == "zebra_finch_julie_elie"
    assert dataset.info.split_paths["test"] is not None
    assert len(dataset) > 0, "Dataset should not be empty"
    assert dataset.sample_rate == 16000


def test_invalid_split() -> None:
    """Test if loading invalid split raises error."""
    with pytest.raises(LookupError):
        ZebraFinchJulieElie(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    # Get same sample through different methods
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))

    # Compare samples (excluding audio for efficiency)
    assert direct_sample["local_path"] == iter_sample["local_path"]
    assert direct_sample["call_type_1"] == iter_sample["call_type_1"]
    assert direct_sample["call_type_2"] == iter_sample["call_type_2"]
    assert direct_sample["age"] == iter_sample["age"]
    assert direct_sample["id"] == iter_sample["id"]
    assert direct_sample["call_type_id"] == iter_sample["call_type_id"]


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

    # Check that only mapped columns are present
    expected_keys = {"label", "audio_path"}
    assert set(sample.keys()) == expected_keys

    # Get the original row to compare values
    original_row = dataset_with_output_mapping._data[0]

    # Verify the mapping and values
    assert sample["label"] == original_row["call_type_1"]
    assert sample["audio_path"] == original_row["local_path"]


def test_sample_rate_resampling() -> None:
    """Test if audio resampling works correctly when sample_rate is specified."""
    target_sr = 16000
    dataset = ZebraFinchJulieElie(split="test", sample_rate=target_sr)

    sample = dataset[0]

    # Test that audio is loaded correctly
    assert "audio" in sample
    audio = sample["audio"]
    assert audio is not None
    assert len(audio.shape) == 1, "Audio should be mono after resampling"


def test_data_root_parameter() -> None:
    """Test if data_root parameter works correctly."""
    # Test default data_root (should be parent of csv_data directory)
    dataset_default = ZebraFinchJulieElie(split="test")
    expected_default = anypath(dataset_default.info.split_paths["test"]).parent.parent
    assert dataset_default.data_root == expected_default

    # Test with custom data_root
    custom_root = "tests/"
    dataset = ZebraFinchJulieElie(split="test", data_root=custom_root)
    assert str(dataset.data_root) == custom_root


def test_string_representation(dataset: Dataset) -> None:
    """Test the string representation of the dataset."""
    str_repr = str(dataset)
    assert "zebra_finch_julie_elie" in str_repr


def test_class_registration() -> None:
    """Test that the dataset class is properly registered."""
    # Test that we can import the class
    from esp_data.datasets import ZebraFinchJulieElie

    # Test that it's in the __all__ list
    import esp_data.datasets as datasets
    assert "ZebraFinchJulieElie" in datasets.__all__

    # Test that the class has the correct decorator
    assert hasattr(ZebraFinchJulieElie, 'info')
    assert hasattr(ZebraFinchJulieElie.info, 'name')


if __name__ == "__main__":
    # Run tests manually if executed directly
    pytest.main([__file__, "-v"])
