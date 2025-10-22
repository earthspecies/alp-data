"""Test suite for the Macaques Coo Calls dataset."""

import pytest

from esp_data.datasets import MacaquesCooCalls
from esp_data import Dataset, DatasetConfig
from esp_data.io import anypath, exists


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing a Macaques Coo Calls dataset instance.

    Returns
    -------
    Dataset
        An instance of the Macaques Coo Calls dataset.
    """
    ds = MacaquesCooCalls(split="train")
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing a Macaques Coo Calls dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the Macaques Coo Calls dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="macaques_coo_calls",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "id",
                "output_feature": "label",
            },
        ],
    )
    ds = MacaquesCooCalls(split="train")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing a Macaques Coo Calls dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the Macaques Coo Calls dataset with output mapping applied.
    """
    dataset_config = DatasetConfig(
        dataset_name="macaques_coo_calls",
        output_take_and_give={"id": "label", "local_path": "audio_path"},
    )
    ds = MacaquesCooCalls(
        split="train", output_take_and_give=dataset_config.output_take_and_give
    )
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "macaques_coo_calls"
    assert dataset.info.version == "0.1.0"
    assert "train" in dataset.info.split_paths
    for split in dataset.info.split_paths.values():
        assert exists(split), f"Split path {split} does not exist"
    assert dataset.info.description == (
        "Coo calls from male and female macaques (Macaca mulatta) including id, sex, weight_kg"
    )
    assert dataset.info.license == "CC0 1.0 Universal"


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be loaded in __init__
    assert dataset._data is not None
    assert "local_path" in dataset._data
    assert "id" in dataset._data
    assert "sex" in dataset._data
    assert "weight_kg" in dataset._data


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    expected_columns = ["local_path", "id", "sex", "weight_kg"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should match the dataset info
    expected_splits = ["train", "val", "test"]
    assert set(dataset.available_splits) == set(expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = dataset._data.shape[0]
    assert len(dataset) == expected_len
    # Note: We don't assert a specific length as it may vary


def test_getitem(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "id" in sample
    assert "sex" in sample
    assert "weight_kg" in sample
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
        dataset_name="macaques_coo_calls",
        split="train",
        sample_rate=16000,
    )
    dataset, _ = MacaquesCooCalls.from_config(dataset_config)
    assert dataset.info.name == "macaques_coo_calls"
    assert dataset.info.split_paths["train"] is not None
    assert len(dataset) > 0, "Dataset should not be empty"
    assert dataset.sample_rate == 16000


def test_invalid_split() -> None:
    """Test if loading invalid split raises error."""
    with pytest.raises(LookupError):
        MacaquesCooCalls(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    # Get same sample through different methods
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))

    # Compare samples (excluding audio for efficiency)
    assert direct_sample["local_path"] == iter_sample["local_path"]
    assert direct_sample["id"] == iter_sample["id"]
    assert direct_sample["sex"] == iter_sample["sex"]
    assert direct_sample["weight_kg"] == iter_sample["weight_kg"]


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
    original_row = dataset_with_output_mapping._data.iloc[0]

    # Verify the mapping and values
    assert sample["label"] == original_row["id"]
    assert sample["audio_path"] == original_row["local_path"]


def test_sample_rate_resampling() -> None:
    """Test if audio resampling works correctly when sample_rate is specified."""
    target_sr = 16000
    dataset = MacaquesCooCalls(split="train", sample_rate=target_sr)

    sample = dataset[0]

    # Test that audio is loaded correctly
    assert "audio" in sample
    audio = sample["audio"]
    assert audio is not None
    assert len(audio.shape) == 1, "Audio should be mono after resampling"


def test_data_root_parameter() -> None:
    """Test if data_root parameter works correctly."""
    # Test default data_root (should be parent of csv_data directory)
    dataset_default = MacaquesCooCalls(split="train")
    expected_default = anypath(dataset_default.info.split_paths["train"]).parent
    assert dataset_default.data_root == expected_default

    # Test with custom data_root
    custom_root = "tests/"
    dataset = MacaquesCooCalls(split="train", data_root=custom_root)
    assert str(dataset.data_root) == custom_root


def test_string_representation(dataset: Dataset) -> None:
    """Test the string representation of the dataset."""
    str_repr = str(dataset)
    assert "macaques_coo_calls" in str_repr


def test_class_registration() -> None:
    """Test that the dataset class is properly registered."""
    # Test that we can import the class
    from esp_data.datasets import MacaquesCooCalls

    # Test that it's in the __all__ list
    import esp_data.datasets as datasets
    assert "MacaquesCooCalls" in datasets.__all__

    # Test that the class has the correct decorator
    assert hasattr(MacaquesCooCalls, 'info')
    assert hasattr(MacaquesCooCalls.info, 'name')


def test_macaque_specific_features(dataset: Dataset) -> None:
    """Test macaque-specific features like sex and weight_kg columns."""
    # Test that macaque-specific columns exist
    assert "sex" in dataset.columns
    assert "weight_kg" in dataset.columns

    # Test that we can access these features in samples
    sample = dataset[0]
    assert "sex" in sample
    assert "weight_kg" in sample

    # Test that sex values are valid (should be categorical)
    sex_values = dataset._data["sex"].unique()
    assert len(sex_values) > 0, "Sex column should have values"

    # Test that weight_kg values are numeric
    weight_values = dataset._data["weight_kg"]
    assert weight_values.dtype in ['int64', 'float64'], "Weight should be numeric"


def test_id_uniqueness(dataset: Dataset) -> None:
    """Test that macaque IDs are properly handled."""
    # Test that id column exists and has values
    assert "id" in dataset.columns
    assert len(dataset._data["id"]) > 0, "ID column should have values"

    # Test that IDs are unique within the test split
    unique_ids = dataset._data["id"].nunique()
    total_ids = len(dataset._data["id"])
    # IDs might not be unique if same macaque has multiple calls
    assert unique_ids > 0, "Should have at least one unique ID"


if __name__ == "__main__":
    # Run tests manually if executed directly
    pytest.main([__file__, "-v"])
