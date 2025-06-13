"""Test suite for the InsectSet459 dataset."""

import pytest

from esp_data.datasets import InsectSet459
from esp_data import Dataset, DatasetConfig


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an InsectSet459 dataset instance.

    Returns
    -------
    Dataset
        An instance of the InsectSet459 dataset.
    """
    ds = InsectSet459(split="validation")
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing an InsectSet459 dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the InsectSet459 dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="insectset_459",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "species_scientific",
                "output_feature": "label",
            },
            {
                "type": "filter",
                "mode": "exclude",
                "property": "genus",
                "values": ["Arunta", "Aleeta"],
            },
        ],
    )
    ds = InsectSet459(split="validation")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing an InsectSet459 dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the InsectSet459 dataset with output mapping applied.
    """
    dataset_config = DatasetConfig(
        dataset_name="insectset_459",
        output_take_and_give={"species_scientific": "species", "family": "fam"},
    )
    ds = InsectSet459(
        split="train", output_take_and_give=dataset_config.output_take_and_give
    )
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "insectset_459"
    assert dataset.info.version == "0.1.0"
    assert "train" in dataset.info.split_paths
    assert "validation" in dataset.info.split_paths


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be _loaded in __init__
    assert dataset._data is not None
    assert "local_path" in dataset._data
    assert "gbifID" in dataset._data


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    expected_columns = ["local_path", "gbifID", "species_scientific", "family", "genus"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should match the dataset info
    expected_splits = ["train", "validation"]
    assert set(dataset.available_splits) == set(expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = dataset._data.shape[0]
    assert len(dataset) == expected_len
    assert len(dataset) == 5307  # Example expected length, adjust as necessary


def test_getitem(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "gbifID" in sample
    assert "species_scientific" in sample
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
        dataset_name="insectset_459",
        split="train",
    )
    dataset, _ = InsectSet459.from_config(dataset_config)
    assert dataset.info.name == "insectset_459"
    assert dataset.info.split_paths["train"] is not None
    assert len(dataset) > 0, "Dataset should not be empty"


def test_invalid_split() -> None:
    """Test if _loading invalid split raises error."""
    with pytest.raises(LookupError):
        InsectSet459(split="invalid_split")


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
    assert "label" in dataset_with_transforms._data.columns

    # Check that the excluded genus is not present
    excluded_genus = "Arunta"
    assert not any(dataset_with_transforms._data["genus"] == excluded_genus), (
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
    assert set(sample.keys()) == {"species", "fam"}

    # Get the original row to compare values
    original_row = dataset_with_output_mapping._data.iloc[0]

    # Verify the mapping and values
    assert sample["species"] == original_row["species_scientific"]
    assert sample["fam"] == original_row["family"]
