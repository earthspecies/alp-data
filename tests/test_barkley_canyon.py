"""Test suite for the BarkleyCanyon dataset."""

import pytest
import numpy as np

from esp_data.datasets import BarkleyCanyon, BarkleyCanyonDetection
from esp_data import Dataset, DatasetConfig
from esp_data.io import anypath, exists


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an BarkleyCanyon dataset instance.

    Returns
    -------
    Dataset
        An instance of the BarkleyCanyon dataset.
    """
    ds = BarkleyCanyon(split="train")
    return ds


@pytest.fixture
def dataset_detection() -> Dataset:
    """Fixture providing an BarkleyCanyonDetection dataset instance.

    Returns
    -------
    Dataset
        An instance of the BarkleyCanyonDetection dataset.
    """
    ds = BarkleyCanyonDetection(split="train")
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing an BarkleyCanyon dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the BarkleyCanyon dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="barkley_canyon",
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
                "values": ["Lagenorhynchus"],
            },
        ],
    )
    ds = BarkleyCanyon(split="train")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_transforms_from_config() -> tuple[Dataset, dict]:
    """Fixture providing an BarkleyCanyon dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the BarkleyCanyon dataset with transformations applied.
    dict
        Metadata from the transformations applied to the dataset.
    """

    dataset_config = DatasetConfig(
        dataset_name="barkley_canyon",
        split="train",
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
                "values": ["Lagenorhynchus"],
            },
        ],
    )
    return BarkleyCanyon.from_config(dataset_config)


@pytest.fixture
def dataset_detection_with_transforms() -> Dataset:
    """Fixture providing an BarkleyCanyonDetection dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the BarkleyCanyonDetection dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="barkley_canyon_detection",
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
                "values": ["Lagenorhynchus"],
            },
        ],
    )
    ds = BarkleyCanyonDetection(split="train")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing an BarkleyCanyon dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the BarkleyCanyon dataset with output mapping applied.
    """
    dataset_config = DatasetConfig(
        dataset_name="barkley_canyon",
        output_take_and_give={"species_scientific": "species", "family": "fam"},
    )
    ds = BarkleyCanyon(
        split="train", output_take_and_give=dataset_config.output_take_and_give
    )
    return ds


@pytest.fixture
def dataset_detection_with_output_mapping() -> Dataset:
    """Fixture providing an BarkleyCanyonDetection dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the BarkleyCanyonDetection dataset with output mapping applied.
    """
    dataset_config = DatasetConfig(
        dataset_name="barkley_canyon_detection",
        output_take_and_give={"species_scientific": "species", "family": "fam"},
    )
    ds = BarkleyCanyonDetection(
        split="train", output_take_and_give=dataset_config.output_take_and_give
    )
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "barkley_canyon"
    assert dataset.info.version == "0.1.0"
    assert "train" in dataset.info.split_paths
    assert "validation" not in dataset.info.split_paths
    assert exists(dataset.info.split_paths["train"])


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
    expected_splits = ["train"]
    assert set(dataset.available_splits) == set(expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = dataset._data.shape[0]
    assert len(dataset) == expected_len


def test_getitem(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert all([x in sample for x in ["Call Type", "species_scientific", "audio"]])


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
        dataset_name="BarkleyCanyon",
        split="train",
    )
    dataset, _ = BarkleyCanyon.from_config(dataset_config)
    assert isinstance(dataset, BarkleyCanyon)
    assert dataset.info.name == "barkley_canyon"
    assert dataset.info.split_paths["train"] is not None
    assert len(dataset) > 0, "Dataset should not be empty"


def test_invalid_split() -> None:
    """Test if _loading invalid split raises error."""
    with pytest.raises(LookupError):
        BarkleyCanyon(split="invalid_split")


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
    excluded_genus = "Lagenorhynchus"
    assert not any(dataset_with_transforms._data["genus"] == excluded_genus), (
        f"Genus '{excluded_genus}' should be excluded from the dataset."
    )


def test_transformations_from_config(dataset_with_transforms_from_config: tuple[Dataset, dict]) -> None:
    """Test if transformations are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The filter transformation excludes specified genera

    """
    ds, metadata = dataset_with_transforms_from_config
    # Check that label column was created
    assert "label" in ds._data.columns

    # Check that the excluded genus is not present
    excluded_genus = "Lagenorhynchus"
    assert not any(ds._data["genus"] == excluded_genus), (
        f"Genus '{excluded_genus}' should be excluded from the dataset."
    )

    assert "label_from_feature" in metadata
    assert "label_map" in metadata["label_from_feature"]
    assert len(metadata["label_from_feature"]["label_map"]) > 0, (
       "Label map should not be empty after transformations."
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
    assert sample["species"] and original_row["species_scientific"]
    assert sample["fam"] and original_row["family"]


# Repeat all tests for BarkleyCanyonDetection dataset

def test_detection_info_property(dataset_detection: Dataset) -> None:
    """Test if the info property returns correct metadata for detection dataset."""
    assert dataset_detection.info.name == "barkley_canyon_detection"
    assert dataset_detection.info.version == "0.1.0"
    assert "train" in dataset_detection.info.split_paths
    assert "validation" not in dataset_detection.info.split_paths
    assert exists(dataset_detection.info.split_paths["train"])


def test_detection_data_property(dataset_detection: Dataset) -> None:
    """Test if the data property returns correct dataframes for detection dataset."""
    # Data should be _loaded in __init__
    assert dataset_detection._data is not None
    assert "local_path" in dataset_detection._data
    assert "gbifID" in dataset_detection._data


def test_detection_columns_property(dataset_detection: Dataset) -> None:
    """Test if the columns property returns correct column names
    for detection dataset.
    """
    # Columns should match the dataframe columns
    expected_columns = ["local_path", "gbifID"]
    assert all(col in dataset_detection.columns for col in expected_columns)


def test_detection_available_splits(dataset_detection: Dataset) -> None:
    """Test if available_splits returns correct split names for detection dataset."""
    # Available splits should match the dataset info
    expected_splits = ["train"]
    assert set(dataset_detection.available_splits) == set(expected_splits)


def test_detection_length(dataset_detection: Dataset) -> None:
    """Test if __len__ returns correct counts for detection dataset."""
    # Length should be sum of all splits
    expected_len = dataset_detection._data.shape[0]
    assert len(dataset_detection) == expected_len


def test_detection_getitem(dataset_detection: Dataset) -> None:
    """Test if __getitem__ returns correct sample format for detection dataset."""
    # Get first sample
    sample = dataset_detection[0]
    assert isinstance(sample, dict)
    assert "species_common" in sample
    assert "species_scientific" in sample
    assert "audio" in sample


def test_detection_iteration(dataset_detection: Dataset) -> None:
    """Test if iteration works correctly for detection dataset."""
    for _, sample in enumerate(dataset_detection):
        assert isinstance(sample, dict)
        # Ensure we can access a known key
        assert "audio" in sample
        break


def test_detection_load_from_config() -> None:
    """Test if detection dataset can be loaded from configuration."""
    dataset_config = DatasetConfig(
        dataset_name="BarkleyCanyonDetection",
        split="train",
    )
    dataset, _ = BarkleyCanyonDetection.from_config(dataset_config)
    assert isinstance(dataset, BarkleyCanyonDetection)
    assert dataset.info.name == "barkley_canyon_detection"
    assert dataset.info.split_paths["train"] is not None
    assert len(dataset) > 0, "Dataset should not be empty"


def test_detection_invalid_split() -> None:
    """Test if loading detection dataset with invalid split raises error."""
    with pytest.raises(LookupError):
        BarkleyCanyonDetection(split="invalid_split")


def test_detection_sample_consistency(dataset_detection: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways for
    detection dataset.
    """
    # Get same sample through different methods
    direct_sample = dataset_detection[0]
    iter_sample = next(iter(dataset_detection))

    # Compare samples
    assert direct_sample["local_path"] == iter_sample["local_path"]


def test_detection_transformations(dataset_detection_with_transforms: Dataset) -> None:
    """Test if transformations are applied correctly for detection dataset.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The filter transformation excludes specified genera

    """
    # Check that label column was created
    assert "label" in dataset_detection_with_transforms._data.columns

    # Check that the excluded genus is not present
    excluded_genus = "Lagenorhynchus"
    assert not any(
        dataset_detection_with_transforms._data["genus"] == excluded_genus
    ), f"Genus '{excluded_genus}' should be excluded from the dataset."


def test_detection_output_take_and_give(
    dataset_detection_with_output_mapping: Dataset,
) -> None:
    """Test if output_take_and_give correctly maps column names for detection dataset.

    This test verifies that:
    1. The output dictionary contains only the mapped columns
    2. The original column names are mapped to the new names
    3. The values are preserved correctly
    """
    # Get a sample
    sample = dataset_detection_with_output_mapping[0]

    # Check that only mapped columns are present
    assert set(sample.keys()) == {"species", "fam"}

    # Get the original row to compare values
    original_row = dataset_detection_with_output_mapping._data.iloc[0]

    # Verify the mapping and values
    assert np.isnan(sample["species"]) and np.isnan(original_row["species_scientific"])
    assert np.isnan(sample["fam"]) and np.isnan(original_row["family"])
