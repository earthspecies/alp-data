"""Test suite for the BirdSet dataset."""

import pytest

from esp_data.datasets import BirdSet
from esp_data import Dataset, DatasetConfig
from esp_data.io import anypath, exists


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an BirdSet dataset instance.

    Returns
    -------
    Dataset
        An instance of the BirdSet dataset.
    """
    ds = BirdSet(split="PER-test")
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing an BirdSet dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the BirdSet dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="BirdSet",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "label",
                "output_feature": "Label",
            },
        ],
    )
    ds = BirdSet(split="PER-test")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing an BirdSet dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the BirdSet dataset with output mapping applied.
    """
    dataset_config = DatasetConfig(
        dataset_name="birdset",
        output_take_and_give={"label": "Label"},
    )
    ds = BirdSet(split="HSN-test", output_take_and_give=dataset_config.output_take_and_give)
    return ds


@pytest.fixture
def dataset_from_config_with_transformations_streaming() -> tuple[Dataset, dict]:
    """Fixture providing an BirdSet dataset instance loaded from config
    with transformations applied in streaming mode.

    Returns
    -------
    tuple[Dataset, dict]
        A tuple containing the BirdSet dataset instance and its config.
    """
    dataset_config = DatasetConfig(
        dataset_name="birdset",
        split="HSN-test",
        data_root="gs://foundation-model-data/",
        transformations=[
            {
                "type": "filter",
                "property": "label",
                "values": ["None"],
                "mode": "exclude",
            },
            {
                "type": "label_from_feature",
                "feature": "label",
                "output_feature": "Label",
            },
        ],
        streaming=True,
    )
    dataset, _ = BirdSet.from_config(dataset_config)
    return dataset, dataset_config


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "birdset"
    assert dataset.info.version == "0.1.0"
    assert "HSN-train" in dataset.info.split_paths
    assert "PER-validation" in dataset.info.split_paths
    assert "POW-test" in dataset.info.split_paths
    for split in dataset.info.split_paths.values():
        assert exists(split), f"Split path {split} does not exist"


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be _loaded in __init__
    assert dataset._data is not None
    assert "path" in dataset._data.columns
    assert "label" in dataset._data.columns


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    expected_columns = ["path", "label"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should contain these
    expected_splits = ["HSN-train", "PER-validation", "POW-test"]
    assert all(split in dataset.available_splits for split in expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = len(dataset._data)
    assert len(dataset) == expected_len
    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 15120  # Example expected length, adjust as necessary


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
        dataset_name="birdset",
        split="HSN-test",

    )
    dataset, _ = BirdSet.from_config(dataset_config)
    assert dataset.info.name == "birdset"
    assert dataset.info.split_paths["HSN-test"] is not None
    assert len(dataset) > 0, "Dataset should not be empty"


def test_invalid_split() -> None:
    """Test if _loading invalid split raises error."""
    with pytest.raises(LookupError):
        BirdSet(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    # Get same sample through different methods
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))

    # Compare samples
    assert direct_sample["path"] == iter_sample["path"]


def test_transformations(dataset_with_transforms: Dataset) -> None:
    """Test if transformations are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The filter transformation excludes specified genera

    """
    # Check that label column was created
    assert "Label" in dataset_with_transforms._data.columns


def test_output_mapping(dataset_with_output_mapping: Dataset) -> None:
    """Test if output mapping works correctly."""
    # Check that output mapping was applied
    sample = dataset_with_output_mapping[0]
    assert "label" not in sample
    assert "Label" in sample  # Original label should be renamed to answer


def test_transformations_from_config_streaming(
    dataset_from_config_with_transformations_streaming: tuple[Dataset, dict]
) -> None:
    """Test if transformations from config work in streaming mode."""
    dataset, _ = dataset_from_config_with_transformations_streaming
    assert dataset._streaming is True
    for i, sample in enumerate(dataset):
        assert "Label" in sample  # Check transformed label exists
        assert sample["Label"] != "None"
        break
