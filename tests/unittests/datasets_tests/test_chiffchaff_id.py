"""Test suite for the ChiffchaffID dataset."""

import pytest

from esp_data.datasets import ChiffchaffId
from esp_data import Dataset, DatasetConfig
from esp_data.io import anypath


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing a ChiffchaffID dataset instance (train within-year split)."""

    ds = ChiffchaffId(split="train_within_year")
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """ChiffchaffID dataset with a label transformation applied."""

    ds = ChiffchaffId(split="train_within_year")

    dataset_config = DatasetConfig(
        dataset_name="chiffchaff_id",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "individual_id",
                "output_feature": "Label",
            },
        ],
    )

    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """ChiffchaffID dataset with output_take_and_give mapping applied."""

    mapping = {"individual_id": "bird_id"}
    ds = ChiffchaffId(split="test_within_year", output_take_and_give=mapping)
    return ds


# -----------------------------------------------------------------------------
# Basic dataset property tests
# -----------------------------------------------------------------------------

def test_info_property(dataset: Dataset) -> None:
    assert dataset.info.name == "chiffchaff_id"
    assert dataset.info.version == "0.1.0"
    assert "train_within_year" in dataset.info.split_paths
    assert "test_within_year" in dataset.info.split_paths
    for split in dataset.info.split_paths.values():
        assert anypath(split).exists(), f"Split path {split} does not exist"


def test_data_property(dataset: Dataset) -> None:
    assert dataset._data is not None
    assert "local_path" in dataset._data
    assert "individual_id" in dataset._data


def test_columns_property(dataset: Dataset) -> None:
    expected_columns = ["local_path", "individual_id"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    expected_splits = [
        "train_within_year",
        "test_within_year",
        "train_across_year",
        "test_across_year",
    ]
    assert set(dataset.available_splits) == set(expected_splits)


def test_length(dataset: Dataset) -> None:
    expected_len = 5107  # Number of rows in within-year FG train split (excludes header)
    assert len(dataset) == expected_len


def test_getitem(dataset: Dataset) -> None:
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "individual_id" in sample
    assert "audio" in sample


def test_iteration(dataset: Dataset) -> None:
    for _, sample in enumerate(dataset):
        assert isinstance(sample, dict)
        assert "audio" in sample
        break


def test_load_from_config() -> None:

    dataset_config = DatasetConfig(
        dataset_name="chiffchaff_id",
        split="train_within_year",
    )
    dataset, _ = ChiffchaffId.from_config(dataset_config)
    assert dataset.info.name == "chiffchaff_id"
    assert dataset.info.split_paths["train_within_year"] is not None
    assert len(dataset) == 5107


def test_invalid_split() -> None:
    with pytest.raises(LookupError):
        ChiffchaffId(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))
    assert direct_sample["local_path"] == iter_sample["local_path"]


def test_transformations(dataset_with_transforms: Dataset) -> None:
    assert "Label" in dataset_with_transforms._data.columns


def test_output_take_and_give(dataset_with_output_mapping: Dataset) -> None:
    sample = dataset_with_output_mapping[0]
    assert set(sample.keys()) == {"bird_id"}

    original_row = dataset_with_output_mapping._data.iloc[0]
    assert sample["bird_id"] == original_row["individual_id"]
