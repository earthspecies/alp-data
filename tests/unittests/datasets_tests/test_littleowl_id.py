"""Test suite for the LittleOwlId dataset."""

import pytest

from alp_data.datasets import LittleOwlId
from alp_data import Dataset, DatasetConfig
from alp_data.io import exists


@pytest.fixture
def dataset() -> Dataset:
    return LittleOwlId(split="train_across_year")


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    ds = LittleOwlId(split="train_across_year")
    cfg = DatasetConfig(
        dataset_name="littleowl_id",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "individual_id",
                "output_feature": "Label",
            }
        ],
    )
    ds.apply_transformations(cfg.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    mapping = {"individual_id": "owl_id"}
    return LittleOwlId(split="test_across_year", output_take_and_give=mapping)


def test_info_property(dataset: Dataset) -> None:
    assert dataset.info.name == "littleowl_id"
    assert dataset.info.version == "0.1.0"
    for split in dataset.info.split_paths.values():
        assert exists(split)


def test_available_splits(dataset: Dataset) -> None:
    expected = ["train_across_year", "test_across_year"]
    assert set(dataset.available_splits) == set(expected)


def test_length(dataset: Dataset) -> None:
    assert len(dataset) == len(dataset._data)


def test_getitem(dataset: Dataset) -> None:
    sample = dataset[0]
    assert "individual_id" in sample
    assert "audio" in sample


def test_iteration(dataset: Dataset) -> None:
    for sample in dataset:
        assert "audio" in sample
        break


def test_transformations(dataset_with_transforms: Dataset) -> None:
    assert "Label" in dataset_with_transforms._data.columns


def test_output_mapping(dataset_with_output_mapping: Dataset) -> None:
    sample = dataset_with_output_mapping[0]
    assert set(sample.keys()) == {"owl_id"}
