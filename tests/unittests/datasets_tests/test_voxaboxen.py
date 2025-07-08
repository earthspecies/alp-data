"""Voxaboxen dataset tests."""

import pytest
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Literal

from esp_data.io import anypath
from esp_data.transforms import register_transform
from esp_data import Dataset, DatasetConfig, Voxaboxen, VoxaboxenEvents


class RenameConfig(BaseModel):
    type: Literal["rename_transform"]
    input_features: list[str]
    output_features: list[str]
    feature_map: dict[str, str] | None = None


class RenameTransform:
    def __init__(self, input_features: list[str], output_features: list[str], feature_map: dict[str, str] | None = None) -> None:
        """Initialize the RenameTransform."""

        if feature_map is None:
            # Create a map from input features to output features
            self.input_features = input_features
            self.output_features = output_features
            if len(self.input_features) != len(self.output_feature):
                raise ValueError("input_features and output_feature must have the same length")

            self.feature_map = dict(zip(self.input_features, self.output_features))
        else:
            # Use the provided feature map
            self.feature_map = feature_map

    @classmethod
    def from_config(cls, cfg: RenameConfig) -> "RenameTransform":
        return cls(**cfg.model_dump(exclude=("type",)))

    def __call__(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        # Rename
        transformed_data = data.rename(columns=self.feature_map)
        return transformed_data, self.feature_map


register_transform(RenameConfig, RenameTransform)


@pytest.fixture
def voxaboxen_dataset() -> Voxaboxen:
    """Fixture providing an AnimalSpeak dataset instance.

    Returns
    -------
    Dataset
        An instance of the AnimalSpeak dataset.
    """
    ds = Voxaboxen(split="hawaii_val", sample_rate=None)
    return ds


@pytest.fixture
def voxaboxen_events_dataset() -> VoxaboxenEvents:
    """Fixture providing a VoxaboxenEvents dataset instance.

    Returns
    -------
    VoxaboxenEvents
        An instance of the VoxaboxenEvents dataset.
    """
    ds = VoxaboxenEvents(split="hawaii_val", sample_rate=None)
    return ds


@pytest.fixture
def voxaboxen_with_transforms() -> Voxaboxen:
    """Fixture providing an AnimalSpeak dataset instance with transformations
    applied.

    Returns
    -------
    Voxaboxen
        An instance of the AnimalSpeak dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="voxaboxen",
        transformations=[
            {
                "type": "rename_transform",
                "input_features": ["fn"],
                "output_features": ["file_name"],
            },
        ],
    )
    ds = Voxaboxen(split="hawaii_val")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def voxaboxen_events_with_transforms() -> Dataset:
    """Fixture providing an AnimalSpeak dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the AnimalSpeak dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="voxaboxen_events",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "Annotation",
                "output_feature": "label",
            },
        ],
        stereo_to_mono="mono",
        segmentation_based=True,
        scale_factor=2,
    )
    ds = VoxaboxenEvents(split="hawaii_val")
    ds.apply_transformations(dataset_config.transformations)
    return ds


def test_info_property(voxaboxen_dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert voxaboxen_dataset.info.name == "voxaboxen"
    assert voxaboxen_dataset.info.version == "0.1.0"
    assert "Anuraset_train" in voxaboxen_dataset.info.split_paths
    for split in voxaboxen_dataset.info.split_paths.values():
        assert anypath(split).exists(), f"Split path {split} does not exist"


def test_data_property(voxaboxen_dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be _loaded in __init__
    assert voxaboxen_dataset._data is not None
    assert "selection_table_fp" in voxaboxen_dataset._data.columns
    assert "fn" in voxaboxen_dataset._data.columns


def test_available_splits(voxaboxen_dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should contain these
    expected_splits = ["Anuraset_train", "humpback_val", "test", "cbi_test", "esc50_validation"]
    assert all(split in voxaboxen_dataset.available_splits for split in expected_splits)


def test_length(voxaboxen_dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = voxaboxen_dataset._data.shape[0]
    assert len(voxaboxen_dataset) == expected_len
    print(f"Dataset length: {len(voxaboxen_dataset)}")
    assert len(voxaboxen_dataset) == 126


def test_getitem(voxaboxen_dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = voxaboxen_dataset[0]
    assert isinstance(sample, dict)
    assert "selection_table" in sample
    assert "audio" in sample


def test_iteration(voxaboxen_dataset: Dataset) -> None:
    """Test if iteration works correctly."""
    for _, sample in enumerate(voxaboxen_dataset):
        assert isinstance(sample, dict)
        # Ensure we can access a known key
        assert "audio" in sample
        break


def test_load_from_config() -> None:
    """Test if dataset can be loaded from configuration."""
    dataset_config = DatasetConfig(
        dataset_name="voxaboxen",
        split="hawaii_val",
    )
    dataset, _ = Voxaboxen.from_config(dataset_config)
    assert dataset.info.name == "voxaboxen"
    assert dataset.info.split_paths["katydids_train"] is not None
    assert len(dataset) > 0, "Dataset should not be empty"


def test_invalid_split() -> None:
    """Test if _loading invalid split raises error."""
    with pytest.raises(LookupError):
        Voxaboxen(split="invalid_split")


def test_dataset_with_transforms(voxaboxen_with_transforms: Dataset) -> None:
    """Test if dataset with transformations works correctly."""
    # Ensure the transformation was applied
    assert "file_name" in voxaboxen_with_transforms._data.columns
    assert "fn" not in voxaboxen_with_transforms._data.columns

    # Check if the transformation works
    sample = voxaboxen_with_transforms[0]
    assert isinstance(sample, dict)
    assert "file_name" in sample
    assert "audio" in sample


## VoxaboxenEvents Tests

def test_voxaboxen_events_info(voxaboxen_events_dataset: Dataset) -> None:
    """Test if the info property returns correct metadata for VoxaboxenEvents."""
    assert voxaboxen_events_dataset.info.name == "voxaboxen_events"
    assert voxaboxen_events_dataset.info.version == "0.1.0"
    assert "hawaii_val" in voxaboxen_events_dataset.info.split_paths
    for split in voxaboxen_events_dataset.info.split_paths.values():
        assert anypath(split).exists(), f"Split path {split} does not exist"


def test_voxaboxen_events_data_property(voxaboxen_events_dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes for detection dataset."""
    # Data should be _loaded in __init__
    assert voxaboxen_events_dataset._data is not None
    assert "local_path" in voxaboxen_events_dataset._data
    assert "gbifID" in voxaboxen_events_dataset._data
    assert voxaboxen_events_dataset._metadata is not None
    assert isinstance(voxaboxen_events_dataset._selection_table_dict, dict)
    assert len(voxaboxen_events_dataset._selection_table_dict) > 0, "Selection table should not be empty"


def test_getitem(voxaboxen_events_dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = voxaboxen_events_dataset[0]
    assert isinstance(sample, dict)
    assert "rev_anchor_anno" in sample
    assert "audio" in sample
    assert "class_anno" in sample
    assert isinstance(sample["class_anno"], np.ndarray)
    assert sample["class_anno"].ndim == 2
