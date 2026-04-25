"""Test suite for the F0 Bioacoustic Benchmark dataset."""

import numpy as np
import pandas as pd
import pytest

from esp_data import Dataset, DatasetConfig
from esp_data.datasets import F0Bioacoustic
from esp_data.datasets.f0_bioacoustic import SPECIES_LABELS, TAXA


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an F0Bioacoustic dataset instance (val split)."""
    ds = F0Bioacoustic(split="val")
    return ds


@pytest.fixture
def dataset_with_transforms_from_config() -> tuple[Dataset, dict]:
    """Fixture providing an F0Bioacoustic dataset with transformations from config."""
    dataset_config = DatasetConfig(
        dataset_name="f0_bioacoustic",
        split="val",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "taxon",
                "output_feature": "label",
            },
        ],
    )
    ds, metadata = F0Bioacoustic.from_config(dataset_config)
    return ds, metadata


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing an F0Bioacoustic dataset with output mapping."""
    ds = F0Bioacoustic(
        split="val",
        output_take_and_give={"taxon": "taxon_name"},
    )
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "f0_bioacoustic"
    assert dataset.info.version == "0.1.0"
    assert "all" in dataset.info.split_paths
    assert "train" in dataset.info.split_paths
    assert "val" in dataset.info.split_paths


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    assert dataset._data is not None
    assert "audio_path" in dataset._data.columns
    assert "f0_contour" in dataset._data.columns
    assert "taxon" in dataset._data.columns


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    expected_columns = ["audio_path", "f0_contour", "taxon"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    expected_splits = ["all", "train", "val"]
    assert set(dataset.available_splits) == set(expected_splits)


def test_available_taxa(dataset: Dataset) -> None:
    """Test if available_taxa returns valid taxa."""
    taxa = dataset.available_taxa
    assert len(taxa) > 0
    assert all(t in TAXA for t in taxa)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    expected_len = dataset._data.unwrap.shape[0]
    assert len(dataset) == expected_len
    assert len(dataset) > 0


def test_getitem(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "audio" in sample
    assert "sample_rate" in sample
    assert "f0_contour" in sample
    assert "taxon" in sample

    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1


def test_f0_contour_parsed(dataset: Dataset) -> None:
    """Test that f0_contour is parsed into a DataFrame."""
    sample = dataset[0]
    f0 = sample["f0_contour"]
    assert isinstance(f0, pd.DataFrame)
    assert "time_s" in f0.columns
    assert "freq_hz" in f0.columns


def test_iteration(dataset: Dataset) -> None:
    """Test if iteration works correctly."""
    for i, sample in enumerate(dataset):
        assert isinstance(sample, dict)
        assert "audio" in sample
        assert "sample_rate" in sample
        assert "f0_contour" in sample
        if i >= 2:
            break


def test_invalid_split() -> None:
    """Test if initializing with invalid split raises error."""
    with pytest.raises(LookupError):
        F0Bioacoustic(split="invalid_split")


def test_invalid_taxa() -> None:
    """Test if initializing with invalid taxa raises error."""
    with pytest.raises(ValueError, match="Unknown taxa"):
        F0Bioacoustic(split="val", taxa=["nonexistent_taxon"])


def test_taxa_filter() -> None:
    """Test filtering by taxa subset."""
    ds = F0Bioacoustic(split="val", taxa=["dolphins"])
    assert len(ds) > 0
    assert ds.available_taxa == ["dolphins"]


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))
    assert direct_sample["audio_path"] == iter_sample["audio_path"]


def test_transformations_from_config(
    dataset_with_transforms_from_config: tuple[Dataset, dict],
) -> None:
    """Test if transformations from config are applied correctly."""
    ds, metadata = dataset_with_transforms_from_config
    assert "label" in ds._data.columns
    assert "label_from_feature" in metadata
    assert "label_map" in metadata["label_from_feature"]


def test_output_take_and_give(dataset_with_output_mapping: Dataset) -> None:
    """Test if output_take_and_give correctly maps column names."""
    sample = dataset_with_output_mapping[0]
    assert "taxon_name" in sample
    assert "taxon" not in sample


def test_data_root_handling(dataset: Dataset) -> None:
    """Test if data_root parameter works correctly."""
    assert dataset.data_root is not None


def test_audio_processing(dataset: Dataset) -> None:
    """Test if audio processing works correctly."""
    sample = dataset[0]
    assert "audio" in sample
    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1
    assert len(sample["audio"]) > 0


def test_str_representation(dataset: Dataset) -> None:
    """Test if string representation works correctly."""
    str_repr = str(dataset)
    assert "f0_bioacoustic" in str_repr
    assert "0.1.0" in str_repr


def test_index_error_handling(dataset: Dataset) -> None:
    """Test if index error handling works correctly."""
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_get_available_labels() -> None:
    """Test get_available_labels returns correct labels."""
    ds = F0Bioacoustic(split="val")
    assert ds.get_available_labels("taxon") == TAXA
    assert ds.get_available_labels("species") == SPECIES_LABELS
    with pytest.raises(ValueError):
        ds.get_available_labels("nonexistent")
