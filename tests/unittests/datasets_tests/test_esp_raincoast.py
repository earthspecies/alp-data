"""Test suite for the ESPRaincoast dataset."""

import pytest

from alp_data.datasets import ESPRaincoast
from alp_data.datasets.esp_raincoast import ESPRaincoastConfig
from alp_data import Dataset, DatasetConfig, dataset_from_config
from alp_data.transforms import LabelFromFeatureConfig, DeduplicateConfig, FilterConfig
from alp_data.io import anypath


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an ESPRaincoast dataset instance.

    Returns
    -------
    Dataset
        An instance of the ESPRaincoast dataset.
    """
    ds = ESPRaincoast(split="full", load_audio_segments=True, mono_method="average")
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing an ESPRaincoast dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the ESPRaincoast dataset with output mapping applied.
    """
    dataset_config = ESPRaincoastConfig(
        dataset_name="esp_raincoast",
        split="full",
        output_take_and_give={"Sound Type": "call_type"},
        load_audio_segments=True,
    )
    ds, _ = dataset_from_config(dataset_config)
    return ds


@pytest.fixture
def dataset_with_sample_rate() -> Dataset:
    """Fixture providing an ESPRaincoast dataset instance with custom sample rate.

    Returns
    -------
    Dataset
        An instance of the ESPRaincoast dataset with custom sample rate.
    """
    return ESPRaincoast(split="full", sample_rate=16000)


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing an ESPRaincoast dataset instance with transformations.

    Returns
    -------
    Dataset
        An instance of the ESPRaincoast dataset with transformations applied.
    """
    dataset_config = ESPRaincoastConfig(
        dataset_name="esp_raincoast",
        split="full",
        transformations=[
            LabelFromFeatureConfig(type="label_from_feature", feature="Sound Type", label_name="label"),
            DeduplicateConfig(type="deduplicate", subset=None),
        ],
        load_audio_segments=True,
        mono_method="average",
    )
    ds, _ = dataset_from_config(dataset_config)
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "esp_raincoast"
    assert dataset.info.version == "0.1.0"
    assert "full" in dataset.info.split_paths


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be loaded in __init__
    assert dataset._data is not None
    assert "local_path" in dataset._data.columns
    assert "Begin Time (s)" in dataset._data.columns
    assert "End Time (s)" in dataset._data.columns


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    expected_columns = ["local_path", "Low Freq (Hz)", "High Freq (Hz)",
                        "Begin Time (s)", "End Time (s)", "Sound Type"]
    assert all(col in dataset.columns for col in expected_columns)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = len(dataset._data)
    assert len(dataset) == expected_len
    print(f"Dataset length: {len(dataset)}")
    # ESPRaincoast should have thousands of samples
    assert len(dataset) > 1000


def test_getitem(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "audio" in sample
    assert "local_path" in sample
    assert "Sound Type" in sample

    # Check audio properties
    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1  # Should be 1D for mono

    # Check time segment properties
    assert isinstance(sample["Begin Time (s)"], float)
    assert isinstance(sample["End Time (s)"], float)
    assert sample["End Time (s)"] > sample["Begin Time (s)"]


def test_iteration(dataset: Dataset) -> None:
    """Test if iteration works correctly."""
    for i, sample in enumerate(dataset):
        assert isinstance(sample, dict)
        # Ensure we can access expected keys
        assert "audio" in sample
        assert "local_path" in sample
        assert "Low Freq (Hz)" in sample
        assert "High Freq (Hz)" in sample
        if i >= 2:  # Only test first few samples
            break


def test_invalid_split() -> None:
    """Test if initializing with invalid split raises error."""
    with pytest.raises(LookupError):
        ESPRaincoast(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    # Get same sample through different methods
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))

    # Compare samples (they should be identical)
    assert direct_sample["local_path"] == iter_sample["local_path"]
    assert direct_sample["Begin Time (s)"] == iter_sample["Begin Time (s)"]


def test_transformations(dataset_with_transforms: Dataset) -> None:
    """Test basic dataset functionality (transformations with list labels not supported)."""
    backend = dataset_with_transforms._data.drop_duplicates()
    assert backend._df.shape == dataset_with_transforms._data._df.shape
    # Check that labels are correctly parsed as lists
    sample = dataset_with_transforms[0]
    assert "label" in sample
    assert isinstance(sample["label"], int)


def test_output_mapping(dataset_with_output_mapping: Dataset) -> None:
    """Test if output mapping works correctly."""
    # Check that output mapping was applied
    sample = dataset_with_output_mapping[0]
    assert "call_type" in sample
    assert "Sound Type" not in sample  # Original column should not be present


def test_str_representation(dataset: Dataset) -> None:
    """Test if string representation works correctly."""
    str_repr = str(dataset)
    assert "esp_raincoast" in str_repr
    assert "0.1.0" in str_repr
    assert "full" in str_repr


def test_index_error_handling(dataset: Dataset) -> None:
    """Test if index error handling works correctly."""
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]  # Should raise IndexError


def test_audio_segment_extraction(dataset: Dataset) -> None:
    """Test if audio segment extraction works correctly."""
    sample = dataset[0]

    # Check that start and end times are used
    assert "Begin Time (s)" in sample
    assert "End Time (s)" in sample
    assert sample["End Time (s)"] > sample["Begin Time (s)"]

    # Check that audio length roughly matches the segment duration
    audio_length = len(sample["audio"])
    segment_duration = sample["End Time (s)"] - sample["Begin Time (s)"]
    # Assuming 16kHz sample rate, allow some tolerance
    expected_length = int(segment_duration * sample["sample_rate"])
    assert abs(audio_length - expected_length) < 2000  # Allow 2000 samples tolerance
