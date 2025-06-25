"""Test suite for the AudioSet dataset."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from esp_data.datasets import AudioSet
from esp_data import Dataset, DatasetConfig

@pytest.fixture
def mock_csv_data():
    """Mock CSV data for AudioSet."""
    return pd.DataFrame({
        'local_path': ['test_audio1.wav', 'test_audio2.wav'],
        'start': [0.0, 1.0],
        'end': [10.0, 11.0],
        'label': ['test_label1', 'test_label2']
    })


@pytest.fixture
def dataset(mock_csv_data) -> Dataset:
    """Fixture providing an AudioSet dataset instance.

    Returns
    -------
    Dataset
        An instance of the AudioSet dataset.
    """
    with patch('esp_data.datasets.audioset.anypath') as mock_anypath, \
         patch('esp_data.datasets.audioset.pd.read_csv') as mock_read_csv:

        # Mock the CSV reading
        mock_read_csv.return_value = mock_csv_data

        # Mock the path existence check
        mock_anypath.return_value.exists.return_value = True
        mock_anypath.return_value.read_text.return_value = mock_csv_data.to_csv(index=False)

        ds = AudioSet(split="train", data_root="gs://esp-ml-datasets/audioset/v0.1.0/raw/")
        return ds


@pytest.fixture
def dataset_with_transforms(mock_csv_data) -> Dataset:
    """Fixture providing an AudioSet dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the AudioSet dataset with transformations applied.
    """
    with patch('esp_data.datasets.audioset.anypath') as mock_anypath, \
         patch('esp_data.datasets.audioset.pd.read_csv') as mock_read_csv:

        # Mock the CSV reading
        mock_read_csv.return_value = mock_csv_data

        # Mock the path existence check
        mock_anypath.return_value.exists.return_value = True
        mock_anypath.return_value.read_text.return_value = mock_csv_data.to_csv(index=False)

        dataset_config = DatasetConfig(
            dataset_name="AudioSet",
            transformations=[
                {
                    "type": "label_from_feature",
                    "feature": "label",
                    "output_feature": "Label",
                },
            ],
        )
        ds = AudioSet(split="train", data_root="gs://esp-ml-datasets/audioset/v0.1.0/raw/")
        ds.apply_transformations(dataset_config.transformations)
        return ds


@pytest.fixture
def dataset_with_output_mapping(mock_csv_data) -> Dataset:
    """Fixture providing an AudioSet dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the AudioSet dataset with output mapping applied.
    """
    with patch('esp_data.datasets.audioset.anypath') as mock_anypath, \
         patch('esp_data.datasets.audioset.pd.read_csv') as mock_read_csv:

        # Mock the CSV reading
        mock_read_csv.return_value = mock_csv_data

        # Mock the path existence check
        mock_anypath.return_value.exists.return_value = True
        mock_anypath.return_value.read_text.return_value = mock_csv_data.to_csv(index=False)

        dataset_config = DatasetConfig(
            dataset_name="AudioSet",
            output_take_and_give={"label": "audio_label"},
        )
        ds = AudioSet(
            split="validation",
            output_take_and_give=dataset_config.output_take_and_give,
            data_root="gs://esp-ml-datasets/audioset/v0.1.0/raw/"
        )
        return ds


@pytest.fixture
def dataset_with_sample_rate(mock_csv_data) -> Dataset:
    """Fixture providing an AudioSet dataset instance with custom sample rate.

    Returns
    -------
    Dataset
        An instance of the AudioSet dataset with custom sample rate.
    """
    with patch('esp_data.datasets.audioset.anypath') as mock_anypath, \
         patch('esp_data.datasets.audioset.pd.read_csv') as mock_read_csv:

        # Mock the CSV reading
        mock_read_csv.return_value = mock_csv_data

        # Mock the path existence check
        mock_anypath.return_value.exists.return_value = True
        mock_anypath.return_value.read_text.return_value = mock_csv_data.to_csv(index=False)

        ds = AudioSet(split="train-balanced", sample_rate=22050, data_root="gs://esp-ml-datasets/audioset/v0.1.0/raw/")
        return ds


def test_info_property() -> None:
    """Test if the info property returns correct metadata."""
    # Test without creating dataset instance to avoid file loading
    assert AudioSet.info.name == "audioset"
    assert AudioSet.info.version == "0.1.0"
    assert "train" in AudioSet.info.split_paths
    assert "validation" in AudioSet.info.split_paths
    assert "train-balanced" in AudioSet.info.split_paths
    assert "train-animal" in AudioSet.info.split_paths
    assert "validation-animal" in AudioSet.info.split_paths
    assert "train-noise" in AudioSet.info.split_paths
    assert "validation-noise" in AudioSet.info.split_paths


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be loaded in __init__
    assert dataset._data is not None
    assert "local_path" in dataset._data
    assert "start" in dataset._data
    assert "end" in dataset._data


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    expected_columns = ["local_path", "start", "end", "label"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should contain these
    expected_splits = [
        "train", "train-balanced", "validation",
        "train-animal", "validation-animal",
        "train-noise", "validation-noise"
    ]
    assert all(split in dataset.available_splits for split in expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = dataset._data.shape[0]
    assert len(dataset) == expected_len
    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 2  # Based on our mock data


def test_getitem_with_mock_audio(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format with mocked audio."""
    with patch('esp_data.datasets.audioset.read_audio') as mock_read_audio, \
         patch('esp_data.datasets.audioset.audio_stereo_to_mono') as mock_stereo_to_mono:

        # Mock audio reading
        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_read_audio.return_value = (mock_audio, 16000)
        mock_stereo_to_mono.return_value = mock_audio

        # Get first sample
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "audio" in sample
        assert "local_path" in sample
        assert "start" in sample
        assert "end" in sample
        assert sample["audio"].dtype == "float32"


def test_iteration_with_mock_audio(dataset: Dataset) -> None:
    """Test if iteration works correctly with mocked audio."""
    with patch('esp_data.datasets.audioset.read_audio') as mock_read_audio, \
         patch('esp_data.datasets.audioset.audio_stereo_to_mono') as mock_stereo_to_mono:

        # Mock audio reading
        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_read_audio.return_value = (mock_audio, 16000)
        mock_stereo_to_mono.return_value = mock_audio

        for _, sample in enumerate(dataset):
            assert isinstance(sample, dict)
            # Ensure we can access a known key
            assert "audio" in sample
            break


def test_load_from_config(mock_csv_data) -> None:
    """Test if dataset can be loaded from configuration."""
    with patch('esp_data.datasets.audioset.anypath') as mock_anypath, \
         patch('esp_data.datasets.audioset.pd.read_csv') as mock_read_csv:

        # Mock the CSV reading
        mock_read_csv.return_value = mock_csv_data

        # Mock the path existence check
        mock_anypath.return_value.exists.return_value = True
        mock_anypath.return_value.read_text.return_value = mock_csv_data.to_csv(index=False)

        dataset_config = DatasetConfig(
            dataset_name="AudioSet",
            split="validation",
            data_root="gs://esp-ml-datasets/audioset/v0.1.0/raw/",
            sample_rate=16000
        )
        dataset, _ = AudioSet.from_config(dataset_config)
        assert dataset.info.name == "AudioSet"
        assert dataset.info.split_paths["validation"] is not None
        assert len(dataset) > 0, "Dataset should not be empty"
        assert dataset.sample_rate == 16000


def test_invalid_split() -> None:
    """Test if loading invalid split raises error."""
    with pytest.raises(LookupError):
        AudioSet(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    with patch('esp_data.datasets.audioset.read_audio') as mock_read_audio, \
         patch('esp_data.datasets.audioset.audio_stereo_to_mono') as mock_stereo_to_mono:

        # Mock audio reading
        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_read_audio.return_value = (mock_audio, 16000)
        mock_stereo_to_mono.return_value = mock_audio

        # Get same sample through different methods
        direct_sample = dataset[0]
        iter_sample = next(iter(dataset))

        # Compare samples
        assert direct_sample["local_path"] == iter_sample["local_path"]


def test_transformations(dataset_with_transforms: Dataset) -> None:
    """Test if transformations are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column

    """
    # Check that label column was created
    assert "Label" in dataset_with_transforms._data.columns


def test_output_mapping(dataset_with_output_mapping: Dataset) -> None:
    """Test if output mapping works correctly."""
    with patch('esp_data.datasets.audioset.read_audio') as mock_read_audio, \
         patch('esp_data.datasets.audioset.audio_stereo_to_mono') as mock_stereo_to_mono:

        # Mock audio reading
        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_read_audio.return_value = (mock_audio, 16000)
        mock_stereo_to_mono.return_value = mock_audio

        # Get a sample and check the mapping
        sample = dataset_with_output_mapping[0]
        assert "audio_label" in sample
        assert "label" not in sample  # Original key should not be present


def test_sample_rate_resampling(dataset_with_sample_rate: Dataset) -> None:
    """Test if sample rate resampling works correctly."""
    with patch('esp_data.datasets.audioset.read_audio') as mock_read_audio, \
         patch('esp_data.datasets.audioset.audio_stereo_to_mono') as mock_stereo_to_mono, \
         patch('esp_data.datasets.audioset.librosa.resample') as mock_resample:

        # Mock audio reading
        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_read_audio.return_value = (mock_audio, 16000)
        mock_stereo_to_mono.return_value = mock_audio
        mock_resample.return_value = mock_audio

        # Get a sample and check the audio sample rate
        sample = dataset_with_sample_rate[0]
        assert "audio" in sample
        # The audio should be resampled to the specified sample rate
        # Note: We can't easily test the actual sample rate without loading the audio
        # but we can verify the sample_rate attribute is set correctly
        assert dataset_with_sample_rate.sample_rate == 22050


def test_data_root_handling(mock_csv_data) -> None:
    """Test if data_root is handled correctly."""
    with patch('esp_data.datasets.audioset.anypath') as mock_anypath, \
         patch('esp_data.datasets.audioset.pd.read_csv') as mock_read_csv:

        # Mock the CSV reading
        mock_read_csv.return_value = mock_csv_data

        # Mock the path existence check
        mock_anypath.return_value.exists.return_value = True
        mock_anypath.return_value.read_text.return_value = mock_csv_data.to_csv(index=False)
        mock_anypath.return_value.parent = "/mock/parent"

        # Test with explicit data_root
        ds_with_root = AudioSet(split="train", data_root="/custom/path")
        assert ds_with_root.data_root == "/custom/path"

        # Test without data_root (should use parent of split path)
        ds_without_root = AudioSet(split="train")
        assert ds_without_root.data_root is not None


def test_audio_processing(dataset: Dataset) -> None:
    """Test if audio processing works correctly."""
    with patch('esp_data.datasets.audioset.read_audio') as mock_read_audio, \
         patch('esp_data.datasets.audioset.audio_stereo_to_mono') as mock_stereo_to_mono:

        # Mock audio reading
        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_read_audio.return_value = (mock_audio, 16000)
        mock_stereo_to_mono.return_value = mock_audio

        sample = dataset[0]

        # Check that audio is present and has correct type
        assert "audio" in sample
        assert sample["audio"].dtype == "float32"

        # Check that audio is mono (AudioSet converts stereo to mono)
        assert len(sample["audio"].shape) == 1  # Should be 1D for mono


def test_str_representation(dataset: Dataset) -> None:
    """Test if string representation is correct."""
    str_repr = str(dataset)
    assert "AudioSet" in str_repr
    assert "v0.1.0" in str_repr
    assert "AudioSet dataset" in str_repr
    assert "YouTube" in str_repr
    assert "Mixed" in str_repr


def test_from_config_with_transformations(mock_csv_data) -> None:
    """Test if from_config works with transformations."""
    with patch('esp_data.datasets.audioset.anypath') as mock_anypath, \
         patch('esp_data.datasets.audioset.pd.read_csv') as mock_read_csv:

        # Mock the CSV reading
        mock_read_csv.return_value = mock_csv_data

        # Mock the path existence check
        mock_anypath.return_value.exists.return_value = True
        mock_anypath.return_value.read_text.return_value = mock_csv_data.to_csv(index=False)

        dataset_config = DatasetConfig(
            dataset_name="AudioSet",
            split="train",
            data_root="gs://esp-ml-datasets/audioset/v0.1.0/raw/",
            transformations=[
                {
                    "type": "label_from_feature",
                    "feature": "label",
                    "output_feature": "Label",
                },
            ],
        )
        dataset, metadata = AudioSet.from_config(dataset_config)
        assert "Label" in dataset._data.columns
        assert isinstance(metadata, dict)


def test_index_error_handling(dataset: Dataset) -> None:
    """Test if index errors are handled correctly."""
    with pytest.raises(IndexError):
        # Try to access an index that doesn't exist
        dataset[len(dataset) + 1]


def test_runtime_error_handling() -> None:
    """Test if runtime errors are handled correctly."""
    # Create a dataset instance without calling _load
    ds = AudioSet.__new__(AudioSet)
    ds._data = None

    with pytest.raises(RuntimeError):
        len(ds)


def test_different_splits(mock_csv_data) -> None:
    """Test if different splits can be loaded."""
    with patch('esp_data.datasets.audioset.anypath') as mock_anypath, \
         patch('esp_data.datasets.audioset.pd.read_csv') as mock_read_csv:

        # Mock the CSV reading
        mock_read_csv.return_value = mock_csv_data

        # Mock the path existence check
        mock_anypath.return_value.exists.return_value = True
        mock_anypath.return_value.read_text.return_value = mock_csv_data.to_csv(index=False)

        splits = ["train", "train-balanced", "validation", "train-animal", "validation-animal"]

        for split in splits:
            ds = AudioSet(split=split, data_root="gs://esp-ml-datasets/audioset/v0.1.0/raw/")
            assert ds.split == split
            assert len(ds) > 0
            assert ds._data is not None


def test_audio_segment_extraction(dataset: Dataset) -> None:
    """Test if audio segments are extracted correctly."""
    with patch('esp_data.datasets.audioset.read_audio') as mock_read_audio, \
         patch('esp_data.datasets.audioset.audio_stereo_to_mono') as mock_stereo_to_mono:

        # Mock audio reading
        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_read_audio.return_value = (mock_audio, 16000)
        mock_stereo_to_mono.return_value = mock_audio

        sample = dataset[0]

        # Check that start and end times are present
        assert "start" in sample
        assert "end" in sample

        # Check that start time is less than end time
        assert sample["start"] < sample["end"]


def test_output_take_and_give_filtering(mock_csv_data) -> None:
    """Test if output_take_and_give acts as a filter."""
    with patch('esp_data.datasets.audioset.anypath') as mock_anypath, \
         patch('esp_data.datasets.audioset.pd.read_csv') as mock_read_csv, \
         patch('esp_data.datasets.audioset.read_audio') as mock_read_audio, \
         patch('esp_data.datasets.audioset.audio_stereo_to_mono') as mock_stereo_to_mono:

        # Mock the CSV reading
        mock_read_csv.return_value = mock_csv_data

        # Mock the path existence check
        mock_anypath.return_value.exists.return_value = True
        mock_anypath.return_value.read_text.return_value = mock_csv_data.to_csv(index=False)

        # Mock audio reading
        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_read_audio.return_value = (mock_audio, 16000)
        mock_stereo_to_mono.return_value = mock_audio

        # Create dataset with specific output mapping
        output_mapping = {"local_path": "path", "start": "start_time"}
        ds = AudioSet(
            split="train",
            output_take_and_give=output_mapping,
            data_root="gs://esp-ml-datasets/audioset/v0.1.0/raw/"
        )

        sample = ds[0]

        # Only mapped keys should be present
        assert "path" in sample
        assert "start_time" in sample
        assert "local_path" not in sample
        assert "start" not in sample
        assert "end" not in sample
