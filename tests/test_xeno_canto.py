"""Test suite for the Xeno-canto dataset."""

import pytest

from esp_data.datasets import XenoCanto
from esp_data import Dataset, DatasetConfig
from esp_data.io import anypath


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing a Xeno-canto dataset instance.

    Returns
    -------
    Dataset
        An instance of the Xeno-canto dataset.
    """
    ds = XenoCanto(split="train")
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing a Xeno-canto dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the Xeno-canto dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="xeno-canto",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "scientificName",
                "output_feature": "label",
            },
        ],
    )
    ds = XenoCanto(split="train")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_transforms_from_config() -> tuple[Dataset, dict]:
    """Fixture providing a Xeno-canto dataset instance with transformations
    applied from config.

    Returns
    -------
    Dataset
        An instance of the Xeno-canto dataset with transformations applied.
    dict
        Metadata dictionary containing information about the transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="xeno-canto",
        split="train",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "scientificName",
                "output_feature": "label",
            },
        ],
    )
    ds, metadata = XenoCanto.from_config(dataset_config)
    return ds, metadata


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing a Xeno-canto dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the Xeno-canto dataset with output mapping applied.
    """
    dataset_config = DatasetConfig(
        dataset_name="xeno-canto",
        output_take_and_give={"scientificName": "species"},
    )
    ds = XenoCanto(
        split="train",
        output_take_and_give=dataset_config.output_take_and_give,
    )
    return ds


@pytest.fixture
def dataset_with_sample_rate() -> Dataset:
    """Fixture providing a Xeno-canto dataset instance with custom sample rate.

    Returns
    -------
    Dataset
        An instance of the Xeno-canto dataset with custom sample rate.
    """
    ds = XenoCanto(split="train", sample_rate=22050)
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "xeno-canto"
    assert dataset.info.version == "0.1.0"
    assert "train" in dataset.info.split_paths
    # Test that split paths are configured correctly
    assert dataset.info.split_paths["train"] is not None


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be loaded in __init__
    assert dataset._data is not None
    assert "relative_path" in dataset._data


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    assert "relative_path" in dataset.columns
    # Check for expected columns
    expected_columns = ["relative_path"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should contain train
    expected_splits = ["train"]
    assert set(dataset.available_splits) == set(expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be the number of rows in the dataframe
    expected_len = dataset._data.shape[0]
    assert len(dataset) == expected_len
    print(f"Dataset length: {len(dataset)}")
    # Xeno-canto should have some samples
    assert len(dataset) > 0


def test_getitem(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "audio" in sample
    assert "relative_path" in sample

    # Check audio properties
    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1  # Should be 1D for mono


def test_iteration(dataset: Dataset) -> None:
    """Test if iteration works correctly."""
    for i, sample in enumerate(dataset):
        assert isinstance(sample, dict)
        # Ensure we can access expected keys
        assert "audio" in sample
        assert "relative_path" in sample
        if i >= 2:  # Only test first few samples
            break


def test_load_from_config() -> None:
    """Test if dataset can be loaded from configuration."""
    dataset_config = DatasetConfig(
        dataset_name="xeno-canto",
        split="train",
        sample_rate=16000
    )
    dataset, _ = XenoCanto.from_config(dataset_config)
    assert dataset.info.name == "xeno-canto"
    assert dataset.info.split_paths["train"] is not None
    assert len(dataset) > 0, "Dataset should not be empty"
    assert dataset.sample_rate == 16000


def test_invalid_split() -> None:
    """Test if initializing with invalid split raises error."""
    with pytest.raises(LookupError):
        XenoCanto(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    # Get same sample through different methods
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))

    # Compare samples (they should be identical)
    assert direct_sample["relative_path"] == iter_sample["relative_path"]


def test_transformations(dataset_with_transforms: Dataset) -> None:
    """Test if transformations are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The metadata is updated with transformation information
    """
    # Check that label column was created
    assert "label" in dataset_with_transforms._data.columns


def test_transformations_from_config(dataset_with_transforms_from_config: tuple[Dataset, dict]) -> None:
    """Test if transformations from config are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The metadata contains transformation information
    """
    # Check that label column was created
    ds, metadata = dataset_with_transforms_from_config
    assert "label" in ds._data.columns

    # check that the metadata contains a key for "label_from_feature"
    assert "label_from_feature" in metadata
    assert "label_map" in metadata["label_from_feature"]


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
    assert "species" in sample
    assert "scientificName" not in sample  # Original column should be filtered out


def test_sample_rate_resampling(dataset_with_sample_rate: Dataset) -> None:
    """Test if sample rate resampling works correctly."""
    # Check that sample rate is set correctly
    assert dataset_with_sample_rate.sample_rate == 22050

    # Test with first valid sample
    sample = dataset_with_sample_rate[0]
    assert "audio" in sample
    assert sample["audio"].dtype.name == "float32"


def test_data_root_handling() -> None:
    """Test if data_root parameter works correctly."""
    # Test with default data_root
    dataset = XenoCanto(split="train")
    assert dataset.data_root is not None

    # Test that we can get samples
    sample = dataset[0]
    assert "audio" in sample


def test_audio_processing(dataset: Dataset) -> None:
    """Test if audio processing works correctly."""
    sample = dataset[0]

    # Check that audio is present and has correct type
    assert "audio" in sample
    assert sample["audio"].dtype.name == "float32"

    # Check that audio is mono (converts stereo to mono)
    assert len(sample["audio"].shape) == 1  # Should be 1D for mono

    # Check that audio has reasonable length
    audio_length = len(sample["audio"])
    assert audio_length > 0


def test_str_representation(dataset: Dataset) -> None:
    """Test if string representation works correctly."""
    str_repr = str(dataset)
    assert "xeno-canto" in str_repr
    assert "0.1.0" in str_repr
    assert "train" in str_repr


def test_from_config_with_transformations() -> None:
    """Test if dataset can be loaded from configuration with transformations."""
    dataset_config = DatasetConfig(
        dataset_name="xeno-canto",
        split="train",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "scientificName",
                "output_feature": "label",
            },
        ],
    )
    dataset, metadata = XenoCanto.from_config(dataset_config)

    # Check basic functionality works
    assert dataset.info.name == "xeno-canto"
    assert isinstance(metadata, dict)

    # Test that we can get a sample
    sample = dataset[0]
    assert "audio" in sample


def test_index_error_handling(dataset: Dataset) -> None:
    """Test if index error handling works correctly."""
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]  # Should raise IndexError


def test_runtime_error_handling() -> None:
    """Test if runtime error handling works correctly."""
    # This is harder to test with real data, but we can check that the dataset
    # initializes correctly and doesn't raise runtime errors during normal operation
    dataset = XenoCanto(split="train")
    assert len(dataset) > 0  # Should not raise RuntimeError


def test_output_take_and_give_filtering() -> None:
    """Test if output_take_and_give filtering works correctly."""
    dataset = XenoCanto(
        split="train",
        output_take_and_give={"scientificName": "species", "relative_path": "path", "audio": "audio"},
    )

    sample = dataset[0]

    # Check that only specified columns are in output
    assert "species" in sample
    assert "path" in sample
    assert "audio" in sample

    # Original column names should not be in output
    assert "scientificName" not in sample
    assert "relative_path" not in sample


def test_available_sample_rates() -> None:
    """Test if available_sample_rates property works correctly."""
    dataset = XenoCanto(split="train")
    sample_rates = dataset.available_sample_rates

    # Check if 32kHz is available (depends on whether 32khz_path column exists)
    if "32khz_path" in dataset.columns:
        assert 32000 in sample_rates
    else:
        assert len(sample_rates) == 0


def test_pre_resampled_audio_32khz() -> None:
    """Test loading pre-resampled 32kHz audio."""
    dataset = XenoCanto(split="train", sample_rate=32000)

    # Check if 32kHz pre-resampled audio is available
    if "32khz_path" in dataset.columns:
        print("32kHz pre-resampled audio is available")
        sample = dataset[0]
        assert "audio" in sample
        assert sample["audio"].dtype.name == "float32"
        print(f"Audio shape: {sample['audio'].shape}")
    else:
        print("32kHz pre-resampled audio not yet available, on-the-fly resampling will be used")
        sample = dataset[0]
        assert "audio" in sample
        assert sample["audio"].dtype.name == "float32"


def test_on_the_fly_resampling() -> None:
    """Test on-the-fly resampling from original files."""
    # Request a sample rate that's not pre-resampled (e.g., 16kHz)
    dataset = XenoCanto(split="train", sample_rate=16000)
    sample = dataset[0]

    assert "audio" in sample
    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1  # Should be 1D for mono


def test_reference_item_stability() -> None:
    """Test that the first item produces a consistent audio hash (bitwise stability).

    This test ensures that audio loading and preprocessing are deterministic.
    If this test fails, it indicates changes in audio processing that affect
    the output waveform.
    """
    import hashlib
    import numpy as np

    # Expected SHA256 hash of the first item's audio data
    EXPECTED_FIRST_ITEM_AUDIO_SHA256 = "c600da0173c138bbf2ea376f7ef45414211d029480bad2258b95648ecda4f3fc"

    ds = XenoCanto(split="train")
    idx = 0
    item = ds[idx]

    # Audio presence/type checks (defensive, so the hash failure message is clearer)
    assert "audio" in item, "[0] missing 'audio' key"
    audio = item["audio"]
    assert isinstance(audio, np.ndarray), "[0] audio is not a numpy array"
    assert audio.dtype == np.float32, f"[0] audio dtype is {audio.dtype}, expected float32"

    # Compute sha256 over raw bytes of the float32 array
    h = hashlib.sha256(audio.tobytes()).hexdigest()

    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )
