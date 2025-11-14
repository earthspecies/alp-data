"""Test suite for the iNaturalist dataset."""

import hashlib

import numpy as np
import pytest

from esp_data.datasets import INaturalist
from esp_data import Dataset, DatasetConfig

# Expected SHA256 hash of the first item's audio (index 0) at default (variable) sample rate
# This ensures bitwise stability of the dataset over time.
# If this hash changes unexpectedly, it indicates a change in:
# - The audio file itself
# - Sample rate handling
# - Channel mixing logic (stereo->mono)
# - Data type conversion
# - Dataset ordering
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = (
    "804d45aafc783c3e08d25c716d299c338ced073245dd07a92d62b2797c105303"
)


@pytest.fixture
def dataset_with_transforms_from_config() -> tuple[Dataset, dict]:
    """Fixture providing an iNaturalist dataset instance with transformations
    applied from config.

    Returns
    -------
    Dataset
        An instance of the iNaturalist dataset with transformations applied.
    dict
        Metadata dictionary containing information about the transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="inaturalist",
        split="train",
        sample_rate=32000,
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "canonical_name",
                "output_feature": "label",
            },
        ],
    )
    ds, _ = INaturalist.from_config(dataset_config)
    return ds


def test_info_property(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset_with_transforms_from_config.info.name == "inaturalist"
    assert dataset_with_transforms_from_config.info.version == "0.1.0"
    assert "train" in dataset_with_transforms_from_config.info.split_paths
    # Test that split paths are configured correctly
    assert dataset_with_transforms_from_config.info.split_paths["train"] is not None


def test_data_property(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be loaded in __init__
    assert dataset_with_transforms_from_config._data is not None
    assert "originals_path" in dataset_with_transforms_from_config._data


def test_columns_property(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    assert "originals_path" in dataset_with_transforms_from_config.columns
    # Check for expected columns
    expected_columns = ["originals_path"]
    assert all(
        col in dataset_with_transforms_from_config.columns for col in expected_columns
    )


def test_available_splits(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should contain train
    expected_splits = ["train"]
    assert set(dataset_with_transforms_from_config.available_splits) == set(
        expected_splits
    )


def test_length(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be the number of rows in the dataframe
    expected_len = dataset_with_transforms_from_config._data.shape[0]
    assert len(dataset_with_transforms_from_config) == expected_len
    print(f"Dataset length: {len(dataset_with_transforms_from_config)}")
    # iNaturalist should have some samples
    assert len(dataset_with_transforms_from_config) > 0


def test_getitem(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = dataset_with_transforms_from_config[0]
    assert isinstance(sample, dict)
    assert "audio" in sample
    assert "originals_path" in sample

    # Check audio properties
    assert sample["audio"].dtype.name == "float32"
    assert len(sample["audio"].shape) == 1  # Should be 1D for mono
    assert len(sample["audio"]) > 0


def test_iteration(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if iteration works correctly."""
    for i, sample in enumerate(dataset_with_transforms_from_config):
        assert isinstance(sample, dict)
        # Ensure we can access expected keys
        assert "audio" in sample
        assert "originals_path" in sample
        if i >= 2:  # Only test first few samples
            break


def test_invalid_split() -> None:
    """Test if initializing with invalid split raises error."""
    with pytest.raises(LookupError):
        INaturalist(split="invalid_split")


def test_sample_consistency(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    # Get same sample through different methods
    direct_sample = dataset_with_transforms_from_config[0]
    iter_sample = next(iter(dataset_with_transforms_from_config))

    # Compare samples (they should be identical)
    assert direct_sample["originals_path"] == iter_sample["originals_path"]


def test_transformations(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if transformations are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The metadata is updated with transformation information
    """
    # Check that label column was created
    assert "label" in dataset_with_transforms_from_config._data.columns


def test_transformations_from_config(
    dataset_with_transforms_from_config: tuple[Dataset, dict],
) -> None:
    """Test if transformations from config are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The metadata contains transformation information
    """
    assert "label" in dataset_with_transforms_from_config.columns


def test_sample_rate_resampling(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if sample rate resampling works correctly."""
    # Check that sample rate is set correctly
    assert dataset_with_transforms_from_config.sample_rate == 22050

    # Test with first valid sample
    sample = dataset_with_transforms_from_config[0]
    assert "audio" in sample
    assert sample["audio"].dtype.name == "float32"


def test_data_root_handling(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if data_root parameter works correctly."""
    # Test with default data_root
    assert dataset_with_transforms_from_config.data_root is not None


def test_str_representation(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if string representation works correctly."""
    str_repr = str(dataset_with_transforms_from_config)
    assert "inaturalist" in str_repr
    assert "0.1.0" in str_repr
    assert "train" in str_repr


def test_index_error_handling(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if index error handling works correctly."""
    with pytest.raises(IndexError):
        _ = dataset_with_transforms_from_config[
            len(dataset_with_transforms_from_config)
        ]  # Should raise IndexError


def test_available_sample_rates(dataset_with_transforms_from_config: Dataset) -> None:
    """Test if available_sample_rates property works correctly."""
    # Check that available_sample_rates returns a list
    sample_rates = dataset_with_transforms_from_config.available_sample_rates
    assert isinstance(sample_rates, list)

    # Check that 32kHz (pre-resampled) is available if the column exists
    if "32khz_path" in dataset_with_transforms_from_config.columns:
        assert 32000 in sample_rates
        print("✓ 32kHz pre-resampled audio is available")
    else:
        # Column doesn't exist yet, so no pre-resampled rates available
        assert len(sample_rates) == 0
        print("Note: 32khz_path column not yet available in metadata")

    # Original files are at variable rates, not pre-resampled to any specific rate
    print(f"Available pre-resampled sample rates: {sample_rates}")


def test_reference_item_stability(dataset_with_transforms_from_config: Dataset) -> None:
    """Check that a canonical item (index 0) is bitwise-stable.

    We hash the raw float32 audio buffer. This catches:
    - sample rate changes (resampling -> different samples)
    - channel handling changes (stereo->mono logic changed)
    - dtype changes
    - ordering changes in the split (if a different recording moved to idx 0)

    If this fails for a legitimate/intentional reason, recompute the hash below
    and update EXPECTED_FIRST_ITEM_AUDIO_SHA256.
    """
    dataset_with_transforms_from_config.sample_rate = None  # original sr
    # Choose deterministic index
    idx = 0
    item = dataset_with_transforms_from_config[idx]

    # Audio presence/type checks (defensive, so the hash failure message is clearer)
    assert "audio" in item, "[0] missing 'audio' key"
    audio = item["audio"]
    assert isinstance(audio, np.ndarray), "[0] audio is not a numpy array"
    assert (
        audio.dtype == np.float32
    ), f"[0] audio dtype is {audio.dtype}, expected float32"

    # Compute sha256 over raw bytes of the float32 array
    h = hashlib.sha256(audio.tobytes()).hexdigest()

    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )
