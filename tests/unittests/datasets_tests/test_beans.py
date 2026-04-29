"""Test suite for the Beans dataset."""

import pytest
import numpy as np

from esp_data.datasets import Beans
from esp_data import Dataset, DatasetConfig
from esp_data.io import exists
from esp_data.utils import create_hash

EXPECTED_FIRST_VAL_ITEM_AUDIO_SHA256 = "595ba74365124bc8872c828c46c2449572b91bd4c3e84288322e5f45e77dd340"
EXPECTED_VAL_ANNOTATIONS_SHA256 = "f843271c815857d318a2c5ba01c625cc95d1506b9c2c95718fb2b6620c8201a7"


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an Beans dataset instance.

    Returns
    -------
    Dataset
        An instance of the Beans dataset.
    """
    ds = Beans(split="validation", streaming=False, backend="pandas")
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing an Beans dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the Beans dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="beans",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "label",
                "output_feature": "Label",
            },
            {
                "type": "filter",
                "mode": "exclude",
                "property": "source_dataset",
                "values": ["speech_commands", "esc50"],
            },
        ],
    )
    ds = Beans(split="validation")
    ds.apply_transformations(dataset_config.transformations)
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "beans"
    assert dataset.info.version == "0.1.0"
    assert "train" in dataset.info.split_paths
    assert "validation" in dataset.info.split_paths
    for split in dataset.info.split_paths.values():
        assert exists(split), f"Split path {split} does not exist"


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be _loaded in __init__
    assert dataset._data is not None
    assert "local_path" in dataset._data.columns
    assert "label" in dataset._data.columns


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    expected_columns = ["local_path", "label", "source_dataset", "file_name"]
    assert all(col in dataset.columns for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should contain these
    expected_splits = ["train", "validation", "test", "cbi_test", "esc50_validation"]
    assert all(split in dataset.available_splits for split in expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = len(dataset._data)
    assert len(dataset) == expected_len
    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 62415  # Example expected length, adjust as necessary


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
        dataset_name="beans",
        split="test",
        streaming=False,
    )
    dataset, _ = Beans.from_config(dataset_config)
    assert dataset.info.name == "beans"
    assert dataset.info.split_paths["train"] is not None
    assert len(dataset) > 0, "Dataset should not be empty"


def test_invalid_split() -> None:
    """Test if _loading invalid split raises error."""
    with pytest.raises(LookupError):
        Beans(split="invalid_split")


def test_transformations(dataset_with_transforms: Dataset) -> None:
    """Test if transformations are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The filter transformation excludes specified genera

    """
    # Check that label column was created
    assert "Label" in dataset_with_transforms._data.columns

    # Check that the excluded genus is not present
    excluded_genus = "esc50"
    source_datasets = [row["source_dataset"] for row in dataset_with_transforms._data]
    assert excluded_genus not in source_datasets, (
        f"Genus '{excluded_genus}' should be excluded from the dataset."
    )


def test_reference_item_stability(dataset: Beans):
    """
    Check that a canonical item (index 0) is bitwise-stable.

    We hash the raw float32 audio buffer. This catches:
    - sample rate changes (resampling -> different samples)
    - channel handling changes (stereo->mono logic changed)
    - dtype changes
    - ordering changes in the split (if a different recording moved to idx 0)

    If this fails for a legitimate/intentional reason, recompute the hash below
    and update EXPECTED_FIRST_VAL_ITEM_AUDIO_SHA256.

    We do the same for the annotations csv.
    """
    # choose deterministic index
    idx = 0
    item = dataset[idx]

    # audio presence/type checks (defensive, so the hash failure message is clearer)
    assert "audio" in item, "[0] missing 'audio' key"
    audio = item["audio"]
    assert isinstance(audio, np.ndarray), "[0] audio is not a numpy array"
    assert (
        audio.dtype == np.float32
    ), f"[0] audio dtype is {audio.dtype}, expected float32"

    # compute sha256 over raw bytes of the float32 array
    h = create_hash(audio.tobytes())

    assert h == EXPECTED_FIRST_VAL_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_FIRST_VAL_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_VAL_ITEM_AUDIO_SHA256 with the new hash."
    )

    # compute sha256 over raw bytes of the float32 array of annotations
    csv_bytes = (
        dataset._data.unwrap.sort_index(axis=0)
        .sort_index(axis=1)
        .to_csv(index=True)
        .encode("utf-8")
    )
    h = create_hash(csv_bytes)

    assert h == EXPECTED_VAL_ANNOTATIONS_SHA256, (
        "Annotation's hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_VAL_ANNOTATIONS_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_VAL_ANNOTATIONS_SHA256 with the new hash."
    )


if __name__ == "__main__":
    # Generate hash
    ds = Beans(split="validation", streaming=False, backend="pandas")

    audio0 = ds[0]["audio"]
    print("dtype:", audio0.dtype, "shape:", audio0.shape)

    h = create_hash(audio0.tobytes())
    print("audio sha256:", h)

    csv_bytes = (
            ds._data.unwrap.sort_index(axis=0)
            .sort_index(axis=1)
            .to_csv(index=True)
            .encode("utf-8")
        )
    h = create_hash(csv_bytes)

    print("annotations sha256:", h)
