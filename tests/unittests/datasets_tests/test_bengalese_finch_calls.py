"""Test suite for the Bengalese Finch Calls dataset."""

import pytest

from esp_data.datasets import BengaleseFinchCalls
from esp_data import Dataset, DatasetConfig
from esp_data.io import exists


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing the default dataset instance."""
    return BengaleseFinchCalls()  # Uses default Bird2_train split


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Dataset instance with a simple transformation applied."""
    cfg = DatasetConfig(
        dataset_name="bengalese_finch_calls",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "call_type",
                "output_feature": "label",
            }
        ],
    )
    ds = BengaleseFinchCalls()  # Uses default Bird2_train split
    ds.apply_transformations(cfg.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Dataset instance where column names are mapped on output."""
    mapping = {"call_type": "call_label", "individual_id": "bird_id", "local_path": "audio_path"}
    return BengaleseFinchCalls(output_take_and_give=mapping)  # Uses default Bird2_train split


# -----------------------------------------------------------------------------
# Basic integrity checks
# -----------------------------------------------------------------------------

def test_info_property(dataset: Dataset) -> None:
    assert dataset.info.name == "Bengalese Finch Calls"
    assert dataset.info.version == "0.1.0"
    # Check that comprehensive splits are available
    assert "Bird2_train" in dataset.info.split_paths  # Default split
    assert "Bird0" in dataset.info.split_paths  # Original birds
    assert "Bird1_train_small" in dataset.info.split_paths  # ML splits
    # Ensure that the default split path exists (locally or remotely)
    split_path = dataset.info.split_paths["Bird2_train"]
    assert exists(split_path), f"Default split path {split_path} does not exist"


def test_data_property(dataset: Dataset) -> None:
    assert dataset._data is not None
    required_columns = [
        "local_path",
        "call_type",
        "individual_id",
    ]
    for col in required_columns:
        assert col in dataset._data.columns


def test_columns_property(dataset: Dataset) -> None:
    assert set(dataset.columns).issuperset({"local_path", "call_type", "individual_id"})


def test_available_splits(dataset: Dataset) -> None:
    available = set(dataset.available_splits)

    # Should have 55 total splits: 11 original + 44 ML splits
    assert len(available) == 55

    # Check original bird splits (11 total)
    original_birds = [f"Bird{i}" for i in range(11)]
    for bird in original_birds:
        assert bird in available

    # Check ML splits for each bird (4 splits × 11 birds = 44 total)
    ml_split_types = ["train", "train_small", "valid", "test"]
    for i in range(11):
        for split_type in ml_split_types:
            split_name = f"Bird{i}_{split_type}"
            assert split_name in available, f"Missing ML split: {split_name}"


def test_length(dataset: Dataset) -> None:
    expected_len = len(dataset._data)
    assert len(dataset) == expected_len
    # Bird2_train should have 18,303 samples based on processing output
    assert len(dataset) == 18303


def test_getitem(dataset: Dataset) -> None:
    sample = dataset[0]
    assert "audio" in sample
    assert "sample_rate" in sample
    assert "call_type" in sample
    assert "individual_id" in sample
    # Audio should be mono
    assert sample["audio"].ndim == 1
    # Individual ID should be Bird2 for the default Bird2_train split
    assert sample["individual_id"] == "Bird2"


def test_iteration(dataset: Dataset) -> None:
    for sample in dataset:
        assert "audio" in sample
        break  # only need to check first sample


def test_load_from_config() -> None:
    cfg = DatasetConfig(dataset_name="bengalese_finch_calls", split="Bird2_train", sample_rate=16000)
    ds, _ = BengaleseFinchCalls.from_config(cfg)
    assert len(ds) == 18303  # Expected Bird2_train size
    assert ds.sample_rate == 16000


def test_invalid_split() -> None:
    with pytest.raises(LookupError):
        BengaleseFinchCalls(split="invalid")


def test_multiple_splits() -> None:
    """Test that we can load different bird splits including ML splits."""
    # Test original bird splits
    ds0 = BengaleseFinchCalls(split="Bird0")
    assert len(ds0) == 7652  # Original Bird0 size
    assert ds0[0]["individual_id"] == "Bird0"

    # Test ML splits for Bird2 (highest diversity)
    ds2_train = BengaleseFinchCalls(split="Bird2_train")
    ds2_train_small = BengaleseFinchCalls(split="Bird2_train_small")
    ds2_valid = BengaleseFinchCalls(split="Bird2_valid")
    ds2_test = BengaleseFinchCalls(split="Bird2_test")

    # Check expected sizes from our processing
    assert len(ds2_train) == 18303
    assert len(ds2_train_small) == 1360  # 80 samples × 17 call types
    assert len(ds2_valid) == 3912
    assert len(ds2_test) == 3912

    # All should be Bird2
    assert ds2_train[0]["individual_id"] == "Bird2"
    assert ds2_train_small[0]["individual_id"] == "Bird2"
    assert ds2_valid[0]["individual_id"] == "Bird2"
    assert ds2_test[0]["individual_id"] == "Bird2"

    # Test another bird's ML splits
    ds1_train = BengaleseFinchCalls(split="Bird1_train")
    assert len(ds1_train) == 25024  # Expected Bird1_train size
    assert ds1_train[0]["individual_id"] == "Bird1"


def test_sample_consistency(dataset: Dataset) -> None:
    direct = dataset[0]
    via_iter = next(iter(dataset))
    # Compare identifiers (ignore audio for speed)
    assert direct["local_path"] == via_iter["local_path"]
    assert direct["call_type"] == via_iter["call_type"]


def test_transformations(dataset_with_transforms: Dataset) -> None:
    assert "label" in dataset_with_transforms._data.columns


def test_output_take_and_give(dataset_with_output_mapping: Dataset) -> None:
    sample = dataset_with_output_mapping[0]
    expected_keys = {"call_label", "bird_id", "audio_path", "audio"}
    assert set(sample.keys()) == expected_keys
    original = dataset_with_output_mapping._data[0]
    assert sample["call_label"] == original["call_type"]
    assert sample["bird_id"] == original["individual_id"]
    assert sample["audio_path"] == original["local_path"]


def test_string_representation(dataset: Dataset) -> None:
    s = str(dataset)
    assert "Bengalese Finch Calls" in s


def test_class_registration() -> None:
    from esp_data.datasets import BengaleseFinchCalls as _Cls  # noqa: N811

    import esp_data.datasets as datasets

    assert "BengaleseFinchCalls" in datasets.__all__
    assert hasattr(_Cls, "info")


def test_call_types_are_preserved() -> None:
    """Test that call types are preserved as strings and cover expected range."""
    ds = BengaleseFinchCalls()  # Uses Bird2_train split
    call_types = set(str(ct) for ct in ds._data.get_unique("call_type"))

    # Bird2 should have 17 call types (highest diversity)
    assert len(call_types) == 17
    # Call types can be numeric strings or letters (e.g., 'a', 'b', 'c', etc.)
    expected_types = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g'}
    assert call_types <= expected_types, f"Unexpected call types found: {call_types - expected_types}"


def test_audio_snippet_structure() -> None:
    """Test that audio snippets are properly structured."""
    ds = BengaleseFinchCalls()  # Uses Bird2_train split
    sample = ds[0]

    # Check that local_path points to expected structure
    assert sample["local_path"].startswith("wav/Bird2/")
    assert sample["local_path"].endswith(".wav")

    # Audio should be loaded and be reasonable length
    audio = sample["audio"]
    assert len(audio) > 0
    assert len(audio) < 48000 * 5  # Should be less than 5 seconds at 48kHz


def test_train_small_splits() -> None:
    """Test that train_small splits work correctly with limited samples per call type."""
    # Test Bird2_train_small (highest diversity)
    ds_small = BengaleseFinchCalls(split="Bird2_train_small")
    assert len(ds_small) == 1360  # 80 samples × 17 call types

    # Check that each call type has at most 80 samples
    call_types = ds_small._data.get_unique("call_type")
    for call_type in call_types:
        # Count occurrences by iterating through the data
        count = sum(1 for row in ds_small._data if row["call_type"] == call_type)
        assert count <= 80, f"Call type {call_type} has {count} samples, expected ≤80"

    # Should have all 17 call types
    assert len(call_types) == 17

    # Test smaller birds' train_small splits
    ds_bird8_small = BengaleseFinchCalls(split="Bird8_train_small")
    assert len(ds_bird8_small) == 320  # 80 samples × 4 call types
    assert len(ds_bird8_small._data.get_unique("call_type")) == 4


def test_call_type_preservation_across_splits() -> None:
    """Test that all call types are preserved in training splits."""
    # Get original Bird2 call types
    ds_original = BengaleseFinchCalls(split="Bird2")
    original_call_types = set(ds_original._data.get_unique("call_type"))

    # Check that train split has all call types
    ds_train = BengaleseFinchCalls(split="Bird2_train")
    train_call_types = set(ds_train._data.get_unique("call_type"))
    assert train_call_types == original_call_types, "Train split missing call types"

    # Check that train_small has all call types
    ds_train_small = BengaleseFinchCalls(split="Bird2_train_small")
    train_small_call_types = set(ds_train_small._data.get_unique("call_type"))
    assert train_small_call_types == original_call_types, "Train_small split missing call types"


def test_split_consistency() -> None:
    """Test that splits are consistent and non-overlapping."""
    # Load all Bird2 splits
    train_ds = BengaleseFinchCalls(split="Bird2_train")
    valid_ds = BengaleseFinchCalls(split="Bird2_valid")
    test_ds = BengaleseFinchCalls(split="Bird2_test")

    # Get sample identifiers (local_path should be unique)
    train_paths = set(row["local_path"] for row in train_ds._data)
    valid_paths = set(row["local_path"] for row in valid_ds._data)
    test_paths = set(row["local_path"] for row in test_ds._data)

    # Check no overlap between splits
    assert len(train_paths & valid_paths) == 0, "Train and valid splits overlap"
    assert len(train_paths & test_paths) == 0, "Train and test splits overlap"
    assert len(valid_paths & test_paths) == 0, "Valid and test splits overlap"

    # Check that total equals original
    total_samples = len(train_paths) + len(valid_paths) + len(test_paths)
    original_ds = BengaleseFinchCalls(split="Bird2")
    assert total_samples == len(original_ds), f"Split totals don't match original: {total_samples} vs {len(original_ds)}"
