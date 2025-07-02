"""Test suite for the Bengalese Finch Calls dataset."""

import pytest

from esp_data.datasets import BengaleseFinchCalls
from esp_data import Dataset, DatasetConfig
from esp_data.io import anypath


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing the default dataset instance."""
    return BengaleseFinchCalls(split="Bird0")


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
    ds = BengaleseFinchCalls(split="Bird0")
    ds.apply_transformations(cfg.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Dataset instance where column names are mapped on output."""
    mapping = {"call_type": "call_label", "individual_id": "bird_id", "local_path": "audio_path"}
    return BengaleseFinchCalls(split="Bird0", output_take_and_give=mapping)


# -----------------------------------------------------------------------------
# Basic integrity checks
# -----------------------------------------------------------------------------

def test_info_property(dataset: Dataset) -> None:
    assert dataset.info.name == "Bengalese Finch Calls"
    assert dataset.info.version == "0.1.0"
    assert "Bird0" in dataset.info.split_paths
    # Ensure that the referenced CSV exists (locally or remotely)
    split_path = dataset.info.split_paths["Bird0"]
    assert anypath(split_path).exists(), f"Split path {split_path} does not exist"


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
    expected_splits = [f"Bird{i}" for i in range(11)]  # Bird0 through Bird10
    assert set(dataset.available_splits) == set(expected_splits)


def test_length(dataset: Dataset) -> None:
    expected_len = dataset._data.shape[0]
    assert len(dataset) == expected_len
    # Bird0 should have 7652 calls based on processing output
    assert len(dataset) == 7652


def test_getitem(dataset: Dataset) -> None:
    sample = dataset[0]
    assert "audio" in sample
    assert "call_type" in sample
    assert "individual_id" in sample
    # Audio should be mono
    assert sample["audio"].ndim == 1
    # Individual ID should be Bird0 for this split
    assert sample["individual_id"] == "Bird0"


def test_iteration(dataset: Dataset) -> None:
    for sample in dataset:
        assert "audio" in sample
        break  # only need to check first sample


def test_load_from_config() -> None:
    cfg = DatasetConfig(dataset_name="bengalese_finch_calls", split="Bird0", sample_rate=16000)
    ds, _ = BengaleseFinchCalls.from_config(cfg)
    assert len(ds) > 0
    assert ds.sample_rate == 16000


def test_invalid_split() -> None:
    with pytest.raises(LookupError):
        BengaleseFinchCalls(split="invalid")


def test_multiple_splits() -> None:
    """Test that we can load different bird splits."""
    # Test Bird0
    ds0 = BengaleseFinchCalls(split="Bird0")
    assert len(ds0) == 7652
    assert ds0[0]["individual_id"] == "Bird0"

    # Test Bird1 (if available)
    try:
        ds1 = BengaleseFinchCalls(split="Bird1")
        assert ds1[0]["individual_id"] == "Bird1"
        assert len(ds1) > 0
    except (FileNotFoundError, LookupError):
        # Skip if Bird1 processing hasn't completed yet
        pass


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
    original = dataset_with_output_mapping._data.iloc[0]
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
    ds = BengaleseFinchCalls(split="Bird0")
    call_types = set(ds._data["call_type"].astype(str))

    # Should have various call types (numeric strings)
    assert len(call_types) > 1
    # All should be numeric strings
    for ct in call_types:
        assert ct.isdigit(), f"Call type {ct} should be a numeric string"


def test_audio_snippet_structure() -> None:
    """Test that audio snippets are properly structured."""
    ds = BengaleseFinchCalls(split="Bird0")
    sample = ds[0]

    # Check that local_path points to expected structure
    assert sample["local_path"].startswith("wav/Bird0/")
    assert sample["local_path"].endswith(".wav")

    # Audio should be loaded and be reasonable length
    audio = sample["audio"]
    assert len(audio) > 0
    assert len(audio) < 48000 * 5  # Should be less than 5 seconds at 48kHz
