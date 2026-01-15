"""Test suite for the InfantMarmosetsVox dataset."""

import numpy as np
import pytest

from esp_data import DatasetConfig
from esp_data.datasets import InfantMarmosetsVox
from esp_data.io import anypath, exists
from esp_data.utils import create_hash


# Dataset-specific constants
NUM_CALLTYPES = 11  # Call types 0-10
NUM_CALLERS = 10  # 5 twin pairs × 2 individuals
NUM_TWINS = 5  # Twin pairs 1-5

# Expected values for integrity tests (computed on 2025-01-15)
# These ensure the dataset hasn't changed unexpectedly
EXPECTED_LEN_ALL = 72921
EXPECTED_LEN_TRAIN = 51044
EXPECTED_LEN_VAL = 14584
EXPECTED_LEN_TEST = 7293

# Hash values for dataset integrity verification
# First audio sample hash and annotations hash per split
_HASHES = {
    "all": (
        "33dd60a95f57b43dc026ec378ecefcc263984ef6d13d1d70d8a4959556e71c7a",
        "f1a94fb8891bc0736e8d9676ddbb0af7361414c818ba2c2d1b4dff4ee2c10864",
    ),
    "train": (
        "765bd9760db5e7d7a4e42fe69718bdb14e7a93714cbbd50ca442cb241984d543",
        "05de390450a00c8d20850d06e9111b820df170c8b9ba4a47b666f8f6910d7a91",
    ),
    "validation": (
        "8d061b7c2572f3077e4951ed429de608c68e38617e71e324adbaf92e411bef36",
        "4d01d8986e58d84cd00da145bf0af393b953d5041c5ab9786a8137ff6582b7e8",
    ),
    "test": (
        "0bddb91326b3df79b44df0bc3bd54ed08d32973936b89950b1ad167819cd2fef",
        "08f1a8448255608f1b6ac835e94002dd501b2a9c40b22aaaf7c4ce65090569ca",
    ),
}

# Columns in the underlying dataframe (not including 'audio' which is added dynamically)
EXPECTED_COLS = [
    "path",
    "start",
    "end",
    "duration",
    "calltypeID",
    "callerID",
    "twinID",
    "vocID",
]


def _compute_dataset_hashes(ds: InfantMarmosetsVox) -> tuple[str, str]:
    """Compute integrity hashes for a dataset split.

    Returns
    -------
    tuple[str, str]
        (first_audio_hash, annotations_hash)
    """
    # Compute first item audio hash
    first_sample = ds[0]
    first_audio = first_sample["audio"].tobytes()
    first_audio_hash = create_hash(first_audio)

    # Compute annotations hash (convert Polars to pandas for consistent hashing)
    df = ds._data.unwrap.to_pandas()
    df = df.sort_index(axis=0).sort_index(axis=1)
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    annotations_hash = create_hash(csv_bytes)

    return first_audio_hash, annotations_hash


@pytest.fixture
def ds() -> InfantMarmosetsVox:
    """Load InfantMarmosetsVox dataset for testing."""
    return InfantMarmosetsVox(split="all", streaming=False, backend="polars")


@pytest.fixture
def ds_train() -> InfantMarmosetsVox:
    """Load InfantMarmosetsVox train split for testing."""
    return InfantMarmosetsVox(split="train", streaming=False, backend="polars")


@pytest.fixture
def first_sample(ds: InfantMarmosetsVox) -> dict:
    """Return the first sample of the dataset."""
    return next(iter(ds))


def test_dataset_integrity(ds: InfantMarmosetsVox) -> None:
    """Test dataset integrity using hash verification."""
    first_audio_hash, annotations_hash = _compute_dataset_hashes(ds)
    expected_audio_hash, expected_annotations_hash = _HASHES["all"]

    assert first_audio_hash == expected_audio_hash, (
        "First item audio hash does not match expected value."
    )
    assert annotations_hash == expected_annotations_hash, (
        "Annotations hash does not match expected value."
    )


def test_info_property(ds: InfantMarmosetsVox) -> None:
    """Test if the info property returns correct metadata."""
    assert ds.info.name == "InfantMarmosetsVox"
    assert ds.info.version == "0.1.0"
    assert "all" in ds.info.split_paths
    assert "train" in ds.info.split_paths
    assert "validation" in ds.info.split_paths
    assert "test" in ds.info.split_paths
    for split in ds.info.split_paths.values():
        assert exists(split), f"Split path {split} does not exist"
    assert "call-type" in ds.info.description.lower()
    assert ds.info.license == "CC-BY-4.0"


def test_dataset_length(ds: InfantMarmosetsVox) -> None:
    """Test the dataset length matches expected value."""
    assert len(ds) == EXPECTED_LEN_ALL, (
        f"Dataset length {len(ds)} does not match expected {EXPECTED_LEN_ALL}"
    )


def test_split_lengths() -> None:
    """Test that split lengths match expected values."""
    ds_train = InfantMarmosetsVox(split="train", streaming=False)
    ds_val = InfantMarmosetsVox(split="validation", streaming=False)
    ds_test = InfantMarmosetsVox(split="test", streaming=False)

    assert len(ds_train) == EXPECTED_LEN_TRAIN, (
        f"Train split length {len(ds_train)} does not match expected {EXPECTED_LEN_TRAIN}"
    )
    assert len(ds_val) == EXPECTED_LEN_VAL, (
        f"Val split length {len(ds_val)} does not match expected {EXPECTED_LEN_VAL}"
    )
    assert len(ds_test) == EXPECTED_LEN_TEST, (
        f"Test split length {len(ds_test)} does not match expected {EXPECTED_LEN_TEST}"
    )

    # Verify splits sum to total (approximately, due to filtering)
    total = len(ds_train) + len(ds_val) + len(ds_test)
    assert total == EXPECTED_LEN_ALL, f"Splits sum {total} != all {EXPECTED_LEN_ALL}"


def test_columns_property(ds: InfantMarmosetsVox) -> None:
    """Test the columns property."""
    cols = ds.columns
    for col in EXPECTED_COLS:
        assert col in cols, f"Expected column '{col}' not found in dataset columns."


def test_data_property(ds: InfantMarmosetsVox) -> None:
    """Test if the data property returns correct dataframes."""
    assert ds._data is not None
    assert "path" in ds._data.columns
    assert "calltypeID" in ds._data.columns
    assert "callerID" in ds._data.columns
    assert "twinID" in ds._data.columns


def test_construction_from_config() -> None:
    """Test the from_config class method."""
    config = {
        "dataset_name": "infant_marmosets_vox",
        "split": "all",
        "streaming": True,
        "backend": "polars",
    }
    config = DatasetConfig.model_validate(config)
    ds, _ = InfantMarmosetsVox.from_config(config)
    assert isinstance(ds, InfantMarmosetsVox), (
        "from_config did not return an InfantMarmosetsVox instance."
    )


def test_transforms_in_from_config() -> None:
    """Test construction with transforms in from_config."""
    config = {
        "dataset_name": "infant_marmosets_vox",
        "split": "train",
        "streaming": False,
        "backend": "polars",
        "transformations": [
            {
                "type": "label_from_feature",
                "feature": "calltypeID",
                "output_feature": "label",
            }
        ],
    }
    config = DatasetConfig.model_validate(config)
    ds, metadata = InfantMarmosetsVox.from_config(config)

    assert "label_from_feature" in metadata, "Transformations metadata not returned."
    assert "label" in ds.columns, (
        "Transformed feature 'label' not found in dataset columns."
    )


def test_available_splits(ds: InfantMarmosetsVox) -> None:
    """Test the available_splits method."""
    splits = ds.available_splits
    expected_splits = ["all", "train", "validation", "test"]
    for split in expected_splits:
        assert split in splits, f"Expected split '{split}' not found in available splits."


def test_split_lookup_error() -> None:
    """Test that an invalid split raises a LookupError."""
    with pytest.raises(LookupError):
        InfantMarmosetsVox(split="invalid_split", streaming=False, backend="polars")


def test_streaming_iter() -> None:
    """Test streaming iteration."""
    ds = InfantMarmosetsVox(split="train", streaming=True, backend="polars")

    # Iterate through first 5 samples
    for i, sample in enumerate(ds):
        if i >= 5:
            break
        assert "audio" in sample, "Sample does not contain 'audio' key."
        assert "calltypeID" in sample, "Sample does not contain 'calltypeID' key."
        assert "callerID" in sample, "Sample does not contain 'callerID' key."


def test_random_samples(ds_train: InfantMarmosetsVox) -> None:
    """Test random samples from the dataset."""
    import random

    n = len(ds_train)
    rng = random.Random(42)
    sample_indices = [rng.randrange(n) for _ in range(min(5, n))]

    for idx in sample_indices:
        item = ds_train[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert len(audio) >= 10, f"[{idx}] audio too short (length={len(audio)})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


def test_getitem(ds_train: InfantMarmosetsVox) -> None:
    """Test if __getitem__ returns correct sample format."""
    sample = ds_train[0]
    assert isinstance(sample, dict)
    assert "calltypeID" in sample
    assert "callerID" in sample
    assert "twinID" in sample
    assert "audio" in sample
    assert "path" in sample

    # Verify audio properties
    audio = sample["audio"]
    assert audio is not None
    assert hasattr(audio, "shape"), "Audio should be a numpy array with shape attribute"
    assert len(audio.shape) == 1, "Audio should be mono (1D array)"


def test_iteration(ds_train: InfantMarmosetsVox) -> None:
    """Test if iteration works correctly."""
    for _, sample in enumerate(ds_train):
        assert isinstance(sample, dict)
        assert "audio" in sample
        break


def test_sample_consistency(ds_train: InfantMarmosetsVox) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    direct_sample = ds_train[0]
    iter_sample = next(iter(ds_train))

    # Compare samples (excluding audio for efficiency)
    assert direct_sample["path"] == iter_sample["path"]
    assert direct_sample["calltypeID"] == iter_sample["calltypeID"]
    assert direct_sample["callerID"] == iter_sample["callerID"]
    assert direct_sample["start"] == iter_sample["start"]
    assert direct_sample["end"] == iter_sample["end"]


def test_output_take_and_give() -> None:
    """Test if output_take_and_give correctly maps column names."""
    ds = InfantMarmosetsVox(
        split="train",
        output_take_and_give={"calltypeID": "label", "audio": "waveform"},
        streaming=False,
    )

    sample = ds[0]

    # Check that only mapped columns are present
    expected_keys = {"label", "waveform"}
    assert set(sample.keys()) == expected_keys

    # Verify the values are correct types
    assert isinstance(sample["label"], (int, np.integer))
    assert isinstance(sample["waveform"], np.ndarray)


def test_sample_rate_resampling() -> None:
    """Test if audio resampling works correctly when sample_rate is specified."""
    # Test with pre-resampled 16kHz
    ds_16k = InfantMarmosetsVox(split="train", sample_rate=16000, streaming=False)
    sample = ds_16k[0]

    assert "audio" in sample
    audio = sample["audio"]
    assert audio is not None
    assert len(audio.shape) == 1, "Audio should be mono after resampling"


def test_available_sample_rates(ds: InfantMarmosetsVox) -> None:
    """Test the available_sample_rates property."""
    rates = ds.available_sample_rates
    assert 44100 in rates, "44100 Hz should be available"
    assert 16000 in rates, "16000 Hz should be available"


def test_data_root_parameter() -> None:
    """Test if data_root parameter works correctly."""
    # Test default data_root (should be parent of CSV path)
    ds_default = InfantMarmosetsVox(split="train")
    expected_default = anypath(ds_default.info.split_paths["train"]).parent
    assert ds_default.data_root == expected_default

    # Test with custom data_root
    custom_root = "tests/"
    ds = InfantMarmosetsVox(split="train", data_root=custom_root)
    assert str(ds.data_root) == custom_root


def test_string_representation(ds: InfantMarmosetsVox) -> None:
    """Test the string representation of the dataset."""
    str_repr = str(ds)
    assert "InfantMarmosetsVox" in str_repr
    assert "all" in str_repr  # split name
    assert "CC-BY-4.0" in str_repr  # license


def test_class_registration() -> None:
    """Test that the dataset class is properly registered."""
    from esp_data.datasets import InfantMarmosetsVox

    # Test that it's in the __all__ list
    import esp_data.datasets as datasets

    assert "InfantMarmosetsVox" in datasets.__all__

    # Test that the class has the correct decorator
    assert hasattr(InfantMarmosetsVox, "info")
    assert hasattr(InfantMarmosetsVox.info, "name")


def test_calltype_names(ds: InfantMarmosetsVox) -> None:
    """Test the calltype_names property."""
    names = ds.calltype_names
    assert isinstance(names, dict)
    assert len(names) == NUM_CALLTYPES, f"Should have {NUM_CALLTYPES} call types (0-10)"

    # Check some specific call types
    assert 0 in names
    assert 1 in names  # Phee
    assert NUM_CALLTYPES - 1 in names  # Last call type (10)

    # Verify all values are strings
    for k, v in names.items():
        assert isinstance(k, int), f"Key {k} should be int"
        assert isinstance(v, str), f"Value {v} should be str"


def test_num_callers(ds: InfantMarmosetsVox) -> None:
    """Test the num_callers property."""
    num = ds.num_callers
    assert isinstance(num, int)
    assert num == NUM_CALLERS, f"Should have {NUM_CALLERS} callers, got {num}"


def test_calltypeID_range(ds_train: InfantMarmosetsVox) -> None:
    """Test that calltypeID values are in expected range (0-10)."""
    sample = ds_train[0]
    calltype = sample["calltypeID"]
    assert 0 <= calltype <= NUM_CALLTYPES - 1, (
        f"calltypeID {calltype} out of expected range 0-{NUM_CALLTYPES - 1}"
    )


def test_callerID_range(ds_train: InfantMarmosetsVox) -> None:
    """Test that callerID values are in expected range (0-9)."""
    sample = ds_train[0]
    caller = sample["callerID"]
    assert 0 <= caller <= NUM_CALLERS - 1, (
        f"callerID {caller} out of expected range 0-{NUM_CALLERS - 1}"
    )


def test_twinID_range(ds_train: InfantMarmosetsVox) -> None:
    """Test that twinID values are in expected range (1-5)."""
    sample = ds_train[0]
    twin = sample["twinID"]
    assert 1 <= twin <= NUM_TWINS, f"twinID {twin} out of expected range 1-{NUM_TWINS}"


def test_duration_positive(ds_train: InfantMarmosetsVox) -> None:
    """Test that duration values are positive."""
    sample = ds_train[0]
    duration = sample["duration"]
    assert duration > 0, f"Duration {duration} should be positive"
    assert sample["end"] > sample["start"], "End should be greater than start"


if __name__ == "__main__":
    # Run tests manually if executed directly
    pytest.main([__file__, "-v"])
