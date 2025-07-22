"""Test suite for the AnimalSpeak dataset."""

import pytest

from esp_data.datasets import AnimalSpeak
from esp_data import Dataset, DatasetConfig
from esp_data.io import anypath


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an AnimalSpeak dataset instance.

    Returns
    -------
    Dataset
        An instance of the AnimalSpeak dataset.
    """
    ds = AnimalSpeak(split="validation", data_root="gs://")
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing an AnimalSpeak dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the AnimalSpeak dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="animalspeak",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "canonical_name",
                "output_feature": "label",
            },
            {
                "type": "filter",
                "mode": "include",
                "property": "source",
                "values": ["xeno-canto", "iNaturalist"],
            },
        ],
        data_root="gs://",
    )
    ds = AnimalSpeak(split="validation")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_transforms_from_config() -> tuple[Dataset, dict]:
    """Fixture providing an AnimalSpeak dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the AnimalSpeak dataset with transformations applied.
    dict
        Metadata dictionary containing information about the transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="animalspeak",
        split="validation",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "canonical_name",
                "output_feature": "label",
            },
            {
                "type": "filter",
                "mode": "include",
                "property": "source",
                "values": ["xeno-canto", "iNaturalist"],
            },
        ],
        data_root="gs://",
    )
    ds, metadata = AnimalSpeak.from_config(dataset_config)
    return ds, metadata


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing an AnimalSpeak dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the AnimalSpeak dataset with output mapping applied.
    """
    dataset_config = DatasetConfig(
        dataset_name="animalspeak",
        output_take_and_give={"canonical_name": "species", "country": "location"},
    )
    ds = AnimalSpeak(
        split="validation",
        output_take_and_give=dataset_config.output_take_and_give,
        data_root="gs://"
    )
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "animalspeak"
    assert dataset.info.version == "0.1.0"
    assert "train" in dataset.info.split_paths
    assert "validation" in dataset.info.split_paths
    # test splits exist
    assert anypath(dataset.info.split_paths["train"]).exists()
    assert anypath(dataset.info.split_paths["validation"]).exists()


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be _loaded in __init__
    assert dataset._data is not None
    assert "genus" in dataset._data
    assert "country" in dataset._data


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    expected_columns = ["local_path", "country", "species_scientific"]
    assert all(col in list(dataset.columns) for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should match the dataset info
    expected_splits = ["train", "validation"]
    assert set(dataset.available_splits) == set(expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = dataset._data.shape[0]
    assert len(dataset) == expected_len


def test_getitem(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "country" in sample
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
        dataset_name="animalspeak",
        split="validation",
    )
    dataset, _ = AnimalSpeak.from_config(dataset_config)
    assert isinstance(dataset, AnimalSpeak)
    assert dataset.info.name == "animalspeak"
    assert dataset.info.split_paths["validation"] is not None


def test_invalid_split() -> None:
    """Test if initializing with invalid split raises error."""
    with pytest.raises(LookupError):
        AnimalSpeak(split="invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    # Get same sample through different methods
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))

    # Compare samples
    assert direct_sample["country"] == iter_sample["country"]


def test_transformations(dataset_with_transforms: Dataset) -> None:
    """Test if transformations are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The filter transformation only keeps specified sources
    3. The metadata is updated with transformation information
    """
    # Check that label column was created
    assert "label" in dataset_with_transforms._data.columns

    # Check that only specified sources are present
    sources = dataset_with_transforms._data["source"].unique()
    assert set(sources).issubset({"xeno-canto", "iNaturalist"})

    # Check that no other sources are present
    assert "Watkins" not in sources


def test_transformations_from_config(dataset_with_transforms_from_config: tuple[Dataset, dict]) -> None:
    """Test if transformations from config are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The filter transformation only keeps specified sources
    """
    # Check that label column was created
    ds, metadata = dataset_with_transforms_from_config
    assert "label" in ds._data.columns

    # Check that only specified sources are present
    sources = ds._data["source"].unique()
    assert set(sources).issubset({"xeno-canto", "iNaturalist"})

    # Check that no other sources are present
    assert "Watkins" not in sources

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
    assert set(sample.keys()) == {"species", "location"}

    # Get the original row to compare values
    original_row = dataset_with_output_mapping._data.iloc[0]

    # Verify the mapping and values
    assert sample["species"] == original_row["canonical_name"]
    assert sample["location"] == original_row["country"]


def test_max_duration_parameter() -> None:
    """Test if max_duration parameter works correctly."""
    # Test with custom max_duration
    dataset = AnimalSpeak(split="validation", max_duration=5.0, data_root="gs://")
    assert dataset.max_duration == 5.0

    # Test default max_duration
    dataset_default = AnimalSpeak(split="validation", data_root="gs://")
    assert dataset_default.max_duration == 10.0


def test_random_window_parameter() -> None:
    """Test if random_window parameter works correctly."""
    # Test with random_window=True (default)
    dataset_random = AnimalSpeak(split="validation", random_window=True, data_root="gs://")
    assert dataset_random.random_window is True
    assert dataset_random._rng is not None  # Should have RNG when random_window=True

    # Test with random_window=False
    dataset_fixed = AnimalSpeak(split="validation", random_window=False, data_root="gs://")
    assert dataset_fixed.random_window is False
    assert dataset_fixed._rng is None  # Should not have RNG when random_window=False


def test_seed_parameter() -> None:
    """Test if seed parameter works correctly for reproducibility."""
    # Test with specific seed
    dataset1 = AnimalSpeak(split="validation", seed=42, data_root="gs://")
    dataset2 = AnimalSpeak(split="validation", seed=42, data_root="gs://")

    assert dataset1.seed == 42
    assert dataset2.seed == 42

    # Both should have RNG with same state initially
    assert dataset1._rng is not None
    assert dataset2._rng is not None


def test_max_duration_audio_limiting() -> None:
    """Test if audio is actually limited by max_duration."""
    # Use a short max_duration to test limiting
    max_duration = 2.0
    dataset = AnimalSpeak(
        split="validation",
        max_duration=max_duration,
        sample_rate=16000,
        random_window=False,  # Use fixed window for consistent testing
        data_root="gs://"
    )

    # Get a sample and check audio duration
    sample = dataset[0]
    audio = sample["audio"]

    # Calculate actual duration (assuming 16kHz sample rate)
    actual_duration = len(audio) / 16000

    # Audio should not exceed max_duration (with small tolerance for rounding)
    assert actual_duration <= max_duration + 0.1, (
        f"Audio duration {actual_duration:.3f}s exceeds max_duration {max_duration}s"
    )


def test_random_window_reproducibility() -> None:
    """Test if random windows are reproducible with same seed."""
    max_duration = 3.0
    seed = 123

    # Create two datasets with same seed
    dataset1 = AnimalSpeak(
        split="validation",
        max_duration=max_duration,
        random_window=True,
        seed=seed,
        sample_rate=16000,
        data_root="gs://"
    )

    dataset2 = AnimalSpeak(
        split="validation",
        max_duration=max_duration,
        random_window=True,
        seed=seed,
        sample_rate=16000,
        data_root="gs://"
    )

    # Get same sample from both datasets
    sample1 = dataset1[0]
    sample2 = dataset2[0]

    # Audio should be identical (same random window selected)
    import numpy as np
    assert np.array_equal(sample1["audio"], sample2["audio"]), (
        "Audio samples should be identical with same seed"
    )


def test_from_config_with_new_parameters() -> None:
    """Test if new parameters work correctly when loaded from config."""
    dataset_config = DatasetConfig(
        dataset_name="animalspeak",
        split="validation",
        max_duration=4.0,
        random_window=False,
        seed=999,
        sample_rate=16000,
        data_root="gs://",
    )

    dataset, _ = AnimalSpeak.from_config(dataset_config)

    # Check all parameters are set correctly
    assert dataset.max_duration == 4.0
    assert dataset.random_window is False
    assert dataset.seed == 999
    assert dataset._rng is None  # Should be None when random_window=False

    # Test that audio respects the duration limit
    sample = dataset[0]
    audio = sample["audio"]
    actual_duration = len(audio) / 16000
    assert actual_duration <= 4.1  # Small tolerance for rounding
