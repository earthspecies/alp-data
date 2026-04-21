"""Test suite for the AnimalSpeak dataset."""

import pytest
import numpy as np

from esp_data.datasets import AnimalSpeak
from esp_data import Dataset, DatasetConfig
from esp_data.io import exists
from esp_data.utils import create_hash


VAL_EXPECTED_FIRST_ITEM_AUDIO_SHA256 = "28061661586286684f61a92f14bf66a7f5a207a48bd8ec0809b1b8e1924d7d99"
VAL_ANNOTATIONS_SHA256 = "3746a67f45c80013ecc7e977e3ee222d2dab7ceba3cde6c8d52bb5ac39aa6c44"
TRAIN_EXPECTED_FIRST_ITEM_AUDIO_SHA256 = "20bedf2fcaf335f711748cb0fa6bbd6fb233dd142ad61209e5dfdbd2e9d09f4c"
TRAIN_ANNOTATIONS_SHA256 = "ad50765e087368b0e435b78d807ffd2858f9d678cf6bf368c9e072de1f53a591"


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an AnimalSpeak dataset instance.

    Returns
    -------
    Dataset
        An instance of the AnimalSpeak dataset.
    """
    ds = AnimalSpeak(split="validation", backend="pandas")
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
    )
    ds = AnimalSpeak(split="validation")
    ds.apply_transformations(dataset_config.transformations)
    return ds



@pytest.fixture
def dataset_with_transforms_streaming_from_config() -> tuple[Dataset, dict]:
    """Fixture providing an AnimalSpeak dataset instance with transformations
    applied in streaming mode.

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
        streaming=True,
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
    )
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "animalspeak"
    assert dataset.info.version == "0.1.0"
    assert "train" in dataset.info.split_paths
    assert "validation" in dataset.info.split_paths
    # test splits exist
    assert exists(dataset.info.split_paths["train"])
    assert exists(dataset.info.split_paths["validation"])


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be _loaded in __init__
    assert dataset._data is not None
    assert "genus" in dataset._data.columns
    assert "country" in dataset._data.columns


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    expected_columns = ["audio_path", "country", "species_scientific"]
    assert all(col in list(dataset.columns) for col in expected_columns)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should match the dataset info
    expected_splits = ["train", "validation"]
    assert set(dataset.available_splits) == set(expected_splits)


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = len(dataset._data)
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
        streaming=True,
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
    sources = dataset_with_transforms._data.get_unique("source")
    assert set(sources).issubset({"xeno-canto", "iNaturalist"})

    # Check that no other sources are present
    assert "Watkins" not in sources


def test_transformations_from_config(dataset_with_transforms_streaming_from_config: tuple[Dataset, dict]) -> None:
    """Test if transformations from config are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The filter transformation only keeps specified sources
    """
    # Check that label column was created
    ds, metadata = dataset_with_transforms_streaming_from_config
    assert "label" in ds._data.columns

    # Check that only specified sources are present
    sources = ds._data.get_unique("source")
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
    original_row = dataset_with_output_mapping._data[0]

    # Verify the mapping and values
    assert sample["species"] == original_row["canonical_name"]
    assert sample["location"] == original_row["country"]


def test_streaming_mode_with_transformations(dataset_with_transforms_streaming_from_config) -> None:
    """Test if transformations are applied correctly in streaming mode.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The filter transformation only keeps specified sources
    3. The metadata is updated with transformation information
    """
    ds, metadata = dataset_with_transforms_streaming_from_config

    # Check that label column was created
    assert "label" in ds._data.columns

    # Check that only specified sources are present
    sources = ds._data.get_unique("source")
    assert set(sources).issubset({"xeno-canto", "iNaturalist"})

    # Check that no other sources are present
    assert "Watkins" not in sources

    # check that the metadata contains a key for "label_from_feature"
    assert "label_from_feature" in metadata
    assert "label_map" in metadata["label_from_feature"]


def test_validation_reference_item_stability(dataset: Dataset) -> None:
    """
    Check that a canonical item (index 0) is bitwise-stable.

    We hash the raw float32 audio buffer. This catches:
    - sample rate changes (resampling -> different samples)
    - channel handling changes (stereo->mono logic changed)
    - dtype changes
    - ordering changes in the split (if a different recording moved to idx 0)

    If this fails for a legitimate/intentional reason, recompute the hash below
    and update EXPECTED_FIRST_ITEM_AUDIO_SHA256.

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
    h1 = create_hash(audio.tobytes())

    assert h1 == VAL_EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {VAL_EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )

    # compute sha256 over raw bytes of the float32 array of annotations
    csv_bytes = (
        dataset._data.unwrap.sort_index(axis=0)
        .sort_index(axis=1)
        .to_csv(index=True)
        .encode("utf-8")
    )
    h2 = create_hash(csv_bytes)

    assert h2 == VAL_ANNOTATIONS_SHA256, (
        "Annotation's hash changed.\n"
        f"Got    {h}\n"
        f"Expect {VAL_ANNOTATIONS_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )


@pytest.mark.skip(reason="This test takes a very long time")
def test_train_reference_item_stability() -> None:
    """
    Check that a canonical item (index 0) is bitwise-stable.

    We hash the raw float32 audio buffer. This catches:
    - sample rate changes (resampling -> different samples)
    - channel handling changes (stereo->mono logic changed)
    - dtype changes
    - ordering changes in the split (if a different recording moved to idx 0)

    If this fails for a legitimate/intentional reason, recompute the hash below
    and update EXPECTED_FIRST_ITEM_AUDIO_SHA256.

    We do the same for the annotations csv.
    """
    ds_train = AnimalSpeak(split="train", sample_rate=16000, backend="pandas")
    # choose deterministic index
    idx = 0
    item = ds_train[idx]

    # audio presence/type checks (defensive, so the hash failure message is clearer)
    assert "audio" in item, "[0] missing 'audio' key"
    audio = item["audio"]
    assert isinstance(audio, np.ndarray), "[0] audio is not a numpy array"
    assert (
        audio.dtype == np.float32
    ), f"[0] audio dtype is {audio.dtype}, expected float32"

    # compute sha256 over raw bytes of the float32 array
    h1 = create_hash(audio.tobytes())

    assert h1 == TRAIN_EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {TRAIN_EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )

    # compute sha256 over raw bytes of the float32 array of annotations
    csv_bytes = (
        ds_train._data.unwrap.sort_index(axis=0)
        .sort_index(axis=1)
        .to_csv(index=True)
        .encode("utf-8")
    )
    h2 = create_hash(csv_bytes)

    assert h2 == TRAIN_ANNOTATIONS_SHA256, (
        "Annotation's hash changed.\n"
        f"Got    {h}\n"
        f"Expect {TRAIN_ANNOTATIONS_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_ITEM_AUDIO_SHA256 with the new hash."
    )


# if __name__ == "__main__":
    # generate hash
    # from esp_data.utils import create_hash
    # ds_train = AnimalSpeak(split="train", sample_rate=16000, backend="pandas")

    # # print("len(ds) =", len(ds))

    # audio0 = ds_train[0]["audio"]
    # print("dtype:", audio0.dtype, "shape:", audio0.shape)

    # h = create_hash(audio0.tobytes())
    # print("sha256:", h)

    # csv_bytes = (
    #         ds_train._data.unwrap.sort_index(axis=0)
    #         .sort_index(axis=1)
    #         .to_csv(index=True)
    #         .encode("utf-8")
    #     )
    # h = create_hash(csv_bytes)

    # print("annotations sha256:", h)
