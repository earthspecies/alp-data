
import numpy as np
import pandas as pd
import pytest

from esp_data import DatasetConfig
from esp_data.datasets import GibbonSolos
from esp_data.utils import create_hash


EXPECTED_LEN_ALL = 18
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = "3ec0bfb03c2a7bc8ab27f3598c792d455178e7161c92efd32aadb7e0154a724f"
ANNOTATIONS_SHA256 = "7d4b21830d79ae121cf4ee765530f2b5303e4825f58019a61cade07300517268"
EXPECTED_COLS = [
    "local_path",
    "file_name",
    "selection_table",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species_scientific",
    "species_common",
    "taxonomic_name",
]


@pytest.fixture
def ds() -> GibbonSolos:
    """Load GibbonSolos dataset for testing."""
    return GibbonSolos(split="all", streaming=False, backend='pandas')


@pytest.fixture
def first_sample(ds: GibbonSolos) -> dict:
    """Return the first sample of the dataset."""
    return next(iter(ds))


def create_dataset_hashes(first_sample: dict, ds: GibbonSolos) -> tuple[int, str, str]:

    # Compute first item audio hash
    first_audio = first_sample["audio"].tobytes()
    first_audio_hash = create_hash(first_audio)

    # Compute annotations hash
    df = ds._data.unwrap.sort_index(axis=0).sort_index(axis=1)
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    annotations_hash = create_hash(csv_bytes)

    return len(ds), first_audio_hash, annotations_hash


@pytest.mark.skipif(
    EXPECTED_LEN_ALL is None,
    reason="Hash values not yet computed. Run hash computation first."
)
def test_dataset_integrity(
    ds: GibbonSolos,
    first_sample: dict,
) -> None:
    """Test the dataset snapshot."""
    len_h, first_audio_h, annotations_h = create_dataset_hashes(first_sample, ds)

    assert len_h == EXPECTED_LEN_ALL, "Dataset length does not match expected value."
    assert (
        first_audio_h == EXPECTED_FIRST_ITEM_AUDIO_SHA256
    ), "First item audio hash does not match expected value."
    assert (
        annotations_h == ANNOTATIONS_SHA256
    ), "Annotations hash does not match expected value."


def test_columns_property(ds: GibbonSolos) -> None:
    """Test the columns property."""
    cols = ds.columns
    for col in EXPECTED_COLS:
        assert col in cols, f"Expected column '{col}' not found in dataset columns."


def test_construction_from_config() -> None:
    """Test the from_config class method."""
    config = {
        "dataset_name": "gibbon_solos",
        "split": "all",
        "streaming": True,
        "backend": "pandas",
    }
    config = DatasetConfig.model_validate(config)
    ds, _ = GibbonSolos.from_config(config)
    assert isinstance(ds, GibbonSolos), "from_config did not return a GibbonSolos instance."


def test_transforms_in_from_config() -> None:
    """Test construction with transforms in from_config."""
    config = {
        "dataset_name": "gibbon_solos",
        "split": "all",
        "streaming": False,
        "backend": "pandas",
        "transformations": [{
            "type": "label_from_feature",
            "feature": "species_common",
            "output_feature": "label"
        }]
    }
    config = DatasetConfig.model_validate(config)
    ds, metadata = GibbonSolos.from_config(config)

    assert "label_from_feature" in metadata, "Transformations metadata not returned."
    assert "label" in ds.columns, "Transformed feature 'label' not found in dataset columns."


def test_available_splits(ds: GibbonSolos) -> None:
    """Test the available_splits method."""
    splits = ds.available_splits
    expected_splits = ["all"]
    for split in expected_splits:
        assert split in splits, f"Expected split '{split}' not found in available splits."


def test_split_lookup_error() -> None:
    """Test that an invalid split raises a LookupError."""
    with pytest.raises(LookupError):
        GibbonSolos(split="invalid_split", streaming=False, backend='pandas')


def test_streaming_iter() -> None:
    ds = GibbonSolos(split="all", streaming=True, backend='polars')

    # iterate through first 5 samples
    for i, sample in enumerate(ds):
        if i >= 3:
            break
        assert "audio" in sample, "Sample does not contain 'audio' key."
        assert "selection_table" in sample, "Sample does not contain 'selection_table' key."


def test_selection_table_is_dataframe(ds: GibbonSolos) -> None:
    """Test that selection_table is properly parsed as a DataFrame."""
    sample = next(iter(ds))
    assert "selection_table" in sample, "Sample does not contain 'selection_table' key."
    assert isinstance(sample["selection_table"], pd.DataFrame), "selection_table should be a pandas DataFrame."


def test_random_samples(ds: GibbonSolos) -> None:
    """Test random samples from the dataset."""
    import random

    n = len(ds)
    rng = random.Random(40)
    sample_indices = [rng.randrange(n) for _ in range(min(2, n))]

    for idx in sample_indices:
        item = ds[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert len(audio) >= 10, f"[{idx}] audio too short (length={len(audio)})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


# if __name__ == "__main__":
#     # For manual hash computation
#     ds_instance = GibbonSolos(split="all", streaming=False, backend='pandas')
#     first_sample_instance = next(iter(ds_instance))
#     len_h, first_audio_h, annotations_h = create_dataset_hashes(first_sample_instance, ds_instance)
#     print("len(ds) hash =", len_h)
#     print("first item audio sha256 =", first_audio_h)
#     print("annotations sha256 =", annotations_h)
