import pytest
from time import time

from esp_data import NatureLMAudio, Dataset

@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an AudioSet dataset instance.

    Returns
    -------
    Dataset
        An instance of the AudioSet dataset.
    """
    ds = NatureLMAudio(split="train")
    return ds


def test_columns_property(dataset: Dataset) -> None:
    """Test if the columns property returns correct column names."""
    # Columns should match the dataframe columns
    expected_columns = ["audio", "source_dataset", "instruction", "output"]
    cols = dataset.columns
    assert all(col in cols for col in expected_columns)


def test_length_property_missing(dataset: Dataset) -> None:
    """Test if the length property returns correct length."""
    with pytest.raises(TypeError):
        # Length property should raise TypeError if not implemented
        _ = len(dataset)


def test_available_splits(dataset: Dataset) -> None:
    """Test if available_splits returns correct split names."""
    # Available splits should contain these
    expected_splits = ["train"]
    assert all(split in dataset.available_splits for split in expected_splits)


def test_iteration_over_one_sample(dataset) -> None:
    t0 = time()
    sample = next(iter(dataset))
    # Don't print the entire sample - the audio array is huge
    sample_info = {k: v for k, v in sample.items() if k != "audio"}
    sample_info["audio_shape"] = sample["audio"].shape
    sample_info["audio_dtype"] = sample["audio"].dtype

    print(f"Time taken to iterate over one sample: {time() - t0:.4f} seconds")
