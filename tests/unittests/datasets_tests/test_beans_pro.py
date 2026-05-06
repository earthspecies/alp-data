"""Test suite for the BeansPro dataset."""


import numpy as np
import pytest

from esp_data import DatasetConfig
from esp_data.datasets import BeansPro
from esp_data.datasets.beans_pro import _normalize_synthetic_conversation_row

SPLITS = list(BeansPro.info.split_paths.keys())
EXPECTED_COLS = [
    "instruction",
    "output",
    "audio_path_original_sample_rate",
    "metadata",
]


@pytest.fixture
def ds() -> BeansPro:
    """Load BeansPro dataset for testing.

    Returns
    -------
    BeansPro
        Streaming BeansPro dataset instance.
    """
    _ds = BeansPro(split="crow-description", streaming=True, backend="pandas")
    return _ds


def test_info_property(ds: BeansPro) -> None:
    """Test if the info property returns correct metadata."""
    assert ds.info.name == "beans_pro"
    assert ds.info.version == "0.1.0"
    assert len(ds.info.split_paths) == len(SPLITS)


def test_columns_property(ds: BeansPro) -> None:
    """Test the columns property."""
    cols = ds.columns
    for col in EXPECTED_COLS:
        assert col in cols, f"Expected column '{col}' not found in dataset columns."


def test_available_splits(ds: BeansPro) -> None:
    """Test the available_splits method."""
    splits = ds.available_splits
    for split in SPLITS:
        assert split in splits, f"Expected split '{split}' not found in available splits."


def test_construction_from_config() -> None:
    """Test the from_config class method."""
    config = DatasetConfig.model_validate({
        "dataset_name": "beans_pro",
        "split": "crow-description",
        "streaming": True,
        "backend": "pandas",
    })
    ds, _ = BeansPro.from_config(config)
    assert isinstance(ds, BeansPro), "from_config did not return a BeansPro instance."


def test_transforms_in_from_config() -> None:
    """Test construction with transforms in from_config."""
    config = DatasetConfig.model_validate({
        "dataset_name": "beans_pro",
        "split": "crow-description",
        "streaming": True,
        "backend": "pandas",
        "transformations": [{
            "type": "label_from_feature",
            "feature": "output",
            "output_feature": "label",
        }],
    })
    ds, metadata = BeansPro.from_config(config)

    assert "label_from_feature" in metadata, "Transformations metadata not returned."
    assert "label" in ds.columns, "Transformed feature 'label' not found in dataset columns."


def test_split_lookup_error() -> None:
    """Test that an invalid split raises a LookupError."""
    with pytest.raises(LookupError):
        BeansPro(split="invalid_split", streaming=False, backend="pandas")


def test_streaming_iter(ds: BeansPro) -> None:
    """Test streaming iteration through first few samples."""
    for i, sample in enumerate(ds):
        if i >= 3:
            break
        assert "audio" in sample, "Sample does not contain 'audio' key."
        assert "instruction" in sample, "Sample does not contain 'instruction' key."


def test_random_samples() -> None:
    """Test random samples from the dataset."""
    ds = BeansPro(split="crow-description", streaming=False, backend="polars")
    import random

    n = len(ds)
    rng = random.Random()
    sample_indices = [rng.randrange(n) for _ in range(min(2, n))]

    for idx in sample_indices:
        item = ds[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert len(audio) >= 10, f"[{idx}] audio too short (length={len(audio)})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


def test_length() -> None:
    """Test that __len__ returns a positive count."""
    ds = BeansPro(split="crow-description", streaming=False, backend="polars")
    assert len(ds) > 0, "Dataset should not be empty"


def test_output_take_and_give() -> None:
    """Test output_take_and_give correctly maps column names."""
    ds = BeansPro(
        split="crow-description",
        output_take_and_give={"output": "answer", "instruction": "prompt"},
        backend="pandas",
    )
    sample = ds[0]
    assert set(sample.keys()) == {"answer", "prompt"}


@pytest.mark.parametrize(
    "split",
    [
        "t1-snr-mcq",
        "t1-snr-binary",
        "t1-snr-regression",
        "t1-description-mcq",
        "t1-caption",
        "t2-captioning",
        "t2-behavior",
    ],
)
def test_synthetic_ids_are_unique(split: str) -> None:
    """Test synthetic splits expose unique ids while preserving source ids."""
    ds = BeansPro(split=split, backend="polars")
    rows = list(ds._data)
    ids = [row["id"] for row in rows]
    source_ids = [row["source_id"] for row in rows]

    assert len(ids) == len(set(ids))
    assert all(id_.startswith(f"{split}:") for id_ in ids)
    assert len(source_ids) == len(rows)
    assert all(source_id is not None for source_id in source_ids)


@pytest.mark.parametrize(
    ("row", "expected_path"),
    [
        (
            {
                "audio_path": "gs://foundation-model-data/synthetic/cropped/powdermill/audio/Recording_1_Segment_34.wav__crop_0293018_0295630.wav",
                "metadata": {"source_dataset": "PowdermillCropped"},
            },
            "gs://foundation-model-data/synthetic/cropped/powdermill/audio/Recording_1_Segment_34.wav__crop_0293018_0295630.wav",
        ),
        (
            {
                "audio_ids": ["AM10_20230718_071000.WAV__crop_0015683_0032419"],
                "audio_paths": [
                    "gs://esp-ml-datasets/birdeep/Audios/AM10/2023_07_18/AM10_20230718_071000.WAV"
                ],
                "metadata": {"source_dataset": "BirdeepCropped"},
            },
            "gs://foundation-model-data/synthetic/cropped/birdeep/audio/AM10_20230718_071000.WAV__crop_0015683_0032419.wav",
        ),
        (
            {
                "audio_ids": ["Recording_1_Segment_27.wav__crop_0243908_0265420"],
                "audio_paths": [
                    "gs://esp-ml-datasets/powdermill/Recording_1/Recording_1_Segment_27.wav"
                ],
                "metadata": {"source_dataset": "PowdermillCropped"},
            },
            "gs://foundation-model-data/synthetic/cropped/powdermill/audio/Recording_1_Segment_27.wav__crop_0243908_0265420.wav",
        ),
        (
            {
                "audio_ids": ["BirdVox-full-night_unit07__crop_8522504_8526617"],
                "audio_paths": [""],
                "metadata": {"source_dataset": "BirdVoxFullNightCropped"},
            },
            "gs://foundation-model-data/synthetic/cropped/birdvox_full_night/audio/BirdVox-full-night_unit07__crop_8522504_8526617.wav",
        ),
        (
            {
                "audio_ids": ["a59f8800-8aff-4075-a597-17e820a07180"],
                "audio_paths": [
                    "data/birdvox_dcase_20k/a59f8800-8aff-4075-a597-17e820a07180.wav"
                ],
                "metadata": {"source_dataset": "BirdVoxFullNightCropped"},
            },
            "gs://foundation-model-data/synthetic/cropped/birdvox_dcase_20k/audio/a59f8800-8aff-4075-a597-17e820a07180.wav",
        ),
    ],
)
def test_synthetic_rows_use_canonical_cropped_audio_uri(
    row: dict,
    expected_path: str,
) -> None:
    """Test cropped-source rows normalize to their canonical cropped audio URI."""
    normalized = _normalize_synthetic_conversation_row(row)

    assert normalized["audio_path_original_sample_rate"] == expected_path
