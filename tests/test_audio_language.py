"""Tests for esp_data.audio_language module."""

from typing import Any, Dict, Iterator, Sequence
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from esp_data.audio_language import AudioLanguageDataset, is_audio_language_dataset
from esp_data.dataset import Dataset, DatasetInfo
from esp_data.prompts import (
    BasePromptTemplate,
    PromptVariant,
    get_prompt,
    register_prompt,
)
from esp_data.prompts.registry import _PROMPT_REGISTRY


# --- Mock Dataset ---


class MockDataset(Dataset):
    """Mock dataset for testing AudioLanguageDataset."""

    info = DatasetInfo(
        name="mock_dataset",
        owner="test",
        split_paths={"train": "mock://train", "test": "mock://test"},
        version="0.1.0",
        description="Mock dataset for testing",
        sources=["Mock"],
        license="MIT",
    )

    def __init__(
        self,
        data: list[dict[str, Any]] | None = None,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self._items = data or [
            {
                "audio": np.random.randn(16000).astype(np.float32),
                "species_common": "American Robin",
                "behavior": "song",
                "other_field": "value1",
            },
            {
                "audio": np.random.randn(16000).astype(np.float32),
                "species_common": "Blue Jay",
                "behavior": "call",
                "other_field": "value2",
            },
            {
                "audio": np.random.randn(16000).astype(np.float32),
                "species_common": "Cardinal",
                "behavior": "alarm",
                "other_field": "value3",
            },
        ]
        self.sample_rate = sample_rate
        self.split = "train"
        # Create a mock _data attribute for ConcatenatedDataset compatibility
        self._data = MagicMock()
        self._data.columns = list(self._items[0].keys()) if self._items else []
        self._data.__len__ = lambda: len(self._items)

    @property
    def columns(self) -> Sequence[str]:
        return list(self._items[0].keys()) if self._items else []

    @property
    def available_splits(self) -> Sequence[str]:
        return ["train", "test"]

    def _load(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self._items):
            raise IndexError(f"Index {idx} out of range")
        return self._items[idx].copy()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for idx in range(len(self)):
            yield self[idx]

    def __str__(self) -> str:
        return f"MockDataset({len(self)} samples)"

    @classmethod
    def from_config(cls, config):
        return cls(), {}


class MockNativeALDataset(MockDataset):
    """Mock dataset that already has prompt/text fields."""

    def __init__(self):
        super().__init__(
            data=[
                {
                    "audio": np.random.randn(16000).astype(np.float32),
                    "prompt": "Describe this sound.",
                    "text": "A bird singing.",
                },
                {
                    "audio": np.random.randn(16000).astype(np.float32),
                    "prompt": "What is this?",
                    "text": "An alarm call.",
                },
            ]
        )


# --- Test Template ---


class SpeciesTemplateForTest(BasePromptTemplate):
    """Test template for species identification."""

    name = "test_species_template"

    def __init__(self, seed: int | None = None):
        super().__init__(
            variants=[
                PromptVariant("What species is this?", "{species_common}"),
                PromptVariant("Identify the species.", "{species_common}"),
            ],
            seed=seed,
        )

# --- Fixtures ---


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    return MockDataset()


@pytest.fixture
def native_al_dataset():
    """Create a mock native audio-language dataset."""
    return MockNativeALDataset()


@pytest.fixture
def test_template():
    """Create and register a test template."""
    template = SpeciesTemplateForTest(seed=42)
    try:
        register_prompt(template)
    except ValueError:
        # Already registered
        pass
    yield template
    try:
        del _PROMPT_REGISTRY["test_species_template"]
    except KeyError:
        pass


# --- Test is_audio_language_dataset ---


class TestIsAudioLanguageDataset:
    """Tests for is_audio_language_dataset function."""

    def test_native_al_dataset(self, native_al_dataset) -> None:
        """Test detection of native audio-language dataset."""
        assert is_audio_language_dataset(native_al_dataset)

    def test_non_al_dataset(self, mock_dataset) -> None:
        """Test detection of non audio-language dataset."""
        assert not is_audio_language_dataset(mock_dataset)

    def test_empty_dataset(self) -> None:
        """Test handling of empty dataset."""
        empty_ds = MockDataset(data=[])
        assert not is_audio_language_dataset(empty_ds)


# --- Test AudioLanguageDataset ---


class TestAudioLanguageDataset:
    """Tests for AudioLanguageDataset wrapper."""

    def test_init_with_template_instance(self, mock_dataset, test_template) -> None:
        """Test initialization with template instance."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt=test_template)
        assert al_ds.template is test_template
        assert al_ds.source is mock_dataset

    def test_init_with_template_name(self, mock_dataset, test_template) -> None:
        """Test initialization with registered template name."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt="test_species_template")
        assert al_ds.template.name == "test_species_template"

    def test_init_with_passthrough(self, native_al_dataset) -> None:
        """Test initialization with passthrough template."""
        al_ds = AudioLanguageDataset(native_al_dataset, prompt="passthrough")
        assert al_ds.template.name == "passthrough"

    def test_len(self, mock_dataset, test_template) -> None:
        """Test __len__ method."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt=test_template)
        assert len(al_ds) == len(mock_dataset)

    def test_getitem_adds_prompt_and_text(self, mock_dataset, test_template) -> None:
        """Test that __getitem__ adds prompt and text keys."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt=test_template)
        sample = al_ds[0]

        assert "prompt" in sample
        assert "text" in sample
        assert sample["text"] == "American Robin"
        assert sample["prompt"] in [
            "What species is this?",
            "Identify the species.",
        ]

    def test_getitem_preserves_audio(self, mock_dataset, test_template) -> None:
        """Test that audio is preserved in output."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt=test_template)
        sample = al_ds[0]

        assert "audio" in sample
        assert isinstance(sample["audio"], np.ndarray)

    def test_getitem_includes_source_fields(self, mock_dataset, test_template) -> None:
        """Test that source fields are included by default."""
        al_ds = AudioLanguageDataset(
            mock_dataset, prompt=test_template, include_source_fields=True
        )
        sample = al_ds[0]

        assert "other_field" in sample
        assert sample["other_field"] == "value1"

    def test_getitem_excludes_source_fields(self, mock_dataset, test_template) -> None:
        """Test that source fields can be excluded."""
        al_ds = AudioLanguageDataset(
            mock_dataset, prompt=test_template, include_source_fields=False
        )
        sample = al_ds[0]

        assert "audio" in sample
        assert "prompt" in sample
        assert "text" in sample
        assert "other_field" not in sample
        assert "species_common" not in sample

    def test_iteration(self, mock_dataset, test_template) -> None:
        """Test iteration over dataset."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt=test_template)

        samples = list(al_ds)
        assert len(samples) == 3

        for sample in samples:
            assert "audio" in sample
            assert "prompt" in sample
            assert "text" in sample

    def test_columns_property(self, mock_dataset, test_template) -> None:
        """Test columns property."""
        al_ds = AudioLanguageDataset(
            mock_dataset, prompt=test_template, include_source_fields=True
        )
        cols = al_ds.columns

        assert "audio" in cols
        assert "prompt" in cols
        assert "text" in cols
        # First three should be audio, prompt, text
        assert cols[:3] == ["audio", "prompt", "text"]

    def test_columns_without_source_fields(self, mock_dataset, test_template) -> None:
        """Test columns when excluding source fields."""
        al_ds = AudioLanguageDataset(
            mock_dataset, prompt=test_template, include_source_fields=False
        )
        cols = al_ds.columns

        assert list(cols) == ["audio", "prompt", "text"]

    def test_available_splits(self, mock_dataset, test_template) -> None:
        """Test available_splits property."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt=test_template)
        assert al_ds.available_splits == mock_dataset.available_splits

    def test_sample_rate_inherited(self, mock_dataset, test_template) -> None:
        """Test sample_rate is inherited from source."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt=test_template)
        assert al_ds.sample_rate == mock_dataset.sample_rate

    def test_info_updated(self, mock_dataset, test_template) -> None:
        """Test that info is updated based on source."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt=test_template)

        assert "mock_dataset" in al_ds.info.name
        assert al_ds.info.sources == mock_dataset.info.sources
        assert al_ds.info.license == mock_dataset.info.license

    def test_str_representation(self, mock_dataset, test_template) -> None:
        """Test string representation."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt=test_template)
        s = str(al_ds)

        assert "mock_dataset" in s
        assert "test_species_template" in s

    def test_passthrough_with_native_dataset(self, native_al_dataset) -> None:
        """Test passthrough template with native AL dataset."""
        al_ds = AudioLanguageDataset(native_al_dataset, prompt="passthrough")
        sample = al_ds[0]

        assert sample["prompt"] == "Describe this sound."
        assert sample["text"] == "A bird singing."

    def test_from_config_raises(self, mock_dataset, test_template) -> None:
        """Test that from_config raises NotImplementedError."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt=test_template)

        with pytest.raises(NotImplementedError):
            AudioLanguageDataset.from_config({})

    def test_unknown_template_raises(self, mock_dataset) -> None:
        """Test that unknown template name raises KeyError."""
        with pytest.raises(KeyError):
            AudioLanguageDataset(mock_dataset, prompt="nonexistent_template_xyz")

    def test_data_attribute_set(self, mock_dataset, test_template) -> None:
        """Test that _data attribute is set for ConcatenatedDataset compatibility."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt=test_template)
        assert al_ds._data is mock_dataset._data


# --- Test Integration with ConcatenatedDataset ---


class TestAudioLanguageWithConcatenation:
    """Tests for AudioLanguageDataset with ConcatenatedDataset.

    Note: Full concatenation tests require real backend data and are covered
    in integration tests. These tests verify the interface compatibility.
    """

    def test_data_attribute_for_concatenation(self, mock_dataset, test_template) -> None:
        """Test that _data attribute is set for ConcatenatedDataset compatibility."""
        al_ds = AudioLanguageDataset(mock_dataset, prompt=test_template)

        # _data should be set to source's _data
        assert al_ds._data is mock_dataset._data

    def test_multiple_al_datasets_have_data(self, test_template) -> None:
        """Test that multiple AudioLanguageDatasets have _data set."""
        ds1 = MockDataset(
            data=[
                {"audio": np.zeros(100), "species_common": "Robin", "behavior": "song"},
            ]
        )
        ds2 = MockDataset(
            data=[
                {"audio": np.zeros(100), "species_common": "Jay", "behavior": "call"},
            ]
        )

        al_ds1 = AudioLanguageDataset(ds1, prompt=test_template)
        al_ds2 = AudioLanguageDataset(ds2, prompt=test_template)

        # Both should have _data set
        assert al_ds1._data is not None
        assert al_ds2._data is not None
        assert al_ds1._data is ds1._data
        assert al_ds2._data is ds2._data
