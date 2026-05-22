"""Tests for WebDatasetBackend and related utilities."""

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import pytest
import soundfile as sf
import webdataset as wds

from esp_data.backends import get_backend
from esp_data.backends.webdataset_backend import WebDatasetBackend, _load_webdataset
from esp_data.exporters import export_dataset
from esp_data.backends.webdataset_utils import audio_decoder, audio_encoder, json_decoder, json_encoder
from esp_data.dataset import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    dataset_from_config,
    register_config,
    register_dataset,
)


def create_audio_sample(sample_rate: int = 16000, duration: float = 0.1) -> np.ndarray:
    """Create a simple audio sample (sine wave).

    Returns
    -------
    np.ndarray
        Float32 sine wave samples.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio.astype(np.float32)


def create_webdataset_tar(
    tmp_path: Path,
    num_samples: int = 5,
    include_audio: bool = True,
    audio_format: str = "flac",
) -> Path:
    """Create a WebDataset tar file with test data.

    Returns
    -------
    Path
        Path to the created tar file.
    """
    tar_path = tmp_path / "shard_000.tar"

    with tarfile.open(tar_path, "w") as tar:
        for i in range(num_samples):
            # Create sample key
            key = f"sample_{i:04d}"

            # Create metadata
            metadata = {
                "id": i,
                "label": f"class_{i % 3}",
                "species": f"species_{i % 2}",
            }

            # Add metadata.json
            metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")
            metadata_info = tarfile.TarInfo(name=f"{key}.metadata.json")
            metadata_info.size = len(metadata_bytes)
            tar.addfile(metadata_info, io.BytesIO(metadata_bytes))

            if include_audio:
                # Create audio
                audio = create_audio_sample()
                audio_buffer = io.BytesIO()
                sf.write(audio_buffer, audio, 16000, format=audio_format.upper())
                audio_bytes = audio_buffer.getvalue()

                # Add audio file
                audio_info = tarfile.TarInfo(name=f"{key}.audio.{audio_format}")
                audio_info.size = len(audio_bytes)
                tar.addfile(audio_info, io.BytesIO(audio_bytes))

    return tar_path


def create_json_webdataset_tar(tmp_path: Path, num_samples: int = 5) -> Path:
    """Create a WebDataset tar file with JSON samples (no audio).

    Returns
    -------
    Path
        Path to the created tar file.
    """
    tar_path = tmp_path / "shard_000.tar"

    with tarfile.open(tar_path, "w") as tar:
        for i in range(num_samples):
            key = f"sample_{i:04d}"
            sample = {
                "id": i,
                "name": f"item_{i}",
                "value": i * 10,
                "category": f"cat_{i % 3}",
            }

            # Add sample.json
            sample_bytes = json.dumps(sample, indent=2).encode("utf-8")
            sample_info = tarfile.TarInfo(name=f"{key}.sample.json")
            sample_info.size = len(sample_bytes)
            tar.addfile(sample_info, io.BytesIO(sample_bytes))

    return tar_path


@pytest.fixture
def audio_tar_dir(tmp_path: Path) -> Path:
    """Create a temp directory with an audio WebDataset.

    Returns
    -------
    Path
        Path to the directory containing the tar file.
    """
    create_webdataset_tar(tmp_path, num_samples=5, include_audio=True)
    return tmp_path


@pytest.fixture
def json_tar_dir(tmp_path: Path) -> Path:
    """Create a temp directory with a JSON WebDataset.

    Returns
    -------
    Path
        Path to the directory containing the tar file.
    """
    create_json_webdataset_tar(tmp_path, num_samples=5)
    return tmp_path


@pytest.fixture
def multi_shard_dir(tmp_path: Path) -> Path:
    """Create a temp directory with multiple shards.

    Returns
    -------
    Path
        Path to the directory containing the tar files.
    """
    # Shard 1
    tar1_path = tmp_path / "shard_000.tar"
    with tarfile.open(tar1_path, "w") as tar:
        for i in range(3):
            key = f"s1_sample_{i:04d}"
            sample_bytes = json.dumps(
                {"id": i, "shard": 1, "sample.json": "skip"}
            ).encode()
            # Use the sample.json naming
            sample_info = tarfile.TarInfo(name=f"{key}.sample.json")
            sample_info.size = len(sample_bytes)
            tar.addfile(sample_info, io.BytesIO(sample_bytes))

    # Shard 2
    tar2_path = tmp_path / "shard_001.tar"
    with tarfile.open(tar2_path, "w") as tar:
        for i in range(3, 6):
            key = f"s2_sample_{i:04d}"
            sample_bytes = json.dumps({"id": i, "shard": 2}).encode()
            sample_info = tarfile.TarInfo(name=f"{key}.sample.json")
            sample_info.size = len(sample_bytes)
            tar.addfile(sample_info, io.BytesIO(sample_bytes))

    return tmp_path


class TestAudioEncoder:
    """Tests for the audio_encoder function."""

    def test_encode_audio_flac(self) -> None:
        """Test encoding audio to FLAC format."""
        audio = create_audio_sample()
        sample = {"audio": audio, "label": "test"}

        encoded = audio_encoder(sample, sample_rate=16000, format="FLAC")

        assert "audio.flac" in encoded
        assert "metadata.json" in encoded
        assert isinstance(encoded["audio.flac"], bytes)
        assert len(encoded["audio.flac"]) > 0

        # Verify metadata doesn't contain audio
        metadata = json.loads(encoded["metadata.json"].decode("utf-8"))
        assert "audio" not in metadata
        assert metadata["label"] == "test"

    def test_encode_audio_wav(self) -> None:
        """Test encoding audio to WAV format."""
        audio = create_audio_sample()
        sample = {"audio": audio}

        encoded = audio_encoder(sample, sample_rate=16000, format="WAV")

        assert "audio.wav" in encoded
        assert "metadata.json" in encoded

    def test_encode_audio_from_list(self) -> None:
        """Test encoding audio from a list."""
        audio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        sample = {"audio": audio_list}

        encoded = audio_encoder(sample, sample_rate=16000)

        assert "audio.flac" in encoded

    def test_encode_audio_missing_key_raises(self) -> None:
        """Test that encoding without audio key raises ValueError."""
        sample = {"label": "test"}

        with pytest.raises(ValueError, match="Sample must contain 'audio' key"):
            audio_encoder(sample)


class TestAudioDecoder:
    """Tests for the audio_decoder function."""

    def test_decode_audio_flac(self) -> None:
        """Test decoding audio from FLAC format."""
        # First encode
        audio = create_audio_sample()
        sample = {"audio": audio, "label": "test", "sample_rate": 16000}
        encoded = audio_encoder(sample, sample_rate=16000, format="FLAC")

        # Then decode
        decoded = audio_decoder(encoded, format="FLAC")

        assert "audio" in decoded
        assert "sample_rate" in decoded
        assert "label" in decoded
        assert decoded["sample_rate"] == 16000
        assert decoded["label"] == "test"
        assert isinstance(decoded["audio"], np.ndarray)
        # Use larger tolerance due to FLAC compression
        np.testing.assert_allclose(decoded["audio"], audio, rtol=1e-2, atol=1e-4)

    def test_decode_audio_wav(self) -> None:
        """Test decoding audio from WAV format."""
        audio = create_audio_sample()
        sample = {"audio": audio}
        encoded = audio_encoder(sample, sample_rate=16000, format="WAV")

        decoded = audio_decoder(encoded, format="WAV")

        assert "audio" in decoded
        assert isinstance(decoded["audio"], np.ndarray)

    def test_decode_missing_audio_raises(self) -> None:
        """Test that decoding without audio key raises ValueError."""
        data = {"metadata.json": b"{}"}

        with pytest.raises(ValueError, match="Sample must contain an audio key"):
            audio_decoder(data)


class TestJsonEncoder:
    """Tests for the json_encoder function."""

    def test_encode_json(self) -> None:
        """Test encoding sample to JSON format."""
        sample = {"id": 1, "name": "test", "values": [1, 2, 3]}

        encoded = json_encoder(sample)

        assert "sample.json" in encoded
        assert isinstance(encoded["sample.json"], bytes)

        # Verify content
        decoded_data = json.loads(encoded["sample.json"].decode("utf-8"))
        assert decoded_data == sample

    def test_encode_json_with_indent(self) -> None:
        """Test encoding JSON with custom indent."""
        sample = {"key": "value"}

        encoded = json_encoder(sample, indent=4)

        content = encoded["sample.json"].decode("utf-8")
        assert "    " in content  # 4-space indent


class TestJsonDecoder:
    """Tests for the json_decoder function."""

    def test_decode_json(self) -> None:
        """Test decoding sample from JSON format."""
        original = {"id": 1, "name": "test"}
        encoded = json_encoder(original)

        decoded = json_decoder(encoded)

        assert decoded == original

    def test_decode_missing_json_raises(self) -> None:
        """Test that decoding without sample.json raises ValueError."""
        data = {"other.txt": b"content"}

        with pytest.raises(ValueError, match="Sample must contain 'sample.json' key"):
            json_decoder(data)


class TestLoadWebdataset:
    """Tests for the _load_webdataset function."""

    def test_load_basic(self, json_tar_dir: Path) -> None:
        """Test basic loading of WebDataset."""
        dataset = _load_webdataset(
            json_tar_dir,
            file_pattern="shard*tar",
            data_processor=json_decoder,
        )

        assert isinstance(dataset, wds.WebDataset)
        samples = list(dataset)
        assert len(samples) == 5

    def test_load_with_shuffle(self, json_tar_dir: Path) -> None:
        """Test loading with shuffle."""
        dataset = _load_webdataset(
            json_tar_dir,
            data_processor=json_decoder,
            shuffle_size=10,
            seed=42,
        )

        samples = list(dataset)
        assert len(samples) == 5

    def test_load_with_batch(self, json_tar_dir: Path) -> None:
        """Test loading with batching."""
        dataset = _load_webdataset(
            json_tar_dir,
            data_processor=json_decoder,
            batch_size=2,
        )

        batches = list(dataset)
        # 5 samples with batch_size=2 -> 3 batches (2, 2, 1)
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[-1]) == 1

    def test_load_multi_shard(self, multi_shard_dir: Path) -> None:
        """Test loading from multiple shards."""
        dataset = _load_webdataset(
            multi_shard_dir,
            data_processor=json_decoder,
        )

        samples = list(dataset)
        assert len(samples) == 6  # 3 + 3 from two shards

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        """Test that loading from empty directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No shard files found"):
            _load_webdataset(tmp_path)


class TestWebDatasetBackend:
    """Tests for the WebDatasetBackend class."""

    def test_init(self, json_tar_dir: Path) -> None:
        """Test initialization with a WebDataset."""
        dataset = _load_webdataset(json_tar_dir, data_processor=json_decoder)
        backend = WebDatasetBackend(dataset)

        assert backend._dataset is not None
        assert backend._columns is None  # Lazy loaded

    def test_from_path(self, json_tar_dir: Path) -> None:
        """Test creating backend from path."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        assert isinstance(backend, WebDatasetBackend)
        assert backend.is_streaming is True

    def test_is_streaming(self, json_tar_dir: Path) -> None:
        """Test that is_streaming is always True."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        assert backend.is_streaming is True

    def test_iter(self, json_tar_dir: Path) -> None:
        """Test iterating over samples."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        samples = list(backend)

        assert len(samples) == 5
        for sample in samples:
            assert isinstance(sample, dict)
            assert "id" in sample
            assert "name" in sample

    def test_columns_property(self, json_tar_dir: Path) -> None:
        """Test columns property (lazy loading)."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        columns = backend.columns

        assert isinstance(columns, list)
        assert "id" in columns
        assert "name" in columns
        assert "value" in columns
        assert "category" in columns

    def test_column_exists(self, json_tar_dir: Path) -> None:
        """Test column existence check."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        assert backend.column_exists("id") is True
        assert backend.column_exists("nonexistent") is False

    def test_unwrap(self, json_tar_dir: Path) -> None:
        """Test unwrap returns the underlying WebDataset."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        unwrapped = backend.unwrap

        assert isinstance(unwrapped, wds.WebDataset)

    def test_repr(self, json_tar_dir: Path) -> None:
        """Test string representation."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        repr_str = repr(backend)

        assert "WebDatasetBackend" in repr_str
        assert "streaming=True" in repr_str

    def test_filter_isin(self, json_tar_dir: Path) -> None:
        """Test filtering with isin."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        # Filter for samples where category is cat_0
        filtered = backend.filter_isin("category", ["cat_0"])
        samples = list(filtered)

        # Samples 0 and 3 should have category=cat_0 (i % 3 == 0)
        assert len(samples) == 2
        for sample in samples:
            assert sample["category"] == "cat_0"

    def test_filter_isin_negate(self, json_tar_dir: Path) -> None:
        """Test filtering with isin negation."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        # Filter for samples where category is NOT cat_0
        filtered = backend.filter_isin("category", ["cat_0"], negate=True)
        samples = list(filtered)

        # Samples 1, 2, 4 should remain
        assert len(samples) == 3
        for sample in samples:
            assert sample["category"] != "cat_0"

    def test_filter_isin_missing_column(self, json_tar_dir: Path) -> None:
        """Test filtering on missing column returns nothing (or all with negate)."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        # Filter on nonexistent column
        filtered = backend.filter_isin("nonexistent", ["value"])
        samples = list(filtered)
        assert len(samples) == 0

        # With negate, should return all
        backend2 = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )
        filtered2 = backend2.filter_isin("nonexistent", ["value"], negate=True)
        samples2 = list(filtered2)
        assert len(samples2) == 5

    def test_dropna(self, tmp_path: Path) -> None:
        """Test dropping samples with None values."""
        # Create tar with some None values
        tar_path = tmp_path / "shard_000.tar"
        with tarfile.open(tar_path, "w") as tar:
            samples_data = [
                {"id": 0, "name": "a", "value": 1},
                {"id": 1, "name": None, "value": 2},  # None in name
                {"id": 2, "name": "c", "value": None},  # None in value
                {"id": 3, "name": "d", "value": 4},
            ]
            for i, sample in enumerate(samples_data):
                key = f"sample_{i:04d}"
                sample_bytes = json.dumps(sample).encode()
                info = tarfile.TarInfo(name=f"{key}.sample.json")
                info.size = len(sample_bytes)
                tar.addfile(info, io.BytesIO(sample_bytes))

        backend = WebDatasetBackend.from_path(
            tmp_path,
            data_processor=json_decoder,
        )

        # Drop samples with None in 'name'
        cleaned = backend.dropna(subset=["name"])
        samples = list(cleaned)

        assert len(samples) == 3  # Only samples 0, 2, 3 have non-None name

    def test_dropna_all_columns(self, tmp_path: Path) -> None:
        """Test dropping samples with None in any column."""
        tar_path = tmp_path / "shard_000.tar"
        with tarfile.open(tar_path, "w") as tar:
            samples_data = [
                {"id": 0, "name": "a"},
                {"id": 1, "name": None},
                {"id": None, "name": "c"},
                {"id": 3, "name": "d"},
            ]
            for i, sample in enumerate(samples_data):
                key = f"sample_{i:04d}"
                sample_bytes = json.dumps(sample).encode()
                info = tarfile.TarInfo(name=f"{key}.sample.json")
                info.size = len(sample_bytes)
                tar.addfile(info, io.BytesIO(sample_bytes))

        backend = WebDatasetBackend.from_path(
            tmp_path,
            data_processor=json_decoder,
        )

        # Drop samples with None in any column
        cleaned = backend.dropna()
        samples = list(cleaned)

        assert len(samples) == 2  # Only samples 0 and 3

    def test_chaining_drop_na_on_columns(self, tmp_path: Path) -> None:
        # do something like backend.dropna(subset=["col1"]).dropna(subset=["col2"])
        """Test chaining dropna on multiple columns."""
        tar_path = tmp_path / "shard_000.tar"
        with tarfile.open(tar_path, "w") as tar:
            samples_data = [
                {"id": 0, "col1": "a", "col2": "x"},
                {"id": 1, "col1": None, "col2": "y"},
                {"id": 2, "col1": "c", "col2": None},
                {"id": 3, "col1": "d", "col2": "z"},
            ]
            for i, sample in enumerate(samples_data):
                key = f"sample_{i:04d}"
                sample_bytes = json.dumps(sample).encode()
                info = tarfile.TarInfo(name=f"{key}.sample.json")
                info.size = len(sample_bytes)
                tar.addfile(info, io.BytesIO(sample_bytes))

        backend = WebDatasetBackend.from_path(
            tmp_path,
            data_processor=json_decoder,
        )
        # Chain dropna on col1 and col2
        cleaned = backend.dropna(subset=["col1"]).dropna(subset=["col2"])
        samples = list(cleaned)
        assert len(samples) == 2  # Only samples 0 and 3
        assert samples[0]["id"] == 0
        assert samples[1]["id"] == 3

    def test_map_column(self, json_tar_dir: Path) -> None:
        """Test mapping column values to new column."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        # Map category to numeric labels
        mapping = {"cat_0": 0, "cat_1": 1, "cat_2": 2}
        mapped = backend.map_column("category", mapping, "label")
        samples = list(mapped)

        assert len(samples) == 5
        for sample in samples:
            assert "label" in sample
            expected_label = mapping[sample["category"]]
            assert sample["label"] == expected_label

    def test_map_column_with_default(self, json_tar_dir: Path) -> None:
        """Test mapping with default value for unmapped keys."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        # Map only cat_0, others get default
        mapping = {"cat_0": "known"}
        mapped = backend.map_column("category", mapping, "mapped", default="unknown")
        samples = list(mapped)

        for sample in samples:
            if sample["category"] == "cat_0":
                assert sample["mapped"] == "known"
            else:
                assert sample["mapped"] == "unknown"

    def test_apply_fn(self, json_tar_dir: Path) -> None:
        """Test applying custom function to samples."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        # Apply function to add a new field
        def add_computed_field(sample: dict) -> dict:
            sample["computed"] = sample["value"] * 2
            return sample

        transformed = backend.apply_fn(add_computed_field)
        samples = list(transformed)

        for sample in samples:
            assert "computed" in sample
            assert sample["computed"] == sample["value"] * 2

    def test_chained_operations(self, json_tar_dir: Path) -> None:
        """Test chaining multiple operations."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        # Chain: map_column -> filter_isin
        mapping = {"cat_0": "class_A", "cat_1": "class_B", "cat_2": "class_C"}
        result = backend.map_column("category", mapping, "class_name").filter_isin(
            "class_name", ["class_A", "class_B"]
        )

        samples = list(result)

        # cat_0 and cat_1 samples should remain (ids: 0, 1, 3, 4)
        assert len(samples) == 4
        for sample in samples:
            assert sample["class_name"] in ["class_A", "class_B"]

    def test_immutability_filter_returns_new_backend(self, json_tar_dir: Path) -> None:
        """Test that filter operations return new backends, leaving original unchanged."""
        base_backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        # Create two independent filtered views from the same base
        cats_backend = base_backend.filter_isin("category", ["cat_0"])
        dogs_backend = base_backend.filter_isin("category", ["cat_1"])

        # Verify they are different objects
        assert base_backend is not cats_backend
        assert base_backend is not dogs_backend
        assert cats_backend is not dogs_backend

        # Verify base_backend has no filters applied
        assert len(base_backend._filter_funcs) == 0

        # Verify each filtered backend has exactly one filter
        assert len(cats_backend._filter_funcs) == 1
        assert len(dogs_backend._filter_funcs) == 1

        # Verify the filters work independently
        cats_samples = list(cats_backend)
        dogs_samples = list(dogs_backend)
        all_samples = list(base_backend)

        assert len(all_samples) == 5  # Original has all samples
        assert len(cats_samples) == 2  # cat_0 appears for ids 0, 3
        assert len(dogs_samples) == 2  # cat_1 appears for ids 1, 4

        # Verify correct filtering
        for sample in cats_samples:
            assert sample["category"] == "cat_0"
        for sample in dogs_samples:
            assert sample["category"] == "cat_1"

    def test_immutability_map_returns_new_backend(self, json_tar_dir: Path) -> None:
        """Test that map operations return new backends, leaving original unchanged."""
        base_backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        # Create mapped backend
        mapping = {"cat_0": "A", "cat_1": "B", "cat_2": "C"}
        mapped_backend = base_backend.map_column("category", mapping, "letter")

        # Verify they are different objects
        assert base_backend is not mapped_backend

        # Verify base_backend has no map functions
        assert len(base_backend._map_funcs) == 0
        assert len(mapped_backend._map_funcs) == 1

        # Verify mapping only applies to mapped_backend
        base_sample = next(iter(base_backend))
        mapped_sample = next(iter(mapped_backend))

        assert "letter" not in base_sample
        assert "letter" in mapped_sample


class TestWebDatasetBackendWithAudio:
    """Tests for WebDatasetBackend with audio data."""

    def test_load_audio_samples(self, audio_tar_dir: Path) -> None:
        """Test loading samples with audio."""
        backend = WebDatasetBackend.from_path(
            audio_tar_dir,
            data_processor=audio_decoder,
        )

        samples = list(backend)

        assert len(samples) == 5
        for sample in samples:
            assert "audio" in sample
            assert "sample_rate" in sample
            assert isinstance(sample["audio"], np.ndarray)
            assert sample["sample_rate"] == 16000


def test_get_webdataset_backend() -> None:
    """Test getting WebDatasetBackend via get_backend."""
    backend_cls = get_backend("webdataset")
    assert backend_cls is WebDatasetBackend


# Create a streaming dataset implementation for testing
@register_config
class StreamingTestConfig(DatasetConfig):
    """Config for streaming test dataset."""

    dataset_name: str = "streaming_test_dataset"
    split: str = "train"
    data_path: str | None = None


@register_dataset
class StreamingTestDataset(Dataset):
    """A test dataset that uses WebDatasetBackend for streaming."""

    info = DatasetInfo(
        name="streaming_test_dataset",
        owner="test",
        split_paths={"train": "placeholder"},
        version="0.1.0",
        description="Test dataset for WebDatasetBackend integration",
        sources=["test"],
        license="test",
    )

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        output_take_and_give: Dict[str, str] | None = None,
        data_processor: Callable | None = None,
    ) -> None:
        """Initialize the streaming dataset."""
        super().__init__(output_take_and_give, backend="webdataset", streaming=True)
        self.split = split
        self._data_path = data_path
        self._data_processor = data_processor or json_decoder
        self._data: WebDatasetBackend | None = None
        self._load()

    def _load(self) -> None:
        """Load the dataset using WebDatasetBackend."""
        self._data = WebDatasetBackend.from_path(
            self._data_path,
            data_processor=self._data_processor,
        )

    @property
    def available_splits(self) -> list[str]:
        """Return available splits."""
        return ["train"]

    @property
    def columns(self) -> list[str]:
        """Return dataset columns."""
        if self._data is None:
            return []
        return self._data.columns

    def __len__(self) -> int:
        """Length not supported for streaming datasets."""
        raise NotImplementedError("Streaming datasets do not support __len__")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples.

        Yields
        ------
        Dict[str, Any]
            One sample dict per iteration.

        Raises
        ------
        RuntimeError
            If the dataset has not been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Dataset not loaded")
        for sample in self._data:
            if self.output_take_and_give:
                yield {
                    new_name: sample[old_name]
                    for old_name, new_name in self.output_take_and_give.items()
                }
            else:
                yield sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Random access not supported for streaming datasets."""
        raise NotImplementedError("Streaming datasets do not support random access")

    def __str__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            Human-readable description of the dataset.
        """
        return f"StreamingTestDataset(path={self._data_path}, streaming=True)"

    @classmethod
    def from_config(
        cls, dataset_config: StreamingTestConfig
    ) -> tuple["StreamingTestDataset", dict]:
        """Create dataset from config.

        Returns
        -------
        tuple[StreamingTestDataset, dict]
            Dataset instance and empty metadata dict.

        Raises
        ------
        ValueError
            If `data_path` is not set in the config.
        """
        if dataset_config.data_path is None:
            raise ValueError("data_path is required for StreamingTestDataset")

        ds = cls(
            data_path=dataset_config.data_path,
            split=dataset_config.split,
            output_take_and_give=dataset_config.output_take_and_give,
        )
        return ds, {}


class TestWebDatasetBackendIntegration:
    """Integration tests for WebDatasetBackend with Dataset class."""

    def test_create_streaming_dataset(self, json_tar_dir: Path) -> None:
        """Test creating a dataset with WebDatasetBackend."""
        dataset = StreamingTestDataset(data_path=str(json_tar_dir))

        assert dataset.streaming is True
        assert isinstance(dataset._data, WebDatasetBackend)

    def test_iterate_streaming_dataset(self, json_tar_dir: Path) -> None:
        """Test iterating over a streaming dataset."""
        dataset = StreamingTestDataset(data_path=str(json_tar_dir))

        # Use explicit iteration instead of list() which calls __len__
        samples = [s for s in iter(dataset)]

        assert len(samples) == 5
        for sample in samples:
            assert "id" in sample
            assert "name" in sample
            assert "value" in sample

    def test_streaming_dataset_columns(self, json_tar_dir: Path) -> None:
        """Test getting columns from streaming dataset."""
        dataset = StreamingTestDataset(data_path=str(json_tar_dir))

        columns = dataset.columns

        assert "id" in columns
        assert "name" in columns
        assert "value" in columns
        assert "category" in columns

    def test_streaming_dataset_len_raises(self, json_tar_dir: Path) -> None:
        """Test that len() raises for streaming dataset."""
        dataset = StreamingTestDataset(data_path=str(json_tar_dir))

        with pytest.raises(NotImplementedError, match="do not support __len__"):
            len(dataset)

    def test_streaming_dataset_getitem_raises(self, json_tar_dir: Path) -> None:
        """Test that __getitem__ raises for streaming dataset."""
        dataset = StreamingTestDataset(data_path=str(json_tar_dir))

        with pytest.raises(NotImplementedError, match="do not support random access"):
            _ = dataset[0]

    def test_streaming_dataset_with_output_mapping(self, json_tar_dir: Path) -> None:
        """Test streaming dataset with output_take_and_give."""
        dataset = StreamingTestDataset(
            data_path=str(json_tar_dir),
            output_take_and_give={"name": "item_name", "value": "item_value"},
        )

        # Use explicit iteration instead of list() which calls __len__
        samples = [s for s in iter(dataset)]

        for sample in samples:
            # Original keys should be remapped
            assert "item_name" in sample
            assert "item_value" in sample
            # Original keys shouldn't be present
            assert "name" not in sample
            assert "value" not in sample

    def test_streaming_dataset_from_config(self, json_tar_dir: Path) -> None:
        """Test creating streaming dataset from config."""
        config = StreamingTestConfig(
            dataset_name="streaming_test_dataset",
            split="train",
            data_path=str(json_tar_dir),
        )

        dataset, metadata = StreamingTestDataset.from_config(config)

        assert isinstance(dataset, StreamingTestDataset)
        assert dataset.streaming is True

        # Use explicit iteration instead of list() which calls __len__
        samples = [s for s in iter(dataset)]
        assert len(samples) == 5

    def test_dataset_from_config_factory(self, json_tar_dir: Path) -> None:
        """Test using dataset_from_config factory with streaming dataset."""
        config = StreamingTestConfig(
            dataset_name="streaming_test_dataset",
            split="train",
            data_path=str(json_tar_dir),
        )

        dataset, _ = dataset_from_config(config)

        assert isinstance(dataset, StreamingTestDataset)
        # Use explicit iteration instead of list() which calls __len__
        samples = [s for s in iter(dataset)]
        assert len(samples) == 5

    def test_streaming_dataset_with_audio(self, audio_tar_dir: Path) -> None:
        """Test streaming dataset with audio data."""
        dataset = StreamingTestDataset(
            data_path=str(audio_tar_dir),
            data_processor=audio_decoder,
        )

        # Use explicit iteration instead of list() which calls __len__
        samples = [s for s in iter(dataset)]

        assert len(samples) == 5
        for sample in samples:
            assert "audio" in sample
            assert "sample_rate" in sample
            assert isinstance(sample["audio"], np.ndarray)


class TestStreamingBackendProtocol:
    """Tests to verify WebDatasetBackend conforms to StreamingBackend protocol."""

    def test_implements_streaming_protocol(self, json_tar_dir: Path) -> None:
        """Test that WebDatasetBackend implements StreamingBackend protocol."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        # Check that it has all required protocol methods
        assert hasattr(backend, "is_streaming")
        assert hasattr(backend, "__iter__")
        assert hasattr(backend, "columns")
        assert hasattr(backend, "column_exists")
        assert hasattr(backend, "unwrap")
        assert hasattr(backend, "filter_isin")
        assert hasattr(backend, "dropna")
        assert hasattr(backend, "map_column")
        assert hasattr(backend, "apply_fn")

        # Verify is_streaming returns True
        assert backend.is_streaming is True

    def test_protocol_methods_return_types(self, json_tar_dir: Path) -> None:
        """Test that protocol methods return correct types."""
        backend = WebDatasetBackend.from_path(
            json_tar_dir,
            data_processor=json_decoder,
        )

        # columns returns list[str]
        assert isinstance(backend.columns, list)
        assert all(isinstance(c, str) for c in backend.columns)

        # column_exists returns bool
        assert isinstance(backend.column_exists("id"), bool)

        # filter_isin returns self (WebDatasetBackend)
        filtered = backend.filter_isin("id", [0, 1])
        assert isinstance(filtered, WebDatasetBackend)

        # __iter__ returns iterator of dicts
        iterator = iter(backend)
        sample = next(iterator)
        assert isinstance(sample, dict)


class TestExportTo:
    """Tests for export_dataset with webdataset format."""

    def test_json_round_trip(self, json_tar_dir: Path, tmp_path: Path) -> None:
        """Test export_dataset then from_path yields identical samples."""
        backend = WebDatasetBackend.from_path(json_tar_dir, data_processor=json_decoder)
        output_dir = tmp_path / "saved"

        n, fmt = export_dataset(iter(backend), str(output_dir), encoder_fn=None)

        assert n == 5
        assert fmt == "webdataset"
        reloaded = WebDatasetBackend.from_path(output_dir, data_processor=json_decoder)
        original = list(backend)
        loaded = list(reloaded)
        assert len(loaded) == 5
        for orig, reloaded_s in zip(original, loaded, strict=True):
            assert orig["id"] == reloaded_s["id"]
            assert orig["name"] == reloaded_s["name"]

    def test_audio_round_trip(self, audio_tar_dir: Path, tmp_path: Path) -> None:
        """Test export_dataset then from_path for audio samples."""
        backend = WebDatasetBackend.from_path(audio_tar_dir, data_processor=audio_decoder)
        output_dir = tmp_path / "saved"

        n, _ = export_dataset(iter(backend), str(output_dir), encoder_fn=audio_encoder)

        assert n == 5
        reloaded = WebDatasetBackend.from_path(output_dir, data_processor=audio_decoder)
        samples = list(reloaded)
        assert len(samples) == 5
        for sample in samples:
            assert "audio" in sample
            assert isinstance(sample["audio"], np.ndarray)

    def test_creates_shard_files(self, json_tar_dir: Path, tmp_path: Path) -> None:
        """Test that export_dataset creates shard tar files with the expected naming."""
        backend = WebDatasetBackend.from_path(json_tar_dir, data_processor=json_decoder)
        output_dir = tmp_path / "saved"

        export_dataset(iter(backend), str(output_dir), encoder_fn=None)

        shard_files = list(output_dir.glob("shard_*.tar"))
        assert len(shard_files) >= 1
        assert (output_dir / "shard_000000.tar").exists()

    def test_returns_sample_count(self, json_tar_dir: Path, tmp_path: Path) -> None:
        """Test that export_dataset return value equals the number of samples written."""
        backend = WebDatasetBackend.from_path(json_tar_dir, data_processor=json_decoder)

        n, _ = export_dataset(iter(backend), str(tmp_path / "saved"), encoder_fn=None)

        assert n == 5

    def test_filtered_backend(self, json_tar_dir: Path, tmp_path: Path) -> None:
        """Test that filters applied before export_dataset are reflected in the output."""
        backend = WebDatasetBackend.from_path(json_tar_dir, data_processor=json_decoder)
        filtered = backend.filter_isin("category", ["cat_0"])
        output_dir = tmp_path / "saved"

        n, _ = export_dataset(iter(filtered), str(output_dir), encoder_fn=None)

        assert n == 2
        reloaded = list(WebDatasetBackend.from_path(output_dir, data_processor=json_decoder))
        assert all(s["category"] == "cat_0" for s in reloaded)

    def test_creates_output_dir(self, json_tar_dir: Path, tmp_path: Path) -> None:
        """Test that export_dataset creates the output directory if it does not exist."""
        backend = WebDatasetBackend.from_path(json_tar_dir, data_processor=json_decoder)
        output_dir = tmp_path / "new" / "nested" / "dir"

        export_dataset(iter(backend), str(output_dir), encoder_fn=None)

        assert output_dir.exists()

    def test_with_explicit_encoder(self, json_tar_dir: Path, tmp_path: Path) -> None:
        """Test export_dataset accepts an explicit encoder_fn."""
        backend = WebDatasetBackend.from_path(json_tar_dir, data_processor=json_decoder)

        n, _ = export_dataset(iter(backend), str(tmp_path / "saved"), encoder_fn=json_encoder)

        assert n == 5
