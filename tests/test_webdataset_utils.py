"""Tests for WebDataset codec utilities (encoders/decoders)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alp_data.backends.webdataset_utils import (
    _parquet_bytes_to_dataframe,
    _tabular_to_parquet_bytes,
    audio_decoder,
    audio_encoder,
    json_decoder,
    json_encoder,
    open_file_for_wds,
)


def create_audio_sample(sample_rate: int = 16000, duration: float = 0.1) -> np.ndarray:
    """Create a simple audio sample (sine wave).

    Returns
    -------
    np.ndarray
        Float32 sine wave samples.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


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
        audio = create_audio_sample()
        sample = {"audio": audio, "label": "test", "sample_rate": 16000}
        encoded = audio_encoder(sample, sample_rate=16000, format="FLAC")

        decoded = audio_decoder(encoded, format="FLAC")

        assert "audio" in decoded
        assert "sample_rate" in decoded
        assert "label" in decoded
        assert decoded["sample_rate"] == 16000
        assert decoded["label"] == "test"
        assert isinstance(decoded["audio"], np.ndarray)
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

        decoded_data = json.loads(encoded["sample.json"].decode("utf-8"))
        assert decoded_data == sample

    def test_encode_json_with_indent(self) -> None:
        """Test encoding JSON with custom indent."""
        sample = {"key": "value"}

        encoded = json_encoder(sample, indent=4)

        content = encoded["sample.json"].decode("utf-8")
        assert "    " in content


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


class TestTabularToParquetBytes:
    """Tests for _tabular_to_parquet_bytes."""

    def test_pandas_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        result = _tabular_to_parquet_bytes(df)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_pyarrow_table(self) -> None:
        import pyarrow as pa

        table = pa.table({"a": [1, 2], "b": [3.0, 4.0]})

        result = _tabular_to_parquet_bytes(table)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported tabular type"):
            _tabular_to_parquet_bytes({"not": "a dataframe"})

    def test_polars_dataframe(self) -> None:
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        result = _tabular_to_parquet_bytes(df)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_polars_lazyframe(self) -> None:
        pl = pytest.importorskip("polars")
        lf = pl.LazyFrame({"a": [1, 2], "b": [3.0, 4.0]})

        result = _tabular_to_parquet_bytes(lf)

        assert isinstance(result, bytes)
        assert len(result) > 0


class TestParquetBytesToDataframe:
    """Tests for _parquet_bytes_to_dataframe."""

    def test_roundtrip_pandas(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        data = _tabular_to_parquet_bytes(df)

        result = _parquet_bytes_to_dataframe(data)

        pd.testing.assert_frame_equal(result, df)

    def test_returns_dataframe(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0]})
        data = _tabular_to_parquet_bytes(df)

        result = _parquet_bytes_to_dataframe(data)

        assert isinstance(result, pd.DataFrame)

    def test_preserves_dtypes(self) -> None:
        df = pd.DataFrame({"int_col": pd.array([1, 2], dtype="int32"), "float_col": [1.5, 2.5]})
        data = _tabular_to_parquet_bytes(df)

        result = _parquet_bytes_to_dataframe(data)

        assert result["int_col"].dtype == df["int_col"].dtype
        assert result["float_col"].dtype == df["float_col"].dtype

    def test_pyarrow_table_roundtrip(self) -> None:
        import pyarrow as pa

        table = pa.table({"a": [10, 20], "b": ["foo", "bar"]})
        data = _tabular_to_parquet_bytes(table)

        result = _parquet_bytes_to_dataframe(data)

        assert list(result["a"]) == [10, 20]
        assert list(result["b"]) == ["foo", "bar"]


class TestAudioSampleRate:
    """Tests that the encoded audio file stays authoritative for the sample rate."""

    def test_stale_metadata_sample_rate_does_not_override(self) -> None:
        """A stale `sample_rate` in the sample must not survive the round trip."""
        audio = create_audio_sample()
        sample = {"audio": audio, "sample_rate": 44100}

        encoded = audio_encoder(sample, sample_rate=16000)

        metadata = json.loads(encoded["metadata.json"].decode("utf-8"))
        assert "sample_rate" not in metadata
        assert audio_decoder(encoded)["sample_rate"] == 16000

    def test_decoder_ignores_sample_rate_in_metadata(self) -> None:
        """A shard written elsewhere with a conflicting rate is decoded from the file."""
        audio = create_audio_sample()
        encoded = audio_encoder({"audio": audio}, sample_rate=16000)
        encoded["metadata.json"] = json.dumps({"sample_rate": 44100}).encode("utf-8")

        assert audio_decoder(encoded)["sample_rate"] == 16000


class TestWdsInternalKeys:
    """Tests that `__key__` / `__url__` survive encoding and decoding."""

    def test_audio_encoder_passes_through_internal_keys(self) -> None:
        audio = create_audio_sample()
        sample = {"audio": audio, "__key__": "abc", "label": "test"}

        encoded = audio_encoder(sample)

        assert encoded["__key__"] == "abc"
        metadata = json.loads(encoded["metadata.json"].decode("utf-8"))
        assert "__key__" not in metadata

    def test_audio_decoder_passes_through_internal_keys(self) -> None:
        audio = create_audio_sample()
        encoded = audio_encoder({"audio": audio, "__key__": "abc"})
        encoded["__url__"] = "shard-000.tar"

        decoded = audio_decoder(encoded)

        assert decoded["__key__"] == "abc"
        assert decoded["__url__"] == "shard-000.tar"

    def test_json_encoder_passes_through_internal_keys(self) -> None:
        encoded = json_encoder({"__key__": "abc", "id": 1})

        assert encoded["__key__"] == "abc"
        assert json.loads(encoded["sample.json"].decode("utf-8")) == {"id": 1}

    def test_json_decoder_passes_through_internal_keys(self) -> None:
        encoded = json_encoder({"__key__": "abc", "id": 1})

        decoded = json_decoder(encoded)

        assert decoded == {"id": 1, "__key__": "abc"}


class TestAudioEncoderSubtype:
    """Tests for the `subtype` and `dtype` arguments of `audio_encoder`."""

    def test_float_subtype_preserves_precision(self) -> None:
        """`subtype="FLOAT"` round-trips more precisely than the PCM_16 default."""
        audio = create_audio_sample()

        default_err = np.abs(
            audio_decoder(audio_encoder({"audio": audio}, format="WAV"), format="WAV")["audio"]
            - audio
        ).max()
        float_err = np.abs(
            audio_decoder(
                audio_encoder({"audio": audio}, format="WAV", subtype="FLOAT"), format="WAV"
            )["audio"]
            - audio
        ).max()

        assert float_err < default_err
        assert float_err == 0.0

    def test_dtype_casts_the_in_memory_array(self) -> None:
        """`dtype` casts the array, so an integer dtype truncates float input."""
        sample = {"audio": [0.1, 0.2, 0.3, 0.4, 0.5]}

        encoded = audio_encoder(sample, dtype="int16", format="WAV")

        decoded = audio_decoder(encoded, dtype="int16", format="WAV")
        np.testing.assert_array_equal(decoded["audio"], np.zeros(5, dtype="int16"))


class TestAudioDecoderFormatMismatch:
    """Tests the error raised when `format` does not match the shard."""

    def test_format_mismatch_error_names_expected_extension(self) -> None:
        audio = create_audio_sample()
        encoded = audio_encoder({"audio": audio}, format="WAV")

        with pytest.raises(ValueError, match=r"ending with '\.flac'"):
            audio_decoder(encoded, format="FLAC")


class TestOpenFileForWds:
    """Tests for `open_file_for_wds`."""

    def test_write_mode_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "dirs" / "shard.tar"

        with open_file_for_wds(target, mode="wb") as f:
            f.write(b"data")

        assert target.read_bytes() == b"data"

    def test_read_mode_does_not_create_parent_dirs(self, tmp_path: Path) -> None:
        """A failed read must not leave empty directory trees behind."""
        target = tmp_path / "nested" / "dirs" / "shard.tar"

        with pytest.raises(FileNotFoundError):
            open_file_for_wds(target, mode="rb")

        assert not (tmp_path / "nested").exists()

    def test_read_mode_opens_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "shard.tar"
        target.write_bytes(b"data")

        with open_file_for_wds(target, mode="rb") as f:
            assert f.read() == b"data"
