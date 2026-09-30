"""Tests for the value serializers used by `alp_data.export`."""

import numpy as np
import pandas as pd
import pytest

from alp_data.export.serializers import (
    decode_audio,
    decode_value,
    encode_audio,
    encode_value,
)


@pytest.fixture
def audio() -> np.ndarray:
    rng = np.random.default_rng(0)
    # Values on the 16-bit grid so that a FLAC round trip is exact.
    return (rng.integers(-32768, 32767, size=16000) / 32768.0).astype(np.float32)


def test_flac_round_trip_is_exact_on_16bit_grid(audio: np.ndarray) -> None:
    data = encode_audio(audio, 16000, "flac")
    decoded, sr = decode_audio(data)
    assert sr == 16000
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, audio)


def test_flac_compresses_a_tonal_signal() -> None:
    t = np.arange(16000) / 16000
    tone = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    data = encode_audio(tone, 16000, "flac")
    assert len(data) < tone.nbytes / 2


def test_wav_round_trip_is_bit_exact_for_arbitrary_floats() -> None:
    audio = np.array([0.123456789, -0.5, 1e-7, 0.99999], dtype=np.float32)
    decoded, sr = decode_audio(encode_audio(audio, 22050, "wav"))
    assert sr == 22050
    np.testing.assert_array_equal(decoded, audio)


def test_unknown_audio_format_raises() -> None:
    with pytest.raises(ValueError, match="audio_format"):
        encode_audio(np.zeros(10, dtype=np.float32), 16000, "mp3")


def test_scalars_and_strings_pass_through_unchanged() -> None:
    for value in (1, 2.5, "x", None, True):
        encoded, kind = encode_value(value)
        assert encoded == value
        assert kind is None
        assert decode_value(encoded, kind) == value


def test_list_of_scalars_passes_through_unchanged() -> None:
    encoded, kind = encode_value(["a", "b"])
    assert encoded == ["a", "b"]
    assert kind is None


def test_dataframe_round_trips_with_dtypes_preserved() -> None:
    df = pd.DataFrame(
        {
            "Begin Time (s)": [0.5, 1.25],
            "Annotation": ["NA", "01"],  # strings that TSV inference would mangle
            "Selection": ["1", "2"],
        }
    )
    encoded, kind = encode_value(df)
    assert isinstance(encoded, bytes)
    assert kind == "dataframe"
    decoded = decode_value(encoded, kind)
    pd.testing.assert_frame_equal(decoded, df)
    assert decoded["Annotation"].tolist() == ["NA", "01"]


def test_empty_dataframe_round_trips() -> None:
    df = pd.DataFrame(
        {
            "Begin Time (s)": pd.Series([], dtype="float64"),
            "Annotation": pd.Series([], dtype="object"),
        }
    )
    encoded, kind = encode_value(df)
    pd.testing.assert_frame_equal(decode_value(encoded, kind), df)


def test_none_passes_through_opaque_kinds() -> None:
    assert decode_value(None, "dataframe") is None
    assert decode_value(None, "ndarray") is None


def test_list_of_numpy_scalars_becomes_a_plain_list() -> None:
    encoded, kind = encode_value([np.int64(1), np.float32(2.5), np.str_("x")])
    assert encoded == [1, 2.5, "x"]
    assert kind is None
    assert all(type(v) in (int, float, str) for v in encoded)


def test_decoded_ndarray_is_writable() -> None:
    encoded, kind = encode_value(np.arange(3))
    decoded = decode_value(encoded, kind)
    assert decoded.flags.writeable
    decoded[0] = 99


def test_ndarray_round_trips_with_dtype_and_shape() -> None:
    arr = np.arange(6, dtype=np.int16).reshape(2, 3)
    encoded, kind = encode_value(arr)
    assert kind == "ndarray"
    assert set(encoded) == {"data", "dtype", "shape"}
    decoded = decode_value(encoded, kind)
    assert decoded.dtype == np.int16
    np.testing.assert_array_equal(decoded, arr)


def test_unsupported_value_raises_type_error() -> None:
    with pytest.raises(TypeError, match="cannot be packed"):
        encode_value(object())
