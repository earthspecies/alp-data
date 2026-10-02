"""Serializers that turn `_process` outputs into pack-friendly values.

Audio arrays become encoded bytes (FLAC or WAV). Every other value is either
stored as a native parquet value (scalars, strings, lists of scalars) or as an
opaque value with a `kind` tag that `decode_value` uses to rebuild it.
"""

from __future__ import annotations

import io
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import soundfile as sf

AudioFormat = Literal["flac", "wav"]
OpaqueKind = Literal["dataframe", "ndarray"]

MAX_CHANNELS = 8
"""Most channels libsndfile writes to FLAC; also the sanity bound for WAV."""

_AUDIO_FORMATS: dict[str, tuple[str, str]] = {
    "flac": ("FLAC", "PCM_16"),
    "wav": ("WAV", "FLOAT"),
}


def encode_audio(audio: np.ndarray, sample_rate: int, audio_format: AudioFormat) -> bytes:
    """Encode an audio array into a byte string.

    Parameters
    ----------
    audio : np.ndarray
        Audio samples, shape `(n,)` or `(n, channels)`.
    sample_rate : int
        Sample rate in Hz.
    audio_format : {"flac", "wav"}
        `"flac"` writes 16-bit PCM FLAC. `"wav"` writes float32 PCM WAV.

    Returns
    -------
    bytes
        The encoded file contents.

    Raises
    ------
    ValueError
        If `audio_format` is not one of the supported formats, or if `audio` is not
        shaped `(n,)` or `(n, channels)` with at most `MAX_CHANNELS` channels.
    """
    try:
        fmt, subtype = _AUDIO_FORMATS[audio_format]
    except KeyError:
        raise ValueError(
            f"Unsupported audio_format {audio_format!r}; expected one of {sorted(_AUDIO_FORMATS)}"
        ) from None
    if audio.ndim not in (1, 2) or (audio.ndim == 2 and audio.shape[1] > MAX_CHANNELS):
        raise ValueError(
            f"Audio has shape {audio.shape}; expected (n,) or (n, channels) with at most "
            f"{MAX_CHANNELS} channels. Channels-first arrays must be transposed by the dataset."
        )
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=fmt, subtype=subtype)
    return buffer.getvalue()


def encode_audio_lossless_if_needed(
    audio: np.ndarray, sample_rate: int, audio_format: AudioFormat
) -> tuple[bytes, AudioFormat]:
    """Encode audio, switching to float32 WAV when 16-bit FLAC would clip it.

    Datasets that resample on the fly can return samples outside `[-1, 1]`
    (see issue #325). FLAC PCM_16 clips those at full scale, so such rows are
    stored as float32 WAV instead, which is exact.

    Parameters
    ----------
    audio : np.ndarray
        Audio samples.
    sample_rate : int
        Sample rate in Hz.
    audio_format : {"flac", "wav"}
        The preferred encoding.

    Returns
    -------
    tuple[bytes, AudioFormat]
        The encoded bytes and the format actually used.
    """
    fmt: AudioFormat = audio_format
    if audio_format == "flac" and audio.size and float(np.max(np.abs(audio))) > 1.0:
        fmt = "wav"
    return encode_audio(audio, sample_rate, fmt), fmt


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Decode bytes produced by `encode_audio`.

    Parameters
    ----------
    data : bytes
        Encoded audio file contents.

    Returns
    -------
    tuple[np.ndarray, int]
        The float32 samples and the sample rate.
    """
    audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    return audio, sample_rate


def encode_value(value: Any) -> tuple[Any, OpaqueKind | None]:  # noqa: ANN401
    """Turn a single `_process` output value into something parquet can hold.

    Parameters
    ----------
    value : Any
        A value returned by a dataset's `_process`, other than the audio.

    Returns
    -------
    tuple[Any, OpaqueKind | None]
        The value to store and a `kind` tag. The tag is `None` for values
        stored natively, `"dataframe"` for a DataFrame written as arrow IPC
        bytes (dtypes preserved), and `"ndarray"` for an array written as a
        dict of raw bytes, dtype string, and shape.

    Raises
    ------
    TypeError
        If the value is of a type that cannot be packed.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value, None
    if isinstance(value, np.generic):
        return value.item(), None
    if isinstance(value, (list, tuple)):
        items = [v.item() if isinstance(v, np.generic) else v for v in value]
        if all(v is None or isinstance(v, (bool, int, float, str)) for v in items):
            return items, None
    if isinstance(value, pd.DataFrame):
        table = pa.Table.from_pandas(value, preserve_index=False)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return sink.getvalue().to_pybytes(), "dataframe"
    if isinstance(value, np.ndarray):
        return (
            {"data": value.tobytes(), "dtype": str(value.dtype), "shape": list(value.shape)},
            "ndarray",
        )
    raise TypeError(f"Value of type {type(value).__name__} cannot be packed")


def decode_value(value: Any, kind: OpaqueKind | None) -> Any:  # noqa: ANN401
    """Inverse of `encode_value`.

    Parameters
    ----------
    value : Any
        The stored value.
    kind : OpaqueKind | None
        The tag returned by `encode_value`. A `None` value passes through
        unchanged whatever the kind, so nullable opaque columns work.

    Returns
    -------
    Any
        The rebuilt value.

    Raises
    ------
    ValueError
        If `kind` is not a known tag.
    """
    if kind is None or value is None:
        return value
    if kind == "dataframe":
        return pa.ipc.open_stream(value).read_all().to_pandas()
    if kind == "ndarray":
        return np.frombuffer(value["data"], dtype=value["dtype"]).reshape(value["shape"]).copy()
    raise ValueError(f"Unknown opaque kind {kind!r}")
