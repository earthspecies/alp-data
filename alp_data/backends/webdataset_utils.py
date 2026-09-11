"""Utilities for encoding and decoding audio and JSON data for use in the WebDataset format."""

import io
import json
from typing import IO, Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from alp_data.io.filesystem import filesystem_from_path
from alp_data.io.paths import AnyPathT, PureCloudPath, anypath


def open_file_for_wds(
    file_path: str | AnyPathT,
    mode: str = "wb",
    block_size: int = 1024 * 1024 * 100,
) -> IO[bytes]:
    """Open a local or cloud file for use with WebDataset readers and writers.

    For local paths opened in a write mode, missing parent directories are
    created first. Read modes never create directories.

    Parameters
    ----------
    file_path: str | AnyPathT
        The file path to open
    mode: str
        The mode in which to open the file (default: "wb")
    block_size: int
        Block size for remote (cloud) files (default: 100 MB). Ignored for
        local paths.

    Returns
    -------
    IO[bytes]
        An open file object for `file_path`, suitable for passing to
        `wds.TarWriter` / `wds.WebDataset` as a file object.
    """
    path_obj = anypath(file_path)

    if not isinstance(path_obj, PureCloudPath):
        # Local filesystem - create parent dirs only when writing
        if any(flag in mode for flag in ("w", "a", "x")):
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        return open(str(path_obj), mode=mode)
    # Remote filesystem (GCS, R2, etc.)
    fs = filesystem_from_path(str(path_obj))
    return fs.open(str(path_obj), mode=mode, block_size=block_size)


def _is_tabular(v: object) -> bool:
    """Return True if `v` is a tabular type (pandas/polars DataFrame or PyArrow Table).
    Used when a sample key is a dataframe type.

    Parameters
    ----------
    v : object
        Value to inspect.

    Returns
    -------
    bool
        True if `v` is a `pd.DataFrame`, `pl.DataFrame`, `pl.LazyFrame`, or `pa.Table`.
    """
    return isinstance(v, (pl.DataFrame, pl.LazyFrame, pd.DataFrame, pa.Table))


def _wds_internal_keys(sample: dict[str, Any]) -> dict[str, Any]:
    """Extract the WebDataset internal entries of a sample.

    WebDataset uses `__`-prefixed keys such as `__key__` and `__url__` to carry
    sample identity and provenance. They are neither payload nor metadata and
    must pass through encoding and decoding unchanged.

    Parameters
    ----------
    sample : dict[str, Any]
        Sample to inspect.

    Returns
    -------
    dict[str, Any]
        The `__`-prefixed entries of `sample`.
    """
    return {k: v for k, v in sample.items() if k.startswith("__")}


def _tabular_to_parquet_bytes(v: object) -> bytes:
    """Serialize a tabular value to Parquet bytes.

    Parameters
    ----------
    v : object
        Tabular value to serialize. Must be one of `pd.DataFrame`,
        `pl.DataFrame`, `pl.LazyFrame`, or `pa.Table`.

    Returns
    -------
    bytes
        Parquet-encoded bytes.

    Raises
    ------
    TypeError
        If `v` is not a supported tabular type.
    """
    # FIXME: this bit is ugly. we're essentially enumerating
    # the DataBackend types. if add another backend, we have to add it here too. maybe we can
    # add a method to the DataBackend interface to convert to arrow table, and then call that here.
    if isinstance(v, pl.LazyFrame):
        v = v.collect()  # Convert LazyFrame to DataFrame for serialization
    if isinstance(v, pl.DataFrame):
        table = v.to_arrow()
        buf = io.BytesIO()
        pq.write_table(table, buf)
        return buf.getvalue()

    if isinstance(v, pd.DataFrame):
        table = pa.Table.from_pandas(v)
    elif isinstance(v, pa.Table):
        table = v
    else:
        raise TypeError(f"Unsupported tabular type: {type(v)}")

    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _parquet_bytes_to_dataframe(data: bytes) -> pd.DataFrame:
    """Deserialize Parquet bytes to a pandas DataFrame.

    Parameters
    ----------
    data : bytes
        Parquet-encoded bytes.

    Returns
    -------
    pd.DataFrame
        Decoded tabular data.
    """
    return pq.read_table(io.BytesIO(data)).to_pandas()


def audio_encoder(
    sample: dict[str, Any],
    sample_rate: int = 16000,
    dtype: str = "float32",
    format: str = "FLAC",
    subtype: str | None = None,
) -> dict[str, Any]:
    """Encode audio data in the sample to a specific format.

    Non-audio tabular fields (`pd.DataFrame`, `pl.DataFrame`, `pl.LazyFrame`,
    `pa.Table`) are stored as individual Parquet files named `{key}.parquet`.
    All remaining non-audio fields are stored in `metadata.json`, except:

    - WebDataset internal keys (`__key__`, `__url__`, ...), which are passed
      through unchanged so the result can be handed to `wds.TarWriter.write`.
    - `sample_rate`, which is a property of the encoded audio file and is
      recovered from it by `audio_decoder`.

    Parameters
    ----------
    sample: dict[str, Any]
        The sample containing audio data
    sample_rate: int
        The sample rate to encode the audio data with (default: 16000)
    dtype: str
        The data type to cast the audio array to before encoding
        (default: "float32"). This only controls the in-memory array; use
        `subtype` to control the precision of the encoded file.
    format: str
        The format to encode the audio data to (e.g., "WAV", "FLAC", "OGG")
        Default is "FLAC".
    subtype: str | None
        The `soundfile` subtype to encode with (e.g., "PCM_16", "PCM_24",
        "FLOAT"). Default is None, which uses `soundfile`'s default subtype for
        `format` - "PCM_16" for both "WAV" and "FLAC", i.e. float input is
        quantized to 16 bits unless a wider subtype is requested here.

    Returns
    -------
    dict
        Dictionary containing the encoded audio data and metadata
        in the WebDataset format.

    Raises
    ------
    ValueError
        If the sample does not contain an "audio" key with audio data.
    """
    if "audio" not in sample:
        raise ValueError("Sample must contain 'audio' key with audio data")

    audio_buffer = io.BytesIO()
    audio = np.asarray(sample["audio"], dtype=dtype)
    sf.write(audio_buffer, audio, sample_rate, format=format, subtype=subtype)

    internal = _wds_internal_keys(sample)
    tabular = {k: v for k, v in sample.items() if k != "audio" and _is_tabular(v)}
    metadata = {
        k: v for k, v in sample.items() if k not in {"audio", "sample_rate", *tabular, *internal}
    }

    data_out = dict(internal)
    data_out[f"audio.{format.lower()}"] = audio_buffer.getvalue()
    for key, tab in tabular.items():
        data_out[f"{key}.parquet"] = _tabular_to_parquet_bytes(tab)

    data_out["metadata.json"] = json.dumps(metadata, indent=2).encode("utf-8")
    return data_out


def audio_decoder(data: dict, dtype: str = "float32", format: str = "FLAC") -> dict[str, Any]:
    """Decode audio data from a WebDataset sample.

    Parquet files stored alongside the audio (e.g., `selection_table.parquet`)
    are decoded back to `pd.DataFrame` and included in the returned sample.
    WebDataset internal keys (`__key__`, `__url__`, ...) are passed through
    unchanged. The returned `sample_rate` is always the rate read from the audio
    file, even if `metadata.json` carries a different one.

    Parameters
    ----------
    data: dict
        The sample containing audio data in WebDataset format
    dtype: str
        The data type of the decoded audio data (default: "float32")
    format: str
        The format of the audio data (default: "FLAC")

    Returns
    -------
    dict
        Dictionary containing the decoded audio data and metadata.
        Tabular fields are returned as `pd.DataFrame`.

    Raises
    ------
    ValueError
        If the sample does not contain an audio key ending with the extension
        implied by `format`.
    """
    suffix = f".{format.lower()}"
    audio_key = next((k for k in data if k.endswith(suffix)), None)
    if not audio_key:
        raise ValueError(
            f"Sample must contain an audio key ending with '{suffix}' "
            f"(format={format!r}); got keys {sorted(data)}"
        )

    audio_buffer = io.BytesIO(data[audio_key])
    audio_data, samplerate = sf.read(audio_buffer, dtype=dtype)

    # Reconstruct sample. Metadata is applied first so that the audio file
    # itself stays authoritative for the sample rate.
    sample = json.loads(data.get("metadata.json", b"{}").decode("utf-8"))
    sample.update(_wds_internal_keys(data))
    sample["audio"] = audio_data
    sample["sample_rate"] = samplerate

    for key, value in data.items():
        if key.endswith(".parquet"):
            field_name = key[: -len(".parquet")]
            sample[field_name] = _parquet_bytes_to_dataframe(value)

    return sample


def json_encoder(
    sample: dict[str, Any],
    indent: int = 2,
) -> dict[str, Any]:
    """Encode a sample to JSON format.

    Tabular fields (`pd.DataFrame`, `pl.DataFrame`, `pl.LazyFrame`, `pa.Table`)
    are stored as individual Parquet files named `{key}.parquet` alongside
    `sample.json`. WebDataset internal keys (`__key__`, `__url__`, ...) are
    passed through unchanged rather than written into `sample.json`, so the
    result can be handed to `wds.TarWriter.write`.

    Parameters
    ----------
    sample: dict[str, Any]
        The sample to encode
    indent: int
        Indentation level for JSON (default: 2)

    Returns
    -------
    dict
        Dictionary containing the encoded sample in JSON format.
        Tabular fields are stored as separate `{key}.parquet` entries.
    """
    internal = _wds_internal_keys(sample)
    tabular = {k: v for k, v in sample.items() if _is_tabular(v)}
    non_tabular = {k: v for k, v in sample.items() if k not in {*tabular, *internal}}

    data_out = dict(internal)
    for key, tab in tabular.items():
        data_out[f"{key}.parquet"] = _tabular_to_parquet_bytes(tab)

    data_out["sample.json"] = json.dumps(non_tabular, indent=indent).encode("utf-8")
    return data_out


def json_decoder(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Decode a sample from JSON format.

    Parquet files stored alongside `sample.json` (e.g., `selection_table.parquet`)
    are decoded back to `pd.DataFrame` and included in the returned sample.
    WebDataset internal keys (`__key__`, `__url__`, ...) are passed through
    unchanged.

    Parameters
    ----------
    data: dict[str, Any]
        The sample containing JSON data

    Returns
    -------
    dict
        Dictionary containing the decoded sample.
        Tabular fields are returned as `pd.DataFrame`.

    Raises
    ------
    ValueError
        If the sample does not contain a "sample.json" key.
    """
    if "sample.json" not in data:
        raise ValueError("Sample must contain 'sample.json' key with JSON data")

    sample = json.loads(data["sample.json"].decode("utf-8"))
    sample.update(_wds_internal_keys(data))

    for key, value in data.items():
        if key.endswith(".parquet"):
            field_name = key[: -len(".parquet")]
            # FIXME: why are we assuming we only return a pandas dataframe ?
            sample[field_name] = _parquet_bytes_to_dataframe(value)

    return sample
