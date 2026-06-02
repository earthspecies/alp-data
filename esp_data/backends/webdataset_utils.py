"""Utilities for encoding and decoding audio and JSON data for use in the WebDataset format."""

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import soundfile as sf

from esp_data.io import AnyPathT, filesystem_from_path
from esp_data.io.paths import PureCloudPath, anypath


def make_file_opener_for_wds(
    file_path: str | AnyPathT,
    mode: str = "wb",
    block_size: int = 1024 * 1024 * 100,
) -> callable:
    """Make a file opener function for WebDataset.

    If local path, create parent dirs if needed.

    Arguments
    ---------
    file_path: str | AnyPathT
        The file path to open
    mode: str
        The mode in which to open the file (default: "wb")
    block_size: int
        Block size for WebDataset (default: 100 MB)

    Returns
    -------
    Callable
        A function that opens the file in the specified mode
        or a file object if the path is local.
    """
    path_obj = anypath(file_path)

    if not isinstance(path_obj, PureCloudPath):
        # Local filesystem - create parent dirs if needed
        parent_dir = path_obj.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        return open(str(path_obj), mode=mode)
    else:
        # Remote filesystem (GCS, R2, etc.)
        fs = filesystem_from_path(str(path_obj))
        return fs.open(str(path_obj), mode=mode, block_size=block_size)


def _is_tabular(v: object) -> bool:
    """Return True if `v` is a tabular type (pandas/polars DataFrame or PyArrow Table).

    Returns
    -------
    bool
        True if `v` is a `pd.DataFrame`, `pl.DataFrame`, `pl.LazyFrame`, or `pa.Table`.
    """
    import pyarrow as pa

    if isinstance(v, (pl.DataFrame, pl.LazyFrame)):
        return True
    return isinstance(v, (pd.DataFrame, pa.Table))


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
    import pyarrow as pa
    import pyarrow.parquet as pq

    if isinstance(v, pl.LazyFrame):
        v = v.collect()
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

    Returns
    -------
    pd.DataFrame
        Decoded tabular data.
    """
    import pyarrow.parquet as pq

    return pq.read_table(io.BytesIO(data)).to_pandas()


def audio_encoder(
    sample: dict[str, Any],
    sample_rate: int = 16000,
    dtype: str = "float32",
    format: str = "FLAC",
) -> dict[str, Any]:
    """Encode audio data in the sample to a specific format.

    Non-audio tabular fields (`pd.DataFrame`, `pl.DataFrame`, `pl.LazyFrame`,
    `pa.Table`) are stored as individual Parquet files named `{key}.parquet`.
    All remaining non-audio fields are stored in `metadata.json`.

    Parameters
    ----------
    sample: dict[str, Any]
        The sample containing audio data
    sample_rate: int
        The sample rate of the audio data
    dtype: str
        The data type of the audio data (default: "float32")
    format: str
        The format to encode the audio data to (e.g., "WAV", "FLAC", "OGG")
        Default is "FLAC".

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

    data_out = {}
    audio_buffer = io.BytesIO()
    audio = sample["audio"]
    if isinstance(audio, (list, tuple)):
        audio = np.array(audio, dtype=dtype)
    elif isinstance(audio, np.ndarray):
        audio = audio.astype(dtype)

    sf.write(audio_buffer, audio, sample_rate, format=format)

    data_out[f"audio.{format.lower()}"] = audio_buffer.getvalue()

    tabular = {k: v for k, v in sample.items() if k != "audio" and _is_tabular(v)}
    metadata = {k: v for k, v in sample.items() if k not in ("audio", *tabular)}

    for key, tab in tabular.items():
        data_out[f"{key}.parquet"] = _tabular_to_parquet_bytes(tab)

    data_out["metadata.json"] = json.dumps(metadata, indent=2).encode("utf-8")
    return data_out


def audio_decoder(data: dict, dtype: str = "float32", format: str = "FLAC") -> dict[str, Any]:
    """Decode audio data from a WebDataset sample.

    Parquet files stored alongside the audio (e.g., `selection_table.parquet`)
    are decoded back to `pd.DataFrame` and included in the returned sample.

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
        If the sample does not contain an audio key ending with .flac, .wav, etc.
    """
    audio_key = next((k for k in data if k.endswith(f".{format.lower()}")), None)
    if not audio_key:
        raise ValueError("Sample must contain an audio key ending with .flac, .wav, etc.")

    audio_buffer = io.BytesIO(data[audio_key])
    audio_data, samplerate = sf.read(audio_buffer, dtype=dtype)

    # Reconstruct sample
    sample = {}
    sample["audio"] = audio_data
    sample["sample_rate"] = samplerate
    md = json.loads(data.get("metadata.json", b"{}").decode("utf-8"))
    sample.update(md)

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
    `sample.json`.

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
    tabular = {k: v for k, v in sample.items() if _is_tabular(v)}
    non_tabular = {k: v for k, v in sample.items() if k not in tabular}

    data_out = {}
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

    for key, value in data.items():
        if key.endswith(".parquet"):
            field_name = key[: -len(".parquet")]
            sample[field_name] = _parquet_bytes_to_dataframe(value)

    return sample
