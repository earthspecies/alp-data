"""Utilities for encoding and decoding audio and JSON data for use in the WebDataset format."""

import functools
import io
import itertools
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import soundfile as sf
import webdataset as wds

from esp_data.io import AnyPathT, filesystem_from_path
from esp_data.io.paths import PureGSPath, PureR2Path


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
    

def audio_encoder(
    sample: dict[str, Any],
    sample_rate: int = 16000,
    dtype: str = "float32",
    format: str = "FLAC",
) -> dict[str, Any]:
    """Encode audio data in the sample to a specific format.

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

    metadata = {k: v for k, v in sample.items() if k != "audio"}
    data_out["metadata.json"] = json.dumps(metadata, indent=2).encode("utf-8")
    return data_out


def audio_decoder(data: dict, dtype: str = "float32", format: str = "FLAC") -> dict[str, Any]:
    """Decode audio data from a WebDataset sample.

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

    return sample


def json_encoder(
    sample: dict[str, Any],
    indent: int = 2,
) -> dict[str, Any]:
    """Encode a sample to JSON format.

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
    """
    json_data = json.dumps(sample, indent=indent).encode("utf-8")
    return {"sample.json": json_data}


def json_decoder(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Decode a sample from JSON format.

    Parameters
    ----------
    data: dict[str, Any]
        The sample containing JSON data

    Returns
    -------
    dict
        Dictionary containing the decoded sample.

    Raises
    ------
    ValueError
        If the sample does not contain a "sample.json" key.
    """
    if "sample.json" not in data:
        raise ValueError("Sample must contain 'sample.json' key with JSON data")

    json_data = json.loads(data["sample.json"].decode("utf-8"))
    return json_data


def write_to_webdataset(
    sample_iter: Iterator[dict[str, Any]] | Iterable[dict[str, Any]],
    path: AnyPathT,
    encoder_fn: Callable | None = audio_encoder,
    shard_pattern: str = "shard_%04d.tar",
    maxcount: int = 100_000,
    maxsize: float = 3e9,
) -> int:
    """Write samples from an iterator or iterable to sharded WebDataset tar files.

    For cloud paths (GCS, R2), samples are written to a temporary local
    directory first, then uploaded to the destination.

    Parameters
    ----------
    sample_iter : Iterator[dict[str, Any]] | Iterable[dict[str, Any]]
        Iterator or iterable of sample dicts to write
    path : AnyPathT
        Destination directory (local or cloud). Created if it does not exist.
        Callers must resolve strings via `anypath` before passing so cloud
        URIs (``gs://``, ``r2://``) are detected correctly.
    encoder_fn : Callable | None, optional
        Function ``dict[str, Any] -> dict[str, bytes]`` in WebDataset format.
        If ``None``, auto-detected from the first sample: samples with an
        ``"audio"`` key use `audio_encoder`, all others use `json_encoder`.
    shard_pattern : str, optional
        Printf-style shard file name pattern, by default ``"shard_%04d.tar"``.
    maxcount : int, optional
        Maximum samples per shard, by default 100 000.
    maxsize : float, optional
        Maximum shard size in bytes, by default 3 GB.

    Returns
    -------
    int
        Number of samples written.
    """

    # Normalise iterables to iterators so next() works unconditionally
    sample_iter = iter(sample_iter)

    # PureR2Path covers both R2 and S3 URIs (anypath("s3://...") returns PureR2Path)
    is_cloud = isinstance(path, (PureGSPath, PureR2Path))

    if encoder_fn is None:
        try:
            first = next(sample_iter)
        except StopIteration:
            return 0
        if "audio" in first:
            sr = first.get("sample_rate", 16000)
            encoder_fn = functools.partial(audio_encoder, sample_rate=sr)
        else:
            encoder_fn = json_encoder
        sample_iter = itertools.chain([first], sample_iter)

    def _write(write_path: Path) -> int:
        write_path.mkdir(parents=True, exist_ok=True)
        pattern = str(write_path / shard_pattern)
        count = 0
        with wds.ShardWriter(pattern, maxcount=maxcount, maxsize=maxsize, verbose=0) as sink:
            for idx, sample in enumerate(sample_iter):
                encoded = encoder_fn(sample)
                encoded["__key__"] = f"sample_{idx:06d}"
                sink.write(encoded)
                count = idx + 1
        return count

    if is_cloud:
        with tempfile.TemporaryDirectory() as tmp_dir:
            count = _write(Path(tmp_dir))
            fs = filesystem_from_path(path)
            for local_file in Path(tmp_dir).iterdir():
                fs.put(str(local_file), str(path / local_file.name))
    else:
        count = _write(Path(str(path)))

    return count