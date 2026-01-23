"""Utilities for encoding and decoding audio and JSON data for use in the WebDataset format."""

import io
import json
from typing import Any

import numpy as np
import soundfile as sf


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
    md = json.loads(data.get("metadata.json", "{}").decode("utf-8"))
    sample.update(md)

    return sample


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
    # Convert audio data to the specified format
    if isinstance(sample["audio"], (list, tuple)):
        # If audio is a list or tuple, convert to numpy array
        sample["audio"] = np.array(sample["audio"], dtype=dtype)
    elif isinstance(sample["audio"], np.ndarray):
        # If audio is already a numpy array, ensure it's the correct dtype
        sample["audio"] = sample["audio"].astype(dtype)

    sf.write(audio_buffer, sample["audio"], sample_rate, format=format)

    data_out[f"audio.{format.lower()}"] = audio_buffer.getvalue()

    # Add metadata (without audio)
    sample = {k: v for k, v in sample.items() if k != "audio"}  # Remove audio key from metadata
    data_out["metadata.json"] = json.dumps(sample, indent=2).encode("utf-8")
    return data_out


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
