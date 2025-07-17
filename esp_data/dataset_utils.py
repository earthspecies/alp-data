import io
import json
from typing import Any, Callable

import numpy as np
import soundfile as sf
import webdataset as wds

from esp_data.io import AnyPathT, anypath


def audio_decoder(data: dict, dtype: str = "float32", format: str = "FLAC") -> dict[str, Any]:
    """Decode audio data from a WebDataset sample.

    Parameters
    ----------
    data: dict
        The sample containing audio data in WebDataset format
    format: str
        The format of the audio data (default: "FLAC")

    Returns
    -------
        dict: Dictionary containing the decoded audio data and metadata.

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
        dict: Dictionary containing the encoded audio data and metadata
            in the WebDataset format.

    Raises
    ------
        ValueError: If the sample does not contain an "audio" key
            with audio data.
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
        dict: Dictionary containing the encoded sample in JSON format.
    """
    json_data = json.dumps(sample, indent=indent).encode("utf-8")
    return {"sample.json": json_data}


def load_webdataset(
    path: str | AnyPathT,
    file_pattern: str = "shard*tar",
    data_processor: Callable = None,
    shuffle_size: int | None = None,
    batch_size: int | None = None,
    shard_shuffle: bool = False,
    shard_shuffle_size: int = 1000,
    split_by_worker: bool = False,
    batch_collate_fn: Callable = None,
    seed: int | bool | None = 42,
) -> wds.WebDataset:
    """Create a pipeline for loading the dataset

    Arguments
    ---------
    path: str | AnyPath
            Path to the directory where the sharded dataset will be stored or
            is already stored.
    file_pattern: str, optional
        Pattern to match the shard files.
    data_processor: Callable, optional
        Function to process the data.
    shuffle_size: int, optional
        Size of the shuffle buffer.
    batch_size: int, optional
        Batch size for processing audio files.
    shard_shuffle: bool, optional
        Whether to shuffle the shards.
    shard_shuffle_size: int, optional
        Size of the shuffle buffer for shards.
    split_by_worker: bool, optional
        Whether to split the dataset by worker.
    batch_collate_fn: Callable, optional
        Function to collate the batch.
    seed Union[int, bool, None]:
        Seed for shuffling. Defaults to True, random seed. If None, means no shuffling!

    Returns
    -------
        wds.WebDataset: WebDataset object

    Raises
    ------
    FileNotFoundError
        If no shard files are found in the specified path.
    """
    path = anypath(path)
    shard_files = list([str(s) for s in path.glob(file_pattern)])

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {path}")

    webds_kwargs = {"shardshuffle": shard_shuffle_size if shard_shuffle else False}
    if shard_shuffle and seed is not None:
        webds_kwargs["seed"] = seed
    if split_by_worker:
        webds_kwargs["workersplitter"] = wds.split_by_worker

    webds = wds.WebDataset(shard_files, **webds_kwargs)

    if shuffle_size:
        webds = webds.shuffle(shuffle_size, seed=seed if seed is not None else 42)
    if data_processor:
        webds = webds.map(data_processor)
    if batch_size is not None:
        webds = webds.batched(batch_size, collation_fn=batch_collate_fn)

    return webds
