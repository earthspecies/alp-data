"""This file offers functionalities necessary to read input streams, like audio."""

import io
import logging

import numpy as np
import soundfile as sf

from .paths import AnyPathT, anypath

logger = logging.getLogger("esp_data")

_AUDIO_FORMATS = (".wav", ".flac", ".ogg", ".mp3")


def _read_audio_from_bytes(audio_bytes: bytes, frames: int = -1, start: int = 0) -> tuple[np.ndarray, int]:
    """
    Reads from an audio buffer while indexing if necessary. By default,
    reads the entire buffer.

    Parameters
    ----------
    audio_bytes : bytes
        The byte string containing the encoded audio data (e.g., WAV, FLAC).
    frames : int, optional
        The number of frames to read. -1 reads all frames from the
        `start` position to the end of the file. Defaults to -1.
    start : int, optional
        The frame index to start reading from. Defaults to 0.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - data (np.ndarray): The audio data as a NumPy array. The shape will
          be (frames,) for mono or (frames, channels) for multi-channel audio.
        - samplerate (int): The sample rate of the audio in Hz.
    """

    with io.BytesIO(audio_bytes) as audio_buffer:
        data, samplerate = sf.read(audio_buffer, frames=frames, start=start)
    return data, samplerate


def read_audio(file_path: str | AnyPathT, frames: int = -1, start: int = 0) -> tuple[np.ndarray, int]:
    """Reads audio data from a file path.

    Handles various path types (local, GCS, R2) via the `anypath` utility.
    Checks if the file extension is a supported audio format.
    Allows specifying a number of frames to read and a starting frame offset.

    Parameters
    ----------
    file_path : str or AnyPathT
        The path string or path object (e.g., Path, GSPath, R2Path) pointing
        to the audio file.
    frames : int, optional
        The number of frames to read. -1 reads all frames from the
        `start` position to the end of the file. Defaults to -1.
    start : int, optional
        The frame index to start reading from. Defaults to 0.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - data (np.ndarray): The audio data as a NumPy array. The shape will
          be (frames,) for mono or (frames, channels) for multi-channel audio.
        - samplerate (int): The sample rate of the audio in Hz.

    Raises
    ------
    ValueError
        If the file extension is not in the supported `_AUDIO_FORMATS`.

    Examples
    --------
    >>> audio, sr = read_audio("tests/samples/noise.wav")
    >>> audio.shape
    (524288,)
    >>> sr
    16000
    """
    file_path = anypath(file_path)
    extension = file_path.suffix

    if extension not in _AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}")

    try:
        with file_path.open("rb") as f:
            return _read_audio_from_bytes(f.read(), frames, start)
    except Exception as e:
        logger.error(f"Error reading audio file {e}")
        raise e
