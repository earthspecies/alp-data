"""This file offers functionalities necessary to read input streams, like audio."""

import io
import logging
from typing import Any, Literal, Optional

import numpy as np
import soundfile as sf

from .paths import AnyPathT, anypath

logger = logging.getLogger("esp_data")

_AUDIO_FORMATS = (".wav", ".flac", ".ogg", ".mp3")


def _read_audio_from_bytes(
    audio_bytes: bytes, frames: int = -1, start: int = 0
) -> tuple[np.ndarray, int]:
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


def read_audio(
    file_path: str | AnyPathT,
    frames: int = -1,
    start: int = 0,
    start_time: float | None = None,
    end_time: float | None = None,
    input_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """Reads audio data from a file path.

    Handles various path types (local, GCS, R2) via the `anypath` utility.
    Checks if the file extension is a supported audio format.
    Allows specifying a number of frames to read and a starting frame offset.
    Allow specifying a starting time (in seconds) to read from with an ending time.
    Frames and time slicing are not compatible.

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
    start_time : float, optional
        Start time in seconds. Defaults to None.
    end_time : float, optional
        End time in seconds. If None, reads to end of file.
    input_sr : int, optional
        Expected sample rate. If provided, used for validation.

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

    if start_time is not None:
        return read_audio_by_time(file_path, start_time, end_time, input_sr)

    try:
        with file_path.open("rb") as f:
            return _read_audio_from_bytes(f.read(), frames, start)
    except Exception as e:
        logger.error(f"Error reading audio file {e}")
        raise e


def audio_stereo_to_mono(
    audio: np.ndarray, mono_method: Literal["keep_first", "average"] = "average"
) -> np.ndarray:
    """Convert stereo audio to mono.

    Parameters
    ----------
    audio : np.ndarray
        The audio data as a NumPy array.
        The channel dimension can be either the first or second dimension.

    mono_method : Literal["keep_first", "average"]
        Method to convert stereo to mono:
        - "keep_first": Keep the first channel.
        - "average": Average both channels.
        Defaults to "average".

    Returns
    -------
    np.ndarray
        The converted mono audio data as a NumPy array with shape (frames,).

    Raises
    ------
    ValueError
        If the audio data is not stereo or if an unsupported
        mono conversion method is provided.

    Examples
    --------
    >>> audio, sr = read_audio("tests/samples/stereo.wav")
    >>> mono_audio = audio_stereo_to_mono(audio, mono_method="average")
    >>> mono_audio.shape
    (10000,)
    """
    if audio.ndim == 1:
        # Already mono, no conversion needed
        return audio

    # Throw error if more than stereo
    if audio.ndim > 2:
        raise ValueError("Audio must be stereo (2 channels) or mono (1 channel).")

    channel_dim = np.argmin(audio.shape)

    # transpose if the channel dimension is not the first one
    if channel_dim != 0:
        audio = audio.T

    if mono_method == "keep_first":
        return audio[0, :]
    elif mono_method == "average":
        return np.mean(audio, axis=0)
    else:
        raise ValueError(f"Unsupported mono conversion method: {mono_method}")


def get_audio_info(
    file_path: str | AnyPathT,
) -> dict[str, Any]:
    """Gets audio file information without loading the data.

    Parameters
    ----------
    file_path : str or AnyPathT
        The path string or path object pointing to the audio file.

    Returns
    -------
    dict[str, Any]
        A dictionary containing:
        - "sr": Sample rate in Hz.
        - "duration": Duration in seconds.
        - "num_frames": Total number of frames.
        - "num_channels": Number of audio channels.
        - "format": File format (e.g., WAV, FLAC).
        - "subtype": Subtype of the audio file.

    Raises
    ------
    ValueError
        If the file extension is not in the supported `_AUDIO_FORMATS`.

    Examples
    --------
    >>> info = get_audio_info("tests/samples/noise.wav")
    >>> info["sr"]
    16000
    """
    file_path = anypath(file_path)
    extension = file_path.suffix

    if extension not in _AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}")
    with file_path.open("rb") as f:
        info = sf.info(f)

    return {
        "sr": info.samplerate,
        "duration": info.duration,
        "num_frames": info.frames,
        "num_channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
    }


def read_audio_by_time(
    file_path: str | AnyPathT,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    input_sr: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """Reads audio data from a file path using time-based parameters.

    Parameters
    ----------
    file_path : str or AnyPathT
        The path string or path object pointing to the audio file.
    start_time : float, optional
        Start time in seconds. Defaults to 0.0.
    end_time : float, optional
        End time in seconds. If None, reads to end of file.
    input_sr : int, optional
        Expected sample rate. If provided, used for validation.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - data (np.ndarray): The audio data as a NumPy array.
        - samplerate (int): The sample rate of the audio in Hz.

    Raises
    ------
    ValueError
        If the file extension is not in the supported `_AUDIO_FORMATS`.

    Examples
    --------
    >>> audio, sr = read_audio_by_time("tests/samples/noise.wav",
    ...     start_time=1.0,
    ...     end_time=2.0)
    >>> audio.shape
    (16000,)
    """
    file_path = anypath(file_path)
    extension = file_path.suffix

    if extension not in _AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}")

    try:
        with file_path.open("rb") as f:
            file_bytes = f.read()

        # Get file info without loading audio data
        info = sf.info(io.BytesIO(file_bytes))
        file_sr = info.samplerate
        total_frames = info.frames

        # Validate input sample rate if provided
        if input_sr is not None and input_sr != file_sr:
            logger.warning(f"Input sample rate {input_sr} doesn't match file sample rate {file_sr}")

        # Convert time to frame indices
        start_frame = int(start_time * file_sr)

        if end_time is not None:
            end_frame = int(end_time * file_sr)
            frames_to_read = end_frame - start_frame
        else:
            frames_to_read = -1  # Read to end

        # Ensure we don't exceed file bounds
        start_frame = max(0, min(start_frame, total_frames))
        if frames_to_read > 0:
            frames_to_read = min(frames_to_read, total_frames - start_frame)

        # Read the actual audio data
        data, samplerate = sf.read(io.BytesIO(file_bytes), frames=frames_to_read, start=start_frame)

        return data, samplerate

    except Exception as e:
        logger.error(f"Error reading audio file {e}")
        raise e
