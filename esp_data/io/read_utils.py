"""This file offers functionalities necessary to read input streams, like audio."""

import json
import logging
import tempfile
from typing import Any, BinaryIO, Literal

import numpy as np
import soundfile as sf
import yaml
from soundfile import LibsndfileError

from esp_data.io.filesystem import filesystem_from_path
from esp_data.io.paths import AnyPathT, anypath

logger = logging.getLogger("esp_data")

_AUDIO_FORMATS = (".wav", ".flac", ".ogg", ".mp3")
_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")
_VIDEO_FORMATS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def _read_audio_from_tmpfile(
    audio_bytes: bytes, format: str, frames: int = -1, start: int = 0
) -> tuple[np.ndarray, int]:
    """
    Read audio from bytes using a temporary file.

    This is needed for formats like MP3 where soundfile/libsndfile cannot
    read from BytesIO objects with format specification.

    Parameters
    ----------
    audio_bytes : bytes
        The byte string containing the encoded audio data.
    format : str
        The audio format (e.g., 'WAV', 'FLAC', 'MP3').
    frames : int, optional
        The number of frames to read. -1 reads all frames from the
        `start` position to the end of the file. Defaults to -1.
    start : int, optional
        The frame index to start reading from. Defaults to 0.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - data (np.ndarray): The audio data as a NumPy array.
        - samplerate (int): The sample rate of the audio in Hz.
    """
    with tempfile.NamedTemporaryFile(suffix=f".{format.lower()}", delete=True) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file.flush()
        data, samplerate = sf.read(tmp_file.name, frames=frames, start=start)
    return data, samplerate


def read_text(
    file_path: str | AnyPathT,
    encoding: str | None = None,
    errors: str | None = None,
) -> str:
    """
    Open the file in text mode, read it, and close the file.

    Parameters
    ----------
    file_path : str or AnyPathT
        The path string or path object pointing to the text file.
    encoding : str or None, optional
        The encoding to use for the file. Defaults to None.
    errors : str or None, optional
        The error handling mode. Defaults to None.

    Returns
    -------
    str
        The contents of the file.
    """

    with filesystem_from_path(file_path).open(
        str(file_path), "rt", encoding=encoding, errors=errors
    ) as f:
        return f.read()


def read_yaml(path: str | AnyPathT) -> object:
    """Read a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : str or AnyPathT
        The path string or path object pointing to the YAML file.

    Returns
    -------
    object
        The contents of the YAML file.

    Raises
    ------
    yaml.YAMLError
        If there is an error parsing the YAML file.
    ValueError
        If the YAML file is empty.
    """
    try:
        with filesystem_from_path(path).open(str(path), "r") as fp:
            result = yaml.safe_load(fp)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file '{path}': {e}") from e

    if result is None:
        raise ValueError(f"YAML file '{path}' is empty")

    return result


def read_json(path: str | AnyPathT) -> object:
    """Read a JSON file and return its contents.

    Parameters
    ----------
    path : str or AnyPathT
        The path string or path object pointing to the JSON file.

    Returns
    -------
    object
        The contents of the JSON file.

    Raises
    ------
    json.JSONDecodeError
        If there is an error parsing the JSON file.
    ValueError
        If the JSON file is empty.
    """
    try:
        with filesystem_from_path(path).open(str(path), "r") as fp:
            result = json.load(fp)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Error parsing JSON file '{path}': {e.msg}", e.doc, e.pos
        ) from e

    if result is None:
        raise ValueError(f"JSON file '{path}' is empty")

    return result


def _read_audio_from_file(
    audio_file: BinaryIO, frames: int = -1, start: int = 0, format: str | None = None
) -> tuple[np.ndarray, int]:
    """Reads from an audio buffer while indexing if necessary. By default,
    reads the entire buffer.

    Parameters
    ----------
    audio_file : BinaryIO
        The audio file-like object containing the encoded audio data.
    frames : int, optional
        The number of frames to read. -1 reads all frames from the
        `start` position to the end of the file. Defaults to -1.
    start : int, optional
        The frame index to start reading from. Defaults to 0.
    format : str or None, optional
        The audio format (e.g., 'WAV', 'FLAC', 'MP3'). If None, soundfile
        will attempt to auto-detect. Defaults to None.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - data (np.ndarray): The audio data as a NumPy array. The shape will
          be (frames,) for mono or (frames, channels) for multi-channel audio.
        - samplerate (int): The sample rate of the audio in Hz.
    """
    try:
        data, samplerate = sf.read(audio_file, frames=frames, start=start)
        return data, samplerate
    except LibsndfileError as e:
        logger.warning(
            "Failed to read audio from file-like object directly, "
            f"falling back to temporary file method: {e}"
        )
        # Fallback to temporary file if BytesIO approach fails.
        # For formats like MP3, soundfile cannot read from BytesIO with format
        # specification due to libsndfile limitations.
        if format:
            audio_file.seek(0)
            return _read_audio_from_tmpfile(audio_file.read(), format, frames, start)


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
    start_time : float | None, optional
        Start time in seconds. Defaults to None.
    end_time : float | None, optional
        End time in seconds. If None, reads to end of file.
    input_sr : int | None, optional
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

    if extension.lower() not in _AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}")

    if start_time is not None:
        return read_audio_by_time(file_path, start_time, end_time, input_sr)

    try:
        fs = filesystem_from_path(file_path)
        with fs.open(str(file_path), "rb") as f:
            audio_format = extension.lstrip(".").upper()
            return _read_audio_from_file(f, frames, start, format=audio_format)
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

    if extension.lower() not in _AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}")

    try:
        with filesystem_from_path(file_path).open(str(file_path), "rb") as f:
            info = sf.info(f)
    except LibsndfileError as e:
        logger.warning(
            "Failed to read audio from file-like object directly, "
            f"falling back to temporary file method: {e}"
        )
        with filesystem_from_path(file_path).open(str(file_path), "rb") as f:
            file_bytes = f.read()
        with tempfile.NamedTemporaryFile(suffix=extension, delete=True) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file.flush()
            info = sf.info(tmp_file.name)

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
    end_time: float | None = None,
    input_sr: int | None = None,
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
    format = extension.lstrip(".").upper()

    if extension.lower() not in _AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}")

    try:
        fs = filesystem_from_path(file_path)
        fp = fs.open(str(file_path), "rb")

        try:
            info = sf.info(fp)
            file_sr = info.samplerate
            total_frames = info.frames
        except LibsndfileError:
            # Use temporary file for MP3/OGG
            with tempfile.NamedTemporaryFile(suffix=extension, delete=True) as tmp_file:
                tmp_file.write(fp.read())
                tmp_file.flush()
                info = sf.info(tmp_file.name)
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
        fp.seek(0)
        try:
            data, samplerate = sf.read(fp, frames=frames_to_read, start=start_frame, format=None)
        except LibsndfileError as e:
            logger.warning(
                "Failed to read audio from file-like object directly, "
                f"falling back to temporary file method: {e}"
            )
            data, samplerate = _read_audio_from_tmpfile(
                fp.read(),
                format=format,
                frames=frames_to_read,
                start=start_frame,
            )

        return data, samplerate

    except Exception as e:
        logger.error(f"Error reading audio file {e}")
        raise e


def read_image(file_path: str | AnyPathT, mode: str | None = "RGB") -> np.ndarray:
    """Read an image file into a NumPy array.

    Handles various path types (local, GCS, R2) via the `filesystem_from_path`
    utility, mirroring `read_audio`. Decoding is done with Pillow.

    Parameters
    ----------
    file_path : str or AnyPathT
        The path string or path object pointing to the image file.
    mode : str or None, optional
        Pillow image mode to convert to before returning (e.g., `"RGB"`,
        `"L"`). If None, the image is returned in its native mode. Defaults
        to `"RGB"`.

    Returns
    -------
    np.ndarray
        The image as a uint8 NumPy array in HWC layout. Shape is
        ``(height, width, channels)`` for multi-channel images or
        ``(height, width)`` for single-channel images.

    Raises
    ------
    ValueError
        If the file extension is not in the supported `_IMAGE_FORMATS`.
    """
    from PIL import Image

    file_path = anypath(file_path)
    extension = file_path.suffix

    if extension.lower() not in _IMAGE_FORMATS:
        raise ValueError(f"Unsupported image format: {extension}")

    try:
        fs = filesystem_from_path(file_path)
        with fs.open(str(file_path), "rb") as f:
            img = Image.open(f)
            if mode is not None:
                img = img.convert(mode)
            return np.asarray(img)
    except Exception as e:
        logger.error(f"Error reading image file {e}")
        raise e


def read_video(
    file_path: str | AnyPathT,
    max_frames: int | None = None,
    target_fps: float | None = None,
    with_audio: bool = True,
) -> dict[str, Any]:
    """Read a video file into frames plus its aligned audio track.

    Handles various path types (local, GCS, R2) via the `filesystem_from_path`
    utility, mirroring `read_audio`. Decoding uses PyAV (`av`), which is an
    optional dependency installed via the ``video`` extra. The import is
    performed lazily so audio-only installs are unaffected.

    PyAV demuxes the video frames and the embedded audio track in a single
    pass, so the returned frames and audio originate from the same file and
    are time-aligned.

    Parameters
    ----------
    file_path : str or AnyPathT
        The path string or path object pointing to the video file.
    max_frames : int or None, optional
        Maximum number of video frames to decode. If None, all frames are
        decoded. Defaults to None.
    target_fps : float or None, optional
        If provided, video frames are subsampled to approximately this frame
        rate (the audio track is never subsampled). If None, every frame is
        returned. Defaults to None.
    with_audio : bool, optional
        Whether to decode and return the aligned audio track. Defaults to True.

    Returns
    -------
    dict[str, Any]
        A dictionary with keys:
        - ``frames`` (np.ndarray): Video frames as a uint8 array in TToHWC
          layout with shape ``(num_frames, height, width, 3)``.
        - ``audio`` (np.ndarray or None): The aligned audio as a float32
          array of shape ``(num_samples,)`` (mono) or
          ``(num_samples, channels)``, or None when ``with_audio`` is False
          or the video has no audio track.
        - ``fps`` (float): The (effective) video frame rate.
        - ``sample_rate`` (int or None): The audio sample rate in Hz, or None
          when no audio was decoded.

    Raises
    ------
    ImportError
        If PyAV (`av`) is not installed. Install the ``video`` extra.
    ValueError
        If the file extension is not in the supported `_VIDEO_FORMATS`.
    """
    try:
        import av
    except ImportError as e:
        raise ImportError(
            "read_video requires PyAV. Install the optional 'video' extra, "
            "e.g. `uv sync --extra video` or `pip install av`."
        ) from e

    file_path = anypath(file_path)
    extension = file_path.suffix

    if extension.lower() not in _VIDEO_FORMATS:
        raise ValueError(f"Unsupported video format: {extension}")

    try:
        fs = filesystem_from_path(file_path)
        with fs.open(str(file_path), "rb") as f:
            with av.open(f) as container:
                frames, fps = _decode_video_frames(container, max_frames, target_fps)
                audio, sample_rate = (None, None)
                if with_audio and container.streams.audio:
                    audio, sample_rate = _decode_video_audio(container)
        return {
            "frames": frames,
            "audio": audio,
            "fps": fps,
            "sample_rate": sample_rate,
        }
    except Exception as e:
        logger.error(f"Error reading video file {e}")
        raise e


def _decode_video_frames(
    container: object,
    max_frames: int | None,
    target_fps: float | None,
) -> tuple[np.ndarray, float]:
    """Decode (optionally subsampled) video frames from an open PyAV container.

    Parameters
    ----------
    container : av.container.InputContainer
        An open PyAV container positioned at the start of the video.
    max_frames : int or None
        Maximum number of frames to return, or None for all frames.
    target_fps : float or None
        Approximate target frame rate for subsampling, or None to keep all
        frames.

    Returns
    -------
    tuple[np.ndarray, float]
        The decoded frames as a uint8 array of shape
        ``(num_frames, height, width, 3)`` and the effective frame rate.
    """
    stream = container.streams.video[0]
    src_fps = float(stream.average_rate) if stream.average_rate else 0.0
    step = 1
    if target_fps is not None and src_fps > target_fps > 0:
        step = max(1, int(round(src_fps / target_fps)))

    frames: list[np.ndarray] = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % step != 0:
            continue
        frames.append(frame.to_ndarray(format="rgb24"))
        if max_frames is not None and len(frames) >= max_frames:
            break

    effective_fps = src_fps / step if src_fps else 0.0
    if frames:
        return np.stack(frames, axis=0), effective_fps
    return np.empty((0, 0, 0, 3), dtype=np.uint8), effective_fps


def _decode_video_audio(
    container: object,
) -> tuple[np.ndarray | None, int | None]:
    """Decode the embedded audio track from an open PyAV container.

    Decoding through PyAV (rather than soundfile) is necessary because
    libsndfile does not read the AAC/MP4 audio commonly muxed into video
    containers. Multi-channel audio is returned interleaved as
    ``(num_samples, channels)``; mono is flattened to ``(num_samples,)``.

    Parameters
    ----------
    container : av.container.InputContainer
        An open PyAV container with at least one audio stream.

    Returns
    -------
    tuple[np.ndarray or None, int or None]
        The float32 audio samples and sample rate, or ``(None, None)`` if the
        audio track could not be decoded.
    """
    try:
        stream = container.streams.audio[0]
        sample_rate = stream.rate
        # Frame decoding has advanced (or exhausted) the demuxer; rewind so the
        # full audio track is read from the start of the container.
        container.seek(0)
        chunks: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            # PyAV returns (channels, samples); transpose to (samples, channels).
            chunks.append(frame.to_ndarray().T)
        if not chunks:
            return None, sample_rate
        audio = np.concatenate(chunks, axis=0).astype(np.float32)
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio[:, 0]
        return audio, sample_rate
    except Exception as e:  # noqa: BLE001  audio is best-effort for video
        logger.warning(f"Could not decode aligned video audio: {e}")
        return None, None
