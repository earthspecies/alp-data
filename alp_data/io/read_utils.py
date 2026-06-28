"""This file offers functionalities necessary to read input streams, like audio."""

import json
import logging
import subprocess
import tempfile
from functools import lru_cache
from typing import Any, BinaryIO, Literal

import google.auth
import librosa
import numpy as np
import soundfile as sf
import yaml
from google.auth.transport.requests import Request
from soundfile import LibsndfileError

from alp_data.io.filesystem import filesystem_from_path
from alp_data.io.paths import AnyPathT, PureGSPath, anypath

logger = logging.getLogger("alp_data")

# Module-level cache of Google credentials. Reused across calls so the token is
# only refreshed once per session (and again whenever it expires).
_gcs_credentials = None


class GCSAuthError(Exception):
    """Raised when Google Cloud credentials cannot be obtained or refreshed."""


class FFmpegSegmentError(Exception):
    """Raised when the ffmpeg-based GCS segment read cannot complete.

    Carries a short, stable `cause` describing the failure category (e.g.
    ``"ffmpeg not installed"``) so callers can warn once per cause before
    falling back to the download-based read path.

    Parameters
    ----------
    cause : str
        Short, stable description of the failure category.
    message : str
        Full human-readable error message.
    """

    def __init__(self, cause: str, message: str) -> None:
        super().__init__(message)
        self.cause = cause


@lru_cache(maxsize=None)
def _warn_ffmpeg_fallback_once(cause: str) -> None:
    """Emit the ffmpeg-fallback warning at most once per distinct `cause`.

    The lru_cache means repeat calls with the same `cause` are no-ops, which
    keeps the log quiet in training/eval loops that read many files while still
    surfacing a genuinely different failure mode.

    Parameters
    ----------
    cause : str
        Short, stable description of why the ffmpeg path was unavailable.
    """
    logger.warning(
        "ffmpeg segment read unavailable (%s); falling back to full download. "
        "This is slower for large remote files.",
        cause,
    )


def get_gcs_token() -> str:
    """Fetch a valid access token using Google Application Default Credentials.

    Relies on the caller having authenticated with GCP, e.g. via
    `gcloud auth application-default login` or an ambient service account. The
    credentials are cached at module level and only refreshed when the current
    token is missing or expired.

    Returns
    -------
    str
        A valid access token for authenticating requests to Google Cloud.

    Raises
    ------
    GCSAuthError
        If credentials cannot be obtained or refreshed.
    """
    global _gcs_credentials
    try:
        if _gcs_credentials is None:
            _gcs_credentials, _ = google.auth.default()
        if not _gcs_credentials.valid:
            _gcs_credentials.refresh(Request())
        return _gcs_credentials.token
    except Exception as e:
        raise GCSAuthError(
            f"Error authenticating with Google Cloud: {e}.\n"
            "Ensure you have run 'gcloud auth application-default login' "
            "and have permission to access the GCS bucket."
        ) from e


def _gcs_path_to_url(file_path: str | AnyPathT) -> str:
    """Convert a GCS path to an HTTPS REST API URL.

    Accepts paths with or without the ``gs://`` prefix. E.g. ``gs://bucket/blob``
    and ``bucket/blob`` both become ``https://storage.googleapis.com/bucket/blob``.

    Parameters
    ----------
    file_path : str or AnyPathT
        The GCS path to convert.

    Returns
    -------
    str
        The corresponding HTTPS URL for accessing the file via the GCS REST API.

    Raises
    ------
    ValueError
        If the path uses an unsupported (``s3://`` or ``r2://``) scheme.
    """
    path_str = str(file_path)
    if path_str.startswith(("s3://", "r2://")):
        # TODO: Add support for R2 cloudflare paths
        raise ValueError(
            f"Unsupported storage scheme in path: {path_str}. Only GCS paths (gs://) are supported."
        )
    if path_str.startswith("gs://"):
        path_str = path_str[len("gs://") :]
    path_str = path_str.lstrip("/")
    return f"https://storage.googleapis.com/{path_str}"


def _read_audio_ffmpeg(
    file_path: str | AnyPathT,
    start_time: float = 0.0,
    end_time: float | None = None,
    anonymous: bool = False,
) -> tuple[np.ndarray, int]:
    """Read an audio segment from GCS using ffmpeg HTTP range requests.

    Streams only the requested segment directly from a GCS bucket via ffmpeg,
    avoiding a full-file download. Output follows the `soundfile.read`
    convention: ``(frames,)`` for mono or ``(frames, channels)`` for
    multi-channel audio.

    Parameters
    ----------
    file_path : str or AnyPathT
        GCS path to the audio file, with or without the ``gs://`` prefix.
    start_time : float, optional
        Start time in seconds. Defaults to 0.0.
    end_time : float or None, optional
        End time in seconds. If None, reads to the end of the file.
    anonymous : bool, optional
        If True, skip credential lookup and access the object without an
        ``Authorization`` header (for public objects). Defaults to False.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - data (np.ndarray): float32 audio data, shape ``(frames,)`` for mono
          or ``(frames, channels)`` for multi-channel.
        - samplerate (int): native sample rate of the audio in Hz.

    Raises
    ------
    FFmpegSegmentError
        If credentials are unavailable, the ffmpeg/ffprobe binaries are not
        installed, or ffprobe/ffmpeg fail to process the audio.
    """
    gcs_url = _gcs_path_to_url(file_path)

    headers_args: list[str] = []
    if not anonymous:
        try:
            token = get_gcs_token()
        except GCSAuthError as e:
            raise FFmpegSegmentError("missing GCS credentials", str(e)) from e
        headers_args = ["-headers", f"Authorization: Bearer {token}\r\n"]

    # Probe native sample rate and channel count.
    probe_cmd = [
        "ffprobe",
        *headers_args,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "csv=p=0",
        gcs_url,
    ]
    try:
        probe_result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise FFmpegSegmentError("ffmpeg not installed", str(e)) from e
    except subprocess.CalledProcessError as e:
        raise FFmpegSegmentError(
            "ffprobe failed", f"ffprobe failed with return code {e.returncode}:\n{e.stderr}"
        ) from e

    probe_parts = probe_result.stdout.strip().split(",")
    sample_rate = int(probe_parts[0])
    channels = int(probe_parts[1])

    command = ["ffmpeg", *headers_args, "-ss", str(start_time), "-i", gcs_url]
    if end_time is not None:
        command += ["-t", str(end_time - start_time)]
    # Output raw PCM float32 to stdout, preserving native channel layout.
    command += ["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]

    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as e:
        raise FFmpegSegmentError("ffmpeg not installed", str(e)) from e
    except subprocess.CalledProcessError as e:
        raise FFmpegSegmentError(
            "ffmpeg decode failed",
            f"ffmpeg failed with return code {e.returncode}:\n{e.stderr.decode()}",
        ) from e

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if channels > 1:
        # Interleaved samples -> (frames, channels), matching soundfile.
        audio = audio.reshape(-1, channels)
    # Copy so the returned array is writable (np.frombuffer is read-only).
    return audio.copy(), sample_rate


def _read_audio_with_librosa(
    audio_file: BinaryIO, frames: int = -1, start: int = 0
) -> tuple[np.ndarray, int]:
    """Read audio from a file-like object using `librosa.load`.

    Used as a fallback when `soundfile.read` cannot decode the buffer directly
    (e.g. some MP3/OGG streams). Channels are preserved (``mono=False``). The
    returned array follows the `soundfile.read` convention: ``(frames,)`` for
    mono or ``(frames, channels)`` for multi-channel.

    Parameters
    ----------
    audio_file : BinaryIO
        The audio file-like object containing the encoded audio data. The
        stream is rewound to position 0 before reading.
    frames : int, optional
        The number of frames to read. -1 reads all frames from the `start`
        position to the end of the file. Defaults to -1.
    start : int, optional
        The frame index to start reading from. Defaults to 0.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - data (np.ndarray): The audio data as a NumPy array.
        - samplerate (int): The sample rate of the audio in Hz.
    """
    audio_file.seek(0)
    data, samplerate = librosa.load(audio_file, sr=None, mono=False)
    # librosa returns (channels, frames) for multi-channel; transpose to match
    # the (frames, channels) convention used by soundfile.
    if data.ndim == 2:
        data = data.T
    if start > 0 or frames > 0:
        end = start + frames if frames > 0 else None
        data = data[start:end]
    return data, samplerate


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

    Decoding is attempted in three stages: `soundfile.read` first, then
    `librosa.load` as a fallback, and finally a temporary-file decode if
    `format` is provided.

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
        return sf.read(audio_file, frames=frames, start=start)
    except LibsndfileError:
        try:
            return _read_audio_with_librosa(audio_file, frames=frames, start=start)
        except Exception:
            if format:
                audio_file.seek(0)
                return _read_audio_from_tmpfile(audio_file.read(), format, frames, start)
            raise


def _read_audio_by_time(
    file_path: str | AnyPathT,
    start_time: float = 0.0,
    end_time: float | None = None,
    input_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """Read an audio segment by time, downloading the file as needed.

    This is the fallback path for time-based segment reads: it opens the file
    (downloading remote files), determines the sample rate, converts the time
    range to frame indices, and decodes via `soundfile.read` with a librosa /
    temporary-file fallback for formats libsndfile cannot read from a buffer.

    Parameters
    ----------
    file_path : str or AnyPathT
        The path string or path object pointing to the audio file.
    start_time : float, optional
        Start time in seconds. Defaults to 0.0.
    end_time : float or None, optional
        End time in seconds. If None, reads to the end of the file.
    input_sr : int or None, optional
        Expected sample rate. If provided, used for validation (a warning is
        logged on mismatch).

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - data (np.ndarray): The audio data as a NumPy array.
        - samplerate (int): The sample rate of the audio in Hz.
    """
    file_path = anypath(file_path)
    extension = file_path.suffix
    format = extension.lstrip(".").upper()

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
        except LibsndfileError:
            try:
                data, samplerate = _read_audio_with_librosa(
                    fp, frames=frames_to_read, start=start_frame
                )
            except Exception:
                fp.seek(0)
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


def read_audio(
    file_path: str | AnyPathT,
    start_time: float | None = None,
    end_time: float | None = None,
    input_sr: int | None = None,
    anonymous: bool = False,
) -> tuple[np.ndarray, int]:
    """Reads audio data from a file path.

    Handles various path types (local, GCS, R2) via the `anypath` utility.
    Reads the entire file by default, or a time range when `start_time` (and
    optionally `end_time`) is given.

    When a time range is requested on a GCS (``gs://``) path, the segment is
    streamed directly via ffmpeg HTTP range requests, avoiding a full-file
    download. If ffmpeg is unavailable (binary missing, no credentials, or a
    decode error), the read falls back to downloading the file and decoding the
    segment, logging a warning once per distinct cause.

    Parameters
    ----------
    file_path : str or AnyPathT
        The path string or path object (e.g., Path, GSPath, R2Path) pointing
        to the audio file.
    start_time : float or None, optional
        Start time in seconds. Defaults to None.
    end_time : float or None, optional
        End time in seconds. If None, reads to end of file.
    input_sr : int or None, optional
        Expected sample rate. If provided, used for validation.
    anonymous : bool, optional
        For the ffmpeg GCS segment path only: if True, access the object
        without credentials (for public objects). Defaults to False.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - data (np.ndarray): The audio data as a NumPy array. The shape will
          be (frames,) for mono or (frames, channels) for multi-channel audio.
        - samplerate (int): The sample rate of the audio in Hz.

    Examples
    --------
    >>> audio, sr = read_audio("tests/samples/noise.wav")
    >>> audio.shape
    (524288,)
    >>> sr
    16000
    """
    file_path = anypath(file_path)

    if start_time is not None:
        if isinstance(file_path, PureGSPath):
            try:
                return _read_audio_ffmpeg(file_path, start_time, end_time, anonymous=anonymous)
            except FFmpegSegmentError as e:
                _warn_ffmpeg_fallback_once(e.cause)
                logger.debug("ffmpeg segment read failed (%s): %s", e.cause, e)
        return _read_audio_by_time(file_path, start_time, end_time, input_sr)

    try:
        fs = filesystem_from_path(file_path)
        with fs.open(str(file_path), "rb") as f:
            audio_format = file_path.suffix.lstrip(".").upper()
            return _read_audio_from_file(f, format=audio_format)
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

    Examples
    --------
    >>> info = get_audio_info("tests/samples/noise.wav")
    >>> info["sr"]
    16000
    """
    file_path = anypath(file_path)
    extension = file_path.suffix

    try:
        with filesystem_from_path(file_path).open(str(file_path), "rb") as f:
            info = sf.info(f)
    except LibsndfileError:
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
