"""Read audio segments from private GCS buckets using FFmpeg with
Google Application Default Credentials.
"""

import logging
import subprocess

import google.auth
import numpy as np
from google.auth.transport.requests import Request

from esp_data.io.paths import AnyPathT

logger = logging.getLogger("esp_data")


def get_gcs_token() -> str:
    """Fetches a fresh access token using Google Application Default Credentials.

    This function relies on the user having authenticated with GCP using
    `gcloud auth application-default login`. It retrieves the default credentials,
    refreshes them to ensure a valid access token, and returns the token string.

    Returns
    -------
    str
        A valid access token for authenticating requests to Google Cloud services.

    Raises
    ------
    Exception
        If there is an error obtaining or refreshing the credentials,
        an exception is raised with a message
        suggesting to run 'gcloud auth application-default login'.
    """
    try:
        credentials, _ = google.auth.default()
        credentials.refresh(Request())
        return credentials.token
    except Exception as e:
        raise Exception(
            f"Error authenticating with Google Cloud: {e}.\n"
            "Ensure you have run 'gcloud auth application-default login'"
            "and have the necessary permissions to access the GCS bucket."
        ) from e


def _gcs_path_to_url(file_path: str | AnyPathT) -> str:
    """Convert a GCS path to an HTTPS REST API URL.

    Accepts paths with or without the ``gs://`` prefix.
    E.g. ``gs://bucket/blob`` and ``bucket/blob`` both become
    ``https://storage.googleapis.com/bucket/blob``.

    Parameters
    ----------
    file_path : str or AnyPathT
        The GCS path to convert.

    Returns
    -------
    str
        The corresponding HTTPS URL for accessing the file via GCS REST API.

    Raises
    ------
    ValueError
        If the path starts with an unsupported scheme.

    """
    path_str = str(file_path)
    if path_str.startswith(("s3://", "r2://")):
        raise ValueError(
            f"Unsupported storage scheme in path: {path_str}. Only GCS paths (gs://) are supported."
        )
    if path_str.startswith("gs://"):
        path_str = path_str[len("gs://") :]
    path_str = path_str.lstrip("/")
    return f"https://storage.googleapis.com/{path_str}"


def read_audio_ffmpeg(
    file_path: str | AnyPathT,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> tuple[np.ndarray, int]:
    """Read an audio segment from GCS using FFmpeg, returning a numpy array.

    Uses FFmpeg to stream audio directly from a private GCS bucket via HTTP,
    avoiding full file downloads. The output matches the signature of
    ``esp_data.io.read_audio``: a tuple of ``(audio_data, sample_rate)``.

    Parameters
    ----------
    file_path : str or AnyPathT
        GCS path to the audio file. Accepts paths with or without the
        ``gs://`` prefix (e.g., ``"gs://bucket/path/to/file.wav"`` or
        ``"bucket/path/to/file.wav"``).
    start_time : float, optional
        Start time in seconds. Defaults to 0.0.
    end_time : float or None, optional
        End time in seconds. If None, reads to end of file.

    Raises
    ------
    RuntimeError
        If FFmpeg or FFprobe fails to process the audio.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - data (np.ndarray): Audio data as a float32 numpy array with shape
          ``(channels, frames)``.
        - sample_rate (int): Native sample rate of the audio file.
    """
    token = get_gcs_token()
    gcs_url = _gcs_path_to_url(file_path)
    auth_header = f"Authorization: Bearer {token}\r\n"

    # Probe native sample rate and channel count
    probe_cmd = [
        "ffprobe",
        "-headers",
        auth_header,
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
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed with return code {e.returncode}:\n{e.stderr}") from e

    probe_parts = probe_result.stdout.strip().split(",")
    sample_rate = int(probe_parts[0])
    channels = int(probe_parts[1])

    command = [
        "ffmpeg",
        "-headers",
        auth_header,
        "-ss",
        str(start_time),
        "-i",
        gcs_url,
    ]

    if end_time is not None:
        duration = end_time - start_time
        command += ["-t", str(duration)]

    # Output raw PCM float32 to stdout, preserving native channel layout
    command += [
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"FFmpeg failed with return code {e.returncode}:\n{e.stderr.decode()}"
        ) from e

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).T
    return audio, sample_rate
