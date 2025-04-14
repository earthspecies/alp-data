import io
import logging

import numpy as np
import soundfile as sf

from .paths import AnyPath

logger = logging.getLogger("esp_data")

_AUDIO_FORMATS = (".wav", ".flac", ".ogg", ".mp3")


def _read_audio_from_bytes(audio_bytes: bytes, frames: int = -1, start: int = 0) -> tuple[np.ndarray, int]:
    """
    Reads from an audio buffer while indexing if necessary. By default,
    reads the entire buffer.
    """
    with io.BytesIO(audio_bytes) as audio_buffer:
        data, samplerate = sf.read(audio_buffer, frames=frames, start=start)
    return data, samplerate


def read_audio(file_path: AnyPath, frames: int = -1, start: int = 0) -> tuple[np.ndarray, int]:
    """
    Reads from an audio file while indexing if necessary. By default,
    reads the entire file.
    """
    file_path = AnyPath(file_path)
    extension = file_path.suffix

    if extension not in _AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}")

    try:
        with file_path.open("rb") as f:
            return _read_audio_from_bytes(f.read(), frames, start)
    except Exception as e:
        logger.error(f"Error reading audio file {e}")
        raise e
