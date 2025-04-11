import io
import logging

import numpy as np
import soundfile as sf

from .paths import AnyPath, anypath

logger = logging.getLogger("esp_data")

_AUDIO_FORMATS = (".wav", ".flac", ".ogg", ".mp3")


def _read_audio_from_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as audio_buffer:
        data, samplerate = sf.read(audio_buffer)
    return data, samplerate


def read_audio(file_path: str | AnyPath) -> tuple[np.ndarray, int]:
    file_path = anypath(file_path)
    extension = file_path.suffix

    if extension not in _AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}")

    try:
        with file_path.open("rb") as f:
            return _read_audio_from_bytes(f.read())
    except Exception as e:
        logger.error(f"Error reading audio file {e}")
        raise e
