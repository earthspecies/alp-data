import io
import json
import logging
import os

import numpy as np
import sounddevice as sd
import soundfile as sf
from cloudpathlib import AnyPath
from gcsfs import GCSFileSystem
from pydub import AudioSegment

logger = logging.getLogger(__name__)

UNCOMPRESSED_AUDIO_FORMATS = ["wav", "flac", "ogg"]
COMPRESSED_AUDIO_FORMATS = ["mp3"]


def read_audio_from_bytes_sf(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as audio_buffer:
        data, samplerate = sf.read(audio_buffer)

    return data, samplerate


def read_audio_from_bytes_pydub(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))

    return np.array(audio.get_array_of_samples()), audio.frame_rate


def play_audio(data: np.ndarray, samplerate: int) -> None:
    sd.play(data, samplerate)
    sd.wait()


# TODO: Add type hint for filesystem
def _read_audio(file_path: AnyPath, fs) -> tuple[np.ndarray, int]:
    file_path = AnyPath(file_path)
    extension = file_path.suffix[1:]

    if extension in UNCOMPRESSED_AUDIO_FORMATS:
        with fs.open(str(file_path), "rb") as f:
            data, samplerate = read_audio_from_bytes_sf(f.read())

    elif extension in COMPRESSED_AUDIO_FORMATS:
        with fs.open(str(file_path), "rb") as f:
            data, samplerate = read_audio_from_bytes_pydub(f.read())

    else:
        raise ValueError(f"Unsupported audio format: {extension}")

    return data, samplerate


def read_audio_from_gs_bucket(file_path: str | os.PathLike, fs: GCSFileSystem = None) -> tuple[np.ndarray, int]:
    """Reads audio file from GCS bucket

    Args:
        file_path (str | os.PathLike): Path to the audio file in the GCS bucket
        fs (GCSFileSystem, optional): GCSFileSystem object. Defaults to None.

    Returns:
        tuple[np.ndarray, int]: Audio data and samplerate
    """
    if fs is None:
        fs = GCSFileSystem()

    return _read_audio(file_path, fs)


def read_jsonl_from_gs_bucket(file_path: str | os.PathLike, fs: GCSFileSystem = None) -> list[dict]:
    """Reads a json file assuming top level key is 'annotation'"""

    if fs is None:
        fs = GCSFileSystem()

    with fs.open(str(file_path)) as f:
        try:
            return json.load(f)["annotation"]

        except Exception as e:
            logger.error(f"Error reading jsonl {e}, trying line by line")
            # read lines
            records = f.readlines()
            return [json.loads(record) for record in records]

        except Exception as e:
            logger.error(f"Error reading jsonl {e}")
            return []
