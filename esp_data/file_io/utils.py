import io
import json
import logging
import os

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from gcsfs import GCSFileSystem
from pydub import AudioSegment
from s3fs import S3FileSystem

from esp_data.paths import AnyPath, is_cloud_path, is_cloudflarer2_path, is_gcs_path, is_s3_path

load_dotenv()


logger = logging.getLogger(__name__)

UNCOMPRESSED_AUDIO_FORMATS = ["wav", "flac", "ogg"]
COMPRESSED_AUDIO_FORMATS = ["mp3"]

# Make a type called FileSystem
FileSystem = GCSFileSystem | S3FileSystem


def make_gcsfs() -> GCSFileSystem:
    return GCSFileSystem(access="full_control")


def make_s3fs() -> S3FileSystem:
    # FIXME: If we ever use AWS S3, this needs to change
    return S3FileSystem(
        key=os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID"),
        secret=os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("CLOUDFLARE_R2_ENDPOINT_URL"),
    )


def make_cloudflarer2fs() -> S3FileSystem:
    return S3FileSystem(
        key=os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID"),
        secret=os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("CLOUDFLARE_R2_ENDPOINT_URL"),
        asynchronous=True,
    )


def make_fs(file_path: str | AnyPath) -> FileSystem | None:
    file_path = AnyPath(file_path)
    if is_gcs_path(file_path):
        return make_gcsfs()
    if is_s3_path(file_path):
        return make_s3fs()
    if is_cloudflarer2_path(file_path):
        return make_cloudflarer2fs()

    logger.info("Could not determine cloud filesystem, returning None = local filesystem")
    return None


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
def _read_audio(file_path: AnyPath, fs=None) -> tuple[np.ndarray, int]:
    file_path = AnyPath(file_path)
    extension = file_path.suffix[1:]

    if extension in UNCOMPRESSED_AUDIO_FORMATS:
        read_func = read_audio_from_bytes_sf

    elif extension in COMPRESSED_AUDIO_FORMATS:
        read_func = read_audio_from_bytes_pydub

    else:
        raise ValueError(f"Unsupported audio format: {extension}")

    try:
        if fs is None:
            with file_path.open("rb") as f:
                return read_func(f.read())

        with fs.open(str(file_path), "rb") as f:
            return read_func(f.read())

    except Exception as e:
        logger.error(f"Error reading audio file {e}")
        raise e


def read_audio_from_bucket(file_path: str | os.PathLike) -> tuple[np.ndarray, int]:
    """Reads audio file from a cloud bucket

    Args:
        file_path (str | os.PathLike): Path to the audio file in a bucket

    Returns:
        tuple[np.ndarray, int]: Audio data and samplerate
    """
    file_path = AnyPath(file_path)
    fs = make_fs(file_path)

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


# no qa
def write_audio_bytes(
    save_path: str | os.PathLike, audio: np.ndarray, samplerate: int, fs: GCSFileSystem = None
) -> None:
    """Writes audio data to a file in GCS bucket

    Args:
        save_path (str | os.PathLike): Path to the save the audio file in the GCS bucket / dir, including extension
        audio (np.ndarray): Audio data
        samplerate (int): Samplerate of the audio
        fs (GCSFileSystem, optional): GCSFileSystem object. Defaults to None.
    """
    if fs is None and is_cloud_path(save_path):
        fs = GCSFileSystem()

    with io.BytesIO() as audio_buffer:
        extension = AnyPath(save_path).suffix[1:]

        if extension in UNCOMPRESSED_AUDIO_FORMATS:
            sf.write(audio_buffer, audio, samplerate, format="wav")
        audio_buffer.seek(0)

        with fs.open(str(save_path), "wb") as f:
            f.write(audio_buffer.read())
