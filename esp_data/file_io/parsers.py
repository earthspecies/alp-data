import io
import logging

import numpy as np
import pandas as pd
import scipy.io
import soundfile as sf
from PIL import Image
from pydub import AudioSegment

from esp_data.paths import AnyPath

UNCOMPRESSED_AUDIO_FORMATS = ["wav", "flac", "ogg"]
COMPRESSED_AUDIO_FORMATS = ["mp3"]


logger = logging.getLogger("esp_data")


def read_audio_from_bytes_sf(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as audio_buffer:
        data, samplerate = sf.read(audio_buffer)

    return data, samplerate


def read_audio_from_bytes_pydub(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))

    return np.array(audio.get_array_of_samples()), audio.frame_rate


def read_audio_bytes(audio_bytes: bytes, extension: str) -> tuple[np.ndarray, int]:
    extension = extension.lower()

    if extension in UNCOMPRESSED_AUDIO_FORMATS:
        read_func = read_audio_from_bytes_sf

    elif extension in COMPRESSED_AUDIO_FORMATS:
        read_func = read_audio_from_bytes_pydub

    else:
        raise ValueError(f"Unsupported audio format: {extension}")

    return read_func(audio_bytes)


def read_audio_bytes_from_path(file_path: AnyPath, fs=None) -> tuple[np.ndarray, int]:
    file_path = AnyPath(file_path)
    extension = file_path.suffix[1:]

    try:
        if fs is None:
            with file_path.open("rb") as f:
                return read_audio_bytes(f.read(), extension)

        with fs.open(str(file_path), "rb") as f:
            return read_audio_bytes(f.read(), extension)

    except Exception as e:
        logger.error(f"Error reading audio file {e}")
        raise e


def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to numpy array.
    Supports common image formats (jpg, png, bmp, etc.)
    Returns image in RGB format with shape (height, width, channels)
    """
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)


# def read_video_from_bytes(video_bytes: bytes) -> tuple[np.ndarray, float]:
#     """Convert video bytes to numpy array.
#     Supports common video formats (mp4, avi, mov, mpeg)
#     Returns:
#         tuple containing:
#         - frames array with shape (frames, height, width, channels)
#         - fps (frames per second)
#     """
#     # Write bytes to temporary file because OpenCV can't read directly from memory
#     temp_file = io.BytesIO(video_bytes)
#     temp_file.write(video_bytes)

#     # Create video capture object
#     video = cv2.VideoCapture(temp_file.name)

#     # Get video properties
#     fps = video.get(cv2.CAP_PROP_FPS)
#     frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Read all frames
#     frames = []
#     for _ in range(frame_count):
#         ret, frame = video.read()
#         if ret:
#             # Convert BGR to RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame)

#     video.release()

#     return np.array(frames), fps


def read_npy_from_bytes(npy_bytes: bytes) -> np.ndarray:
    """Convert numpy .npy file bytes to numpy array"""
    with io.BytesIO(npy_bytes) as bio:
        return np.load(bio)


def read_npz_from_bytes(npz_bytes: bytes) -> dict:
    """Convert numpy .npz file bytes to dict of numpy arrays"""
    with io.BytesIO(npz_bytes) as bio:
        npz_file = np.load(bio)
        # Convert to regular dict because npz file must be closed
        return {key: npz_file[key] for key in npz_file.files}


def read_mat_from_bytes(mat_bytes: bytes) -> dict:
    """Convert MATLAB .mat file bytes to dict of numpy arrays"""
    with io.BytesIO(mat_bytes) as bio:
        return scipy.io.loadmat(bio)


def read_csv_from_bytes(csv_bytes: bytes, **pandas_kwargs) -> pd.DataFrame:
    """Convert CSV file bytes to pandas DataFrame.
    Additional keyword arguments are passed to pd.read_csv()
    """
    return pd.read_csv(io.BytesIO(csv_bytes), **pandas_kwargs)


# Dictionary mapping file extensions to parser functions
PARSER_MAP = {
    # Images
    "jpg": read_image_from_bytes,
    "jpeg": read_image_from_bytes,
    "png": read_image_from_bytes,
    "bmp": read_image_from_bytes,
    "gif": read_image_from_bytes,
    # Videos
    # "mp4": read_video_from_bytes,
    # "avi": read_video_from_bytes,
    # "mov": read_video_from_bytes,
    # "mpeg": read_video_from_bytes,
    # Numpy files
    "npy": read_npy_from_bytes,
    "npz": read_npz_from_bytes,
    # MATLAB files
    "mat": read_mat_from_bytes,
    # CSV files
    "csv": read_csv_from_bytes,
}


def read_bytes_to_array(file_bytes: bytes, extension: str, **kwargs) -> np.ndarray | tuple | dict | pd.DataFrame:
    """Generic function to convert file bytes to numpy array based on file extension.

    Args:
        file_bytes: Raw bytes of the file
        extension: File extension without dot (e.g. 'jpg', 'mp4', 'npy')
        **kwargs: Additional keyword arguments passed to specific parser functions

    Returns:
        Numpy array, tuple, dict or DataFrame depending on the file type:
        - Images: np.ndarray with shape (height, width, channels)
        - Videos: tuple(np.ndarray with shape (frames, height, width, channels), fps)
        - Numpy files: np.ndarray or dict of arrays for .npz
        - MATLAB files: dict of arrays
        - CSV files: pandas DataFrame

    Raises:
        ValueError: If file extension is not supported
    """
    extension = extension.lower()
    if extension not in PARSER_MAP:
        raise ValueError(f"Unsupported file extension: {extension}")

    parser = PARSER_MAP[extension]
    return parser(file_bytes, **kwargs)
