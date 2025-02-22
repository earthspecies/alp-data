"""Module for managing local and cloud storage files. Operations like:

- Read and write files
- Copy files
- Delete files
- Download and upload files
- Get file size
- Check if file exists
- Create a file
- Make parent directory
"""

import logging
import os
import warnings

import numpy as np
from google.cloud.storage import Client as GSClient

from esp_data.paths import AnyPath, is_gcs_path, is_local_path
from esp_data.utils import run_as_async

warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

logger = logging.getLogger(__name__)
logger.propagate = True


class File:
    """Class for managing a local file or one in a storage bucket."""

    def __init__(self, file_path: str | AnyPath):
        self.file_path = AnyPath(file_path)
        self._is_local = is_local_path(file_path)

    @property
    def is_local(self):
        return self._is_local

    @property
    def exists(self):
        return self.file_path.exists()

    def create(self, exist_ok: bool = True) -> None:
        self.file_path.touch(exist_ok=exist_ok)

    def open(self, mode: str, **open_kwargs):
        if not self.exists:
            raise FileNotFoundError(f"File does not exist: {self.file_path}")
        return self.file_path.open(mode, **open_kwargs)

    def read_bytes(self) -> bytes:
        """Read the contents of the file."""
        if not self.exists:
            raise FileNotFoundError(f"File does not exist: {self.file_path}")
        return self.file_path.read_bytes()

    def read_text(self) -> str:
        """Read the contents of the file as text."""
        if not self.exists:
            raise FileNotFoundError(f"File does not exist: {self.file_path}")
        return self.file_path.read_text()

    def make_parent_dir(self, exist_ok: bool = True) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=exist_ok)

    def delete(self, confirm: bool = True):
        if confirm:
            if not input(f"Are you sure you want to delete {str(self)}? (y/n): ").lower() == "y":
                return
        self.file_path.unlink()

    def download_to(self, file_path: str | os.PathLike | AnyPath) -> None:
        """Download the file to a local path."""
        if not is_local_path(str(file_path)):
            raise ValueError("File path must be a local path.")
        self.file_path.download_to(AnyPath(file_path))

    def upload_from(self, file_path: str | os.PathLike | AnyPath) -> None:
        """Upload a local file to the bucket."""
        if not is_local_path(file_path):
            raise ValueError("File path must be a local path.")
        self.file_path.upload_from(str(file_path))

    def copy_to(self, destination: str | os.PathLike | AnyPath, overwrite: bool = True):
        """Copy the file to another location."""
        if AnyPath(destination).is_dir():
            destination = AnyPath(destination) / self.file_path.name
        self.file_path.copy(str(destination), force_overwrite_to_cloud=overwrite)

    async def async_copy_to(self, destination: str | os.PathLike | AnyPath, overwrite: bool = True):
        """Copy the file to another location asynchronously."""
        await run_as_async(self.copy_to, destination, overwrite)

    async def async_download_to(self, file_path: str | os.PathLike | AnyPath):
        """Download the file to a local path asynchronously."""
        await run_as_async(self.download_to, file_path)

    async def async_upload_from(self, file_path: str | os.PathLike | AnyPath):
        """Upload a local file to the bucket asynchronously."""
        await run_as_async(self.upload_from, file_path)

    def size(self) -> int:
        """Return the size of the file in bytes."""
        return self.file_path.stat().st_size

    def __repr__(self):
        return f"File({str(self.file_path)})"

    def __str__(self):
        return str(self.file_path)


class GSFile(File):
    def __init__(self, file_path: str | AnyPath):
        super().__init__(file_path)

        if not is_gcs_path(file_path):
            raise ValueError("File path must be a cloud path for Google Cloud Storage.")

        self.file_path = AnyPath(file_path)
        self.client = GSClient()
        file_path_parts = self.file_path.parts[1:]  # remove the gs:// prefix
        self.gs_bucket_name = file_path_parts[0]
        self.file_subpath = AnyPath("/".join(file_path_parts[1:]))

    def upload_from_bytes_or_str(self, contents: bytes | str) -> None:
        """Upload content from memory to a file in Google Cloud Storage.

        Args:
            bucket_name (str | os.PathLike): The name of the bucket.
            destination_file_name (str): The name of the file in the bucket.
            contents (bytes | str): The contents of the file.

        Raises:
            ValueError: If the bucket name is not a cloud path.
        """
        if self.is_local:
            raise ValueError("File path must be a cloud path for uploading to a bucket.")

        bucket = self.client.bucket(self.gs_bucket_name)
        blob = bucket.blob(str(self.file_subpath))

        blob.upload_from_string(contents)
        logger.info(f"Uploaded content to {self.file_path.name}.")

    async def async_upload_from_bytes_or_str(self, contents: bytes | str) -> None:
        """Upload content from memory to a file in Google Cloud Storage asynchronously."""
        await run_as_async(self.upload_from_bytes_or_str, contents)


class GSAudioFile(GSFile):
    def __init__(self, file_path: str | AnyPath):
        super().__init__(file_path)

    def _read_with_soundfile(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        from .parsers import read_audio_from_bytes_sf

        return read_audio_from_bytes_sf(audio_bytes)

    def _read_with_pydub(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        from .parsers import read_audio_from_bytes_pydub

        return read_audio_from_bytes_pydub(audio_bytes)

    def read_audio(self) -> tuple[np.ndarray, int]:
        """Read the audio file from Google Cloud Storage."""
        if not self.exists:
            raise FileNotFoundError(f"File does not exist: {self.file_path}")

        extension = self.file_path.suffix
        audio_bytes = self.read_bytes()

        if extension in [".wav", ".flac", ".ogg"]:
            return self._read_with_soundfile(audio_bytes)
        elif extension == ".mp3":
            return self._read_with_pydub(audio_bytes)
        else:
            raise ValueError(f"Unsupported audio format: {extension}")

    async def async_read_audio(self) -> tuple[np.ndarray, int]:
        """Read the audio file from Google Cloud Storage asynchronously."""
        return await run_as_async(self.read_audio)
