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

from cloudpathlib import AnyPath, CloudPath
from google.cloud.storage import Client as GSClient

from ..utils import is_local_path, is_gcs_path

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
        self.file_path.copy(str(destination), force_overwrite_to_cloud=True)

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

        self.file_path = CloudPath(file_path)
        self.client = GSClient()
        bucket_path_parts = str(self.file_path).replace("gs://", "").split("/")
        self.gs_bucket_name = bucket_path_parts[0]
        self.bucket_subpath = AnyPath("/".join(bucket_path_parts[1:]))

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
        blob = bucket.blob(str(self.bucket_subpath))

        blob.upload_from_string(contents)
        logger.info(f"Uploaded content to {self.file_path.name}.")
