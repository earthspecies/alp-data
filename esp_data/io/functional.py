"""This is a functional implementation of fileio using the FileSystem approach.

The idea is to simplify the api, moving towards a functional, stateless approach.

We prefer using the FileSystem approach, defaulting to cloudpathlib if it doesn't work.
"""

import logging
import os
from functools import cache
from typing import Literal

import fsspec
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from esp_data.utils import read_gcp_secret

from .paths import AnyPath, GSPath, R2Path

logger = logging.getLogger("esp_data")


@cache
def filesystem(
    protocol: Literal["gcs", "gs", "r2", "local"] = "local",
    **kwargs,
):
    if protocol in ["gcs", "gs"]:
        return GCSFileSystem(**kwargs)
    elif protocol == "r2":
        return S3FileSystem(
            key=read_gcp_secret("cloudflare_r2_bucket_readwrite_access_key_id"),
            secret=read_gcp_secret("cloudflare_r2_bucket_readwrite_secret_access_key"),
            endpoint_url=read_gcp_secret("cloudflare_r2_bucket_readwrite_endpoint_url"),
            asynchronous=False,
            **kwargs,
        )
    elif protocol == "local":
        return fsspec.filesystem("local", **kwargs)
    else:
        raise ValueError(f"Unknown backend: {protocol}. Supported backends are: gcs, r2.")


def filesystem_from_path(path: str | os.PathLike | AnyPath):
    path = AnyPath(path)
    if path.is_local:
        return filesystem("local")
    elif isinstance(path, GSPath):
        return filesystem("gcs")
    elif isinstance(path, R2Path):
        return filesystem("r2")
    else:
        raise ValueError(f"Unknown path type: {path}. Supported types are: local, gcs, r2.")


def cp_to_cloud(
    src: str | os.PathLike | AnyPath,
    dst: str | os.PathLike | AnyPath,
) -> bool:
    """Move a file from the source to the destination.

    Args:
        src (str | os.PathLike | AnyPath): The path to the source file.
        dst (str | os.PathLike | AnyPath): The path to the destination file.

    Returns:
        bool: True if the file was moved successfully.
    """
    src = AnyPath(src)
    dst = AnyPath(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source {src} does not exist")

    if dst.exists():
        raise FileExistsError(f"Destination {dst} already exists")

    if src.is_cloud or dst.is_local:
        raise TypeError("Source must be a local path and destination must be a cloud path")

    try:
        fs = filesystem(dst)
        fs.put(src, dst)
        return dst.exists()
    except Exception as e:
        raise IOError(f"Failed to move file {src} to {dst} using both methods: {e}") from e


def yield_files(dir_path: str | os.PathLike | AnyPath, pattern: str = "*"):
    """Yield files in the given directory.

    Args:
        dir_path (str | os.PathLike | AnyPath): The path to the directory.
        pattern (str, optional): A pattern to match the files. Defaults to "*".
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False, which is using cloudpathlib.

    Yields:
        str: A file path if successful.
    """
    dir_path = AnyPath(dir_path)

    if not dir_path.exists():
        logger.warning(f"Directory {dir_path} does not exist, aborting.")
        return

    if dir_path.is_local:
        for f in dir_path.rglob(pattern):
            if f.is_file() and str(f) != str(dir_path):
                yield str(f)
        return

    # FileSystem approach as fallback or if specifically requested
    try:
        fs = _make_fs(dir_path)
        glob_path = strip_cloud_prefix(dir_path / pattern)
        cloud_prefix = AnyPath(dir_path).cloud_prefix

        for f in fs.glob(glob_path):
            if fs.isfile(f) and f != glob_path:
                yield (cloud_prefix + f)

    except Exception as e:
        raise IOError(f"Failed to yield files in {dir_path} using both methods: {e}") from e
