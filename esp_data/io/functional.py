"""This is a functional implementation of fileio using the FileSystem approach.

The idea is to simplify the api, moving towards a functional, stateless approach.

We prefer using the FileSystem approach, defaulting to cloudpathlib if it doesn't work.
"""

import logging
import os
import shutil
from functools import cache
from typing import Literal

import fsspec
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from esp_data.utils import read_gcp_secret

from .paths import AnyPath, GSPath, strip_cloud_prefix

logger = logging.getLogger("esp_data")


@cache
def get_fs(
    protocol: Literal["gcs", "gs", "r2", "local"] = "local",
    **kwargs,
) -> GCSFileSystem | S3FileSystem | "fsspec.implementations.local.LocalFileSystem":
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


# def _make_gcsfs() -> GCSFileSystem:
#     return GCSFileSystem(access="full_control")


# def _make_cloudflarer2fs() -> S3FileSystem:
#     return S3FileSystem(
#         key=os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID"),
#         secret=os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
#         endpoint_url=os.getenv("CLOUDFLARE_R2_ENDPOINT_URL"),
#         asynchronous=False,
#     )


# def _make_fs(f: str | AnyPath) -> GCSFileSystem | S3FileSystem | None:
#     f = AnyPath(f)
#     if isinstance(f, GSPath):
#         return _make_gcsfs()
#     if isinstance(f, R2Path):
#         return _make_cloudflarer2fs()
#     else:
#         logger.info("Could not determine cloud filesystem, returning None = local filesystem")
#         return None


def cp_to_cloud(
    src: str | os.PathLike | AnyPath,
    dst: str | os.PathLike | AnyPath,
) -> bool:
    """Move a file from the source to the destination.

    Args:
        source (str | os.PathLike | AnyPath): The path to the source file.
        destination (str | os.PathLike | AnyPath): The path to the destination file.

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
        fs = get_fs(dst)
        fs.put(src, dst)
        return dst.exists()
    except Exception as e:
        raise IOError(f"Failed to move file {src} to {dst} using both methods: {e}") from e


def yield_files(dir_path: str | os.PathLike | AnyPath, pattern: str = "*", use_fs: bool = False):
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

    if not use_fs:
        try:
            for f in dir_path.rglob(pattern):
                if f.is_file() and str(f) != str(dir_path):
                    yield str(f)
            return
        except Exception as e:
            logger.warning(f"Could not yield files using AnyPath method: {e}, trying FileSystem approach.")

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


def delete_dir(dir_path: str | os.PathLike | AnyPath, use_fs: bool = False) -> bool:
    """Delete the directory at the given path.

    Args:
        dir_path (str | os.PathLike | AnyPath): The path to the directory, local or cloud.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.

    Returns:
        bool: True if the directory was deleted successfully
    """
    dir_path = AnyPath(dir_path)

    if not dir_path.exists():
        logger.info(f"Directory {dir_path} does not exist, aborting.")
        return False

    if dir_path.is_local:
        try:
            shutil.rmtree(dir_path)
            return True
        except Exception as e:
            if not use_fs:
                raise IOError(f"Failed to delete directory {dir_path}: {e}") from e

    if not use_fs:
        try:
            for f in yield_files(dir_path):
                AnyPath(f).unlink()
            return True
        except Exception as e:
            logger.warning(f"Could not delete directory using AnyPath method: {e}, trying FileSystem approach.")

    try:
        fs = _make_fs(dir_path)
        files = list(yield_files(dir_path))
        # Remove all files
        if files:
            fs.rm(files)
        return True
    except Exception as e:
        raise IOError(f"Failed to delete directory {dir_path} using both methods: {e}") from e


def copy(
    source: str | os.PathLike | AnyPath, destination: str | os.PathLike | AnyPath, use_fs: bool = False, **gcloud_kwargs
) -> bool:
    """Copy a file/directory from source to destination.

    If both the source and destination are local paths, the standard file operations are used.
    If both the source and destination are cloud paths on GCP, the gcloud storage rsync command is used.

    Args:
        source (str | os.PathLike | AnyPath): The source path.
        destination (str | os.PathLike | AnyPath): The destination path.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.
        **gcloud_kwargs: Additional keyword arguments to pass to the gcloud storage rsync command.

    Returns:
        bool: True if the copy was successful.
    """
    source = AnyPath(source)
    destination = AnyPath(destination)

    if not source.exists():
        raise FileNotFoundError(f"Source {source} does not exist")

    # Create parent directories for destination if it's a local path
    if destination.is_local:
        os.makedirs(os.path.dirname(str(destination)), exist_ok=True)

    # If both paths are local, use standard file operations
    if source.is_local and destination.is_local:
        try:
            if source.is_file():
                shutil.copy2(source, destination)
            else:
                shutil.copytree(source, destination)
            return True
        except Exception as e:
            raise IOError(f"Failed to copy local {source} to {destination}: {e}") from e

    is_upload = source.is_local and not destination.is_local
    is_download = not source.is_local and destination.is_local
    # For cloud paths, determine which path needs the filesystem
    cloud_path = destination if not destination.is_local else source

    if not use_fs:
        try:
            if is_upload:
                destination.upload_from(source, force_overwrite_to_cloud=True)
            elif is_download:
                source.download_to(destination)
            else:
                source.copy(destination, force_overwrite_to_cloud=True)
            return True
        except Exception as e:
            logger.warning(f"Could not copy using AnyPath method: {e}, trying FileSystem approach.")

    try:
        fs = _make_fs(cloud_path)
        source_str = strip_cloud_prefix(source)
        destination_str = strip_cloud_prefix(destination)

        if is_upload:
            fs.put(source_str, destination_str, recursive=True)
        elif is_download:
            fs.get(source_str, destination_str, recursive=True)
        else:
            if isinstance(source, GSPath) and isinstance(destination, GSPath):
                # call gcloud_rsync
                gcloud_rsync(source, destination, **gcloud_kwargs)
            else:
                raise ValueError("Cloud to cloud copy is only supported for GCP paths")

        return True
    except Exception as e:
        raise IOError(f"Failed to copy {source} to {destination} using both methods: {e}") from e
