"""This file offers functionalities necessary to manipulate different sort of filesystems"""

import logging
from functools import cache
from typing import Literal, Union

import fsspec
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from esp_data.utils import read_gcp_secret

from .paths import AnyPathT, GSPath, R2Path, anypath

logger = logging.getLogger("esp_data")


@cache
def filesystem(
    protocol: Literal["gcs", "gs", "r2", "local"] = "local",
    **kwargs: dict,
) -> Union[GCSFileSystem, S3FileSystem, fsspec.filesystem]:
    """Initializes and returns a cached filesystem instance.

    This function acts as a factory for creating filesystem objects based on the
    specified protocol. It supports Google Cloud Storage ('gcs', 'gs'),
    Cloudflare R2 ('r2'), and the local filesystem ('local').

    For the 'r2' protocol, it automatically retrieves the necessary credentials
    (access key ID, secret access key, endpoint URL) from GCP Secret Manager.

    The results are cached so subsequent calls with the same protocol and keyword
    arguments will return the identical filesystem instance.

    Parameters
    ----------
        protocol: Literal["gcs", "gs", "r2", "local"]
            The type of filesystem to initialize. Defaults to "local".
            Supported values are "gcs", "gs", "r2", "local".
        **kwargs: dict
            Additional keyword arguments to pass directly to the
            underlying filesystem constructor (e.g., GCSFileSystem, S3FileSystem).

    Raises
    ------
    ValueError
        If an unsupported protocol is provided.

    Returns
    -------
        An filesystem object corresponding to the specified protocol
        (e.g., GCSFileSystem, S3FileSystem, LocalFileSystem).

    Examples
    --------
    >>> import fsspec
    >>> local_fs = filesystem("local")
    >>> isinstance(local_fs, fsspec.implementations.local.LocalFileSystem)
    True
    >>> local_fs_again = filesystem("local")
    >>> local_fs is local_fs_again
    True
    """
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


def filesystem_from_path(path: str | AnyPathT) -> Union[GCSFileSystem, S3FileSystem, fsspec.filesystem]:
    """Determines and returns the appropriate cached filesystem based on the path.

    Uses the `anypath` utility to normalize the input path and identify its
    protocol (local, GCS, R2). It then calls the `filesystem` factory function
    to retrieve the corresponding cached fsspec-compatible filesystem instance.

    Parameters
    ----------
    path : str or AnyPathT
        The path string or path object (e.g., Path, GSPath, R2Path) whose
        protocol determines the filesystem to return.

    Returns
    -------
    An filesystem object corresponding to the specified protocol
        (e.g., GCSFileSystem, S3FileSystem, LocalFileSystem).

    Raises
    ------
    ValueError
        If the path type derived from `anypath` is not recognized as
        local, GSPath, or R2Path.

    Examples
    --------
    >>> # gcs_fs = filesystem_from_path("gs://esp-ci-cd-tests/esp-data-tests/file1.txt")
    >>> # isinstance(gcs_fs, GCSFileSystem) # Should be True if configured
    True
    """
    path = anypath(path)
    if path.is_local:
        return filesystem("local")
    elif isinstance(path, GSPath):
        return filesystem("gcs")
    elif isinstance(path, R2Path):
        return filesystem("r2")
    else:
        raise ValueError(f"Unknown path type: {path}. Supported types are: local, gcs, r2.")
