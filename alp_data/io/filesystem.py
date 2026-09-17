"""
This file offers functionalities necessary to manipulate different sort of filesystems
"""

import logging
from functools import cache
from typing import Literal

import fsspec
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from alp_data.utils import read_gcp_secret

from .paths import AnyPathT, PureGSPath, PureHTTPPath, PureHTTPSPath, PureR2Path, anypath

logger = logging.getLogger("alp_data")


@cache
def filesystem(
    protocol: Literal["gcs", "gs", "r2", "local", "http", "https"] = "local",
    **kwargs: dict,
) -> GCSFileSystem | S3FileSystem | LocalFileSystem | HTTPFileSystem:
    """Initializes and returns a cached filesystem instance.

    This function acts as a factory for creating filesystem objects based on the
    specified protocol. It supports Google Cloud Storage ('gcs', 'gs'),
    Cloudflare R2 ('r2'), plain HTTP(S) endpoints ('http', 'https'), and the local
    filesystem ('local').

    Both 'http' and 'https' return an `HTTPFileSystem`; fsspec registers a single
    implementation for the two schemes, so the scheme is carried by the URL passed
    to the filesystem rather than by the filesystem object itself.

    An `HTTPFileSystem` is not equivalent to the bucket backends. Plain HTTP has
    no listing API, so `fsspec` emulates one by fetching the URL and scraping
    `<a href=...>` links out of the response when it is HTML:

    - `ls()` and `glob()` work only against a server that returns an HTML index
      (an Apache/nginx autoindex, for example).
    - On an object-storage endpoint such as a Cloudflare R2 public bucket there
      is no index document, so `ls()` raises `FileNotFoundError` and `glob()`
      returns an empty list **without raising**. Prefer an explicit manifest of
      URLs over discovering files by glob.
    - Reads are the supported operation. `alp_data.io.rm` rejects HTTP(S) paths,
      and `alp_data.io.exists` goes through `info()` (a HEAD) rather than
      `HTTPFileSystem.exists` (a GET).

    For the 'r2' protocol, it automatically retrieves the necessary credentials
    (access key ID, secret access key, endpoint URL) from GCP Secret Manager.

    The results are cached so subsequent calls with the same protocol and keyword
    parameters will return the identical filesystem instance.

    Parameters
    ----------
        protocol: Literal["gcs", "gs", "r2", "http", "https", "local"]
            The type of filesystem to initialize. Defaults to "local".
            Supported values are "gcs", "gs", "r2", "http", "https", "local".
        **kwargs: dict
            Additional keyword parameters to pass directly to the
            underlying filesystem constructor (e.g., GCSFileSystem, S3FileSystem).

    Raises
    ------
    ValueError
        If an unsupported protocol is provided.

    Returns
    -------
        An filesystem object corresponding to the specified protocol
        (e.g., GCSFileSystem, S3FileSystem, LocalFileSystem, HTTPFileSystem).

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
    elif protocol in ["http", "https"]:
        return fsspec.filesystem("http", **kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {protocol}. Supported backends are: gcs, gs, r2, http, https, local."
        )


def filesystem_from_path(
    path: str | AnyPathT,
) -> GCSFileSystem | S3FileSystem | LocalFileSystem | HTTPFileSystem:
    """Determines and returns the appropriate cached filesystem based on the path.

    Uses the `anypath` utility to normalize the input path and identify its
    protocol (local, GCS, R2, HTTP, HTTPS). It then calls the `filesystem` factory
    function to retrieve the corresponding cached fsspec-compatible filesystem
    instance.

    Parameters
    ----------
    path : str or AnyPathT
        The path string or path object (e.g., Path, GSPath, R2Path) whose
        protocol determines the filesystem to return.

    Returns
    -------
    An filesystem object corresponding to the specified protocol
        (e.g., GCSFileSystem, S3FileSystem, LocalFileSystem, HTTPFileSystem).

    Examples
    --------
    >>> # gcs_fs = filesystem_from_path("gs://esp-ci-cd-tests/esp-data-tests/file1.txt")
    >>> # isinstance(gcs_fs, GCSFileSystem) # Should be True if configured
    True
    """
    path = anypath(str(path))

    if isinstance(path, PureGSPath):
        return filesystem("gcs")
    elif isinstance(path, PureR2Path):
        return filesystem("r2")
    elif isinstance(path, PureHTTPSPath):
        return filesystem("https")
    elif isinstance(path, PureHTTPPath):
        return filesystem("http")
    else:
        return filesystem("local")
