import logging
from functools import cache
from typing import Literal

import fsspec
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from esp_data.utils import read_gcp_secret

from .paths import AnyPathT, GSPath, R2Path, anypath

logger = logging.getLogger("esp_data")


@cache
def filesystem(
    protocol: Literal["gcs", "gs", "r2", "local"] = "local",
    **kwargs,
):
    """Create a filesystem object based on the protocol.

    PARAMETERS
    ----------
    protocol : str
        The protocol to use. Supported protocols are "gcs", "r2", and "local".
    kwargs : dict
        Additional keyword arguments to pass to the filesystem constructor.
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


def filesystem_from_path(path: str | AnyPathT):
    """TODO"""
    path = anypath(path)
    if path.is_local:
        return filesystem("local")
    elif isinstance(path, GSPath):
        return filesystem("gcs")
    elif isinstance(path, R2Path):
        return filesystem("r2")
    else:
        raise ValueError(f"Unknown path type: {path}. Supported types are: local, gcs, r2.")
