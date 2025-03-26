import logging
import os
from typing import TypeVar

from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from esp_data.paths import AnyPath, is_gcs_path, is_r2_path, is_s3_path

logger = logging.getLogger("esp_data")


# Make a type called FileSystem
FileSystem = TypeVar("FileSystem", GCSFileSystem, S3FileSystem)


def _make_gcsfs() -> GCSFileSystem:
    return GCSFileSystem(access="full_control")


def _make_s3fs() -> S3FileSystem:
    # FIXME: If we ever use AWS S3, this needs to change
    return S3FileSystem(
        key=os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID"),
        secret=os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("CLOUDFLARE_R2_ENDPOINT_URL"),
    )


def _make_cloudflarer2fs() -> S3FileSystem:
    return S3FileSystem(
        key=os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID"),
        secret=os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("CLOUDFLARE_R2_ENDPOINT_URL"),
        asynchronous=False,
    )


def make_fs(file_path: str | AnyPath) -> FileSystem | None:
    file_path = AnyPath(file_path)
    if is_gcs_path(file_path):
        return _make_gcsfs()
    if is_s3_path(file_path):
        return _make_s3fs()
    if is_r2_path(file_path):
        return _make_cloudflarer2fs()

    logger.info("Could not determine cloud filesystem, returning None = local filesystem")
    return None
