import json
import logging
import os
from typing import TypeVar

from dotenv import load_dotenv
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from esp_data.paths import AnyPath, is_cloudflarer2_path, is_gcs_path, is_s3_path

load_dotenv()


logger = logging.getLogger(__name__)


# Make a type called FileSystem
FileSystem = TypeVar("FileSystem", GCSFileSystem, S3FileSystem)


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
        asynchronous=False,
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
