"""Path definitions that get rid of some issues with the AnyPath type hint."""

import os
import warnings
from functools import lru_cache
from pathlib import Path

import cloudpathlib
from cloudpathlib import S3Path
from google.cloud.storage.client import Client as GSClient

warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")


def is_gcs_path(path: str | Path | os.PathLike) -> bool:
    return str(path).startswith("gs://")


def is_s3_path(path: str | Path | os.PathLike) -> bool:
    return str(path).startswith("s3://")


def is_r2_path(path: str | Path | os.PathLike) -> bool:
    # FIXME: This is a temporary solution
    return "r2://" in str(path)


def is_local_path(path: str | Path | os.PathLike) -> bool:
    return not (is_gcs_path(path) or is_s3_path(path) or is_r2_path(path))


def is_cloud_path(path: str | Path | os.PathLike) -> bool:
    return is_gcs_path(path) or is_s3_path(path) or is_r2_path(path)


def strip_cloud_prefix(path: str | Path | os.PathLike) -> str:
    return str(path).replace("gs://", "").replace("s3://", "").replace("r2://", "")


def make_gscs_storage_options() -> dict:
    return {"project": os.getenv("GCP_DEFAULT_PROJECT")}


def make_s3_storage_options() -> dict:
    return {
        "client_kwargs": {
            "aws_access_key_id": os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
            "endpoint_url": os.getenv("CLOUDFLARE_R2_ENDPOINT_URL"),
        }
    }


def make_storage_options(path: str | os.PathLike) -> dict | None:
    if is_gcs_path(path):
        return make_gscs_storage_options()
    elif is_s3_path(path) or is_r2_path(path):
        return make_s3_storage_options()
    else:
        return None


# TODO (milad) maybe we want to use a singleton pattern here using __new__? I think
# class variables are instantiated when class statement is executed so they will make
# imports a tiny bit slower.
@lru_cache(maxsize=1)
def _get_gs_client():
    return cloudpathlib.GSClient(storage_client=GSClient())


@lru_cache(maxsize=1)
def _get_r2_client():
    return cloudpathlib.S3Client(
        aws_access_key_id=os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("CLOUDFLARE_R2_ENDPOINT_URL"),
    )


class GSPath(cloudpathlib.GSPath):
    """
    A wrapper for the GSPath class that provides a default client to the constructor.
    This is necessary due to a bug in cloudpathlib (v0.20.0) which assumes that the
    GOOGLE_APPLICATION_CREDENTIALS environment variable always points to a service
    account. This assumption is incorrect when using Workload Identity Federation, which
    we in our Github Action. Here, we fallback to the actual Google library for a
    default client that handles this correctly.

    For more details, see: https://github.com/drivendataorg/cloudpathlib/issues/390
    """

    def __init__(self, cloud_path: str | cloudpathlib.GSPath, client=_get_gs_client()):
        super().__init__(cloud_path, client=client)


class R2Path(cloudpathlib.S3Path):
    def __init__(self, cloud_path: str | cloudpathlib.S3Path, client=_get_r2_client()):
        if isinstance(cloudpathlib, str):
            cloud_path = cloudpathlib.replace("r2://", "s3://")

        super().__init__(cloud_path, client=client)


class AnyPath:
    """A class that returns the correct path object based on the path string."""

    def __new__(cls, path: str | Path | GSPath | R2Path) -> Path | GSPath | R2Path:
        """This is a factory function. It returns the correct path object based on the path string.
        Solves the issue of disappearing // in the path string when using cloudpathlib.AnyPath.

        Args:
            path (str | Path | GSPath | R2Path): The path to a file or directory.

        Returns:
            Path | GSPath | S3Path: The correct path object based on the path string.
        """

        if isinstance(path, (Path, GSPath, R2Path | S3Path)):
            path = str(path)

        if is_gcs_path(path):
            return GSPath(str(path))
        elif is_s3_path(path):
            # Since we are currently not using AWS we assume that all S3 paths are R2 paths.
            # TODO This must be changed if we start using AWS.
            return R2Path(str(path), client=_get_r2_client())
        elif is_r2_path(path):
            return R2Path(str(path))
        else:
            return Path(path)
