"""Path definitions that get rid of some issues with the AnyPath type hint."""

import os
from pathlib import PosixPath
from typing import Optional

import cloudpathlib
from cloudpathlib import GSClient, S3Client, S3Path
from google.cloud.storage.client import Client as GS_Client_Official

from esp_data.utils import cached_class_property, read_gcp_secret

_DEFAULT_GCP_PROJECT = "okapi-274503"


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

    storage_options: dict = {"project": _DEFAULT_GCP_PROJECT}
    is_cloud: bool = True
    is_local: bool = False

    def __init__(self, cloud_path: str | cloudpathlib.GSPath, client: Optional[GSClient] = None):
        if not client:
            client = GSPath.__client
        super().__init__(cloud_path, client=client)

    @cached_class_property
    def __client(cls) -> cloudpathlib.GSClient:
        return cloudpathlib.GSClient(storage_client=GS_Client_Official())

    @property
    def no_prefix(self) -> str:
        return str(self)[len(self.cloud_prefix) :]


class R2Path(cloudpathlib.S3Path):
    is_cloud: bool = True
    is_local: bool = False

    def __init__(self, cloud_path: str | cloudpathlib.S3Path, client: Optional[S3Client] = None):
        if not client:
            client = R2Path.__client

        if isinstance(cloud_path, str):
            cloud_path = cloud_path.replace("r2://", "s3://")

        super().__init__(cloud_path, client=client)

    @cached_class_property
    def __client(cls) -> S3Client:
        return cloudpathlib.S3Client(
            aws_access_key_id=read_gcp_secret("cloudflare_r2_bucket_readwrite_access_key_id"),
            aws_secret_access_key=read_gcp_secret("cloudflare_r2_bucket_readwrite_secret_access_key"),
            endpoint_url=read_gcp_secret("cloudflare_r2_bucket_readwrite_endpoint_url"),
        )

    @cached_class_property
    def storage_options(self) -> dict:
        return {
            "client_kwargs": {
                "aws_access_key_id": read_gcp_secret("cloudflare_r2_bucket_readwrite_access_key_id"),
                "aws_secret_access_key": read_gcp_secret("cloudflare_r2_bucket_readwrite_secret_access_key"),
                "endpoint_url": read_gcp_secret("cloudflare_r2_bucket_readwrite_endpoint_url"),
            }
        }

    @property
    def no_prefix(self) -> str:
        return str(self)[len(self.cloud_prefix) :]


class Path(PosixPath):
    # TODO: Path is a factory class and we're dropping support for WindowsPath class. Let's see if we can bring it back.
    storage_options = None

    is_cloud: bool = False
    is_local: bool = True


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

        if _is_gcs_path(path):
            return GSPath(str(path))
        elif _is_s3_path(path):
            # Since we are currently not using AWS we assume that all S3 paths are R2 paths.
            # TODO This must be changed if we start using AWS.
            return R2Path(str(path))
        elif _is_r2_path(path):
            return R2Path(str(path))
        else:
            return Path(path)


def _is_gcs_path(path: str | Path | os.PathLike) -> bool:
    return str(path).startswith("gs://")


def _is_s3_path(path: str | Path | os.PathLike) -> bool:
    return str(path).startswith("s3://")


def _is_r2_path(path: str | Path | os.PathLike) -> bool:
    # FIXME: This is a temporary solution
    return "r2://" in str(path)


def strip_cloud_prefix(path: str | Path | os.PathLike) -> str:
    return str(path).replace("gs://", "").replace("s3://", "").replace("r2://", "")
