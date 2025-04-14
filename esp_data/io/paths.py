import os
from pathlib import PosixPath
from typing import Optional, TypeAlias

import cloudpathlib
from cloudpathlib import GSClient, S3Client
from google.cloud.storage.client import Client as GS_Client_Official

from esp_data.utils import cached_class_property, read_gcp_secret

_DEFAULT_GCP_PROJECT = "okapi-274503"


# Define a type alias for cloud paths
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
    """TODO"""
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
    """TODO"""
    # TODO: Path is a factory class and we're dropping support for WindowsPath class. Let's see if we can bring it back.

    storage_options = None

    is_cloud: bool = False
    is_local: bool = True


# TODO (milad) Python 3.12 introduces `type`. It will probably deprecate TypeAlias at
# some point. We should use that instead when 3.12 is not too new anymore.
AnyPathT: TypeAlias = Path | GSPath | R2Path
"""TODO: AnyPathT"""

def anypath(path: str | Path | GSPath | R2Path) -> AnyPathT:
    """A factory function that returns the correct path object based on the path string.

    Args:
        path (str | Path | GSPath | R2Path): The path to a local or Bucket file

    Returns:
        AnyPathT: The correct path object based on the path string.
    """

    path = str(path)

    if path.startswith("gs://"):
        return GSPath(path)
    elif path.startswith("s3://") or path.startswith("r2://"):
        # Since we are currently not using AWS we assume that all S3 paths are R2 paths.
        # TODO This must be changed if we start using AWS.
        return R2Path(path)
    else:
        return Path(path)


def strip_cloud_prefix(path: str | Path | os.PathLike) -> str:
    """Strip the cloud prefix from a path."""
    return str(path).replace("gs://", "").replace("s3://", "").replace("r2://", "")
