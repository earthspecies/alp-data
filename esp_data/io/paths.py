"""This file offers path functionalities for homogeneous resource access."""

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

    Arguments
    ----------
    cloud_path : str or cloudpathlib.GSPath
        The Google Cloud Storage path string (e.g., "gs://bucket/blob") or
        an existing `cloudpathlib.GSPath` instance.
    client : cloudpathlib.GSClient, optional
        An explicit `cloudpathlib.GSClient` instance to use. If None, the
        default cached client (`__client`) configured with the official
        Google Cloud Storage library is used. Defaults to None.

    Examples
    --------
    >>> p = GSPath("gs://esp-ci-cd-tests/esp-data-tests/file1.txt")
    >>> isinstance(p, GSPath)
    True
    >>> isinstance(p, cloudpathlib.GSPath)
    True
    >>> str(p)
    'gs://esp-ci-cd-tests/esp-data-tests/file1.txt'
    >>> p.bucket
    'esp-ci-cd-tests'
    """

    storage_options: dict = {"project": _DEFAULT_GCP_PROJECT}
    is_cloud: bool = True
    is_local: bool = False

    def __init__(self, cloud_path: str | cloudpathlib.GSPath, client: Optional[GSClient] = None) -> None:
        """Initializes the GSPath instance."""
        if not client:
            client = GSPath.__client
        super().__init__(cloud_path, client=client)

    @cached_class_property
    def __client(cls) -> cloudpathlib.GSClient:
        """Gets the cached default cloudpathlib GSClient.

        Returns
        -------
        cloudpathlib.GSClient
            The default client instance.
        """
        return cloudpathlib.GSClient(storage_client=GS_Client_Official())

    @property
    def no_prefix(self) -> str:
        """str: The path string without the 'gs://' prefix."""
        return str(self)[len(self.cloud_prefix) :]


class R2Path(cloudpathlib.S3Path):
    """A cloudpathlib.S3Path wrapper for Cloudflare R2 storage paths.

    Handles paths starting with `r2://` by internally converting them to `s3://` for compatibility
    with `cloudpathlib.S3Path`.
    Automatically configures a default client using R2 credentials fetched from GCP Secret Manager.

    Arguments
    ----------
    cloud_path : str or cloudpathlib.S3Path
        The Cloudflare R2 path string (e.g., "r2://bucket/key" or "s3://...")
        or an existing `cloudpathlib.S3Path` instance.
    client : cloudpathlib.S3Client, optional
        An explicit `cloudpathlib.S3Client` instance to use. If None, the
        default cached client (`__client`) configured with R2 credentials
        is used. Defaults to None.
    """

    is_cloud: bool = True
    is_local: bool = False

    def __init__(self, cloud_path: str | cloudpathlib.S3Path, client: Optional[S3Client] = None) -> None:
        """Initializes the R2Path, converting 'r2://' prefix if needed."""
        if not client:
            client = R2Path.__client

        if isinstance(cloud_path, str):
            cloud_path = cloud_path.replace("r2://", "s3://")

        super().__init__(cloud_path, client=client)

    @cached_class_property
    def __client(cls) -> S3Client:
        """Gets the cached default S3Client configured for R2.

        Returns
        -------
        cloudpathlib.S3Client
            The default client instance for R2.
        """
        return cloudpathlib.S3Client(
            aws_access_key_id=read_gcp_secret("cloudflare_r2_bucket_readwrite_access_key_id"),
            aws_secret_access_key=read_gcp_secret("cloudflare_r2_bucket_readwrite_secret_access_key"),
            endpoint_url=read_gcp_secret("cloudflare_r2_bucket_readwrite_endpoint_url"),
        )

    @cached_class_property
    def storage_options(self) -> dict:
        """Gets cached R2 credentials.

        Returns
        -------
        dict
            A dictionary containing R2 client keyword arguments.
        """
        return {
            "client_kwargs": {
                "aws_access_key_id": read_gcp_secret("cloudflare_r2_bucket_readwrite_access_key_id"),
                "aws_secret_access_key": read_gcp_secret("cloudflare_r2_bucket_readwrite_secret_access_key"),
                "endpoint_url": read_gcp_secret("cloudflare_r2_bucket_readwrite_endpoint_url"),
            }
        }

    @property
    def no_prefix(self) -> str:
        """str: The path string without the 's3://' prefix."""
        return str(self)[len(self.cloud_prefix) :]


class Path(PosixPath):
    """TODO: write the docstring once the class is consolidated."""

    # TODO: Path is a factory class and we're dropping support for WindowsPath class. Let's see if we can bring it back.
    storage_options = None

    is_cloud: bool = False
    is_local: bool = True


# TODO (milad) Python 3.12 introduces `type`. It will probably deprecate TypeAlias at
# some point. We should use that instead when 3.12 is not too new anymore.
AnyPathT: TypeAlias = Path | GSPath | R2Path


def anypath(path: str | Path | GSPath | R2Path) -> AnyPathT:
    """Creates the appropriate path object based on the input path string or object.

    This factory function inspects the input `path` to determine if it's a Google
    Cloud Storage path, an S3-compatible path (assumed to be Cloudflare R2),
    or a local path. It then returns an instance of the corresponding path class
    (`GSPath`, `R2Path`, or `Path`).

    Arguments
    ----------
    path : str | Path | GSPath | R2Path
        The path string (e.g., "/local/file.txt", "gs://bucket/blob", "r2://bucket/key")
        or an existing `Path`, `GSPath`, or `R2Path` object.

    Returns
    -------
    AnyPathT
        An instance of `Path` for local paths, `GSPath` for Google Cloud Storage
        paths, or `R2Path` for Cloudflare R2 paths (including those starting
        with "s3://").


    Examples
    --------
    >>> local_p = anypath("tests/samples/noise.wav")
    >>> isinstance(local_p, Path)
    True
    >>> print(local_p)
    tests/samples/noise.wav
    >>> gs_p = anypath("gs://esp-ci-cd-tests/esp-data-tests/file1.txt")
    >>> isinstance(gs_p, GSPath)
    True
    >>> print(gs_p)
    gs://esp-ci-cd-tests/esp-data-tests/file1.txt
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
