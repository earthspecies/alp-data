"""Path definitions that get rid of some issues with the AnyPath type hint."""

import os
from pathlib import Path

from cloudpathlib import GSPath, S3Client, S3Path
from dotenv import load_dotenv

load_dotenv()


def is_gcs_path(path: str | Path | os.PathLike) -> bool:
    return str(path).startswith("gs://")


def is_s3_path(path: str | Path | os.PathLike) -> bool:
    return str(path).startswith("s3://")


def is_cloudflarer2_path(path: str | Path | os.PathLike) -> bool:
    # FIXME: This is a temporary solution
    return "r2://" in str(path)


def is_local_path(path: str | Path | os.PathLike) -> bool:
    return not (is_gcs_path(path) or is_s3_path(path) or is_cloudflarer2_path(path))


def is_cloud_path(path: str | Path | os.PathLike) -> bool:
    return is_gcs_path(path) or is_s3_path(path) or is_cloudflarer2_path(path)


def strip_cloud_prefix(path: str | Path | os.PathLike) -> str:
    return str(path).replace("gs://", "").replace("s3://", "").replace("r2://", "")


def _make_r2_path_with_auth(path: str | os.PathLike | Path) -> S3Path:
    c = S3Client(
        aws_access_key_id=os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("CLOUDFLARE_R2_ENDPOINT_URL"),
    )
    return c.S3Path(path)


class AnyPath:
    """A class that returns the correct path object based on the path string."""

    def __new__(cls, path: str | os.PathLike | Path) -> Path | GSPath | S3Path:
        """This is a factory function. It returns the correct path object based on the path string.
        Solves the issue of disappearing // in the path string when using cloudpathlib.AnyPath.

        Args:
            path (str | os.PathLike | Path | AnyPath): The path to a file or directory.

        Returns:
            Path | GSPath | S3Path: The correct path object based on the path string.
        """
        if isinstance(path, cls):
            return path

        elif is_gcs_path(path):
            return GSPath(path)

        elif is_s3_path(path):
            # FIXME: Since we are not going to use AWS, we can use the same constructor for R2 and S3
            return _make_r2_path_with_auth(path)

        elif is_cloudflarer2_path(path):
            # Since Cloudflare R2 uses the S3 api, only distinguishing itself on the endpoint_url,
            # we can use the S3Path class for Cloudflare R2 paths.
            p = str(path).replace("r2://", "s3://")
            return _make_r2_path_with_auth(p)

        # local path
        return Path(path)
