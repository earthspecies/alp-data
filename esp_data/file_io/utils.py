import logging
import os
import subprocess
from typing import TypeVar

from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from esp_data.paths import AnyPath, is_gcs_path, is_r2_path, is_s3_path

logger = logging.getLogger("esp_data")


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
    if is_r2_path(file_path):
        return make_cloudflarer2fs()

    logger.info("Could not determine cloud filesystem, returning None = local filesystem")
    return None


def gsutil_ls(path: str | AnyPath, options: list[str] | None = None) -> list[str]:
    """List contents of a Google Cloud Storage path.

    Args:
          path (str): The path to list.
          options (list[str], optional): Additional options to pass to `gsutil ls`. Defaults to None.

    Examples:
        # List all files in the bucket
        >>> gsutil_ls("gs://mybucket")

        # List all files with the txt extension with long listing, human readable format
        # recursively
        >>> gsutil_ls("gs://mybucket/*txt", ["-lh", "-R"])

    OPTIONS (from `gsutil ls --help`):
        -l          Prints long listing (owner, length).

        -L          Prints even more detail than -l.

                    Note: If you use this option with the (non-default) XML API it
                    generates an additional request per object being listed, which
                    makes the -L option run much more slowly and cost more than the
                    default JSON API.

        -d          List matching subdirectory names instead of contents, and do not
                    recurse into matching subdirectories even if the -R option is
                    specified.

        -b          Prints info about the bucket when used with a bucket URL.

        -h          When used with -l, prints object sizes in human readable format
                    (e.g., 1 KiB, 234 MiB, 2 GiB, etc.)

        -p proj_id  Specifies the project ID or project number to use for listing
                    buckets.

        -R, -r      Requests a recursive listing, performing at least one listing
                    operation per subdirectory. If you have a large number of
                    subdirectories and do not require recursive-style output ordering,
                    you may be able to instead use wildcards to perform a flat
                    listing, e.g.  ``gsutil ls gs://mybucket/**``, which generally
                    performs fewer listing operations.

        -a          Includes non-current object versions / generations in the listing
                    (only useful with a versioning-enabled bucket). If combined with
                    -l option also prints metageneration for each listed object.

        -e          Include ETag in long listing (-l) output.

    Returns:
          list[str]: The list of files and directories in the path.
    """
    options = options or []
    p = subprocess.run(
        [
            "gsutil",
            "ls",
            *options,
            str(path),
        ],
        capture_output=True,
        text=True,
    )

    if p.returncode != 0:
        logger.error(f"gsutil ls failed with error: {p.stderr}")
        return []

    # split the output by newline and remove the last empty string
    return p.stdout.split("\n")[:-1]
