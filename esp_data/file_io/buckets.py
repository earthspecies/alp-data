"""Module for storage bucket management and file management."""

import asyncio
import concurrent.futures
import logging
import os
import subprocess
import warnings
from typing import Generator

from cloudpathlib import CloudPath
from dotenv import load_dotenv
from google.cloud.storage import Client as GSClient
from google.cloud.storage import transfer_manager

from esp_data.paths import AnyPath, is_cloud_path, is_local_path, strip_cloud_prefix

from .utils import make_cloudflarer2fs, make_fs, make_gcsfs

load_dotenv()
# TODO: This is a temporary fix to suppress the warning about using end user credentials
# see here: https://github.com/googleapis/google-auth-library-python/issues/271 for info
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# propagate logs to parent logger
logger.propagate = True


class Bucket:
    """Class for managing a storage bucket. Not to be used for local directories, use AnyPath for that.

    NOTE: Even though the name suggests its only used for buckets, this class can also be used to
    manage dirs in buckets.

    Args:
        bucket_path: The path to the bucket.
    """

    def __init__(self, bucket_path: str | os.PathLike | AnyPath):
        if not is_cloud_path(str(bucket_path)):
            raise ValueError("Bucket path must be a cloud path.")

        self.bucket_path = CloudPath(bucket_path)

    @property
    def exists(self):
        return self.bucket_path.exists()

    def list_files(self, recursive: bool = True, as_str: bool = False) -> list[CloudPath]:
        if recursive:
            iterator = self.bucket_path.rglob("*")
        else:
            # WARNING: this seems to pass the root dir as well, even
            # though it is not a file
            iterator = self.bucket_path.glob("*")

        return [str(file) if as_str else file for file in iterator if file.is_file()]

    def list_dirs(self, recursive: bool = True, as_str: bool = False) -> list[CloudPath]:
        if recursive:
            iterator = self.bucket_path.rglob("*")
        else:
            iterator = self.bucket_path.iterdir()

        return [str(dir) if as_str else dir for dir in iterator if dir.is_dir()]

    def subdir_as_bucket(self, subdir: str | os.PathLike | AnyPath) -> "Bucket":
        """Create a subdirectory as a Bucket object.
        e.g. if the bucket path is gs://esp-ci-cd-tests/esp-data-tests/,
        and subdir is "subdir", then the new bucket path will be
        gs://esp-ci-cd-tests/esp-data-tests/subdir.
        """
        subpath = self.bucket_path / AnyPath(subdir)

        return Bucket(subpath)

    def find_paths_with_extension(self, extension: str) -> Generator[CloudPath, None, None]:
        for file in self.bucket_path.rglob(f"*{extension}"):
            yield CloudPath(file)

    def find_paths_containing(self, substring: str) -> Generator[CloudPath, None, None]:
        for file in self.bucket_path.rglob(f"*{substring}*"):
            yield CloudPath(file)

    def delete_dir(self, dir_path: str | os.PathLike, confirm: bool = True):
        """Delete a directory in the bucket."""
        if is_cloud_path(dir_path):
            raise ValueError("dir path must be relative to the bucket.")
        dir_path = self.bucket_path / AnyPath(dir_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {dir_path}")

        if confirm:
            if not input(f"Are you sure you want to delete {dir_path}? (y/n): ").lower() == "y":
                return

        try:
            dir_path.rmtree()
            logger.info(f"Deleted dir {dir_path} from {self.bucket_path}")
        except Exception as e:
            raise RuntimeError(f"Error deleting {dir_path}: {e}")

    def download_to(self, destination: str | os.PathLike) -> None:
        if not is_local_path(destination):
            raise ValueError("Destination path must be a local path.")

        try:
            self.bucket_path.download_to(destination)
        except Exception as e:
            raise RuntimeError(f"Error downloading to {destination}: {e}")
        logger.info(f"Downloaded to {destination}")

    def move_dir(
        self,
        source: str | os.PathLike,
        destination: str | os.PathLike,
        overwrite: bool = False,
        keep_parent: bool = False,
    ) -> None:
        """Move a source folder in this bucket to a destination folder within the same bucket.

        Args:
            source: The source folder to move.
            destination: The destination folder.
            overwrite: Whether to overwrite files if they already exist in the destination.
            keep_parent: Whether to keep the parent directory of the source folder in the destination.

        Raises:
            FileNotFoundError: If the source path does not exist.
            FileExistsError: If a file already exists in the destination and overwrite is False.

        Usage:
            bucket = Bucket("gs://esp-ci-cd-tests/esp-data-tests/")
            bucket.move_dir("temp_folder", "non_empty", overwrite=True, keep_parent=True)
        """
        source = self.bucket_path / AnyPath(source)
        destination = self.bucket_path / AnyPath(destination)

        if not source.exists():
            raise FileNotFoundError(f"Source path does not exist: {source}")

        # move the dir by renaming each file recursively
        for file in source.rglob("*"):
            if file.is_dir():
                continue
            # set new path
            if keep_parent:
                new_path = destination / file.relative_to(source.parent)
            else:
                new_path = destination / file.relative_to(source)
            # check if file exists
            if new_path.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {new_path}")
            elif new_path.exists() and overwrite:
                file.replace(new_path)
            else:
                file.rename(new_path)

    def __repr__(self):
        return f"Bucket({self.bucket_path})"

    def __str__(self):
        return str(self.bucket_path)


class GSBucket(Bucket):
    """Class for managing a Google Cloud Storage bucket.
    Uses some methods from the google-cloud-storage library not available in cloudpathlib.
    """

    def __init__(self, bucket_path):
        super().__init__(bucket_path)
        self.client = GSClient()
        bucket_path_parts = self.bucket_path.parts[1:]  # remove the gs:// prefix
        self.gs_bucket_name = bucket_path_parts[0]
        self.bucket_subpath = AnyPath("/".join(bucket_path_parts[1:]))

    async def async_upload_dir(self, source_dir: str | os.PathLike, workers: int = 8) -> None:
        # creates a new event loop for the async operation
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, self.upload_dir, source_dir, workers)

    def upload_dir(self, source_dir: str | os.PathLike, workers: int = 8) -> None:
        """Upload every file in source_dir, including all files in subdirs to the bucket.
        Does not keep the directory structure. E.g. if source_dir is "dir1", and it contains
        "dir2/file.txt", the file will be uploaded to the bucket as "dir2/file.txt" not "dir1/dir2/file.txt".
        Source script: https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python

        Args:
            source_dir: The dir to upload. Files in the
                dir and its subdirs will be uploaded. An empty string
                means "the current working dir".
            workers: The maximum number of processes to use for the operation. The
                performance impact of this value depends on the use case, but smaller
                files usually benefit from a higher number of processes. Each
                additional process occupies some CPU and memory resources until
                finished. Threads can be used instead of processes by passing
                `worker_type=transfer_manager.THREAD`.
        """
        bucket = self.client.bucket(self.gs_bucket_name)

        # First, recursively get all files in `dir` as Path objects.
        dir_as_path_obj = AnyPath(source_dir)
        file_paths = []
        for path in dir_as_path_obj.rglob("*"):
            if not path.is_file():
                continue
            file_paths.append(str(path.relative_to(source_dir)))

        logger.info("Found {} files.".format(len(file_paths)))

        # Start the upload.
        results = transfer_manager.upload_many_from_filenames(
            bucket,
            file_paths,
            source_directory=source_dir,
            max_workers=workers,
            blob_name_prefix=str(self.bucket_subpath) + "/",
        )

        for name, result in zip(file_paths, results):
            if isinstance(result, Exception):
                logger.error("Failed to upload {} due to exception: {}".format(name, result))
            else:
                logger.info("Uploaded {} to {}.".format(name, bucket.name))

    async def async_upload_data_as_str(self, file_names: list[str | os.PathLike], contents: list[str | bytes]) -> None:
        """Upload a list of str | byte data to the bucket.

        Args:
            file_names: The names of the files to be added to the bucket.
            contents: The contents of the files.
        """
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, self.upload_data_as_str, file_names, contents)

    def upload_data_as_str(self, file_names: list[str | os.PathLike], contents: list[str | bytes]) -> None:
        """Upload a list of byte or string data to the bucket. If string, it will be encoded as UTF-8.

        Args:
            file_names: The names of the files to be added to the bucket.
            contents: The contents of the files.
        """
        bucket = self.client.bucket(self.gs_bucket_name)

        for file_name, content in zip(file_names, contents):
            blob = bucket.blob(str(self.bucket_subpath / file_name))
            blob.upload_from_string(content)
            logger.info(f"{file_name} with contents {content} uploaded to {bucket.name}.")

    def _rsync(
        self,
        destination: AnyPath,
        delete_unmatched: bool = False,
        continue_on_error: bool = True,
        recursive: bool = True,
        gzip_in_flight: str = "",
    ) -> None:
        """Sync the bucket directory with a local / cloud directory.

        Args:
            destination: The local directory to sync with.
            delete_unmatched: Whether to delete files in the destination that are not in the source. Default is False
            continue_on_error: Whether to continue syncing if an error occurs. Default is True
            recursive: Whether to sync recursively. Default is True
            gzip_in_flight: Whether to compress files with the given extensions in flight for faster transfer.
                For e.g. gzip_in_flight="txt,csv,jpg". Default is "" which means none. If "all",
                will try to compress everything, but this may be counterproductive for files that should not be
                compressed.

        Raises:
            subprocess.CalledProcessError: If rsync command fails
        """
        # uses gcloud storage rsync command
        cmd = ["gcloud", "storage", "rsync", str(self), str(destination)]

        if recursive:
            cmd.append("--recursive")
        if delete_unmatched:
            cmd.append("--delete-unmatched-destination-objects")
        if continue_on_error:
            cmd.append("--continue-on-error")

        if gzip_in_flight == "all":
            cmd.append("--gzip-in-flight-all")
        elif gzip_in_flight != "":
            cmd.append(f"--gzip-in-flight={gzip_in_flight}")

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()

        if p.returncode != 0:
            error_msg = stderr.decode("utf-8").strip()
            logger.error(f"Rsync failed with return code {p.returncode}: {error_msg}")
            logger.error(f"Command used: {' '.join(cmd)}")
            if not continue_on_error:
                raise subprocess.CalledProcessError(p.returncode, cmd, stdout, stderr)
        else:
            # Log success message with some stats if available
            output = stdout.decode("utf-8").strip()
            if output:
                logger.info(f"Rsync completed successfully: {output}")
            else:
                logger.info("Rsync completed successfully")

    async def async_rsync(
        self,
        other: str | os.PathLike | Bucket,
        self_is_source: bool = True,
        delete_unmatched: bool = False,
        continue_on_error: bool = True,
        recursive: bool = True,
        gzip_in_flight: str = "",
    ) -> None:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(
                pool,
                self.rsync,
                other,
                self_is_source,
                delete_unmatched,
                continue_on_error,
                recursive,
                gzip_in_flight,
            )

    def rsync(
        self,
        other: str | os.PathLike | Bucket,
        self_is_source: bool = True,
        delete_unmatched: bool = False,
        continue_on_error: bool = True,
        recursive: bool = True,
        gzip_in_flight: str = "",
    ) -> None:
        """Sync this bucket directory with a local / cloud directory.

        Args:
            other: The local / cloud directory to sync with.
            self_is_source: Whether the bucket is the source. Default is True
            delete_unmatched: Whether to delete files in the destination that are not in the source. Default is False
            continue_on_error: Whether to continue syncing if an error occurs. Default is True
            recursive: Whether to sync recursively. Default is True
            gzip_in_flight: Whether to compress files with the given extensions in flight for faster transfer.
                For e.g. gzip_in_flight="txt,csv,jpg". Default is "" which means none. If "all",
                will try to compress everything, but this may be counterproductive for files that should not be
                compressed.

        Raises:
            FileNotFoundError: If destination path does not exist
            subprocess.CalledProcessError: If rsync command fails
        """
        if self_is_source:
            other = AnyPath(str(other))
            if not other.exists():
                raise FileNotFoundError(f"Destination path does not exist: {other}")
            self._rsync(other, delete_unmatched, continue_on_error, recursive, gzip_in_flight)
        else:
            if not isinstance(other, Bucket):
                try:
                    other = Bucket(other)
                except ValueError:
                    raise ValueError("Other must be a Bucket object or a valid cloud path.")
            other._rsync(self, delete_unmatched, continue_on_error, recursive, gzip_in_flight)


class FSBucket:
    """A Bucket that uses an underlying filesystem class.

    Args:

        bucket_path: The path to the bucket.
        fs: The filesystem object to use for the bucket.

    """

    def __init__(self, bucket_path: str | os.PathLike | AnyPath, fs=None):
        if fs is None:
            fs = make_fs(bucket_path)

        bucket_path = strip_cloud_prefix(str(bucket_path))  # remove any cloud prefix, fsspec doesn't like it
        self.bucket_path = AnyPath(bucket_path)  # not using CloudPath because it doesn't work with r2 paths
        self.fs = fs

    def _exists(self, path: str | os.PathLike | AnyPath) -> bool:
        return self.fs.exists(str(path))

    @property
    def exists(self):
        return self.fs.exists(str(self.bucket_path))

    def fopen(self, file_path: str | os.PathLike, mode: str = "rb", **fopen_kwargs):
        """Open a file in the bucket."""
        file_path = AnyPath(file_path)

        return self.fs.open(str(file_path), mode, **fopen_kwargs)

    def list_dirs(self, recursive: bool = True) -> list[AnyPath]:
        if recursive:
            search_str = "**"
        else:
            search_str = "*"

        return [AnyPath(dir) for dir in self.fs.glob(str(self.bucket_path / search_str)) if self.fs.isdir(dir)]

    def list_files(self, recursive: bool = True) -> list[AnyPath]:
        if recursive:
            search_str = "**"
        else:
            search_str = "*"

        return [AnyPath(file) for file in self.fs.glob(str(self.bucket_path / search_str)) if self.fs.isfile(file)]

    def find_files_with_extension(self, extension: str) -> list[AnyPath]:
        return [
            AnyPath(file) for file in self.fs.glob(str(self.bucket_path / f"**/*{extension}")) if self.fs.isfile(file)
        ]

    def find_files_containing(self, substring: str) -> list[AnyPath]:
        return [
            AnyPath(file) for file in self.fs.glob(str(self.bucket_path / f"**/*{substring}*")) if self.fs.isfile(file)
        ]

    def subdir_as_bucket(self, subdir: str | os.PathLike | AnyPath) -> "FSBucket":
        """Create a subdirectory as a Bucket object.
        e.g. if the bucket path is gs://esp-ci-cd-tests/esp-data-tests/,
        and subdir is "subdir", then the new bucket path will be
        gs://esp-ci-cd-tests/esp-data-tests/subdir.
        """
        subpath = self.bucket_path / AnyPath(subdir)

        return FSBucket(subpath, self.fs)  # I'm not sure if this is the right way to do this, using the same fs ?

    def upload_dir(self, source_dir: str | os.PathLike | AnyPath, destination: str | os.PathLike | AnyPath) -> None:
        """Upload a local directory to a destination folder in this bucket."""
        if not is_local_path(source_dir):
            raise ValueError("Source dir must be a local path.")

        try:
            self.fs.put(str(source_dir), str(self.bucket_path / AnyPath(destination)), recursive=True)
        except Exception as e:
            raise RuntimeError(f"Error uploading {source_dir} to {destination}: {e}")
        logger.info(f"Uploaded {source_dir} to {destination}")

    def download_to(self, destination: str | os.PathLike) -> None:
        """Download the bucket contents to a local path. Does recursive download."""
        if not is_local_path(destination):
            raise ValueError("Destination path must be a local path.")

        try:
            self.fs.get(str(self.bucket_path), str(destination), recursive=True)
        except Exception as e:
            raise RuntimeError(f"Error downloading to {destination}: {e}")

        logger.info(f"Downloaded to {destination}")

    def move_dir(self, source: str | os.PathLike, destination: str | os.PathLike, confirm: bool = False) -> None:
        """Move a source folder in this bucket to a destination folder within the same bucket.
        CAUTION: This will overwrite files in the destination if they have the same name as files in the source.
        AND it will delete the source folder after moving.

        Args:
            source: The source folder to move.
            destination: The destination folder.
            confirm: Whether to confirm before moving. Default is False

        Raises:
            FileNotFoundError: If the source path does not exist.
        """
        source = self.bucket_path / AnyPath(source)
        destination = self.bucket_path / AnyPath(destination)

        if confirm:
            if not input(f"Are you sure you want to move {source} to {destination}? (y/n): ").lower() == "y":
                return

        try:
            self.fs.mv(str(source), str(destination))
            logger.info(f"Moved {source} to {destination}")
        except Exception as e:
            raise RuntimeError(f"Error moving {source} to {destination}: {e}")

    def delete_dir(self, dir_path: str | os.PathLike, confirm: bool = True):
        """Delete a directory in the bucket. When = "", this means the whole bucket!"""
        if is_cloud_path(dir_path):
            raise ValueError("dir path must be relative to the bucket.")
        dir_path = self.bucket_path / AnyPath(dir_path) / "**"

        if confirm:
            if not input(f"Are you sure you want to delete {dir_path}? (y/n): ").lower() == "y":
                return

        try:
            # first find all files in the dir and delete them
            self.fs.rm(str(dir_path), recursive=True)
            logger.info(f"Deleted dir {dir_path} from {self.bucket_path}")

        except Exception as e:
            raise RuntimeError(f"Error deleting {dir_path}: {e}")

    async def async_upload_dir(self, source_dir: str | os.PathLike, destination: str | os.PathLike) -> None:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, self.upload_dir, source_dir, destination)

    async def async_download_to(self, destination: str | os.PathLike) -> None:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, self.download_to, destination)

    async def async_move_dir(
        self, source: str | os.PathLike, destination: str | os.PathLike, confirm: bool = False
    ) -> None:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, self.move_dir, source, destination, confirm)

    def __repr__(self):
        return f"R2Bucket({self.bucket_path})"

    def __str__(self):
        return str(self.bucket_path)


class GSBucketV2(FSBucket):
    """This version of Google cloud storage uses the GCSFileSystem from gcsfs.
    It seems more efficient than cloudpathlib and google-cloud-storage.

    Args:
        bucket_path: The path to the bucket. This may either be the bucket name
        or a path like gs://bucket_name/some_folder.
    """

    def __init__(self, bucket_path: str | os.PathLike | AnyPath):
        self.bucket_path = AnyPath(strip_cloud_prefix(str(bucket_path)))
        self.fs = make_gcsfs()


class R2Bucket(FSBucket):
    """A Cloudflare R2 bucket that uses the aws s3 api (s3fs filesystem).

    Args:
        bucket_path: The path to the bucket. This may either be the bucket name
        or a path like r2://bucket_name/some_folder.
    """

    def __init__(self, bucket_path: str | os.PathLike | AnyPath):
        self.bucket_path = AnyPath(strip_cloud_prefix(str(bucket_path)))
        self.fs = make_cloudflarer2fs()
