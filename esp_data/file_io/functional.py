"""This is a functional implementation of fileio using the FileSystem approach.

The idea is to simplify the api, moving towards a functional, stateless approach.

We prefer using the FileSystem approach, defaulting to cloudpathlib if it doesn't work.
"""

import logging
import os
import shutil

from cloudpathlib import GSPath

from esp_data.paths import AnyPath, is_local_path, strip_cloud_prefix

from .utils import make_fs

logger = logging.getLogger("esp_data")


def move_file(
    source: str | os.PathLike | AnyPath, destination: str | os.PathLike | AnyPath, use_fs: bool = False
) -> bool:
    """Move a file from the source to the destination.

    Args:
        source (str | os.PathLike | AnyPath): The path to the source file.
        destination (str | os.PathLike | AnyPath): The path to the destination file.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.

    Returns:
        bool: True if the file was moved successfully.
    """
    source = AnyPath(source)
    destination = AnyPath(destination)

    if not source.exists():
        raise FileNotFoundError(f"Source {source} does not exist")

    # Handle local path move directly
    if is_local_path(source) and is_local_path(destination):
        try:
            shutil.move(source, destination)
            return True
        except Exception as e:
            raise IOError(f"Failed to move local file {source} to {destination}: {e}") from e

    # For cloud paths, determine which path needs the filesystem
    cloud_path = destination if not is_local_path(destination) else source

    if not use_fs:
        try:
            source.rename(destination)
            return True
        except Exception as e:
            logger.warning(f"Could not move file using AnyPath method: {e}, trying FileSystem approach.")

    try:
        fs = make_fs(cloud_path)
        source_str = strip_cloud_prefix(source)
        destination_str = strip_cloud_prefix(destination)
        fs.mv(source_str, destination_str)
        return True
    except Exception as e:
        raise IOError(f"Failed to move file {source} to {destination} using both methods: {e}") from e


def yield_files(dir_path: str | os.PathLike | AnyPath, pattern: str = "*", use_fs: bool = False):
    """Yield files in the given directory.

    Args:
        dir_path (str | os.PathLike | AnyPath): The path to the directory.
        pattern (str, optional): A pattern to match the files. Defaults to "*".
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False, which is using cloudpathlib.

    Yields:
        str: A file path if successful.
    """
    dir_path = AnyPath(dir_path)

    if not dir_path.exists():
        logger.warning(f"Directory {dir_path} does not exist, aborting.")
        return

    if is_local_path(dir_path):
        for f in dir_path.rglob(pattern):
            if f.is_file() and str(f) != str(dir_path):
                yield str(f)
        return

    if not use_fs:
        try:
            for f in dir_path.rglob(pattern):
                if f.is_file() and str(f) != str(dir_path):
                    yield str(f)
            return
        except Exception as e:
            logger.warning(f"Could not yield files using AnyPath method: {e}, trying FileSystem approach.")

    # FileSystem approach as fallback or if specifically requested
    try:
        fs = make_fs(dir_path)
        glob_path = strip_cloud_prefix(dir_path / pattern)
        cloud_prefix = AnyPath(dir_path).cloud_prefix

        for f in fs.glob(glob_path):
            if fs.isfile(f) and f != glob_path:
                yield (cloud_prefix + f)

    except Exception as e:
        raise IOError(f"Failed to yield files in {dir_path} using both methods: {e}") from e


def delete_dir(dir_path: str | os.PathLike | AnyPath, use_fs: bool = False) -> bool:
    """Delete the directory at the given path.

    Args:
        dir_path (str | os.PathLike | AnyPath): The path to the directory, local or cloud.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.

    Returns:
        bool: True if the directory was deleted successfully
    """
    dir_path = AnyPath(dir_path)

    if not dir_path.exists():
        logger.info(f"Directory {dir_path} does not exist, aborting.")
        return False

    if is_local_path(dir_path):
        try:
            shutil.rmtree(dir_path)
            return True
        except Exception as e:
            if not use_fs:
                raise IOError(f"Failed to delete directory {dir_path}: {e}") from e

    if not use_fs:
        try:
            for f in yield_files(dir_path):
                AnyPath(f).unlink()
            return True
        except Exception as e:
            logger.warning(f"Could not delete directory using AnyPath method: {e}, trying FileSystem approach.")

    try:
        fs = make_fs(dir_path)
        files = list(yield_files(dir_path))
        # Remove all files
        if files:
            fs.rm(files)
        return True
    except Exception as e:
        raise IOError(f"Failed to delete directory {dir_path} using both methods: {e}") from e


def makedirs(dir_path: str | os.PathLike | AnyPath, use_fs: bool = False, exist_ok: bool = True) -> bool:
    """Create a directory at the given path.

    CAUTION: Most cloud storage services do not allow creation of empty directories
    (because they are not real filesystems).
    So, this function will create a temporary file called ".temp" with 0 bytes
    in the directory to check if it can be created.

    Args:
        dir_path (str | os.PathLike | AnyPath): The path to the directory.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.
        exist_ok (bool, optional): If True, do not raise an exception if the directory already exists. Defaults to True.

    Returns:
        bool: True if the directory was created successfully
    """
    dir_path = AnyPath(dir_path)

    if is_local_path(dir_path):
        try:
            dir_path.mkdir(parents=True, exist_ok=exist_ok)
            return True
        except Exception as e:
            raise IOError(f"Failed to create local directory {dir_path}: {e}") from e

    if not use_fs:
        try:
            # Cloud paths - create a temporary file to ensure directory exists
            temp_file = dir_path / ".temp"
            temp_file.touch()
            dir_path.mkdir(parents=True, exist_ok=exist_ok)
            return True
        except Exception as e:
            logger.warning(f"Could not create directory using AnyPath method: {e}, trying FileSystem approach.")

    try:
        fs = make_fs(dir_path)
        dir_path_str = strip_cloud_prefix(dir_path)
        temp_file_str = strip_cloud_prefix(dir_path / ".temp")
        # Create a temp file for cloud storage services
        fs.touch(temp_file_str)
        fs.mkdirs(dir_path_str, exist_ok=exist_ok)
        return True
    except Exception as e:
        raise IOError(f"Failed to create directory {dir_path} using both methods: {e}") from e


def copy(
    source: str | os.PathLike | AnyPath, destination: str | os.PathLike | AnyPath, use_fs: bool = False, **gcloud_kwargs
) -> bool:
    """Copy a file/directory from source to destination.

    If both the source and destination are local paths, the standard file operations are used.
    If both the source and destination are cloud paths on GCP, the gcloud storage rsync command is used.

    Args:
        source (str | os.PathLike | AnyPath): The source path.
        destination (str | os.PathLike | AnyPath): The destination path.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.
        **gcloud_kwargs: Additional keyword arguments to pass to the gcloud storage rsync command.

    Returns:
        bool: True if the copy was successful.
    """
    source = AnyPath(source)
    destination = AnyPath(destination)

    if not source.exists():
        raise FileNotFoundError(f"Source {source} does not exist")

    # Create parent directories for destination if it's a local path
    if is_local_path(destination):
        os.makedirs(os.path.dirname(str(destination)), exist_ok=True)

    # If both paths are local, use standard file operations
    if is_local_path(source) and is_local_path(destination):
        try:
            if source.is_file():
                shutil.copy2(source, destination)
            else:
                shutil.copytree(source, destination)
            return True
        except Exception as e:
            raise IOError(f"Failed to copy local {source} to {destination}: {e}") from e

    is_upload = is_local_path(source) and not is_local_path(destination)
    is_download = not is_local_path(source) and is_local_path(destination)
    # For cloud paths, determine which path needs the filesystem
    cloud_path = destination if not is_local_path(destination) else source

    if not use_fs:
        try:
            if is_upload:
                destination.upload_from(source, force_overwrite_to_cloud=True)
            elif is_download:
                source.download_to(destination)
            else:
                source.copy(destination, force_overwrite_to_cloud=True)
            return True
        except Exception as e:
            logger.warning(f"Could not copy using AnyPath method: {e}, trying FileSystem approach.")

    try:
        fs = make_fs(cloud_path)
        source_str = strip_cloud_prefix(source)
        destination_str = strip_cloud_prefix(destination)

        if is_upload:
            fs.put(source_str, destination_str, recursive=True)
        elif is_download:
            fs.get(source_str, destination_str, recursive=True)
        else:
            if isinstance(source, GSPath) and isinstance(destination, GSPath):
                # call gcloud_rsync
                gcloud_rsync(source, destination, **gcloud_kwargs)
            else:
                raise ValueError("Cloud to cloud copy is only supported for GCP paths")

        return True
    except Exception as e:
        raise IOError(f"Failed to copy {source} to {destination} using both methods: {e}") from e
