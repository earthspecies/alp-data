"""This is a functional implementation of fileio using the FileSystem approach.

The idea is to simplify the api, moving towards a functional, stateless approach.

We prefer using the FileSystem approach, defaulting to cloudpathlib if it doesn't work.
"""

import logging
import os
import shutil
import subprocess
from io import StringIO
from typing import Any

from cloudpathlib import GSPath

from esp_data.paths import AnyPath, is_local_path, strip_cloud_prefix

from .utils import make_fs

logger = logging.getLogger("esp_data")


# def create_file_local(file_path: str | os.PathLike | AnyPath, data: bytes = None, exist_ok: bool = True) -> bool:
#     """Create a file at the given path.

#     Arguments
#     ---------
#         file_path (str | os.PathLike | AnyPath): The path to the file.
#         data (bytes, optional): The data to write to the file. Defaults to None.
#         exist_ok (bool, optional): If True, do not raise an exception if the file already exists. Defaults to True.

#     Returns
#     -------
#         bool: True if the file was created successfully

#     Example
#     -------
#         >>> create_file_local("local_file.txt", data=b"Hello, world!")
#         >>> with open("local_file.txt", "rb") as f:
#         ...     print(f.read())
#         Hello, world!
#     """
#     file_path = AnyPath(file_path)

#     parent_dir = file_path.parent
#     parent_dir.mkdir(parents=True, exist_ok=True)
#     file_path.touch(exist_ok=exist_ok)
#     if data:
#         file_path.write_bytes(data)
#     return True


# def create_file_fs(file_path: str | os.PathLike | AnyPath, data: bytes = None) -> bool:
#     """Create a file at the given cloud path.

#     Arguments
#     ---------
#         file_path (str | os.PathLike | AnyPath): The path to the file.
#         data (bytes, optional): The data to write to the file. Defaults to None.
#         exist_ok (bool, optional): If True, do not raise an exception if the file already exists. Defaults to True.

#     Returns
#     -------
#         bool: True if the file was created successfully

#     Example
#     -------
#         >>> create_file_fs("gs://esp-ci-cd-tests/esp-data-tests/create_file_fs_test.txt", data=b"Hello, world!")
#         True
#     """
#     file_path = AnyPath(file_path)

#     fs = make_fs(file_path)
#     file_path_str = strip_cloud_prefix(file_path)
#     with fs.open(file_path_str, "wb") as f:
#         f.write(data or b"")
#     return True


# def create_file(
#     file_path: str | os.PathLike | AnyPath, use_fs: bool = False, data: bytes = None, exist_ok: bool = True
# ) -> bool:
#     """Create a file at the given path.

#     Args:
#         file_path (str | os.PathLike | AnyPath): The path to the file.
#         use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False, which is using cloudpathlib.
#         data (bytes, optional): The data to write to the file. Defaults to None.
#         exist_ok (bool, optional): If True, do not raise an exception if the file already exists. Defaults to True.

#     Returns:
#         bool: True if the file was created successfully
#     """
#     file_path = AnyPath(file_path)

#     # Handle local path creation directly
#     if is_local_path(file_path):
#         return create_file_local(file_path, data, exist_ok)

#     # Try AnyPath method first if not using filesystem or if filesystem creation failed
#     if not use_fs:
#         try:
#             file_path.touch(exist_ok=exist_ok)
#             if data:
#                 file_path.write_bytes(data)
#             return True
#         except Exception as e:
#             logger.warning(f"Could not create file using AnyPath method: {e}, trying FileSystem approach.")

#     # FileSystem approach as fallback or if specifically requested
#     try:
#         return create_file_fs(file_path, data)
#     except Exception as e:
#         raise IOError(f"Failed to create file at {file_path} using both methods: {e}") from e


def open_file(file_path: str | os.PathLike | AnyPath, mode: str, use_fs: bool = False, **open_kwargs) -> Any:
    """Open the file at the given path.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file.
        mode (str): The mode to open the file in.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False, which is using cloudpathlib.
        **open_kwargs: Additional keyword arguments to pass to the open function.

    Returns:
        Any: The file object.
    """
    file_path = AnyPath(file_path)

    if is_local_path(file_path):
        return file_path.open(mode, **open_kwargs)

    if not use_fs:
        try:
            return file_path.open(mode, **open_kwargs)
        except Exception as e:
            logger.warning(f"Could not open file using AnyPath method: {e}, trying FileSystem approach.")

    try:
        fs = make_fs(file_path)
        file_path_str = strip_cloud_prefix(file_path)
        return fs.open(str(file_path_str), mode, **open_kwargs)
    except Exception as e:
        raise IOError(f"Failed to open file at {file_path} using both methods: {e}") from e


# def exists(file_path: str | os.PathLike | AnyPath, use_fs: bool = False) -> bool:
#     """Check if the file exists.

#     Args:
#         file_path (str | os.PathLike | AnyPath): The path to the file.
#         use_fs (bool, optional): If True, use the FileSystem approach. Defaults to True, which is using cloudpathlib.

#     Returns:
#         bool: True if the file exists.
#     """
#     file_path = AnyPath(file_path)

#     if is_local_path(file_path):
#         return file_path.exists()

#     if not use_fs:
#         try:
#             return file_path.exists()
#         except Exception as e:
#             logger.warning(
#                 f"Could not check for existence of file using AnyPath method: {e}, trying FileSystem approach."
#             )

#     try:
#         fs = make_fs(file_path)
#         file_path = strip_cloud_prefix(file_path)
#         return fs.exists(file_path)
#     except Exception as e:
#         raise IOError(f"Failed to check file exists at {file_path} using both methods: {e}") from e


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


def list_files(dir_path: str | os.PathLike | AnyPath, pattern: str = "*", use_fs: bool = False) -> list[str]:
    """List files in the given directory.

    Args:
        dir_path (str | os.PathLike | AnyPath): The path to the directory.
        pattern (str, optional): A pattern to match the files. Defaults to "*".
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False, which is using cloudpathlib.

    Returns:
        list[str]: A list of file paths if successful.
    """
    return [f for f in yield_files(dir_path, pattern, use_fs)]


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


def read_bytes(file_path: str | os.PathLike | AnyPath, use_fs: bool = False) -> bytes:
    """Read the contents of the file.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file, local or cloud.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.

    Returns:
        bytes: The contents of the file if successful.
    """
    file_path = AnyPath(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    if is_local_path(file_path):
        return file_path.read_bytes()

    if not use_fs:
        try:
            return file_path.read_bytes()
        except Exception as e:
            logger.warning(f"Could not read file using AnyPath method: {e}, trying FileSystem approach.")

    try:
        fs = make_fs(file_path)
        file_path_str = strip_cloud_prefix(file_path)
        return fs.read_bytes(file_path_str)
    except Exception as e:
        raise IOError(f"Failed to read file {file_path} using both methods: {e}") from e


def read_text(file_path: str | os.PathLike | AnyPath, use_fs: bool = False) -> str:
    """Read the contents of the file as text.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file, local or cloud.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.

    Returns:
        str: The contents of the file as text if successful.
    """
    file_path = AnyPath(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    if is_local_path(file_path):
        return file_path.read_text()

    if not use_fs:
        try:
            return file_path.read_text()
        except Exception as e:
            logger.warning(f"Could not read file as text using AnyPath method: {e}, trying FileSystem approach.")

    try:
        fs = make_fs(file_path)
        file_path_str = strip_cloud_prefix(file_path)
        return fs.read_text(file_path_str)
    except Exception as e:
        raise IOError(f"Failed to read file {file_path} as text using both methods: {e}") from e


def write_bytes(file_path: str | os.PathLike | AnyPath, data: bytes, use_fs: bool = False) -> bool:
    """Write the data to the file.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file, local or cloud.
        data (bytes): The data to write to the file.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.

    Returns:
        bool: True if the data was written successfully.
    """
    file_path = AnyPath(file_path)

    if is_local_path(file_path):
        file_path.write_bytes(data)
        return True

    if not use_fs:
        try:
            file_path.write_bytes(data)
            return True
        except Exception as e:
            logger.warning(f"Could not write bytes using AnyPath method: {e}, trying FileSystem approach.")

    try:
        fs = make_fs(file_path)
        file_path_str = strip_cloud_prefix(file_path)
        fs.write_bytes(file_path_str, data)
        return True
    except Exception as e:
        raise IOError(f"Failed to write bytes to {file_path} using both methods: {e}") from e


def write_text(
    file_path: str | os.PathLike | AnyPath, text: str, use_fs: bool = False, encoding: str = None, newline=None
) -> bool:
    """Write the text to the file.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file, local or cloud.
        text (str): The text to write to the file.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.
        encoding (str, optional): The encoding to use. Defaults to None.
        newline ([type], optional): The newline character to use. Defaults to None.

    Returns:
        bool: True if the text was written successfully
    """
    file_path = AnyPath(file_path)

    if is_local_path(file_path):
        file_path.write_text(text, encoding=encoding, newline=newline)
        return True

    if not use_fs:
        try:
            file_path.write_text(text, encoding=encoding, newline=newline)
            return True
        except Exception as e:
            logger.warning(f"Could not write text using AnyPath method: {e}, trying FileSystem approach.")

    try:
        fs = make_fs(file_path)
        file_path_str = strip_cloud_prefix(file_path)
        fs.write_text(file_path_str, text, encoding=encoding, newline=newline)
        return True
    except Exception as e:
        raise IOError(f"Failed to write text to {file_path} using both methods: {e}") from e


def delete_file(file_path: str | os.PathLike | AnyPath, use_fs: bool = False) -> bool:
    """Delete the file at the given path.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file, local or cloud.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.

    Returns:
        bool: True if the file was deleted successfully
    """
    file_path = AnyPath(file_path)

    if not file_path.exists():
        logger.warning(f"File {file_path} does not exist, aborting.")
        return False

    if is_local_path(file_path):
        file_path.unlink()
        return True

    if not use_fs:
        try:
            file_path.unlink()
            return True
        except Exception as e:
            logger.warning(f"Could not delete file using AnyPath method: {e}, trying FileSystem approach.")

    try:
        fs = make_fs(file_path)
        file_path_str = strip_cloud_prefix(file_path)
        fs.rm(file_path_str)
        return True
    except Exception as e:
        raise IOError(f"Failed to delete file {file_path} using both methods: {e}") from e


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
        files = list_files(dir_path)
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


def upload(source: str | os.PathLike | AnyPath, destination: str | os.PathLike | AnyPath, use_fs: bool = False) -> bool:
    """Upload a file/directory from local path to destination (typically cloud).
    Please end dirs to upload to with a trailing /.

    Args:
        source (str | os.PathLike | AnyPath): The local path to upload from.
        destination (str | os.PathLike | AnyPath): The destination path.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.

    Returns:
        bool: True if the upload was successful.

    Example:
        # Upload a file to a cloud bucket
        upload("local_file.txt", "gs://bucket_name/path/to/file.txt")

        # Upload a dir to a cloud bucket
        upload("local_dir/", "gs://bucket_name/path/to/dir/")
    """
    return copy(source, destination, use_fs)


def download(
    source: str | os.PathLike | AnyPath, destination: str | os.PathLike | AnyPath, use_fs: bool = False
) -> bool:
    """Download a file/directory from source (typically cloud) to local path.
    Please end dirs to download with a trailing /.

    Args:
        source (str | os.PathLike | AnyPath): The source path to download from.
        destination (str | os.PathLike | AnyPath): The local destination path.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.

    Returns:
        bool: True if the download was successful.

    Example:
        # Download a file from a cloud bucket
        download("gs://bucket_name/path/to/file.txt", "local_file.txt")

        # Download a dir from a cloud bucket
        download("gs://bucket_name/path/to/dir/", "local_dir/")
    """
    return copy(source, destination, use_fs)


def gcloud_rsync(
    source: str | os.PathLike | AnyPath,
    destination: str | os.PathLike | AnyPath,
    avoid_copy_if_same: bool = False,
    delete_unmatched: bool = False,
    continue_on_error: bool = True,
    recursive: bool = True,
    gzip_in_flight: str = "",
) -> None:
    """Sync the bucket directory with a local / cloud directory.

    Args:
        source (str | os.PathLike | AnyPath): The source path to sync from.
        destination (str | os.PathLike | AnyPath): The destination path to sync to.
        avoid_copy_if_same: Whether to avoid copying files that are the same in source and destination. Default is False
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
    cmd = ["gcloud", "storage", "rsync", str(source), str(destination)]

    if recursive:
        cmd.append("--recursive")
    if delete_unmatched:
        cmd.append("--delete-unmatched-destination-objects")
    if continue_on_error:
        cmd.append("--continue-on-error")
    if avoid_copy_if_same:
        cmd.append("--no-clobber")

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


def write_rows_to_csv(rows: list[dict], *, file_path: str | AnyPath, mode: str = "a", use_fs: bool = False) -> None:
    """Write a list of dicts to a remote CSV file. Allows appending to the file.

    Args:
        rows (list[dict]): The list of rows to write.
        file_path (str | AnyPath): The path to the CSV file.
        mode (str, optional): The mode to open the file in. Defaults to "a".
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.
    """
    import csv

    if not rows:
        logger.warning("No rows to write, aborting.")
        return

    fs = None
    if use_fs:
        fs = make_fs(file_path)

    fieldnames = list(rows[0].keys())
    # Write batch to string buffer
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    if not exists(file_path):
        writer.writeheader()

    writer.writerows(rows)

    if fs is None:
        with AnyPath(file_path).open(mode=mode) as f:
            f.write(output.getvalue())
    else:
        with fs.open(file_path, mode=mode) as f:
            f.write(output.getvalue())
