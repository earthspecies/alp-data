"""This is a functional implementation of fileio using the FileSystem approach.

The idea is to simplify the api, moving towards a functional, stateless approach.

We prefer using the FileSystem approach, defaulting to cloudpathlib if it doesn't work.
"""

import csv
import logging
import os
from io import StringIO
from typing import Any

from esp_data.paths import AnyPath, is_cloud_path, strip_cloud_prefix

from .utils import make_fs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_file(
    file_path: str | os.PathLike | AnyPath, use_fs: bool = False, data: bytes = None, exist_ok: bool = True
) -> bool:
    """Create a file at the given path.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False, which is using cloudpathlib.
        data (bytes, optional): The data to write to the file. Defaults to None.
        exist_ok (bool, optional): If True, do not raise an exception if the file already exists. Defaults to True.

    Returns:
        bool: True if the file was created successfully
    """
    file_path = AnyPath(file_path)
    fs = None
    if use_fs:
        fs = make_fs(file_path)

    try:
        if use_fs and fs is None:
            logger.warning("Using AnyPath method to create file.")

        if not use_fs or fs is None:
            file_path.touch(exist_ok=exist_ok)
            if data:
                file_path.write_bytes(data)
            return True

        file_path = strip_cloud_prefix(file_path)
        with fs.open(file_path, "wb") as f:
            f.write(data or b"")

        return True
    except Exception as e:
        raise e


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
    fs = None
    if use_fs:
        fs = make_fs(file_path)

    try:
        if use_fs and fs is None:
            logger.warning("Using AnyPath method to open file.")

        if not use_fs or fs is None:
            return file_path.open(mode, **open_kwargs)

        file_path = strip_cloud_prefix(file_path)
        return fs.open(str(file_path), mode, **open_kwargs)

    except Exception as e:
        raise e


def exists(file_path: str | os.PathLike | AnyPath, use_fs: bool = False) -> bool:
    """Check if the file exists.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to True, which is using cloudpathlib.

    Returns:
        bool: True if the file exists.
    """
    file_path = AnyPath(file_path)
    fs = None
    if use_fs:
        fs = make_fs(file_path)

    try:
        if use_fs and fs is None:
            logger.warning("Using AnyPath method to check if file exists.")

        if not use_fs or fs is None:
            return file_path.exists()

        file_path = strip_cloud_prefix(file_path)
        return fs.exists(file_path)

    except Exception as e:
        raise e


def list_files(dir_path: str | os.PathLike | AnyPath, pattern: str = "**", use_fs: bool = False) -> list[str]:
    """List files in the given directory.

    Args:
        dir_path (str | os.PathLike | AnyPath): The path to the directory.
        pattern (str, optional): A pattern to match the files. Defaults to "**" which lists all
        files and dirs recursively.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False, which is using cloudpathlib.

    Returns:
        list[str]: A list of file paths if successful.
    """
    dir_path = AnyPath(dir_path)
    fs = None
    if use_fs:
        fs = make_fs(dir_path)

    if not exists(dir_path):
        logger.warning(f"Directory {dir_path} does not exist, aborting.")
        return []

    try:
        if use_fs and fs is None:
            logger.warning("Using AnyPath method to list files.")

        if not use_fs or fs is None:
            return [str(f) for f in dir_path.rglob(pattern or "*") if f.is_file() and str(f) != str(dir_path)]

        dir_path = strip_cloud_prefix(dir_path / pattern)

        return [f for f in fs.glob(dir_path) if fs.isfile(f) and f != dir_path]
    except Exception as e:
        raise e


def read_bytes(file_path: str | os.PathLike | AnyPath) -> bytes:
    """Read the contents of the file.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file, local or cloud.

    Returns:
        bytes: The contents of the file if successful.
    """
    file_path = AnyPath(file_path)
    fs = make_fs(file_path)

    if not exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")

    try:
        if fs is None:
            return file_path.read_bytes()

        file_path = strip_cloud_prefix(file_path)
        return fs.read_bytes(file_path)

    except Exception as e:
        raise e


def find_files(dir_path: str | os.PathLike | AnyPath, pattern: str) -> list[str]:
    """Find files in the given directory matching the pattern. Alias for list_files.

    Examples:
        - match all files in folder "data": find_files("data", "*")
        - match all files in folder "data" with extension ".txt": find_files("data", "*.txt")
        - match all files in folder "data" with extension ".txt" recursively: find_files("data", "**/*.txt")

    Args:
        dir_path (str | os.PathLike | AnyPath): The path to the directory.
        pattern (str): A pattern to match the files.

    Returns:
        list[str]: A list of file paths if successful.
    """
    return list_files(dir_path, pattern=pattern)


def read_text(file_path: str | os.PathLike | AnyPath) -> str:
    """Read the contents of the file as text.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file, local or cloud.

    Returns:
        str: The contents of the file as text if successful.
    """
    file_path = AnyPath(file_path)
    fs = make_fs(file_path)

    if not exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")

    try:
        if fs is None:
            return file_path.read_text()

        file_path = strip_cloud_prefix(file_path)
        return fs.read_text(file_path)

    except Exception as e:
        raise e


def write_bytes(file_path: str | os.PathLike | AnyPath, data: bytes, use_fs: bool = False) -> bool:
    """Write the data to the file.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file, local or cloud.
        data (bytes): The data to write to the file.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False, which is using cloudpathlib.

    Returns:
        bool: True if the data was written successfully.
    """
    file_path = AnyPath(file_path)
    fs = None
    if use_fs:
        fs = make_fs(file_path)

    # TODO: for some reason writing to a file doesn't work with fs, need to investigate
    # error is: PermissionError: The request signature we calculated does not match the signature you provided.
    # Check your secret access key and signing method.
    # fs = make_fs(file_path)

    try:
        if use_fs and fs is None:
            logger.warning("Using AnyPath method to write bytes to file.")

        if not use_fs or fs is None:
            file_path.write_bytes(data)
            return True

        file_path = strip_cloud_prefix(file_path)
        fs.write_bytes(file_path, data)
        return True

    except Exception as e:
        raise e


def write_text(
    file_path: str | os.PathLike | AnyPath, text: str, use_fs: bool = False, encoding: str = None, newline=None
) -> bool:
    """Write the text to the file.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file, local or cloud.
        text (str): The text to write to the file.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False, which is using cloudpathlib.
        encoding (str, optional): The encoding to use. Defaults to None.
        newline ([type], optional): The newline character to use. Defaults to None.

    Returns:
        bool: True if the text was written successfully
    """
    file_path = AnyPath(file_path)
    fs = None
    if use_fs:
        fs = make_fs(file_path)

    try:
        if use_fs and fs is None:
            logger.warning("Using AnyPath method to write text to file.")

        if not use_fs or fs is None:
            file_path.write_text(text, encoding=encoding, newline=newline)
            return True

        file_path = strip_cloud_prefix(file_path)
        fs.write_text(file_path, text, encoding=encoding, newline=newline)
        return True

    except Exception as e:
        raise e


def delete_file(file_path: str | os.PathLike | AnyPath, use_fs: bool = False) -> bool:
    """Delete the file at the given path.

    Args:
        file_path (str | os.PathLike | AnyPath): The path to the file, local or cloud.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False, which is using cloudpathlib.

    Returns:
        bool: True if the file was deleted successfully
    """
    file_path = AnyPath(file_path)

    if not exists(file_path):
        logger.warning(f"File {file_path} does not exist, aborting.")
        return False

    fs = None
    if use_fs:
        fs = make_fs(file_path)

    try:
        if use_fs and fs is None:
            logger.warning("Using AnyPath method to delete file.")

        if not use_fs or fs is None:
            file_path.unlink()
            return True

        fs.rm(strip_cloud_prefix(file_path))
        return True

    except Exception as e:
        raise e


def delete_dir(dir_path: str | os.PathLike | AnyPath) -> bool:
    """Delete the directory at the given path.

    Args:
        dir_path (str | os.PathLike | AnyPath): The path to the directory, local or cloud.

    Returns:
        bool: True if the directory was deleted successfully
    """
    dir_path = AnyPath(dir_path)
    fs = make_fs(dir_path)

    if not exists(dir_path):
        logger.info(f"Directory {dir_path} does not exist, aborting.")
        return False

    try:
        if fs is None:
            # only empty dirs removed
            dir_path.rmdir()
            return True

        # list everything here
        files = list_files(dir_path)
        # pop the element that matches the dir_path
        dir_path = strip_cloud_prefix(dir_path)
        fs.rm(files)

        return True
    except Exception as e:
        raise e


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
    fs = None
    if use_fs:
        fs = make_fs(dir_path)

    try:
        if use_fs and fs is None:
            logger.warning("Using AnyPath method to create directory.")

        if not use_fs or fs is None:
            # make a temporary file there
            AnyPath(dir_path / "temp").touch()
            dir_path.mkdir(parents=True, exist_ok=exist_ok)
            # delete the temporary file
            AnyPath(dir_path / "temp").unlink()
            return True

        # make a temporary file theret
        tmp_file = strip_cloud_prefix(dir_path / ".temp")
        fs.touch(tmp_file)
        fs.mkdirs(strip_cloud_prefix(dir_path), exist_ok=exist_ok)

        return True
    except Exception as e:
        raise e


def upload(source: str | os.PathLike | AnyPath, destination: str | AnyPath, use_fs: bool = False) -> bool:
    """Upload a file / dir to the destination. Please end dirs to upload to with a trailing /.

    Args:
        source (str | os.PathLike | AnyPath): The path to the file to upload.
        destination (str | os.PathLike | AnyPath): The destination path, local or cloud.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.

    Returns:
        bool: True if the file was uploaded successfully.

    Example:
        # Upload a file to a cloud bucket
        upload_to("local_file.txt", "gs://bucket_name/path/to/file.txt")

        # Upload a dir to a cloud bucket
        upload_to("local_dir/", "gs://bucket_name/path/to/dir/")
    """
    source = AnyPath(source)
    destination = AnyPath(destination)
    fs = None
    if use_fs:
        fs = make_fs(destination)

    try:
        if use_fs and fs is None:
            logger.warning("Using AnyPath method to upload file.")

        if not use_fs or fs is None:
            destination.upload_from(source, force_overwrite_to_cloud=True)
            return True

        destination = strip_cloud_prefix(destination)
        source = strip_cloud_prefix(source)
        fs.put(source, destination, recursive=True)

        return True
    except Exception as e:
        raise e


def download(source: str | os.PathLike, destination: str | os.PathLike, use_fs: bool = False) -> bool:
    """Download a file / dir to the destination. Please end dirs to download with a trailing /.

    Args:
        source (str | os.PathLike): The path to the file / dir to download.
        destination (str | os.PathLike): The destination path, local or cloud.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.

    Returns:
        bool: True if the file was downloaded successfully.

    Example:
        # Download a file from a cloud bucket
        download_to("gs://bucket_name/path/to/file.txt", "local_file.txt")

        # Download a dir from a cloud bucket
        download_to("gs://bucket_name/path/to/dir/", "local_dir/")
    """
    source = AnyPath(source)
    destination = AnyPath(destination)
    fs = None
    if use_fs:
        fs = make_fs(source)

    try:
        if use_fs and fs is None:
            logger.warning("Using AnyPath method to download")

        if not use_fs or fs is None:
            source.download_to(destination)
            return True

        source = strip_cloud_prefix(source)
        destination = strip_cloud_prefix(destination)
        fs.get(source, destination, recursive=True)

        return True
    except Exception as e:
        raise e


def copy(source: str | os.PathLike | AnyPath, destination: str | os.PathLike | AnyPath, use_fs: bool = False) -> bool:
    """Copy a file / dir to the destination. Please end dirs to copy with a trailing /.

    Args:
        source (str | os.PathLike | AnyPath): The path to the file / dir to copy.
        destination (str | os.PathLike | AnyPath): The destination path, local or cloud.
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.

    Returns:
        bool: True if the file / dir was copied successfully.

    Example:
        # Copy a file to a cloud bucket
        copy("local_file.txt", "gs://bucket_name/path/to/file.txt")

        # Copy a dir to a cloud bucket
        copy("local_dir/", "gs://bucket_name/path/to/dir/")
    """
    source = AnyPath(source)
    destination = AnyPath(destination)

    if is_cloud_path(source) and is_cloud_path(destination) and source is not destination:
        raise ValueError(
            """Source and destination must be of the same type if they are both cloud paths.
             For e.g., we cannot copy from S3 to GCS"""
        )

    if not exists(source):
        logger.warning(f"Source {source} does not exist, aborting.")
        return False

    fs = None
    if use_fs:
        fs = make_fs(source)

    try:
        if use_fs and fs is None:
            logger.warning("Using AnyPath method to copy file.")

        if not use_fs or fs is None:
            source.copy_to(destination)
            return True

        fs.put(strip_cloud_prefix(source), strip_cloud_prefix(destination))

        return True
    except Exception as e:
        raise e


def write_rows_to_csv(rows: list[dict], *, file_path: str | AnyPath, mode: str = "a", use_fs: bool = False) -> None:
    """Write a list of dicts to a CSV file. Allows appending to the file.

    Args:
        rows (list[dict]): The list of rows to write.
        file_path (str | AnyPath): The path to the CSV file.
        mode (str, optional): The mode to open the file in. Defaults to "a".
        use_fs (bool, optional): If True, use the FileSystem approach. Defaults to False.
    """
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
