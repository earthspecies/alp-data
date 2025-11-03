"""Integration tests for filesystem operations with path objects.

This module tests the integration between path objects (from paths.py) and
filesystem operations (from filesystem.py).
"""

from pathlib import Path

import pytest

from esp_data.io import anypath, filesystem_from_path


def test_anypath_local_path_with_file_operations():
    """Test anypath with local paths and file I/O operations."""
    path = anypath("tests/samples/file1.txt")
    assert isinstance(path, Path)
    assert path.is_file()
    assert path.read_text().strip() == "hello"
    assert path.exists()


@pytest.mark.parametrize(
    "cloud_path",
    [
        "gs://esp-ci-cd-tests/esp-data-tests/file1.txt",
        "r2://esp-ci-cd-tests/esp-data-tests/file1.txt",
    ],
)
def test_cloud_filesystem_operations(cloud_path):
    """Test filesystem operations (upload, info, read, delete) with cloud paths."""
    path = anypath(cloud_path)
    fs = filesystem_from_path(path)

    fs.put("tests/samples/file1.txt", str(path))

    info = fs.info(str(path))
    assert info["size"] == 6
    assert info["type"] == "file"
    with fs.open(str(path), "rb") as f:
        assert f.read() == b"hello\n"

    fs.rm(str(path))
    assert not fs.exists(str(path))


def test_filesystem_from_path():
    """Test filesystem_from_path creates appropriate filesystem objects for different path types."""
    # Test with GCS path
    gs_path = anypath("gs://bucket/file.txt")
    fs = filesystem_from_path(gs_path)
    assert fs is not None

    # Test with R2 path
    r2_path = anypath("r2://bucket/file.txt")
    fs = filesystem_from_path(r2_path)
    assert fs is not None

    # Test with local path
    local_path = anypath("local/file.txt")
    fs = filesystem_from_path(local_path)
    assert fs is not None
