"""Integration tests for filesystem operations with path objects.

This module tests the integration between path objects (from paths.py) and
filesystem operations (from filesystem.py).
"""

import uuid
from pathlib import Path

import pytest

from esp_data.io import anypath, filesystem_from_path
from esp_data.io.paths import PureGSPath, PureR2Path


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
        f"gs://esp-ci-cd-tests/esp-data-tests/test-{uuid.uuid4()}.txt",
        f"r2://esp-ci-cd-tests/esp-data-tests/test-{uuid.uuid4()}.txt",
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


def test_unescaped_glob_patterns_rejected():
    """Test that paths with unescaped glob patterns are rejected."""
    # Test asterisk
    with pytest.raises(ValueError, match="Glob patterns are not supported"):
        PureGSPath("gs://bucket/folder/*.txt")

    # Test question mark
    with pytest.raises(ValueError, match="Glob patterns are not supported"):
        PureGSPath("gs://bucket/what?.txt")

    # Test bracket
    with pytest.raises(ValueError, match="Glob patterns are not supported"):
        PureGSPath("gs://bucket/file[1-3].txt")

    # Test R2 path with glob pattern
    with pytest.raises(ValueError, match="Glob patterns are not supported"):
        PureR2Path("s3://bucket/*.txt")


def test_escaped_glob_characters():
    """Test that escaped glob characters are allowed and properly unescaped."""
    # Test escaped asterisk
    p1 = PureGSPath("gs://bucket/folder/file\\*.txt")
    assert str(p1) == "gs://bucket/folder/file*.txt"

    # Test escaped question mark
    p2 = PureGSPath("gs://bucket/what\\?.txt")
    assert str(p2) == "gs://bucket/what?.txt"

    # Test escaped bracket
    p3 = PureGSPath("gs://bucket/file\\[1\\].txt")
    assert str(p3) == "gs://bucket/file[1].txt"

    # Test multiple escaped characters
    p4 = PureGSPath("gs://bucket/my-file\\*-v\\?.txt")
    assert str(p4) == "gs://bucket/my-file*-v?.txt"

    # Test R2 path with escaped characters
    p5 = PureR2Path("s3://bucket/file\\*.txt")
    assert str(p5) == "s3://bucket/file*.txt"

    # Test path operations preserve literal glob characters
    p6 = PureGSPath("gs://bucket/folder") / "file\\*.txt"
    assert str(p6) == "gs://bucket/folder/file*.txt"


def test_escaped_and_unescaped_mixed():
    """Test that mixing escaped and unescaped glob characters works correctly."""
    # Escaped glob should work
    p1 = PureGSPath("gs://bucket/file\\*.txt")
    assert str(p1) == "gs://bucket/file*.txt"

    # Unescaped glob should fail
    with pytest.raises(ValueError, match="Glob patterns are not supported"):
        PureGSPath("gs://bucket/file*.txt")

    # Multiple escaped globs should work
    p2 = PureGSPath("gs://bucket/\\*\\?\\[test\\].txt")
    assert str(p2) == "gs://bucket/*?[test].txt"
