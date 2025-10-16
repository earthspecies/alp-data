"""Tests for the PureGSPath class."""

import pytest
from esp_data.io import PureGSPath


def test_pure_gcs_path_creation():
    """Test creating PureGSPath instances."""
    # Test basic GCS path
    path = PureGSPath("gs://my-bucket/folder/file.txt")
    assert str(path) == "gs://my-bucket/folder/file.txt"

    # Test bucket-only path
    path = PureGSPath("gs://my-bucket")
    assert str(path) == "gs://my-bucket"

    # Test nested path
    path = PureGSPath("gs://my-bucket/folder/subfolder/file.txt")
    assert str(path) == "gs://my-bucket/folder/subfolder/file.txt"

    # Test that non-GCS paths raise ValueError
    with pytest.raises(ValueError, match="Path must start with 'gs://': /local/path"):
        PureGSPath("/local/path")


def test_bucket_property():
    """Test the bucket property."""
    path = PureGSPath("gs://my-bucket/folder/file.txt")
    assert path.bucket == "my-bucket"

    path = PureGSPath("gs://another-bucket")
    assert path.bucket == "another-bucket"

    # Test bucket-only path
    path = PureGSPath("gs://bucket-only")
    assert path.bucket == "bucket-only"


def test_path_manipulation():
    """Test standard pathlib operations on PureGSPath."""
    path = PureGSPath("gs://my-bucket/folder/file.txt")

    # Test name property
    assert path.name == "file.txt"

    # Test suffix property
    assert path.suffix == ".txt"

    # Test stem property
    assert path.stem == "file"

    # Test parent property
    parent = path.parent
    assert str(parent) == "gs://my-bucket/folder"

    # Test parts property
    parts = path.parts
    assert parts == ('gs://my-bucket/', 'folder', 'file.txt')


def test_path_joining():
    """Test joining paths with PureGSPath."""
    base = PureGSPath("gs://my-bucket/folder")
    joined = base / "subfolder" / "file.txt"
    assert str(joined) == "gs://my-bucket/folder/subfolder/file.txt"

    # Test joining with string
    joined = base / "file.txt"
    assert str(joined) == "gs://my-bucket/folder/file.txt"

    # Test joining with absolute path
    joined = base / "/absolute/path"
    assert str(joined) == "gs://my-bucket/absolute/path"

    # Joining from bucket-only should insert root separator
    bucket_only = PureGSPath("gs://my-bucket")
    assert str(bucket_only / "bar/text.text") == "gs://my-bucket/bar/text.text"

    # Joining Posix pathlike should behave the same
    from pathlib import PosixPath, PurePosixPath

    assert str(bucket_only / PosixPath("bar/text.text")) == "gs://my-bucket/bar/text.text"
    assert str(bucket_only / PurePosixPath("bar/text.text")) == "gs://my-bucket/bar/text.text"


def test_path_comparison():
    """Test path comparison operations."""
    path1 = PureGSPath("gs://my-bucket/folder/file.txt")
    path2 = PureGSPath("gs://my-bucket/folder/file.txt")
    path3 = PureGSPath("gs://my-bucket/folder/other.txt")

    assert path1 == path2
    assert path1 != path3
    assert path1 < path3  # Based on string comparison

    # Test comparison with string
    assert path1 != "gs://my-bucket/folder/file.txt"


def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test non-string input
    with pytest.raises(TypeError, match="Expected string, got <class 'int'>"):
        PureGSPath(123)

    # Test empty string
    with pytest.raises(ValueError, match="Path must start with 'gs://': "):
        PureGSPath("")

    # Test invalid protocol
    with pytest.raises(ValueError, match="Path must start with 'gs://': s3://bucket/file"):
        PureGSPath("s3://bucket/file")


def test_edge_cases():
    """Test edge cases and special scenarios."""
    # Test path with multiple consecutive slashes
    path = PureGSPath("gs://my-bucket//folder///file.txt")
    assert path.bucket == "my-bucket"

    # Test path ending with slash
    path = PureGSPath("gs://my-bucket/folder/")
    assert path.bucket == "my-bucket"
    assert path.name == "folder"

    # Test path with dots
    path = PureGSPath("gs://my-bucket/folder/../file.txt")
    assert path.bucket == "my-bucket"


def test_drive_root_anchor_parts():
    p = PureGSPath("gs://bucket/folder/file.txt")
    # drive/root/anchor
    assert p.drive == "gs://bucket"
    assert p.root == "/"
    assert p.anchor == "gs://bucket/"

    # parts: first element is anchor when rooted
    assert p.parts[0] == "gs://bucket/"
    assert p.parts[-1] == "file.txt"

    # bucket-only path: no root, anchor is just drive
    b = PureGSPath("gs://bucket")
    assert b.drive == "gs://bucket"
    assert b.root == ""
    assert b.anchor == "gs://bucket"
    assert b.parts == ("gs://bucket",)


def test_as_uri_and_is_absolute():
    p = PureGSPath("gs://bucket/folder/file.txt")
    assert p.is_absolute() is True
    assert p.as_uri() == "gs://bucket/folder/file.txt"

    # bucket-only is not absolute (no root) -> as_uri should fail
    b = PureGSPath("gs://bucket")
    assert b.is_absolute() is False
    with pytest.raises(ValueError):
        _ = b.as_uri()


def test_parents_and_parent():
    p = PureGSPath("gs://bucket/a/b/c.txt")
    assert str(p.parent) == "gs://bucket/a/b"
    # parents sequence
    assert str(p.parents[0]) == "gs://bucket/a/b"
    assert str(p.parents[1]) == "gs://bucket/a"
    assert str(p.parents[2]) == "gs://bucket/"


def test_match_and_relative_to():
    p = PureGSPath("gs://bucket/f1/f2/file.txt")
    assert p.match("**/*.txt")
    assert p.match("f1/*/file.txt")

    rel = p.relative_to("gs://bucket/")
    assert isinstance(rel, PureGSPath)
    assert str(rel) == "f1/f2/file.txt"


def test_joinpath_equivalence():
    b = PureGSPath("gs://bucket/base")
    assert str(b.joinpath("sub", "file.txt")) == "gs://bucket/base/sub/file.txt"
    # joining an absolute subpath (bucket-rooted)
    assert str(b / "/abs/thing") == "gs://bucket/abs/thing"
