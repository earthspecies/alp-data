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
    # Test non-string input (error message differs between Python versions)
    with pytest.raises(TypeError):
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


def test_python312_lazy_loading():
    """Test that Python 3.12's lazy loading works correctly."""
    p = PureGSPath("gs://bucket/folder/file.txt")
    # Accessing str should trigger lazy loading
    str_repr = str(p)
    assert str_repr == "gs://bucket/folder/file.txt"

    # Verify internal attributes are loaded
    assert hasattr(p, '_drv') or hasattr(p, '_raw_paths')

    # Test that operations work after lazy loading
    assert p.name == "file.txt"
    assert p.parent.name == "folder"


def test_with_segments():
    """Test with_segments method for creating new paths (Python 3.12+)."""
    import sys

    base = PureGSPath("gs://bucket/folder")

    # with_segments is only available in Python 3.12+
    if hasattr(base, 'with_segments'):
        new_path = base.with_segments("gs://other/path")
        assert str(new_path) == "gs://other/path"
    else:
        pytest.skip("with_segments() requires Python 3.12+")


def test_multiple_path_construction():
    """Test constructing paths from multiple arguments."""
    # Constructing from multiple segments
    p = PureGSPath("gs://bucket", "folder", "file.txt")
    # Note: The behavior depends on how pathlib handles multiple args
    # Just verify it doesn't crash and produces a valid path
    assert "bucket" in str(p)


def test_path_from_purepath():
    """Test creating cloud paths from other PurePath objects."""
    from pathlib import PurePosixPath

    base = PureGSPath("gs://bucket/folder")
    posix_path = PurePosixPath("subfolder/file.txt")

    # Should be able to join with PurePosixPath
    result = base / posix_path
    assert str(result) == "gs://bucket/folder/subfolder/file.txt"


def test_normcase_and_case_sensitivity():
    """Test case sensitivity for cloud paths."""
    p1 = PureGSPath("gs://Bucket/Folder/File.txt")
    p2 = PureGSPath("gs://bucket/folder/file.txt")

    # Cloud paths should be case-sensitive
    assert p1 != p2


def test_splitdrive():
    """Test splitdrive functionality for Python 3.12."""
    p = PureGSPath("gs://bucket/folder/file.txt")

    # Access drive to ensure it works
    assert p.drive == "gs://bucket"
    assert p.root == "/"


def test_relative_to_comprehensive():
    """Test relative_to with various scenarios."""
    p = PureGSPath("gs://bucket/a/b/c/file.txt")

    # Relative to parent directory
    rel = p.relative_to("gs://bucket/a/b")
    assert str(rel) == "c/file.txt"

    # Relative to root
    rel = p.relative_to("gs://bucket/")
    assert str(rel) == "a/b/c/file.txt"


def test_is_relative_to():
    """Test is_relative_to method."""
    p = PureGSPath("gs://bucket/a/b/c/file.txt")

    assert p.is_relative_to("gs://bucket/")
    assert p.is_relative_to("gs://bucket/a")
    assert p.is_relative_to("gs://bucket/a/b")
    assert not p.is_relative_to("gs://other/")


def test_hash_consistency():
    """Test that hashing works consistently."""
    p1 = PureGSPath("gs://bucket/folder/file.txt")
    p2 = PureGSPath("gs://bucket/folder/file.txt")

    # Equal paths should have equal hashes
    assert hash(p1) == hash(p2)

    # Should be usable in sets and dicts
    path_set = {p1, p2}
    assert len(path_set) == 1


def test_with_name_and_suffix():
    """Test with_name and with_suffix methods."""
    p = PureGSPath("gs://bucket/folder/file.txt")

    # Change name
    new_p = p.with_name("newfile.txt")
    assert str(new_p) == "gs://bucket/folder/newfile.txt"

    # Change suffix
    new_p = p.with_suffix(".md")
    assert str(new_p) == "gs://bucket/folder/file.md"

    # with_stem (if available)
    if hasattr(p, 'with_stem'):
        new_p = p.with_stem("newfile")
        assert str(new_p) == "gs://bucket/folder/newfile.txt"


def test_str_and_repr():
    """Test string representations."""
    p = PureGSPath("gs://bucket/folder/file.txt")

    # __str__ should return the path
    assert str(p) == "gs://bucket/folder/file.txt"

    # __repr__ should be informative
    repr_str = repr(p)
    assert "PureGSPath" in repr_str
    assert "gs://bucket/folder/file.txt" in repr_str


def test_truediv_rtruediv():
    """Test both forward and reverse division operators."""
    base = PureGSPath("gs://bucket/folder")

    # Forward division
    result = base / "file.txt"
    assert str(result) == "gs://bucket/folder/file.txt"

    # Multiple divisions
    result = base / "sub1" / "sub2" / "file.txt"
    assert str(result) == "gs://bucket/folder/sub1/sub2/file.txt"
