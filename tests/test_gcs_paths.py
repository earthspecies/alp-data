"""Tests for the PureGCSPath class."""

import pytest
# from esp_data.io.gcs_pathlib_delete import PureGCSPath, gcs_path
from esp_data.io import PureGCSPath

def test_pure_gcs_path_creation():
    """Test creating PureGCSPath instances."""
    # Test basic GCS path
    path = PureGCSPath("gs://my-bucket/folder/file.txt")
    assert str(path) == "gs://my-bucket/folder/file.txt"

    # Test bucket-only path
    path = PureGCSPath("gs://my-bucket")
    assert str(path) == "gs://my-bucket"

    # Test nested path
    path = PureGCSPath("gs://my-bucket/folder/subfolder/file.txt")
    assert str(path) == "gs://my-bucket/folder/subfolder/file.txt"

    # Test that non-GCS paths raise ValueError
    with pytest.raises(ValueError, match="Path must start with 'gs://': /local/path"):
        PureGCSPath("/local/path")


def test_bucket_property():
    """Test the bucket property."""
    path = PureGCSPath("gs://my-bucket/folder/file.txt")
    assert path.bucket == "my-bucket"

    path = PureGCSPath("gs://another-bucket")
    assert path.bucket == "another-bucket"

    # Test bucket-only path
    path = PureGCSPath("gs://bucket-only")
    assert path.bucket == "bucket-only"


def test_object_path_property():
    """Test the object_path property."""
    path = PureGCSPath("gs://my-bucket/folder/file.txt")
    assert path.object_path == "folder/file.txt"

    path = PureGCSPath("gs://my-bucket/folder/subfolder/file.txt")
    assert path.object_path == "folder/subfolder/file.txt"

    # Test bucket-only path
    path = PureGCSPath("gs://my-bucket")
    assert path.object_path == ""


# def test_is_gcs_path_property():
#     """Test the is_gcs_path property."""
#     path = PureGCSPath("gs://my-bucket/folder/file.txt")
#     assert path.is_gcs_path is True

#     path = PureGCSPath("gs://my-bucket")
#     assert path.is_gcs_path is True


def test_with_bucket_method():
    """Test the with_bucket method."""
    path = PureGCSPath("gs://old-bucket/folder/file.txt")
    new_path = path.with_bucket("new-bucket")
    assert str(new_path) == "gs://new-bucket/folder/file.txt"
    assert new_path.bucket == "new-bucket"

    # Test bucket-only path
    path = PureGCSPath("gs://old-bucket")
    new_path = path.with_bucket("new-bucket")
    assert str(new_path) == "gs://new-bucket"


def test_relative_to_bucket_method():
    """Test the relative_to_bucket method."""
    path = PureGCSPath("gs://my-bucket/folder/file.txt")
    rel_path = path.relative_to_bucket()
    assert rel_path == "/folder/file.txt"

    # Test bucket-only path
    path = PureGCSPath("gs://my-bucket")
    rel_path = path.relative_to_bucket()
    assert rel_path == "/"


def test_path_manipulation():
    """Test standard pathlib operations on PureGCSPath."""
    path = PureGCSPath("gs://my-bucket/folder/file.txt")

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
    assert len(parts) >= 2  # At least gs://bucket and the object path


def test_path_joining():
    """Test joining paths with PureGCSPath."""
    base = PureGCSPath("gs://my-bucket/folder")
    joined = base / "subfolder" / "file.txt"
    assert str(joined) == "gs://my-bucket/folder/subfolder/file.txt"

    # Test joining with string
    joined = base / "file.txt"
    assert str(joined) == "gs://my-bucket/folder/file.txt"

    # Test joining with absolute path
    joined = base / "/absolute/path"
    assert str(joined) == "gs://my-bucket/absolute/path"


def test_path_comparison():
    """Test path comparison operations."""
    path1 = PureGCSPath("gs://my-bucket/folder/file.txt")
    path2 = PureGCSPath("gs://my-bucket/folder/file.txt")
    path3 = PureGCSPath("gs://my-bucket/folder/other.txt")

    assert path1 == path2
    assert path1 != path3
    assert path1 < path3  # Based on string comparison

    # Test comparison with string
    assert path1 != "gs://my-bucket/folder/file.txt"


def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test non-string input
    with pytest.raises(TypeError, match="Expected string, got <class 'int'>"):
        PureGCSPath(123)

    # Test empty string
    with pytest.raises(ValueError, match="Path must start with 'gs://': "):
        PureGCSPath("")

    # Test invalid protocol
    with pytest.raises(ValueError, match="Path must start with 'gs://': s3://bucket/file"):
        PureGCSPath("s3://bucket/file")


def test_edge_cases():
    """Test edge cases and special scenarios."""
    # Test path with multiple consecutive slashes
    path = PureGCSPath("gs://my-bucket//folder///file.txt")
    assert path.bucket == "my-bucket"
    assert path.object_path == "folder/file.txt"  # Should normalize slashes

    # Test path ending with slash
    path = PureGCSPath("gs://my-bucket/folder/")
    assert path.bucket == "my-bucket"
    assert path.object_path == "folder"
    assert path.name == "folder"

    # Test path with dots
    path = PureGCSPath("gs://my-bucket/folder/../file.txt")
    assert path.bucket == "my-bucket"
    assert path.object_path == "folder/../file.txt"  # Dots are preserved as-is


def test_drive_root_anchor_parts():
    p = PureGCSPath("gs://bucket/folder/file.txt")
    # drive/root/anchor
    assert p.drive == "gs://bucket"
    assert p.root == "/"
    assert p.anchor == "gs://bucket/"

    # parts: first element is anchor when rooted
    assert p.parts[0] == "gs://bucket/"
    assert p.parts[-1] == "file.txt"

    # bucket-only path: no root, anchor is just drive
    b = PureGCSPath("gs://bucket")
    assert b.drive == "gs://bucket"
    assert b.root == ""
    assert b.anchor == "gs://bucket"
    assert b.parts == ("gs://bucket",)


def test_as_uri_and_is_absolute():
    p = PureGCSPath("gs://bucket/folder/file.txt")
    assert p.is_absolute() is True
    assert p.as_uri() == "gs://bucket/folder/file.txt"

    # bucket-only is not absolute (no root) -> as_uri should fail
    b = PureGCSPath("gs://bucket")
    assert b.is_absolute() is False
    with pytest.raises(ValueError):
        _ = b.as_uri()


def test_parents_and_parent():
    p = PureGCSPath("gs://bucket/a/b/c.txt")
    assert str(p.parent) == "gs://bucket/a/b"
    # parents sequence
    assert str(p.parents[0]) == "gs://bucket/a/b"
    assert str(p.parents[1]) == "gs://bucket/a"
    assert str(p.parents[2]) == "gs://bucket/"


def test_match_and_relative_to():
    p = PureGCSPath("gs://bucket/f1/f2/file.txt")
    assert p.match("**/*.txt")
    assert p.match("f1/*/file.txt")

    rel = p.relative_to("gs://bucket/")
    assert isinstance(rel, PureGCSPath)
    assert str(rel) == "f1/f2/file.txt"


def test_joinpath_equivalence():
    b = PureGCSPath("gs://bucket/base")
    assert str(b.joinpath("sub", "file.txt")) == "gs://bucket/base/sub/file.txt"
    # joining an absolute subpath (bucket-rooted)
    assert str(b / "/abs/thing") == "gs://bucket/abs/thing"



if __name__ == "__main__":
    pytest.main([__file__])
