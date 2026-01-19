"""Tests for PureCloudPath base class and all cloud path implementations."""

import os
import pytest
from pathlib import Path, PurePosixPath, PureWindowsPath

from esp_data.io import PureGSPath, PureR2Path, PureS3Path, anypath
from esp_data.io.paths import PureCloudPath


class TestPureCloudPathBase:
    """Test the PureCloudPath base class functionality."""

    def test_cloud_prefix_validation(self):
        """Test that subclasses must define cloud_prefix."""
        with pytest.raises(ValueError, match="cloud_prefix must be defined in subclass"):
            class InvalidCloudPath(PureCloudPath):
                pass
            InvalidCloudPath("test")

    def test_abstract_base_class(self):
        """Test that PureCloudPath cannot be instantiated directly."""
        with pytest.raises(ValueError, match="cloud_prefix must be defined in subclass"):
            PureCloudPath("test")


class TestPureS3Path:
    """Test PureS3Path functionality."""

    def test_s3_path_creation(self):
        """Test creating PureS3Path instances."""
        path = PureS3Path("s3://my-bucket/folder/file.txt")
        assert str(path) == "s3://my-bucket/folder/file.txt"
        assert path.bucket == "my-bucket"

    def test_s3_bucket_only(self):
        """Test S3 bucket-only path."""
        path = PureS3Path("s3://my-bucket")
        assert str(path) == "s3://my-bucket"
        assert path.bucket == "my-bucket"

    def test_s3_invalid_protocol(self):
        """Test that non-S3 paths raise ValueError."""
        with pytest.raises(ValueError, match="Path must start with 's3://': gs://bucket/file"):
            PureS3Path("gs://bucket/file")

    def test_s3_path_operations(self):
        """Test standard path operations on PureS3Path."""
        path = PureS3Path("s3://my-bucket/folder/file.txt")
        assert path.name == "file.txt"
        assert path.suffix == ".txt"
        assert path.stem == "file"
        assert str(path.parent) == "s3://my-bucket/folder"

    def test_s3_path_joining(self):
        """Test joining paths with PureS3Path."""
        base = PureS3Path("s3://my-bucket/folder")
        joined = base / "subfolder" / "file.txt"
        assert str(joined) == "s3://my-bucket/folder/subfolder/file.txt"

    def test_s3_path_comparison(self):
        """Test path comparison operations."""
        path1 = PureS3Path("s3://my-bucket/folder/file.txt")
        path2 = PureS3Path("s3://my-bucket/folder/file.txt")
        path3 = PureS3Path("s3://my-bucket/folder/other.txt")

        assert path1 == path2
        assert path1 != path3
        assert path1 < path3


class TestPureR2Path:
    """Test PureR2Path functionality."""

    def test_r2_path_creation(self):
        """Test creating PureR2Path instances."""
        path = PureR2Path("s3://my-bucket/folder/file.txt")
        assert str(path) == "s3://my-bucket/folder/file.txt"
        assert path.bucket == "my-bucket"

    def test_r2_bucket_only(self):
        """Test R2 bucket-only path."""
        path = PureR2Path("s3://my-bucket")
        assert str(path) == "s3://my-bucket"
        assert path.bucket == "my-bucket"

    def test_r2_invalid_protocol(self):
        """Test that non-S3 paths raise ValueError."""
        with pytest.raises(ValueError, match="Path must start with 's3://': gs://bucket/file"):
            PureR2Path("gs://bucket/file")

    def test_r2_path_operations(self):
        """Test standard path operations on PureR2Path."""
        path = PureR2Path("s3://my-bucket/folder/file.txt")
        assert path.name == "file.txt"
        assert path.suffix == ".txt"
        assert path.stem == "file"
        assert str(path.parent) == "s3://my-bucket/folder"

    def test_r2_path_joining(self):
        """Test joining paths with PureR2Path."""
        base = PureR2Path("s3://my-bucket/folder")
        joined = base / "subfolder" / "file.txt"
        assert str(joined) == "s3://my-bucket/folder/subfolder/file.txt"


class TestCrossSchemeValidation:
    """Test validation of cross-scheme path operations."""

    def test_gs_path_rejects_s3_join(self):
        """Test that GS path rejects joining with S3 URL."""
        gs_path = PureGSPath("gs://my-bucket/folder")
        with pytest.raises(ValueError, match="Cannot join gs:// path with s3://.*incompatible cloud schemes"):
            gs_path / "s3://other-bucket/file.txt"

    def test_s3_path_rejects_gs_join(self):
        """Test that S3 path rejects joining with GS URL."""
        s3_path = PureS3Path("s3://my-bucket/folder")
        with pytest.raises(ValueError, match="Cannot join s3:// path with gs://.*incompatible cloud schemes"):
            s3_path / "gs://other-bucket/file.txt"

    def test_gs_path_accepts_same_scheme_join(self):
        """Test that GS path accepts joining with another GS URL (absolute replacement)."""
        gs_path = PureGSPath("gs://my-bucket/folder")
        result = gs_path / "gs://other-bucket/file.txt"
        assert str(result) == "gs://other-bucket/file.txt"

    def test_s3_path_accepts_same_scheme_join(self):
        """Test that S3 path accepts joining with another S3 URL (absolute replacement)."""
        s3_path = PureS3Path("s3://my-bucket/folder")
        result = s3_path / "s3://other-bucket/file.txt"
        assert str(result) == "s3://other-bucket/file.txt"

    def test_r2_path_rejects_gs_join(self):
        """Test that R2 path rejects joining with GS URL."""
        r2_path = PureR2Path("s3://my-bucket/folder")
        with pytest.raises(ValueError, match="Cannot join s3:// path with gs://.*incompatible cloud schemes"):
            r2_path / "gs://other-bucket/file.txt"


class TestPathInteractions:
    """Test interactions between PureCloudPath and standard pathlib objects."""

    def test_cloud_path_with_posix_path(self):
        """Test joining cloud paths with PosixPath objects."""
        from pathlib import PosixPath, PurePosixPath

        # Test with PurePosixPath
        gs_path = PureGSPath("gs://bucket/folder")
        posix_path = PurePosixPath("subfolder/file.txt")
        result = gs_path / posix_path
        assert str(result) == "gs://bucket/folder/subfolder/file.txt"

        # Test with PosixPath
        posix_path = PosixPath("subfolder/file.txt")
        result = gs_path / posix_path
        assert str(result) == "gs://bucket/folder/subfolder/file.txt"

    def test_cloud_path_with_os_pathlike(self):
        """Test that cloud paths work with os.PathLike protocol."""
        class CustomPathLike:
            def __fspath__(self):
                return "subfolder/file.txt"

        gs_path = PureGSPath("gs://bucket/folder")
        custom_path = CustomPathLike()
        result = gs_path / custom_path
        assert str(result) == "gs://bucket/folder/subfolder/file.txt"

    def test_cloud_path_equality_with_different_types(self):
        """Test equality between cloud paths and other types."""
        gs_path = PureGSPath("gs://bucket/file.txt")

        # Should not equal string
        assert gs_path != "gs://bucket/file.txt"

        # Should not equal different cloud path types
        s3_path = PureS3Path("s3://bucket/file.txt")
        assert gs_path != s3_path

        # Should not equal Path objects
        local_path = Path("gs://bucket/file.txt")
        assert gs_path != local_path

    def test_cloud_path_in_sets_and_dicts(self):
        """Test that cloud paths work correctly in sets and dictionaries."""
        gs_path1 = PureGSPath("gs://bucket/file.txt")
        gs_path2 = PureGSPath("gs://bucket/file.txt")
        gs_path3 = PureGSPath("gs://bucket/other.txt")

        # Test sets
        path_set = {gs_path1, gs_path2, gs_path3}
        assert len(path_set) == 2  # gs_path1 and gs_path2 are equal

        # Test dictionaries
        path_dict = {gs_path1: "value1", gs_path3: "value3"}
        assert path_dict[gs_path2] == "value1"  # gs_path2 should be same key as gs_path1

    def test_cloud_path_ordering(self):
        """Test ordering of cloud paths."""
        gs_path1 = PureGSPath("gs://bucket/a.txt")
        gs_path2 = PureGSPath("gs://bucket/b.txt")
        gs_path3 = PureGSPath("gs://bucket/a.txt")

        assert gs_path1 < gs_path2
        assert gs_path2 > gs_path1
        assert gs_path1 <= gs_path3
        assert gs_path1 >= gs_path3

    def test_cloud_path_with_absolute_posix_path(self):
        """Test joining with absolute PosixPath replaces path after bucket."""
        from pathlib import PosixPath

        gs_path = PureGSPath("gs://bucket/folder")
        abs_posix = PosixPath("/absolute/path/file.txt")
        result = gs_path / abs_posix
        assert str(result) == "gs://bucket/absolute/path/file.txt"

    def test_cloud_path_construction_from_multiple_args(self):
        """Test constructing cloud paths from multiple arguments."""
        # Test with mixed types
        gs_path = PureGSPath("gs://bucket", "folder", "file.txt")
        assert str(gs_path) == "gs://bucket/folder/file.txt"

        # Test with cloud path and string segments
        base = PureGSPath("gs://bucket")
        result = PureGSPath(base, "folder", "file.txt")
        assert str(result) == "gs://bucket/folder/file.txt"

    def test_cloud_path_rtruediv(self):
        """Test reverse division operator creates new path from left operand."""
        gs_path = PureGSPath("gs://bucket/file.txt")
        result = "prefix" / gs_path
        assert str(result) == "gs://bucket/file.txt"

    def test_cloud_path_with_relative_components(self):
        """Test cloud paths preserve relative path components like '..' without normalization."""
        gs_path = PureGSPath("gs://bucket/folder")
        result = gs_path / ".." / "other" / "file.txt"
        assert str(result) == "gs://bucket/folder/../other/file.txt"


class TestAnyPathFunction:
    """Test the anypath factory function."""

    def test_anypath_with_gs_path(self):
        """Test anypath with Google Cloud Storage paths."""
        path = anypath("gs://bucket/file.txt")
        assert isinstance(path, PureGSPath)
        assert str(path) == "gs://bucket/file.txt"

    def test_anypath_with_s3_path(self):
        """Test anypath with S3 paths."""
        path = anypath("s3://bucket/file.txt")
        assert isinstance(path, PureR2Path)
        assert str(path) == "s3://bucket/file.txt"

    def test_anypath_with_r2_path(self):
        """Test anypath with R2 paths."""
        path = anypath("r2://bucket/file.txt")
        assert isinstance(path, PureR2Path)
        assert str(path) == "s3://bucket/file.txt"  # R2 paths are converted to S3

    def test_anypath_with_local_path(self):
        """Test anypath with local paths."""
        path = anypath("local/file.txt")
        assert isinstance(path, Path)
        assert str(path) == "local/file.txt"

    def test_anypath_with_absolute_local_path(self):
        """Test anypath with absolute local paths."""
        path = anypath("/absolute/path/file.txt")
        assert isinstance(path, Path)
        assert str(path) == "/absolute/path/file.txt"

    def test_anypath_with_existing_path_objects(self):
        """Test anypath returns equivalent object when passed existing path objects."""
        # Test with existing PureGSPath
        gs_path = PureGSPath("gs://bucket/file.txt")
        result = anypath(gs_path)
        assert result == gs_path

        # Test with existing Path
        local_path = Path("local/file.txt")
        result = anypath(local_path)
        assert result == local_path

    def test_anypath_with_relative_paths(self):
        """Test anypath with relative paths."""
        path = anypath("relative/path/file.txt")
        assert isinstance(path, Path)
        assert str(path) == "relative/path/file.txt"

    def test_anypath_with_empty_string(self):
        """Test anypath with empty string."""
        path = anypath("")
        assert isinstance(path, Path)
        assert str(path) == "."


class TestPureGSPath:
    """Test PureGSPath-specific functionality."""

    def test_gs_path_creation(self):
        """Test creating PureGSPath instances with various formats."""
        path = PureGSPath("gs://my-bucket/folder/file.txt")
        assert str(path) == "gs://my-bucket/folder/file.txt"
        assert path.bucket == "my-bucket"
        assert path.name == "file.txt"
        assert path.suffix == ".txt"
        assert path.stem == "file"
        assert str(path.parent) == "gs://my-bucket/folder"

    def test_gs_bucket_only(self):
        """Test bucket-only path without trailing slash."""
        path = PureGSPath("gs://my-bucket")
        assert str(path) == "gs://my-bucket"
        assert path.bucket == "my-bucket"

    def test_gs_drive_root_anchor_properties(self):
        """Test drive, root, and anchor properties for GS paths."""
        # Path with folder structure
        p = PureGSPath("gs://bucket/folder/file.txt")
        assert p.drive == "gs://bucket"
        assert p.root == "/"
        assert p.anchor == "gs://bucket/"

        # Bucket-only path has no root
        b = PureGSPath("gs://bucket")
        assert b.drive == "gs://bucket"
        assert b.root == ""
        assert b.anchor == "gs://bucket"

    def test_gs_path_is_absolute(self):
        """Test is_absolute and as_uri for GS paths."""
        # Path with root is absolute
        p = PureGSPath("gs://bucket/folder/file.txt")
        assert p.is_absolute() is True
        assert p.as_uri() == "gs://bucket/folder/file.txt"

        # Bucket-only path is not absolute (no root)
        b = PureGSPath("gs://bucket")
        assert b.is_absolute() is False
        with pytest.raises(ValueError, match="relative path can't be expressed as a file URI"):
            b.as_uri()

    def test_gs_parts_property(self):
        """Test parts property for GS paths."""
        path = PureGSPath("gs://my-bucket/folder/file.txt")
        parts = path.parts
        assert parts == ('gs://my-bucket/', 'folder', 'file.txt')

    def test_gs_path_joining_from_bucket(self):
        """Test joining paths from bucket-only path."""
        bucket_only = PureGSPath("gs://my-bucket")
        joined = bucket_only / "bar/text.txt"
        assert str(joined) == "gs://my-bucket/bar/text.txt"

    def test_gs_path_joining_with_absolute(self):
        """Test joining with absolute path replaces path after bucket."""
        base = PureGSPath("gs://my-bucket/folder")
        joined = base / "/absolute/path"
        assert str(joined) == "gs://my-bucket/absolute/path"

    def test_gs_parents_property(self):
        """Test parents property returns correct hierarchy and tests _ParentsSequence."""
        # Test with deep path structure
        path = PureGSPath("gs://my-bucket/a/b/c/file.txt")
        parents = path.parents

        # Check length via __len__
        assert len(parents) == 4

        # Check each parent in sequence via __getitem__
        assert str(parents[0]) == "gs://my-bucket/a/b/c"
        assert str(parents[1]) == "gs://my-bucket/a/b"
        assert str(parents[2]) == "gs://my-bucket/a"
        assert str(parents[3]) == "gs://my-bucket/"

        # Test iteration over parents (uses __iter__)
        parent_list = list(parents)
        assert len(parent_list) == 4
        assert str(parent_list[0]) == "gs://my-bucket/a/b/c"
        assert str(parent_list[-1]) == "gs://my-bucket/"

        # Test out of bounds index
        with pytest.raises(IndexError, match="index out of range"):
            _ = parents[4]

        # Test with shallow path
        shallow_path = PureGSPath("gs://my-bucket/file.txt")
        shallow_parents = shallow_path.parents
        assert len(shallow_parents) == 1
        assert str(shallow_parents[0]) == "gs://my-bucket/"

        # Test with bucket root - should have no parents
        root_path = PureGSPath("gs://my-bucket/")
        root_parents = root_path.parents
        assert len(root_parents) == 0

        # Test __repr__
        assert "PureGSPath.parents" in repr(parents)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_cloud_path_with_empty_string(self):
        """Test cloud path creation with empty string raises ValueError."""
        with pytest.raises(ValueError, match="Path is empty"):
            PureGSPath("")

    def test_cloud_path_with_none(self):
        """Test cloud path creation with None raises TypeError."""
        with pytest.raises(TypeError):
            PureGSPath(None)

    def test_cloud_path_with_invalid_type(self):
        """Test cloud path creation with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="argument should be a str or os.PathLike object"):
            PureGSPath(123)

    def test_cloud_path_with_list(self):
        """Test cloud path creation with list raises TypeError."""
        with pytest.raises(TypeError, match="argument should be a str or os.PathLike object"):
            PureGSPath(["gs://bucket/file.txt"])

    def test_cloud_path_with_multiple_slashes(self):
        """Test cloud path normalizes multiple consecutive slashes."""
        path = PureGSPath("gs://bucket//folder///file.txt")
        assert str(path) == "gs://bucket/folder/file.txt"

    def test_cloud_path_with_dots(self):
        """Test cloud path preserves dot components (no normalization)."""
        path = PureGSPath("gs://bucket/folder/../other/file.txt")
        assert str(path) == "gs://bucket/folder/../other/file.txt"

    def test_cloud_path_with_trailing_slash(self):
        """Test cloud path with trailing slash."""
        path = PureGSPath("gs://bucket/folder/")
        assert str(path) == "gs://bucket/folder/"
        assert path.name == "folder"

    def test_cloud_path_bucket_only_with_trailing_slash(self):
        """Test bucket-only path with trailing slash."""
        path = PureGSPath("gs://bucket/")
        assert str(path) == "gs://bucket/"
        assert path.bucket == "bucket"

    def test_cloud_path_with_special_characters(self):
        """Test cloud path with special characters."""
        path = PureGSPath("gs://bucket/folder with spaces/file.txt")
        assert str(path) == "gs://bucket/folder with spaces/file.txt"

    def test_cloud_path_with_unicode(self):
        """Test cloud path with unicode characters."""
        path = PureGSPath("gs://bucket/folder/文件.txt")
        assert str(path) == "gs://bucket/folder/文件.txt"

    def test_cloud_path_with_name_invalid(self):
        """Test with_name with invalid name."""
        path = PureGSPath("gs://bucket/folder/file.txt")
        with pytest.raises(ValueError, match="Invalid name"):
            path.with_name("invalid/name")

    def test_cloud_path_with_suffix_invalid(self):
        """Test with_suffix with invalid suffix."""
        path = PureGSPath("gs://bucket/folder/file.txt")
        with pytest.raises(ValueError, match="Invalid suffix"):
            path.with_suffix("invalid.suffix")

    def test_cloud_path_with_stem(self):
        """Test with_stem method."""
        path = PureGSPath("gs://bucket/folder/file.txt")
        new_path = path.with_stem("newname")
        assert str(new_path) == "gs://bucket/folder/newname.txt"

    def test_cloud_path_suffixes_property(self):
        """Test suffixes property with multiple extensions."""
        path = PureGSPath("gs://bucket/folder/file.tar.gz")
        assert path.suffixes == [".tar", ".gz"]
        assert path.suffix == ".gz"
        assert path.stem == "file.tar"

    def test_cloud_path_match_with_glob(self):
        """Test match method with glob patterns."""
        path = PureGSPath("gs://bucket/folder/file.txt")
        assert path.match("**/*.txt")
        assert path.match("folder/*.txt")
        assert not path.match("**/*.pdf")

    def test_cloud_path_match_empty_pattern(self):
        """Test match method with empty pattern."""
        path = PureGSPath("gs://bucket/folder/file.txt")
        with pytest.raises(ValueError, match="empty pattern"):
            path.match("")

    def test_cloud_path_as_uri_not_absolute(self):
        """Test as_uri raises ValueError for non-absolute paths (bucket-only)."""
        path = PureGSPath("gs://bucket")  # Not absolute (no root)
        with pytest.raises(ValueError, match="relative path can't be expressed as a file URI"):
            path.as_uri()

    def test_cloud_path_as_posix(self):
        """Test as_posix method."""
        path = PureGSPath("gs://bucket/folder/file.txt")
        assert path.as_posix() == "gs://bucket/folder/file.txt"

    def test_cloud_path_joinpath_multiple_args(self):
        """Test joinpath with multiple arguments."""
        path = PureGSPath("gs://bucket")
        result = path.joinpath("folder", "subfolder", "file.txt")
        assert str(result) == "gs://bucket/folder/subfolder/file.txt"

    def test_cloud_path_parents_sequence(self):
        """Test parents sequence returns correct hierarchy of parent paths."""
        path = PureGSPath("gs://bucket/a/b/c/file.txt")
        parents = path.parents
        assert len(parents) == 4
        assert str(parents[0]) == "gs://bucket/a/b/c"
        assert str(parents[1]) == "gs://bucket/a/b"
        assert str(parents[2]) == "gs://bucket/a"
        assert str(parents[3]) == "gs://bucket/"

    def test_cloud_path_parents_sequence_index_error(self):
        """Test parents sequence with invalid index."""
        path = PureGSPath("gs://bucket/file.txt")
        parents = path.parents
        with pytest.raises(IndexError, match="index out of range"):
            _ = parents[1]

    def test_cloud_path_fspath_protocol(self):
        """Test __fspath__ protocol."""
        path = PureGSPath("gs://bucket/file.txt")
        assert os.fspath(path) == "gs://bucket/file.txt"

    def test_cloud_path_repr(self):
        """Test __repr__ method."""
        path = PureGSPath("gs://bucket/file.txt")
        repr_str = repr(path)
        assert "PureGSPath" in repr_str
        assert "gs://bucket/file.txt" in repr_str

    def test_cloud_path_hash_consistency(self):
        """Test hash consistency."""
        path1 = PureGSPath("gs://bucket/file.txt")
        path2 = PureGSPath("gs://bucket/file.txt")
        path3 = PureGSPath("gs://bucket/other.txt")

        assert hash(path1) == hash(path2)
        assert hash(path1) != hash(path3)

    def test_cloud_path_different_types_not_equal(self):
        """Test that different cloud path types are not equal even with same string representation."""
        gs_path = PureGSPath("gs://bucket/file.txt")
        s3_path = PureS3Path("s3://bucket/file.txt")
        r2_path = PureR2Path("s3://bucket/file.txt")

        assert gs_path != s3_path
        assert gs_path != r2_path
        assert s3_path != r2_path  # Different classes even with same prefix



class TestOSPathLikeIntegration:
    """Test os.PathLike protocol integration."""

    def test_cloud_path_is_pathlike(self):
        """Test that cloud paths implement os.PathLike protocol."""
        path = PureGSPath("gs://bucket/file.txt")
        assert isinstance(path, os.PathLike)

    def test_cloud_path_with_os_fspath(self):
        """Test cloud paths work with os.fspath function."""
        path = PureGSPath("gs://bucket/file.txt")
        assert os.fspath(path) == "gs://bucket/file.txt"

    def test_cloud_path_has_fspath_method(self):
        """Test cloud paths have __fspath__ method for PathLike protocol."""
        gs_path = PureGSPath("gs://bucket/file.txt")
        assert hasattr(gs_path, '__fspath__')
        assert gs_path.__fspath__() == "gs://bucket/file.txt"
