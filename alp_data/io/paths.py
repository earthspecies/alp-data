"""Path classes for cloud URIs.

This module provides "Pure" path classes for cloud URIs (gs://, s3://, etc.) with a
pathlib-like module. These classes handle path manipulation without providing any IO
functionality.

The implementation is standalone and doesn't depend on pathlib internals, making it
compatible across Python 3.10+ versions. This is necessary as there are significant
changes to pathlib in Python 3.12.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import TypeAlias

# `ALP_DATA_HOME` is the current name; `ESP_DATA_HOME` is kept as a deprecated
# fallback for the esp-data -> alp-data rename so existing deployments keep working.
DATA_HOME = (
    os.environ.get("ALP_DATA_HOME") or os.environ.get("ESP_DATA_HOME") or "gs://esp-data-274503"
)


class PureCloudPath:
    """Base class for cloud path manipulation.

    This class provides a PurePath-like interface for cloud URIs without inheriting from
    pathlib.PurePath. We implement everything here because the internal implementation
    of pathlib has changed between versions, especially in 3.12, and we want to support
    various Python versions.

    Subclasses must set the `cloud_prefix` class attribute (e.g., "gs://", "s3://").

    Examples:
        >>> class PureGSPath(PureCloudPath):
        ...     cloud_prefix = "gs://"
        >>> p = PureGSPath("gs://bucket/folder/file.txt")
        >>> p.bucket
        'bucket'
        >>> str(p.parent)
        'gs://bucket/folder'
        >>> str(p / "newfile.txt")
        'gs://bucket/folder/file.txt/newfile.txt'
    """

    cloud_prefix: str = ""
    __slots__ = ("_path",)

    def __init__(self, *args: str | PureCloudPath | os.PathLike) -> None:
        """Initialize a cloud path from one or more path segments.

        Args:
            *args: Path segments to join. Can be strings, PureCloudPath instances,
                   or os.PathLike objects.

        Raises:
            TypeError: If no arguments are provided or if an argument is not a string,
                      PureCloudPath, or os.PathLike object.
            ValueError: If the path doesn't start with the expected cloud_prefix,
                       or if cloud_prefix is not defined in the subclass.
        """
        if not self.__class__.cloud_prefix:
            raise ValueError("cloud_prefix must be defined in subclass")

        if not args:
            raise TypeError("__init__() missing required arguments")

        # Convert all args to strings and join them
        path_parts: list[str] = []
        for arg in args:
            if isinstance(arg, PureCloudPath):
                path_parts.append(str(arg))
            elif isinstance(arg, os.PathLike):
                path_parts.append(os.fspath(arg))
            elif isinstance(arg, str):
                path_parts.append(arg)
            else:
                raise TypeError(
                    f"argument should be a str or os.PathLike object, not {type(arg).__name__!r}"
                )

        # Join the parts
        if len(path_parts) == 1:
            self._validate_path(path_parts[0])
            self._path = self._normalize(path_parts[0])
        else:
            result = path_parts[0]
            for part in path_parts[1:]:
                result = self._join_paths(result, part)

            self._validate_path(result)
            self._path = self._normalize(result)

    def _validate_path(self, path: str) -> None:
        """Validate that a path is appropriate for this cloud type.

        Args:
            path: The path string to validate.

        Raises:
            ValueError: If the path is invalid for this cloud type.
        """
        if not path:
            raise ValueError("Path is empty")

        # Reject absolute POSIX paths (starting with /) from direct construction.
        # They are still used internally for "absolute within bucket" path joining.
        if path.startswith("/"):
            raise ValueError(f"Path must start with '{self.cloud_prefix}': {path}")

        if not path.startswith(self.cloud_prefix):
            raise ValueError(f"Path must start with '{self.cloud_prefix}': {path}")

    def _normalize(self, path: str) -> str:
        """Normalize a path by removing redundant separators and dots.

        Args:
            path: The path to normalize.

        Returns:
            The normalized path.
        """
        if not path:
            return path

        # Remember if path ends with /
        ends_with_slash = path.endswith("/") and len(path) > 1

        # Split into scheme+bucket and path components
        prefix_len = len(self.cloud_prefix)
        remainder = path[prefix_len:]

        if "/" in remainder:
            bucket, _, rest = remainder.partition("/")
            # Normalize the path part
            parts = [p for p in rest.split("/") if p and p != "."]
            if parts:
                result = f"{self.cloud_prefix}{bucket}/{'/'.join(parts)}"
            else:
                # Path was like gs://bucket/ with no other parts
                result = f"{self.cloud_prefix}{bucket}/"
            # Preserve trailing slash for directories
            if ends_with_slash and not result.endswith("/"):
                result += "/"
            return result
        else:
            # Just bucket (with or without trailing /)
            bucket = remainder.rstrip("/")
            if ends_with_slash:
                return f"{self.cloud_prefix}{bucket}/"
            return f"{self.cloud_prefix}{bucket}"

    def _join_paths(self, base: str, other: str) -> str:
        """Join two path strings.

        Args:
            base: The base path.
            other: The path to join to the base.

        Returns:
            The joined path.
        """
        if not other:
            return base

        # If other is an absolute path starting with /, treat it as "absolute within
        # the bucket". This enables interoperability with pathlib (when joining with
        # absolute PosixPath objects) and allows explicit path replacement:
        #   PureGSPath("gs://bucket/folder") / "/new/path" -> "gs://bucket/new/path"
        # The bucket is preserved but everything after it is replaced.
        if other.startswith("/"):
            # Replace path after bucket with the new absolute path
            prefix_len = len(self.cloud_prefix)
            remainder = base[prefix_len:]
            if "/" in remainder:
                bucket = remainder.split("/")[0]
            else:
                bucket = remainder
            # Remove leading slash from other and append
            return f"{self.cloud_prefix}{bucket}{other}"

        # If other looks like a URI (contains ://), return it as absolute replacement
        if "://" in other:
            return other

        # Otherwise, append as relative path
        if base.endswith("/"):
            return base + other
        else:
            return base + "/" + other

    def __str__(self) -> str:
        """Return the string representation of the path.

        Returns:
            The path as a string.
        """
        return self._path

    def __repr__(self) -> str:
        """Return a readable representation of the path.

        Returns:
            A string representation of the path object.
        """
        return f"{self.__class__.__name__}({self._path!r})"

    def __fspath__(self) -> str:
        """Return the file system path representation.

        Returns:
            The file system path as a string.
        """
        return self._path

    def __truediv__(self, other: str | PureCloudPath | os.PathLike) -> PureCloudPath:
        """Join paths using the / operator.

        Args:
            other: The path segment to append.

        Returns:
            A new path with the segments joined.

        Raises:
            ValueError: If other is a URI with an incompatible scheme.
        """
        if isinstance(other, (PureCloudPath, os.PathLike)):
            other = os.fspath(other)

        if not isinstance(other, str):
            return NotImplemented

        # Validate that if other is a URI, it matches our scheme
        if "://" in other and not other.startswith(self.cloud_prefix):
            raise ValueError(
                f"Cannot join {self.cloud_prefix} path with {other}: incompatible cloud schemes"
            )

        new_path = self._join_paths(self._path, other)

        # Create new instance without validation for internal paths
        result = object.__new__(self.__class__)
        result._path = self._normalize(new_path)
        return result

    def __rtruediv__(self, other: str | os.PathLike) -> PureCloudPath:
        """Support path / cloud_path.

        Args:
            other: The left-hand path segment.

        Returns:
            A new path with the segments joined.
        """
        if isinstance(other, os.PathLike):
            other = os.fspath(other)

        if not isinstance(other, str):
            return NotImplemented

        # This creates a new path starting from other
        return self.__class__(other, self._path)

    def __eq__(self, other: object) -> bool:
        """Check equality with another path.

        Args:
            other: The object to compare with.

        Returns:
            True if paths are equal, False otherwise.
        """
        if isinstance(other, PureCloudPath):
            return self._path == other._path and type(self) is type(other)
        return NotImplemented

    def __hash__(self) -> int:
        """Return hash of the path.

        Returns:
            The hash value of the path.
        """
        return hash((self._path, self.__class__))

    def __lt__(self, other: PureCloudPath) -> bool:
        """Compare paths lexicographically.

        Args:
            other: The path to compare with.

        Returns:
            True if this path is less than the other path.
        """
        if not isinstance(other, PureCloudPath):
            return NotImplemented
        return self._path < other._path

    def __le__(self, other: PureCloudPath) -> bool:
        """Compare paths lexicographically.

        Args:
            other: The path to compare with.

        Returns:
            True if this path is less than or equal to the other path.
        """
        if not isinstance(other, PureCloudPath):
            return NotImplemented
        return self._path <= other._path

    def __gt__(self, other: PureCloudPath) -> bool:
        """Compare paths lexicographically.

        Args:
            other: The path to compare with.

        Returns:
            True if this path is greater than the other path.
        """
        if not isinstance(other, PureCloudPath):
            return NotImplemented
        return self._path > other._path

    def __ge__(self, other: PureCloudPath) -> bool:
        """Compare paths lexicographically.

        Args:
            other: The path to compare with.

        Returns:
            True if this path is greater than or equal to the other path.
        """
        if not isinstance(other, PureCloudPath):
            return NotImplemented
        return self._path >= other._path

    @property
    def bucket(self) -> str:
        """Extract the bucket name from the cloud path.

        Returns:
            The bucket name.

        Examples:
            >>> PureGSPath("gs://my-bucket/file.txt").bucket
            'my-bucket'
            >>> PureGSPath("gs://another-bucket").bucket
            'another-bucket'
        """
        remainder = self._path[len(self.cloud_prefix) :]
        if "/" in remainder:
            return remainder.split("/")[0]
        return remainder

    @property
    def drive(self) -> str:
        """The drive component (scheme://bucket).

        Returns:
            The drive string (e.g., "gs://bucket").
        """
        return f"{self.cloud_prefix}{self.bucket}"

    @property
    def root(self) -> str:
        """The root component (/ if path extends beyond bucket).

        Returns:
            "/" if path has components after bucket, "" otherwise.
        """
        remainder = self._path[len(self.cloud_prefix) :]
        return "/" if "/" in remainder else ""

    @property
    def anchor(self) -> str:
        """The drive and root concatenated.

        Returns:
            The anchor string (e.g., "gs://bucket/").
        """
        return self.drive + self.root

    @property
    def parts(self) -> tuple[str, ...]:
        """The path components as a tuple.

        Returns:
            Tuple of path components, with the first being the anchor if present.

        Examples:
            >>> PureGSPath("gs://bucket/folder/file.txt").parts
            ('gs://bucket/', 'folder', 'file.txt')
            >>> PureGSPath("gs://bucket").parts
            ('gs://bucket',)
        """
        remainder = self._path[len(self.cloud_prefix) :]
        if "/" in remainder:
            bucket, _, rest = remainder.partition("/")
            path_parts = [p for p in rest.split("/") if p]
            return (f"{self.cloud_prefix}{bucket}/",) + tuple(path_parts)
        else:
            # Bucket only
            return (f"{self.cloud_prefix}{remainder}",)

    @property
    def name(self) -> str:
        """The final path component.

        Returns:
            The final component, or empty string for bucket-only paths.
        """
        parts = self.parts
        if not parts:
            return ""

        # If last part is the anchor, return empty
        anchor = self.anchor
        if parts[-1] == anchor or parts[-1] == anchor.rstrip("/"):
            return ""

        return parts[-1]

    @property
    def suffix(self) -> str:
        """The file extension of the final component.

        Returns:
            The suffix including the dot, or empty string.
        """
        name = self.name
        if not name:
            return ""

        i = name.rfind(".")
        if 0 < i < len(name) - 1:
            return name[i:]
        return ""

    @property
    def suffixes(self) -> list[str]:
        """All file extensions of the final component.

        Returns:
            List of suffixes including the dots.
        """
        name = self.name
        if not name or name.endswith("."):
            return []

        name = name.lstrip(".")
        return ["." + suffix for suffix in name.split(".")[1:]]

    @property
    def stem(self) -> str:
        """The final component without its suffix.

        Returns:
            The stem of the filename.
        """
        name = self.name
        if not name:
            return ""

        i = name.rfind(".")
        if 0 < i < len(name) - 1:
            return name[:i]
        return name

    @property
    def parent(self) -> PureCloudPath:
        """The logical parent of this path.

        Returns:
            A new path object representing the parent directory.
        """
        parts = self.parts
        if not parts or len(parts) == 1:
            # Can't go higher than bucket
            return self

        # Reconstruct from parts
        anchor = parts[0]  # Keep the trailing / if it's there
        if len(parts) == 2:
            # Parent is the bucket with root (ensure it has /)
            bucket_name = self.bucket
            result = object.__new__(self.__class__)
            result._path = f"{self.cloud_prefix}{bucket_name}/"
            return result
        else:
            # Join anchor with middle parts
            parent_parts = list(parts[1:-1])
            parent_path = anchor.rstrip("/") + "/" + "/".join(parent_parts)
            result = object.__new__(self.__class__)
            result._path = parent_path
            return result

    @property
    def parents(self) -> _ParentsSequence:
        """A sequence of this path's logical parents.

        Returns:
            A sequence object providing access to parent paths.
        """
        return _ParentsSequence(self)

    def is_absolute(self) -> bool:
        """Check if the path is absolute.

        A cloud path is absolute if it has both a bucket and a root (/).

        Returns:
            True if the path is absolute, False otherwise.
        """
        return bool(self.drive and self.root)

    def as_uri(self) -> str:
        """Return the path as a URI.

        For cloud paths, this is the same as the string representation.

        Returns:
            The URI string.

        Raises:
            ValueError: If the path is not absolute.
        """
        if not self.is_absolute():
            raise ValueError("relative path can't be expressed as a file URI")
        return self._path

    def as_posix(self) -> str:
        """Return the path with forward slashes.

        Returns:
            The path string (cloud paths always use forward slashes).
        """
        return self._path

    def joinpath(self, *args: str | PureCloudPath | os.PathLike) -> PureCloudPath:
        """Join this path with one or more path segments.

        Args:
            *args: Path segments to join.

        Returns:
            A new path with all segments joined.
        """
        result = self
        for arg in args:
            result = result / arg
        return result

    def with_name(self, name: str) -> PureCloudPath:
        """Return a new path with the filename changed.

        Args:
            name: The new filename.

        Returns:
            A new path with the filename replaced.

        Raises:
            ValueError: If the path has no filename.
        """
        if not self.name:
            raise ValueError(f"{self!r} has an empty name")

        if "/" in name or name == "." or name == "..":
            raise ValueError(f"Invalid name {name!r}")

        parent_path = self.parent
        return parent_path / name

    def with_suffix(self, suffix: str) -> PureCloudPath:
        """Return a new path with the file suffix changed.

        Args:
            suffix: The new suffix (including the dot), or empty to remove.

        Returns:
            A new path with the suffix replaced.

        Raises:
            ValueError: If the suffix is invalid or path has no name.
        """
        if not self.name:
            raise ValueError(f"{self!r} has an empty name")

        if suffix and not suffix.startswith("."):
            raise ValueError(f"Invalid suffix {suffix!r}")

        if suffix == ".":
            raise ValueError(f"Invalid suffix {suffix!r}")

        if "/" in suffix:
            raise ValueError(f"Invalid suffix {suffix!r}")

        name = self.name
        old_suffix = self.suffix

        if old_suffix:
            new_name = name[: -len(old_suffix)] + suffix
        else:
            new_name = name + suffix

        return self.with_name(new_name)

    def with_stem(self, stem: str) -> PureCloudPath:
        """Return a new path with the stem changed.

        Args:
            stem: The new stem.

        Returns:
            A new path with the stem replaced.
        """
        return self.with_name(stem + self.suffix)

    def match(self, pattern: str) -> bool:
        """Match this path against a glob pattern.

        Args:
            pattern: The glob pattern to match against.

        Returns:
            True if the path matches the pattern, False otherwise.

        Raises:
            ValueError: If the pattern is empty.
        """
        if not pattern:
            raise ValueError("empty pattern")

        # Parse pattern parts
        pattern_parts = [p for p in pattern.split("/") if p]
        path_parts = list(self.parts)

        # If pattern starts with **, match from anywhere
        if pattern.startswith("**/"):
            # Try matching from each position
            for i in range(len(path_parts)):
                if self._match_parts(path_parts[i:], pattern_parts[1:]):
                    return True
            return False

        # Match from the end
        if len(pattern_parts) > len(path_parts):
            return False

        # Match the last N parts
        return self._match_parts(path_parts[-len(pattern_parts) :], pattern_parts)

    def _match_parts(self, path_parts: list[str], pattern_parts: list[str]) -> bool:
        """Match path parts against pattern parts.

        Args:
            path_parts: The path components to match.
            pattern_parts: The pattern components to match against.

        Returns:
            True if all parts match, False otherwise.
        """
        if len(path_parts) != len(pattern_parts):
            return False

        for path_part, pattern_part in zip(path_parts, pattern_parts, strict=True):
            # Remove trailing / from path part for matching
            path_part = path_part.rstrip("/")

            if pattern_part == "**":
                # ** matches anything
                continue
            elif "*" in pattern_part or "?" in pattern_part or "[" in pattern_part:
                # Use fnmatch for wildcard matching
                if not fnmatch.fnmatch(path_part, pattern_part):
                    return False
            else:
                # Exact match required
                if path_part != pattern_part:
                    return False

        return True


class _ParentsSequence:
    """Sequence of parent paths."""

    __slots__ = ("_path",)

    def __init__(self, path: PureCloudPath) -> None:
        self._path = path

    def __len__(self) -> int:
        """Return the number of parents.

        Returns:
            The number of parent paths.
        """
        parts = self._path.parts
        if not parts:
            return 0
        # Don't count the path itself, and stop at the anchor
        return max(0, len(parts) - 1)

    def __getitem__(self, idx: int) -> PureCloudPath:
        """Get the parent at the given index.

        Args:
            idx: The index of the parent to retrieve.

        Returns:
            The parent path at the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")

        current = self._path
        for _ in range(idx + 1):
            current = current.parent

        return current

    def __repr__(self) -> str:
        return f"<{self._path.__class__.__name__}.parents>"


class PureGSPath(PureCloudPath):
    """Google Cloud Storage path."""

    cloud_prefix = "gs://"
    __slots__ = ()


class PureS3Path(PureCloudPath):
    """Amazon S3 path."""

    cloud_prefix = "s3://"
    __slots__ = ()


class PureR2Path(PureCloudPath):
    """Cloudflare R2 path (uses S3 protocol)."""

    cloud_prefix = "s3://"
    __slots__ = ()


# TODO (milad) Python 3.12 introduces `type`. It will probably deprecate TypeAlias at
# some point. We should use that instead when 3.12 is not too new anymore.
AnyPathT: TypeAlias = Path | PureGSPath | PureR2Path


def anypath(path: str | AnyPathT) -> AnyPathT:
    """Creates the appropriate path object based on the input path string or object.

    This factory function inspects the input `path` to determine if it's a Google Cloud
    Storage path, an S3-compatible path (assumed to be Cloudflare R2), or a local path.
    It then returns an instance of the corresponding path class (`PureGSPath`,
    `PureR2Path`, or `Path`).

    Parameters
    ----------
    path : str | AnyPathT
        The path string (e.g., "/local/file.txt", "gs://bucket/blob", "r2://bucket/key")
        or an existing `Path`, `PureGSPath`, or `PureR2Path` object.

    Returns
    -------
    AnyPathT
        An instance of `Path` for local paths, `PureGSPath` for Google Cloud Storage
        paths, or `PureR2Path` for Cloudflare R2 paths (including those starting with
        "s3://").


    Examples
    --------
    >>> local_p = anypath("tests/samples/noise.wav")
    >>> isinstance(local_p, Path)
    True
    >>> print(local_p)
    tests/samples/noise.wav
    >>> gs_p = anypath("gs://esp-ci-cd-tests/esp-data-tests/file1.txt")
    >>> isinstance(gs_p, PureGSPath)
    True
    >>> print(gs_p)
    gs://esp-ci-cd-tests/esp-data-tests/file1.txt
    """

    path = str(path)

    if path.startswith("gs://"):
        return PureGSPath(path)
    elif path.startswith("r2://"):
        return PureR2Path("s3://" + path.removeprefix("r2://"))
    elif path.startswith("s3://"):
        # Since we are currently not using AWS we assume that all S3 paths are R2 paths.
        # TODO This must be changed if we start using AWS.
        return PureR2Path(path)
    else:
        return Path(path)
