"""Path utilities for homogeneous resource access.

This module provides extensions of pathlib's PurePath classes to support URI-like cloud
paths. These enable path manipulation without providing any IO functionality. This
allows using os.Pathlike objects in places where IO is not needed and providing a more
unified interface for local/cloud paths. IO operations are handled by fsspec libraries
(e.g. gcsfs, s3fs, etc.).

The implementation uses a custom "flavour" that defines cloud-specific path semantics
(similar to how pathlib handles POSIX vs Windows paths). This approach is compatible
with Python 3.10+ with minimal version-specific code (only in __init__).
"""

from __future__ import annotations

import fnmatch
import os
import posixpath
import re
from pathlib import Path, PurePath
from typing import Any, TypeAlias

# Python 3.12+ added __init__() to PurePath; 3.10/3.11 only use __new__()


class _CloudFlavour:
    """A minimal pathlib flavour for generic cloud-like paths.

    Models cloud paths (e.g., gs://bucket/path) by treating "scheme://bucket" as
    the drive component, similar to how Windows paths have drive letters.
    """

    sep = "/"
    altsep = ""
    has_drv = True  # model "<scheme>://bucket" as a drive
    pathmod = posixpath

    def __init__(self, scheme: str) -> None:
        if not scheme or not scheme.endswith("://"):
            raise ValueError("scheme must end with '://'")
        self._scheme = scheme

    def join(self, *args: Any) -> str:  # noqa: ANN401
        """Join path components.

        Supports both Python 3.11 (single iterable argument) and
        Python 3.12+ (multiple string arguments) calling conventions.

        Returns:
            The joined path string.
        """
        # Python 3.12+ passes multiple string args, 3.11 passes a single iterable
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return self.sep.join(args[0])
        return self.sep.join(args)

    def normcase(self, s: str) -> str:
        """Cloud paths are case-sensitive, so return unchanged.

        Returns:
            The unchanged string.
        """
        return s

    def isabs(self, path: str) -> bool:
        """A cloud path is absolute if it has both a drive (bucket) and root.

        Returns:
            True if path is absolute, False otherwise.
        """
        drv, root, _ = self.splitroot(path)
        return bool(drv and root)

    def splitdrive(self, path: str) -> tuple[str, str]:
        """Split path into drive (scheme://bucket) and rest.

        Returns:
            Tuple of (drive, rest).
        """
        drv, root, rel = self.splitroot(path)
        if root or rel:
            rest = root + self.sep.join(rel) if isinstance(rel, list) else root + rel
            return drv, rest
        return drv, ""

    def parse_parts(self, parts: list[str]) -> tuple[str, str, list[str]]:
        """Parse path parts into (drive, root, parts_list).

        This method is called by PurePath to normalize constructor arguments.

        Returns:
            Tuple of (drive, root, parts_list).
        """
        parsed: list[str] = []
        drv = root = ""

        # Process parts in reverse order (pathlib convention)
        for part in reversed(parts):
            if not part:
                continue

            drv, root, rel = self.splitroot(part)

            # Split relative part and add non-empty, non-dot components
            if self.sep in rel:
                for x in reversed(rel.split(self.sep)):
                    if x and x != ".":
                        parsed.append(x)
            elif rel and rel != ".":
                parsed.append(rel)

            # If we found a drive or root, we're done
            if drv or root:
                break

        # Reconstruct parts list
        if drv or root:
            parsed.append(drv + root)
        parsed.reverse()
        return drv, root, parsed

    def splitroot(self, part: str) -> tuple[str, str, str]:
        """Split path into (drive, root, relative) components.

        For cloud paths:
        - drive: "scheme://bucket" (e.g., "gs://my-bucket")
        - root: "/" if path continues after bucket, else ""
        - relative: path after the bucket and root

        Returns:
            Tuple of (drive, root, relative).

        Examples:
            "gs://bucket/path/file" -> ("gs://bucket", "/", "path/file")
            "gs://bucket" -> ("gs://bucket", "", "")
            "/local/path" -> ("", "/", "local/path")
            "relative/path" -> ("", "", "relative/path")
        """
        # Check if path starts with our scheme
        if not part.startswith(self._scheme):
            # Not a cloud path - treat as POSIX-style
            if part.startswith(self.sep):
                return "", self.sep, part.lstrip(self.sep)
            return "", "", part

        # Extract bucket and rest from "scheme://bucket/rest"
        remainder = part[len(self._scheme) :]
        if not remainder:
            return "", "", ""

        slash_idx = remainder.find("/")
        if slash_idx == -1:
            # Just "scheme://bucket" with no trailing path
            return f"{self._scheme}{remainder}", "", ""

        # "scheme://bucket/path"
        bucket = remainder[:slash_idx]
        path_part = remainder[slash_idx + 1 :]
        return f"{self._scheme}{bucket}", self.sep, path_part.lstrip(self.sep)

    def join_parsed_parts(
        self, drv: str, root: str, parts: list[str], drv2: str, root2: str, parts2: list[str]
    ) -> tuple[str, str, list[str]]:
        """Join two parsed path representations.

        This implements the logic for path joining (e.g., path1 / path2).
        Special handling for bucket-only paths ensures 'gs://bucket' / 'file'
        produces 'gs://bucket/file' not 'gs://bucketfile'.

        Returns:
            Tuple of (drive, root, parts) for the joined path.
        """
        if root2:
            # Second path has a root - it replaces the path portion but keeps drive
            if not drv2 and drv:
                return drv, root2, [drv + root2] + parts2[1:]
        elif drv2:
            # Second path has a drive
            if drv2 == drv or self.casefold(drv2) == self.casefold(drv):
                # Same drive - append parts
                return drv, root, parts + parts2[1:]
        else:
            # Second path is relative - append to first
            # Special case: bucket-only path needs root inserted
            if drv and not root:
                bucket_root = drv + self.sep
                parts = [bucket_root] if not parts else [bucket_root] + parts[1:]
                root = self.sep
            return drv, root, parts + parts2

        # Second path is absolute and different - return it
        return drv2, root2, parts2

    def casefold(self, s: str) -> str:
        """Cloud paths are case-sensitive.

        Returns:
            The unchanged string.
        """
        return s

    def casefold_parts(self, parts: list[str]) -> list[str]:
        """Cloud paths are case-sensitive.

        Returns:
            The unchanged list.
        """
        return parts

    def compile_pattern(self, pattern: str) -> Any:  # noqa: ANN401
        """Compile a glob pattern for matching.

        Returns:
            A compiled pattern matcher function.
        """
        return re.compile(fnmatch.translate(pattern)).fullmatch

    def is_reserved(self, parts: list[str]) -> bool:
        """Cloud paths have no reserved names.

        Returns:
            Always False.
        """
        return False

    def make_uri(self, path: PurePath) -> str:
        """Convert path to URI (for cloud paths, already a URI).

        Returns:
            The path as a URI string.

        Raises:
            ValueError: If path is not absolute.
        """
        if not path.is_absolute():
            raise ValueError("relative path can't be expressed as a file URI")
        return str(path)


class PureCloudPath(PurePath):
    """Generic PurePath for cloud schemes.

    Subclasses must set the ``cloud_prefix`` class attribute (e.g., "gs://", "s3://").
    This class works with Python 3.10+ by using a custom _CloudFlavour that handles
    version differences in the join() method and a version-aware __init__().

    Examples:
        >>> class PureGSPath(PureCloudPath):
        ...     cloud_prefix = "gs://"
        ...     _flavour = _CloudFlavour("gs://")
        >>> p = PureGSPath("gs://bucket/folder/file.txt")
        >>> p.bucket
        'bucket'
        >>> str(p.parent)
        'gs://bucket/folder'
    """

    cloud_prefix: str = ""
    __slots__ = ()

    def __new__(cls, *args: str) -> PureCloudPath:
        """Create a new cloud path instance with validation.

        Raises:
            ValueError: If cloud_prefix is not defined in subclass, or if path
                doesn't start with the expected prefix.
        """
        if not getattr(cls, "cloud_prefix", None):
            raise ValueError("cloud_prefix must be defined in subclass")

        # Validate input for user-facing constructions
        if args:
            first_arg = args[0] if args else ""
            if isinstance(first_arg, str) and first_arg:
                # Check for wrong cloud schemes
                other_schemes = ["gs://", "s3://", "r2://", "http://", "https://"]
                has_other_scheme = any(
                    first_arg.startswith(scheme)
                    for scheme in other_schemes
                    if scheme != cls.cloud_prefix
                )

                if has_other_scheme:
                    raise ValueError(f"Path must start with '{cls.cloud_prefix}': {first_arg}")

                # Allow patterns (for glob matching) and relative paths (for internal use)
                is_pattern = any(c in first_arg for c in ["*", "?", "["])
                is_relative = not first_arg.startswith(
                    cls.cloud_prefix
                ) and not first_arg.startswith("/")

                # Validate user-facing paths (not patterns, not relative)
                if not is_pattern and not is_relative:
                    if not first_arg.startswith(cls.cloud_prefix):
                        raise ValueError(f"Path must start with '{cls.cloud_prefix}': {first_arg}")
            elif not first_arg and len(args) == 1:
                # Empty string
                raise ValueError(f"Path must start with '{cls.cloud_prefix}': {first_arg}")

        return super().__new__(cls, *args)

    @property
    def bucket(self) -> str:
        """Extract the bucket name from the cloud path.

        Returns:
            The bucket name, or empty string if not a valid cloud path.

        Examples:
            >>> PureGSPath("gs://my-bucket/file.txt").bucket
            'my-bucket'
            >>> PureGSPath("gs://another-bucket").bucket
            'another-bucket'
        """
        drv = self.drive
        prefix = self.__class__.cloud_prefix
        if drv.startswith(prefix):
            return drv[len(prefix) :]
        return ""

    def as_uri(self) -> str:
        """Return the path as a URI.

        For cloud paths, the path string is already a valid URI.

        Returns:
            The path as a URI string.

        Raises:
            ValueError: If the path is not absolute.
        """
        if not self.is_absolute():
            raise ValueError("relative path can't be expressed as a file URI")
        return str(self)

    # Ensure absolute subpaths (like "/abs") replace the path portion but keep the drive
    # across Python versions (not relying on private internals in 3.12+).
    def joinpath(self, *args: Any) -> PureCloudPath:  # noqa: ANN401
        current: PureCloudPath = self
        for arg in args:
            # Convert pathlikes to strings similar to pathlib
            if isinstance(arg, PurePath):
                a_str = str(arg)
            else:
                a_str = os.fspath(arg)  # type: ignore[arg-type]
            if isinstance(a_str, str) and a_str.startswith("/"):
                # Replace path after bucket with absolute subpath, keeping drive
                current = type(self)(current.drive + a_str)
            else:
                # Delegate to base PurePath join for relative segments
                current = PurePath.joinpath(current, a_str)  # type: ignore[assignment]
        return current

    def __truediv__(self, key: Any) -> PureCloudPath:  # noqa: ANN401
        return self.joinpath(key)


_s3_flavour = _CloudFlavour(scheme="s3://")
_gcs_flavour = _CloudFlavour(scheme="gs://")
_r2_flavour = _CloudFlavour(scheme="s3://")


class PureGSPath(PureCloudPath):
    cloud_prefix = "gs://"
    _flavour = _gcs_flavour
    __slots__ = ()


class PureS3Path(PureCloudPath):
    cloud_prefix = "s3://"
    _flavour = _s3_flavour
    __slots__ = ()


class PureR2Path(PureCloudPath):
    cloud_prefix = "s3://"
    _flavour = _r2_flavour
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
