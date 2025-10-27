"""Path utilities for homogeneous resource access.

This module provides extensions of pathlib's PurePath classes to support URI-like cloud
paths. These enable path manipulation without providing any IO functionality. This
allows using os.Pathlike objects in places where IO is not needed and providing a more
unified interface for local/cloud paths. IO operations are handled by fsspec libraries
(e.g. gcsfs, s3fs, etc.).

The implementation mirrors pathlib's design and supports both Python 3.11 and 3.12+.
Some code duplication was necessary since pathlib doesn't expose its private `_Flavour`
classes in 3.11, and uses a different architecture in 3.12+.
"""

from __future__ import annotations

import fnmatch
import os
import posixpath
import re
import sys
from pathlib import Path, PurePath
from typing import Iterable, Tuple, TypeAlias

# Python version detection for compatibility
_PY312_PLUS = sys.version_info >= (3, 12)


class _CloudFlavour:
    """A minimal pathlib flavour for generic cloud-like paths.

    It provides only the subset of the contract required by ``PurePath``.
    Compatible with both Python 3.11 and 3.12+ pathlib implementations.
    """

    sep = "/"
    altsep = ""
    has_drv = True  # model "<scheme>://bucket" as a drive
    pathmod = posixpath

    def __init__(self, scheme: str) -> None:
        if not scheme or not scheme.endswith("://"):
            raise ValueError("scheme must end with '://'")
        self._scheme = scheme

    def join(self, *args: str) -> str:
        """Join path components, compatible with both Python 3.11 and 3.12.

        Args:
            *args: The path components to join.

        Returns:
            The joined path.
        """
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            # Python 3.11 style: single iterable argument
            return self.sep.join(args[0])
        # Python 3.12 style: multiple string arguments
        return self.sep.join(args)

    def normcase(self, s: str) -> str:
        """Return normalized case version of the path (Python 3.12).

        Args:
            s: The path to normalize.

        Returns:
            The normalized path.
        """
        return s  # Cloud paths are case-sensitive

    def isabs(self, path: str) -> bool:
        """Check if path is absolute (Python 3.12).

        Args:
            path: The path to check.

        Returns:
            True if the path is absolute, False otherwise.
        """
        # A cloud path is absolute if it has both a drive (bucket) and root
        drv, root, _ = self.splitroot(path)
        return bool(drv and root)

    def splitdrive(self, path: str) -> tuple[str, str]:
        """Split path into drive and rest (Python 3.12).

        Args:
            path: The path to split.

        Returns:
            A tuple of (drive, rest).
        """
        # For cloud paths, the drive includes the scheme and bucket
        drv, root, rel = self.splitroot(path)
        # Reconstruct the path without the drive
        if root or rel:
            rest = (
                root + self.sep.join([rel])
                if isinstance(rel, str)
                else root + self.sep.join(rel)
                if rel
                else root
            )
            return drv, rest
        return drv, ""

    def parse_parts(self, parts: Iterable[str]) -> tuple[str, str, list[str]]:
        parsed: list[str] = []
        sep = self.sep
        drv = root = ""
        it = reversed(list(parts))
        for part in it:
            if not part:
                continue
            drv, root, rel = self.splitroot(part)
            if sep in rel:
                for x in reversed(rel.split(sep)):
                    if x and x != ".":
                        parsed.append(x)
            else:
                if rel and rel != ".":
                    parsed.append(rel)
            if drv or root:
                if not drv:
                    for part in it:
                        if not part:
                            continue
                        drv = self.splitroot(part)[0]
                        if drv:
                            break
                break
        if drv or root:
            parsed.append(drv + root)
        parsed.reverse()
        return drv, root, parsed

    def _split_scheme(self, s: str) -> Tuple[str, str]:
        """Split a string that may start with ``<scheme>://`` into (bucket, rest).

        Parameters:
            s: The string to split.

        Returns:
            A tuple of (bucket, rest_without_leading_slash). If no scheme present,
            returns ("", s).
        """
        if not s.startswith(self._scheme):
            return "", s
        rem = s[len(self._scheme) :]
        if not rem:
            return "", ""
        slash = rem.find("/")
        if slash == -1:
            return rem, ""
        return rem[:slash], rem[slash + 1 :]

    def splitroot(self, part: str) -> tuple[str, str, str]:
        """Return (drv, root, rel) like pathlib flavours.

        Parameters:
            part: The path part to split.

        Returns:
            A tuple of (drv, root, rel) where:
            - drv: "<scheme>://bucket" when scheme and bucket are present, else ""
            - root: "/" if there is an explicit root after the bucket, else ""
            - rel: remaining relative portion without leading slashes
        """
        bucket, rest = self._split_scheme(part)
        if not bucket:
            if part.startswith(self.sep):
                return "", self.sep, part.lstrip(self.sep)
            return "", "", part

        if rest or part.startswith(f"{self._scheme}{bucket}/"):
            return f"{self._scheme}{bucket}", self.sep, rest.lstrip(self.sep)
        return f"{self._scheme}{bucket}", "", ""

    def join_parsed_parts(
        self, drv: str, root: str, parts: list[str], drv2: str, root2: str, parts2: list[str]
    ) -> tuple[str, str, list[str]]:
        if root2:
            if not drv2 and drv:
                return drv, root2, [drv + root2] + parts2[1:]
        elif drv2:
            if drv2 == drv or self.casefold(drv2) == self.casefold(drv):
                return drv, root, parts + parts2[1:]
        else:
            # If left side is bucket-only (has drive but no root), emulate
            # pathlib behavior by inserting a root between drive and parts.
            # This ensures `PureGCSPath('gs://foo') / 'bar'` becomes
            # `gs://foo/bar` instead of `gs://foobar`.
            if drv and not root:
                # construct a minimal bucket-root element as first part
                bucket_root = drv + self.sep
                if parts:
                    # replace existing first element (drive without root)
                    parts = [bucket_root] + parts[1:]
                else:
                    parts = [bucket_root]
                root = self.sep
            return drv, root, parts + parts2
        return drv2, root2, parts2

    def casefold(self, s: str) -> str:
        return s

    def casefold_parts(self, parts: list[str]) -> list[str]:
        return parts

    def compile_pattern(self, pattern: str):  # noqa: ANN202
        return re.compile(fnmatch.translate(pattern)).fullmatch

    def is_reserved(self, parts) -> bool:  # noqa: ANN001
        return False

    def make_uri(self, path: PurePath) -> str:
        if not path.is_absolute():
            raise ValueError("relative path can't be expressed as a file URI")
        return str(path)


class PureCloudPath(PurePath):
    """Generic PurePath for cloud schemes. Subclasses must set ``cloud_prefix``.

    Compatible with both Python 3.11 and 3.12+ pathlib implementations.
    """

    cloud_prefix: str = ""
    __slots__ = ()

    def __new__(cls, *args: str) -> PureCloudPath:
        if not getattr(cls, "cloud_prefix", None):
            raise ValueError("cloud_prefix must be defined in subclass")

        # For user-facing construction, validate that paths start with cloud_prefix
        # But allow relative paths for internal operations (like match patterns)
        if args:
            # Check if this looks like a user-facing construction
            # (single string argument that doesn't start with '/' or contain wildcards)
            first_arg = args[0]
            if isinstance(first_arg, str):
                # Empty strings should be rejected
                if not first_arg and len(args) == 1:
                    raise ValueError(f"Path must start with '{cls.cloud_prefix}': {first_arg}")

                # Check if it starts with a different cloud scheme
                other_schemes = ["gs://", "s3://", "r2://", "http://", "https://"]
                has_other_scheme = any(
                    first_arg.startswith(scheme)
                    for scheme in other_schemes
                    if scheme != cls.cloud_prefix
                )

                # Allow relative paths (for internal use) and wildcard patterns
                is_pattern = any(c in first_arg for c in ["*", "?", "["])
                is_relative = (
                    not first_arg.startswith(cls.cloud_prefix)
                    and not first_arg.startswith("/")
                    and not has_other_scheme
                )

                # Reject paths with wrong cloud scheme
                if has_other_scheme:
                    raise ValueError(f"Path must start with '{cls.cloud_prefix}': {first_arg}")

                # Only validate if it looks like user input (not a pattern, not clearly relative)
                if not is_pattern and not is_relative and first_arg:  # Added check for non-empty
                    if len(args) == 1:
                        if not first_arg.startswith(cls.cloud_prefix):
                            raise ValueError(
                                f"Path must start with '{cls.cloud_prefix}': {first_arg}"
                            )
                    else:
                        joined = "".join(str(a) for a in args)
                        if not joined.startswith(cls.cloud_prefix):
                            raise ValueError(f"Path must start with '{cls.cloud_prefix}': {joined}")

        return super().__new__(cls, *args)

    if _PY312_PLUS:

        def __init__(self, *args):
            """Initialize for Python 3.12+, storing raw paths."""
            # Store raw paths for lazy parsing (Python 3.12 style)
            paths = []
            for arg in args:
                if isinstance(arg, PurePath):
                    if arg._flavour is not self._flavour:
                        paths.append(arg.as_posix())
                    else:
                        paths.extend(arg._raw_paths)
                else:
                    try:
                        path = os.fspath(arg)
                    except TypeError:
                        path = arg
                    if not isinstance(path, str):
                        raise TypeError(
                            "argument should be a str or an os.PathLike "
                            "object where __fspath__ returns a str, "
                            f"not {type(path).__name__!r}"
                        )
                    paths.append(path)

            # Handle absolute path joining: if joining with a path starting with "/",
            # it should replace the path after the bucket, not append to it
            if len(paths) > 1:
                result_paths = [paths[0]]
                for path in paths[1:]:
                    if path.startswith("/"):
                        # Extract bucket from previous path if it's a cloud path
                        prev = result_paths[-1]
                        if self.cloud_prefix in prev:
                            drv, _, _ = self._flavour.splitroot(prev)
                            if drv:
                                # Replace with bucket + new absolute path
                                result_paths = [drv + path]
                                continue
                    result_paths.append(path)
                paths = result_paths

            self._raw_paths = paths

        def _load_parts(self):
            """Parse the raw paths into drive, root, and tail components (Python 3.12)."""
            paths = self._raw_paths
            if len(paths) == 0:
                path = ""
            elif len(paths) == 1:
                path = paths[0]
            else:
                path = self._flavour.join(*paths)
            drv, root, tail = self._parse_path(path)
            self._drv = drv
            self._root = root
            self._tail_cached = tail

        @classmethod
        def _parse_path(cls, path: str) -> tuple[str, str, list[str]]:
            """Parse a path string into (drive, root, tail) components (Python 3.12)."""
            if not path:
                return "", "", []
            sep = cls._flavour.sep
            drv, root, rel = cls._flavour.splitroot(path)
            if not rel:
                return drv, root, []
            parsed = [sys.intern(str(x)) for x in rel.split(sep) if x and x != "."]
            return drv, root, parsed

    @property
    def bucket(self) -> str:
        """Extract the bucket name from the path."""
        drv = self.drive
        prefix = self.__class__.cloud_prefix
        if drv.startswith(prefix):
            return drv[len(prefix) :]
        return ""

    def as_uri(self) -> str:
        """Return the path as a URI.

        For cloud paths, the path itself is already a URI, so we just
        return the string representation.

        Returns:
            The path as a URI.
        """
        if not self.is_absolute():
            raise ValueError("relative path can't be expressed as a file URI")
        return str(self)


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
