"""
# TODO (milad) update

This file offers path functionalities for homogeneous resource access.

This file provides extensions of pathlib's PurePath classes to support URI-like cloud
paths. These enable path manipulation without providing any IO functionality. This
allows using os.Pathlike objects in places where IO is not needed and providing a more
unified interface for local/cloud paths. IO operations are handled by fsspec libraries
(e.g. gcsfs, s3fs, etc.).

The implementation mirrors pathlib's design. Some code duplication was necessary since
pathlib doesn't expose its private `_Flavour` classes.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from pathlib import Path, PurePath
from typing import Iterable, Tuple, TypeAlias

# TODO what should happen when you do PureGCSPath("gs:///")
# TODO Path("/foo/bar") / PureGCSPath("gs://baz")


class _CloudFlavour:
    """A minimal pathlib flavour for generic cloud-like paths.

    It provides only the subset of the contract required by ``PurePath``.
    """

    sep = "/"
    altsep = ""
    has_drv = True  # model "<scheme>://bucket" as a drive
    pathmod = posixpath

    def __init__(self, scheme: str) -> None:
        if not scheme or not scheme.endswith("://"):
            raise ValueError("scheme must end with '://'")
        self._scheme = scheme
        self.join = self.sep.join

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
    """Generic PurePath for cloud schemes. Subclasses must set ``cloud_prefix``."""

    cloud_prefix: str = ""
    __slots__ = ()

    def __new__(cls, *args: str) -> PureCloudPath:
        if not getattr(cls, "cloud_prefix", None):
            raise ValueError("cloud_prefix must be defined in subclass")
        if len(args) == 1:
            arg = args[0]
            if not isinstance(arg, str):
                raise TypeError(f"Expected string, got {type(arg)}")
            if not arg.startswith(cls.cloud_prefix):
                raise ValueError(f"Path must start with '{cls.cloud_prefix}': {arg}")
        else:
            joined = "".join(str(a) for a in args)
            if not joined.startswith(cls.cloud_prefix):
                raise ValueError(f"Path must start with '{cls.cloud_prefix}': {joined}")
        return super().__new__(cls, *args)

    @property
    def bucket(self) -> str:
        drv = self.drive
        prefix = self.__class__.cloud_prefix
        if drv.startswith(prefix):
            return drv[len(prefix) :]
        return ""

    @property
    def object_path(self) -> str:
        parts = self.parts
        if not parts:
            return ""
        if self.anchor:
            if len(parts) > 1:
                return "/".join(parts[1:])
            return ""
        return "/".join(parts)

    def with_bucket(self, bucket: str) -> PureCloudPath:
        if not bucket:
            raise ValueError("bucket must be a non-empty string")
        prefix = self.__class__.cloud_prefix
        if self.object_path:
            obj = self.object_path.lstrip("/")
            return self.__class__(f"{prefix}{bucket}/{obj}")
        return self.__class__(f"{prefix}{bucket}")

    def relative_to_bucket(self) -> str:
        obj = self.object_path
        return obj if obj.startswith("/") else (f"/{obj}" if obj else "/")


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
