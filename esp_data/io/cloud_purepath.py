"""PurePath-compatible support for URI-like cloud paths.

This module provides a minimal, stdlib-compatible, pure-path implementation for
scheme-based paths such as ``gs://`` and ``s3://``. It mirrors enough of
``pathlib``'s internal flavour protocol to interoperate with ``pathlib.PurePath``
APIs (no I/O).

Design goals:
- Keep changes minimal by reusing ``pathlib.PurePath`` with a custom flavour
- POSIX-like separators ("/")
- Treat ``<scheme>://bucket`` as the drive, and "/" as root when anchored
- Support ``.name``, ``.suffix``, ``.stem``, ``.parts``, ``.parents``, joins, etc.

You can create specific subclasses by setting only ``cloud_prefix`` (scheme),
for example ``GCSPurePath`` with ``cloud_prefix = "gs://"``.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from pathlib import PurePath
from typing import Iterable, Tuple


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
        self,
        drv: str,
        root: str,
        parts: list[str],
        drv2: str,
        root2: str,
        parts2: list[str],
    ) -> tuple[str, str, list[str]]:
        if root2:
            if not drv2 and drv:
                return drv, root2, [drv + root2] + parts2[1:]
        elif drv2:
            if drv2 == drv or self.casefold(drv2) == self.casefold(drv):
                return drv, root, parts + parts2[1:]
        else:
            return drv, root, parts + parts2
        return drv2, root2, parts2

    # --- Case/Matching/URI ---
    def casefold(self, s: str) -> str:
        return s

    def casefold_parts(self, parts: list[str]) -> list[str]:
        return parts

    def compile_pattern(self, pattern: str):
        return re.compile(fnmatch.translate(pattern)).fullmatch

    def is_reserved(self, parts: list[str]) -> bool:
        return False

    def make_uri(self, path: PurePath) -> str:
        if not path.is_absolute():
            raise ValueError("relative path can't be expressed as a file URI")
        return str(path)


class PureCloudPath(PurePath):
    """Generic PurePath for cloud schemes. Subclasses must set ``cloud_prefix``.

    Example: subclass with ``cloud_prefix = "gs://"``.
    """

    cloud_prefix: str = ""
    __slots__ = ()

    @classmethod
    def _get_flavour(cls) -> _CloudFlavour:
        if not getattr(cls, "cloud_prefix", None):
            raise ValueError("cloud_prefix must be defined in subclass")
        # Cache a flavour per subclass
        flavour_attr = "_flavour_instance"
        fl = getattr(cls, flavour_attr, None)
        if fl is None or getattr(fl, "_scheme", None) != cls.cloud_prefix:
            fl = _CloudFlavour(cls.cloud_prefix)
            setattr(cls, flavour_attr, fl)
        return fl

    # pathlib looks up the class attribute _flavour
    @property  # type: ignore[override]
    def _flavour(self):  # noqa: N802 (match stdlib name)
        return self.__class__._get_flavour()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure each subclass has a bound _flavour class attribute for speed
        try:
            cls._flavour = cls._get_flavour()  # type: ignore[attr-defined]
        except Exception:
            # Defer errors until first instantiation if cloud_prefix missing
            pass

    def __new__(cls, *args):
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

    def with_bucket(self, bucket: str):
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


class PureGCSPath(PureCloudPath):
    cloud_prefix = "gs://"
    __slots__ = ()


class PureS3Path(PureCloudPath):
    cloud_prefix = "s3://"
    __slots__ = ()


def is_cloud_pure_path(p: PurePath | str) -> bool:
    s = str(p)
    return s.startswith("gs://") or s.startswith("s3://")
