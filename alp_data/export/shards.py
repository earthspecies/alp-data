"""Tar shard writer that records where every blob lands.

A shard is an uncompressed tar with one member per packed row. The writer
tracks the byte offset and size of each member so that a reader can fetch a
blob with one range read, without opening the archive as a tar. Identical
bytes within one shard are written once and share a location.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from dataclasses import dataclass
from types import TracebackType
from typing import IO

from alp_data.io import AnyPathT, anypath, filesystem_from_path


def member_name(source_index: int, ext: str) -> str:
    """Name of the tar member holding the blob for a source row.

    Parameters
    ----------
    source_index : int
        Row index in the source dataset.
    ext : str
        File extension without the dot, e.g. `"flac"`.

    Returns
    -------
    str
        A zero-padded name such as `"000000003.flac"`.
    """
    return f"{source_index:09d}.{ext}"


@dataclass(frozen=True)
class ShardEntry:
    """Where a blob lives inside a shard.

    Attributes
    ----------
    offset : int
        Byte offset of the blob's first byte within the shard file.
    size : int
        Blob length in bytes.
    sha256 : str
        Hex digest of the blob.
    """

    offset: int
    size: int
    sha256: str


class _HashingWriter:
    """File-like wrapper that counts and hashes everything written through it."""

    def __init__(self, fileobj: IO[bytes]) -> None:
        self._fileobj = fileobj
        self.size = 0
        self._hasher = hashlib.sha256()

    def write(self, data: bytes) -> int:
        self._hasher.update(data)
        self.size += len(data)
        return self._fileobj.write(data)

    def tell(self) -> int:
        return self.size

    def close(self) -> None:
        self._fileobj.close()

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


class ShardWriter:
    """Write blobs into one tar shard atomically.

    The archive is written to `<path>.tmp` and renamed to `path` when the
    context exits cleanly. If the context exits with an exception, the temp
    file is removed and `path` is never created. Any `anypath` target works;
    writes go through fsspec.

    Parameters
    ----------
    path : str | AnyPathT
        Final location of the shard.

    Attributes
    ----------
    size : int
        Total bytes written, available after the context exits.
    sha256 : str
        Hex digest of the whole shard file, available after the context exits.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> with tempfile.TemporaryDirectory() as d:
    ...     with ShardWriter(pathlib.Path(d) / "shard-00000.tar") as w:
    ...         entry = w.add(0, b"hello", "flac")
    ...     entry.size
    5
    """

    def __init__(self, path: str | AnyPathT) -> None:
        self.path = anypath(str(path))
        self._fs = filesystem_from_path(self.path)
        self._tmp_path = str(self.path) + ".tmp"
        self._fileobj: _HashingWriter | None = None
        self._tar: tarfile.TarFile | None = None
        self._seen: dict[str, ShardEntry] = {}
        self.size = 0
        self.sha256 = ""

    def __enter__(self) -> ShardWriter:
        self._fs.makedirs(str(self.path.parent), exist_ok=True)
        self._fileobj = _HashingWriter(self._fs.open(self._tmp_path, "wb"))
        self._tar = tarfile.open(fileobj=self._fileobj, mode="w")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._tar is not None:
            self._tar.close()
        if self._fileobj is not None:
            self._fileobj.close()
            self.size = self._fileobj.size
            self.sha256 = self._fileobj.hexdigest()
        if exc_type is None:
            self._fs.mv(self._tmp_path, str(self.path))
        else:
            self._fs.rm(self._tmp_path)

    def add(self, source_index: int, data: bytes, ext: str) -> ShardEntry:
        """Append a blob, or reuse an earlier identical one.

        Parameters
        ----------
        source_index : int
            Row index in the source dataset; becomes the member name.
        data : bytes
            The encoded blob.
        ext : str
            Member extension without the dot.

        Returns
        -------
        ShardEntry
            Location of the blob within this shard.

        Raises
        ------
        RuntimeError
            If called outside the `with` block.
        """
        if self._tar is None:
            raise RuntimeError("ShardWriter.add called outside of its context")
        digest = hashlib.sha256(data).hexdigest()
        if digest in self._seen:
            return self._seen[digest]
        info = tarfile.TarInfo(member_name(source_index, ext))
        info.size = len(data)
        # tarfile copies `info` on write and never fills `offset_data`, so the
        # data offset is the archive position plus the serialized header size.
        header_len = len(info.tobuf(self._tar.format, self._tar.encoding, self._tar.errors))
        offset = self._tar.offset + header_len
        self._tar.addfile(info, io.BytesIO(data))
        entry = ShardEntry(offset=offset, size=len(data), sha256=digest)
        self._seen[digest] = entry
        return entry
