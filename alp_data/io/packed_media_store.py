"""Range reads out of packed media shards, with one cached handle per shard."""

from __future__ import annotations

from typing import Any

from alp_data.io.filesystem import filesystem_from_path
from alp_data.io.paths import AnyPathT, anypath


class PackedMediaStore:
    """Read blobs from tar shards by `(shard, offset, size)`.

    Shard handles are opened on first use and kept open, so a long-running
    reader holds as many handles as there are shards it has touched. The
    store pickles without its handles, which makes it safe to hand to spawned
    DataLoader workers: each worker opens its own.

    Parameters
    ----------
    media_dir : str | AnyPathT
        Directory holding the shard files. Any `anypath` target.
    shard_names : list[str]
        File names of the shards, indexed by shard number.
    """

    def __init__(self, media_dir: str | AnyPathT, shard_names: list[str]) -> None:
        self.media_dir = anypath(str(media_dir))
        self.shard_names = list(shard_names)
        self._handles: dict[int, Any] = {}

    @property
    def open_handles(self) -> int:
        """Number of shard handles currently open in this process."""
        return len(self._handles)

    def read(self, shard: int, offset: int, size: int) -> bytes:
        """Read one blob.

        Parameters
        ----------
        shard : int
            Shard number.
        offset : int
            Byte offset within the shard.
        size : int
            Number of bytes to read.

        Returns
        -------
        bytes
            The blob.
        """
        handle = self._handles.get(shard)
        if handle is None:
            fs = filesystem_from_path(self.media_dir)
            path = str(self.media_dir / self.shard_names[shard])
            kwargs = {} if fs.protocol in ("file", "local") else {"cache_type": "none"}
            handle = fs.open(path, "rb", **kwargs)
            self._handles[shard] = handle
        handle.seek(offset)
        return handle.read(size)

    def close(self) -> None:
        """Close every open shard handle."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state
