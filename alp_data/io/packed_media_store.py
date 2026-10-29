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
    shard_sizes : list[int] | None, optional
        Byte size of each shard, indexed like `shard_names`. On object stores a
        known size is passed to `open`, which saves fsspec a metadata request per
        shard. Unknown sizes are looked up by fsspec as usual.
    """

    def __init__(
        self,
        media_dir: str | AnyPathT,
        shard_names: list[str],
        shard_sizes: list[int] | None = None,
    ) -> None:
        self.media_dir = anypath(str(media_dir))
        self.shard_names = list(shard_names)
        self.shard_sizes = None if shard_sizes is None else [int(n) for n in shard_sizes]
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
            protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
            kwargs: dict[str, Any] = {}
            if protocol not in ("file", "local"):
                kwargs["cache_type"] = "none"
                if self.shard_sizes is not None:
                    kwargs["size"] = self.shard_sizes[shard]
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
