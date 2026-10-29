"""Tests for `PackedMediaStore`."""

import pickle
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from alp_data.export.shards import ShardWriter
from alp_data.io.packed_media_store import PackedMediaStore


def _write_two_shards(media_dir: Path) -> tuple[list[str], list[tuple[int, int, int]]]:
    media_dir.mkdir()
    locations = []
    names = ["shard-00000.tar", "shard-00001.tar"]
    for shard_idx, name in enumerate(names):
        with ShardWriter(media_dir / name) as writer:
            entry = writer.add(shard_idx, f"blob in shard {shard_idx}".encode(), "flac")
            locations.append((shard_idx, entry.offset, entry.size))
    return names, locations


def test_read_returns_exact_bytes(tmp_path: Path) -> None:
    names, locations = _write_two_shards(tmp_path / "media")
    store = PackedMediaStore(tmp_path / "media", names)
    assert store.read(*locations[0]) == b"blob in shard 0"
    assert store.read(*locations[1]) == b"blob in shard 1"


def test_handles_are_opened_lazily_and_cached_per_shard(tmp_path: Path) -> None:
    names, locations = _write_two_shards(tmp_path / "media")
    store = PackedMediaStore(tmp_path / "media", names)
    assert store.open_handles == 0
    store.read(*locations[0])
    store.read(*locations[0])
    assert store.open_handles == 1
    store.read(*locations[1])
    assert store.open_handles == 2


def test_pickling_drops_open_handles(tmp_path: Path) -> None:
    names, locations = _write_two_shards(tmp_path / "media")
    store = PackedMediaStore(tmp_path / "media", names)
    store.read(*locations[0])
    clone = pickle.loads(pickle.dumps(store))
    assert clone.open_handles == 0
    assert clone.read(*locations[1]) == b"blob in shard 1"


def test_known_shard_size_is_passed_to_remote_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On object stores the open call gets `size=` so fsspec skips the metadata request."""
    import alp_data.io.packed_media_store as mod

    names, locations = _write_two_shards(tmp_path / "media")
    sizes = [(tmp_path / "media" / n).stat().st_size for n in names]
    calls: list[dict] = []

    class FakeRemoteFS:
        protocol = ("gs", "gcs")

        def open(self, path: str, mode: str = "rb", **kwargs: Any) -> BinaryIO:  # noqa: ANN401
            calls.append(kwargs)
            return open(path, mode)

    monkeypatch.setattr(mod, "filesystem_from_path", lambda _p: FakeRemoteFS())
    store = PackedMediaStore(tmp_path / "media", names, shard_sizes=sizes)
    assert store.read(*locations[1]) == b"blob in shard 1"
    assert calls == [{"cache_type": "none", "size": sizes[1]}]


def test_unknown_shard_size_omits_size_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import alp_data.io.packed_media_store as mod

    names, locations = _write_two_shards(tmp_path / "media")
    calls: list[dict] = []

    class FakeRemoteFS:
        protocol = "s3"

        def open(self, path: str, mode: str = "rb", **kwargs: Any) -> BinaryIO:  # noqa: ANN401
            calls.append(kwargs)
            return open(path, mode)

    monkeypatch.setattr(mod, "filesystem_from_path", lambda _p: FakeRemoteFS())
    store = PackedMediaStore(tmp_path / "media", names)
    store.read(*locations[0])
    assert calls == [{"cache_type": "none"}]
