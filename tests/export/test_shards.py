"""Tests for the tar shard writer used by `alp_data.export`."""

import tarfile
from pathlib import Path

import pytest

from alp_data.export.shards import ShardWriter, member_name


def _read_range(path: Path, offset: int, size: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def test_member_name_is_zero_padded_source_index() -> None:
    assert member_name(3, "flac") == "000000003.flac"


def test_bytes_are_readable_at_recorded_offset(tmp_path: Path) -> None:
    shard = tmp_path / "shard-00000.tar"
    with ShardWriter(shard) as writer:
        first = writer.add(0, b"first blob", "flac")
        second = writer.add(1, b"second, longer blob", "flac")
    assert _read_range(shard, first.offset, first.size) == b"first blob"
    assert _read_range(shard, second.offset, second.size) == b"second, longer blob"


def test_identical_bytes_share_one_member(tmp_path: Path) -> None:
    shard = tmp_path / "shard-00000.tar"
    with ShardWriter(shard) as writer:
        a = writer.add(0, b"same", "flac")
        b = writer.add(1, b"same", "flac")
        c = writer.add(2, b"different", "flac")
    assert (a.offset, a.size, a.sha256) == (b.offset, b.size, b.sha256)
    assert c.offset > a.offset
    with tarfile.open(shard) as tar:
        assert tar.getnames() == ["000000000.flac", "000000002.flac"]


def test_members_are_named_by_source_index(tmp_path: Path) -> None:
    shard = tmp_path / "shard-00000.tar"
    with ShardWriter(shard) as writer:
        writer.add(41, b"x", "flac")
        writer.add(7, b"y", "wav")
    with tarfile.open(shard) as tar:
        assert tar.getnames() == ["000000041.flac", "000000007.wav"]


def test_final_file_appears_only_on_close(tmp_path: Path) -> None:
    shard = tmp_path / "shard-00000.tar"
    with ShardWriter(shard) as writer:
        writer.add(0, b"x", "flac")
        assert not shard.exists()
    assert shard.exists()
    assert list(tmp_path.iterdir()) == [shard]


def test_failure_inside_context_leaves_nothing_behind(tmp_path: Path) -> None:
    shard = tmp_path / "shard-00000.tar"
    with pytest.raises(RuntimeError):
        with ShardWriter(shard) as writer:
            writer.add(0, b"x", "flac")
            raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []


def test_writer_reports_shard_size_and_digest(tmp_path: Path) -> None:
    import hashlib

    shard = tmp_path / "shard-00000.tar"
    with ShardWriter(shard) as writer:
        writer.add(0, b"x" * 100, "flac")
    assert writer.size == shard.stat().st_size
    assert writer.sha256 == hashlib.sha256(shard.read_bytes()).hexdigest()
