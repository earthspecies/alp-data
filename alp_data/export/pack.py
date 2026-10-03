"""The pack format: uncompressed tar shards of audio blobs plus a parquet table.

`pack` runs the shared export loop with `TarSink`, which writes one tar
member per row and records where it landed on the table, so a reader can
fetch any sample with one range read. See `alp_data.export.runner` for what
the loop does and `PackedDataset` for how a pack is read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import pyarrow as pa
import yaml
from fsspec import AbstractFileSystem

from alp_data.dataset import ChainedDatasetConfig
from alp_data.export.columns import (
    AUDIO_FORMAT_COL,
    OFFSET_COL,
    SHA256_COL,
    SHARD_COL,
    SIZE_COL,
    SOURCE_INDEX_COL,
)
from alp_data.export.runner import (
    PARTS_DIR,
    ExportConfig,
    ExportJob,
    FinaliseContext,
    OnError,
    join,
    read_parquet,
    read_shard_metadata,
    run_export,
    shard_metadata,
    write_parquet,
)
from alp_data.export.serializers import AudioFormat
from alp_data.export.shards import ShardWriter
from alp_data.io import AnyPathT

CONFIG_FILE = "config.yaml"
TABLE_FILE = "table.parquet"
ERRORS_FILE = "pack_errors.parquet"
MEDIA_DIR = "media"

_LARGE_SHARD_BYTES = 4 * 1024**3

logger = logging.getLogger("alp_data")


def shard_name(shard: int) -> str:
    """File name of shard number `shard`.

    Parameters
    ----------
    shard : int
        Shard number.

    Returns
    -------
    str
        A name such as `"shard-00003.tar"`.
    """
    return f"shard-{shard:05d}.tar"


def _part_name(shard: int) -> str:
    return f"table-{shard:05d}.parquet"


def pack(
    config: ExportConfig | ChainedDatasetConfig,
    out_path: str | AnyPathT,
    *,
    samples_per_shard: int = 1000,
    audio_format: AudioFormat = "flac",
    num_workers: int = 1,
    on_error: OnError = "raise",
    audio_key: str | None = None,
    sample_rate_key: str | None = None,
) -> AnyPathT:
    """Pack a configured dataset into `out_path`.

    Parameters
    ----------
    config : DatasetConfig | ConcatConfig | ChainedDatasetConfig
        What to pack. A chained config is packed as a concatenation, with a
        warning, because a pack is a single table.
    out_path : str | AnyPathT
        Destination directory. Any `anypath` target: local, `gs://`, `s3://`.
    samples_per_shard : int
        Rows per tar shard. Shard `j` holds source rows `j * N` to `(j + 1) * N`.
    audio_format : {"flac", "wav"}
        Blob encoding. FLAC is 16-bit PCM; WAV is float32 and bit-exact.
    num_workers : int
        Spawned worker processes. Each rebuilds the dataset from `config` and
        packs whole shards. `1` runs in the calling process.
    on_error : {"raise", "skip"}
        What to do when `ds[i]` fails after retries. `"skip"` drops the row,
        records it in `pack_errors.parquet`, and counts it in `config.yaml`.
    audio_key : str | None
        Output key holding the audio array. Defaults to `"audio"`, or to what
        the config's `output_take_and_give` maps `"audio"` to.
    sample_rate_key : str | None
        Output key holding the sample rate. Defaults to `"sample_rate"` when
        present, else the config's `sample_rate` is used for every row.

    Returns
    -------
    AnyPathT
        `out_path`, as an `anypath`.

    Notes
    -----
    Rerunning with the same arguments resumes: shards that already have both
    their tar and their table slice are skipped, and a pack whose
    `config.yaml` exists is left untouched.
    """
    return run_export(
        config,
        out_path,
        TarSink(),
        samples_per_shard=samples_per_shard,
        audio_format=audio_format,
        num_workers=num_workers,
        on_error=on_error,
        audio_key=audio_key,
        sample_rate_key=sample_rate_key,
    )


@dataclass
class TarSink:
    """Export sink that writes the pack layout."""

    errors_file: str = ERRORS_FILE

    def is_complete(self, fs: AbstractFileSystem, out: AnyPathT) -> bool:
        return fs.exists(join(out, CONFIG_FILE))

    def prepare(self, fs: AbstractFileSystem, out: AnyPathT) -> None:
        fs.makedirs(join(out, MEDIA_DIR), exist_ok=True)

    def shard_is_finished(self, fs: AbstractFileSystem, out: AnyPathT, job: ExportJob) -> bool:
        return fs.exists(join(out, MEDIA_DIR, shard_name(job.shard))) and fs.exists(
            join(out, PARTS_DIR, _part_name(job.shard))
        )

    def open_shard(self, fs: AbstractFileSystem, out: AnyPathT, job: ExportJob) -> _TarShard:
        return _TarShard(fs, out, job)

    def finalise(self, fs: AbstractFileSystem, out: AnyPathT, ctx: FinaliseContext) -> None:
        parts = [join(out, PARTS_DIR, _part_name(s)) for s in range(ctx.num_shards)]
        table = pl.concat([read_parquet(fs, p) for p in parts], how="diagonal_relaxed")
        with fs.open(join(out, TABLE_FILE), "wb") as f:
            table.write_parquet(f)

        opaque: dict[str, str] = {}
        by_shard = {r.shard: r for r in ctx.results}
        shards = []
        for s in range(ctx.num_shards):
            if s in by_shard:
                opaque.update(by_shard[s].opaque_columns)
                shards.append(by_shard[s].info)
            else:
                # Finished by an earlier run: its part file carries what the
                # worker reported then.
                kinds, info = read_shard_metadata(fs, parts[s])
                opaque.update(kinds)
                shards.append(info)

        fallback = sum(int(s.get("num_lossless_fallback", 0)) for s in shards)
        meta = ctx.provenance(opaque, table.height, fallback)
        meta["shards"] = [
            {k: v for k, v in s.items() if k in ("name", "size", "sha256")} for s in shards
        ]
        with fs.open(join(out, CONFIG_FILE), "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)


@dataclass
class _TarShard:
    fs: AbstractFileSystem
    out: AnyPathT
    job: ExportJob
    rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.writer = ShardWriter(join(self.out, MEDIA_DIR, shard_name(self.job.shard)))
        self.writer.__enter__()

    def add(self, source_index: int, row: dict[str, Any], audio: bytes, ext: str) -> None:
        entry = self.writer.add(source_index, audio, ext)
        row[SOURCE_INDEX_COL] = source_index
        row[AUDIO_FORMAT_COL] = ext
        row[SHARD_COL] = self.job.shard
        row[OFFSET_COL] = entry.offset
        row[SIZE_COL] = entry.size
        row[SHA256_COL] = entry.sha256
        self.rows.append(row)

    def close(self, opaque_columns: dict[str, str], stats: dict[str, Any]) -> dict[str, Any]:
        self.writer.__exit__(None, None, None)
        if self.writer.size > _LARGE_SHARD_BYTES:
            logger.warning(
                "Shard %d is %.1f GB; consider a smaller samples_per_shard.",
                self.job.shard,
                self.writer.size / 1024**3,
            )
        info = {
            "name": shard_name(self.job.shard),
            "size": self.writer.size,
            "sha256": self.writer.sha256,
            **stats,
        }
        part = join(self.out, PARTS_DIR, _part_name(self.job.shard))
        table = _rows_to_table(self.rows)
        table = table.replace_schema_metadata(shard_metadata(opaque_columns, info))
        write_parquet(self.fs, part, table)
        return info

    def abort(self) -> None:
        self.writer.__exit__(RuntimeError, RuntimeError("aborted"), None)


def _rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    if rows:
        return pa.Table.from_pylist(rows)
    return pa.table(
        {
            SOURCE_INDEX_COL: pa.array([], pa.int64()),
            SHARD_COL: pa.array([], pa.int64()),
            OFFSET_COL: pa.array([], pa.int64()),
            SIZE_COL: pa.array([], pa.int64()),
            SHA256_COL: pa.array([], pa.string()),
            AUDIO_FORMAT_COL: pa.array([], pa.string()),
        }
    )
