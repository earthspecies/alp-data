"""Export to Hugging Face style parquet.

The Hub's native audio layout is parquet with the encoded audio embedded in a
struct of `bytes` and `path`, and the column declared as an `Audio` feature in
the parquet schema metadata. `to_hf` writes that shape either from a dataset
config, through the shared export loop, or from an existing pack, by copying
its blobs without decoding.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

import pyarrow as pa
import yaml
from fsspec import AbstractFileSystem

from alp_data.dataset import ChainedDatasetConfig
from alp_data.export.columns import (
    BOOKKEEPING_COLS,
    OFFSET_COL,
    SHARD_COL,
    SIZE_COL,
    SOURCE_INDEX_COL,
)
from alp_data.export.packed_dataset import PackedDataset
from alp_data.export.runner import (
    ExportConfig,
    ExportJob,
    FinaliseContext,
    OnError,
    join,
    run_export,
    write_parquet,
)
from alp_data.export.serializers import AudioFormat, decode_audio
from alp_data.export.shards import member_name
from alp_data.io import AnyPathT, anypath, filesystem_from_path

README_FILE = "README.md"
ERRORS_FILE = "export_errors.jsonl"

AUDIO_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def to_hf(
    source: ExportConfig | ChainedDatasetConfig | str | PathLike | AnyPathT,
    out_dir: str | AnyPathT,
    *,
    samples_per_shard: int = 1000,
    audio_format: AudioFormat = "flac",
    num_workers: int = 1,
    on_error: OnError = "raise",
    audio_key: str | None = None,
    sample_rate_key: str | None = None,
) -> AnyPathT:
    """Write a dataset as Hugging Face style parquet files.

    Parameters
    ----------
    source : DatasetConfig | ConcatConfig | ChainedDatasetConfig | path
        A dataset config, exported through the same loop as `pack`, or the
        path of an existing pack, whose blobs are copied without decoding.
    out_dir : str | AnyPathT
        Destination directory. Files are named `<split>-NNNNN-of-MMMMM.parquet`.
        A `README.md` records the source config and provenance.
    samples_per_shard : int
        Rows per parquet file.
    audio_format : {"flac", "wav"}
        Blob encoding when exporting from a config. Ignored for a pack, whose
        encoding is kept.
    num_workers : int
        Spawned worker processes when exporting from a config.
    on_error : {"raise", "skip"}
        What to do when a row fails. `"skip"` records it in `export_errors.jsonl`,
        a name the Hub's parquet loader ignores.
    audio_key, sample_rate_key : str | None
        Output keys, as for `pack`.

    Returns
    -------
    AnyPathT
        `out_dir`, as an `anypath`.

    Notes
    -----
    The parquet metadata declares only the audio column as an `Audio`
    feature; every other column is inferred by the reader. Opaque columns
    (TSV strings, array structs) are passed through unchanged. Nothing here
    imports the `datasets` library.
    """
    if isinstance(source, (str, PathLike)) or type(source).__name__.startswith("Pure"):
        return _pack_to_hf(source, out_dir, samples_per_shard)
    return run_export(
        source,
        out_dir,
        HFSink(),
        samples_per_shard=samples_per_shard,
        audio_format=audio_format,
        num_workers=num_workers,
        on_error=on_error,
        audio_key=audio_key,
        sample_rate_key=sample_rate_key,
    )


def file_name(split: str, shard: int, num_shards: int) -> str:
    """Name of one parquet file in the Hub convention.

    Parameters
    ----------
    split : str
        Split name.
    shard, num_shards : int
        This file's number and the total.

    Returns
    -------
    str
        A name such as `"train-00002-of-00010.parquet"`.
    """
    return f"{split}-{shard:05d}-of-{num_shards:05d}.parquet"


def _audio_metadata(audio_key: str, sample_rate: int | None) -> dict[bytes, bytes]:
    feature: dict[str, Any] = {"_type": "Audio"}
    if sample_rate is not None:
        feature["sampling_rate"] = int(sample_rate)
    return {b"huggingface": json.dumps({"info": {"features": {audio_key: feature}}}).encode()}


def _to_arrow(columns: dict[str, list[Any]], audio_key: str, sample_rate: int | None) -> pa.Table:
    arrays = {
        name: pa.array(values, type=AUDIO_TYPE if name == audio_key else None)
        for name, values in columns.items()
    }
    return pa.table(arrays).replace_schema_metadata(_audio_metadata(audio_key, sample_rate))


@dataclass
class HFSink:
    """Export sink that writes Hub-native parquet files."""

    errors_file: str = ERRORS_FILE

    def is_complete(self, fs: AbstractFileSystem, out: AnyPathT) -> bool:
        return fs.exists(join(out, README_FILE))

    def prepare(self, fs: AbstractFileSystem, out: AnyPathT) -> None:
        fs.makedirs(str(out), exist_ok=True)

    def shard_is_finished(
        self, fs: AbstractFileSystem, out: AnyPathT, shard: int, num_shards: int
    ) -> bool:
        return fs.exists(join(out, file_name(_split_of(out, fs), shard, num_shards)))

    def open_shard(self, fs: AbstractFileSystem, out: AnyPathT, job: ExportJob) -> _HFShard:
        return _HFShard(fs, out, job)

    def finalise(self, fs: AbstractFileSystem, out: AnyPathT, ctx: FinaliseContext) -> None:
        opaque: dict[str, str] = {}
        num_rows = 0
        for result in ctx.results:
            opaque.update(result.opaque_columns)
            num_rows += result.num_rows
        split = _split_name(ctx.config)
        files = []
        for s in range(ctx.num_shards):
            name = file_name(split, s, ctx.num_shards)
            files.append({"name": name, "size": fs.size(join(out, name))})
        meta = ctx.provenance(opaque, num_rows)
        meta["files"] = files
        _write_readme(fs, out, meta)


def _split_name(config: ExportConfig) -> str:
    return getattr(config, "split", None) or "train"


def _split_of(out: AnyPathT, fs: AbstractFileSystem) -> str:
    # The split is a property of the config, which the sink does not hold; the
    # runner asks about finished shards before any file exists in a fresh run,
    # and on a rerun every file carries the same split prefix.
    for entry in fs.ls(str(out), detail=False):
        name = str(entry).rsplit("/", 1)[-1]
        if name.endswith(".parquet") and "-of-" in name:
            return name.rsplit("-", 3)[0]
    return "train"


def _write_readme(fs: AbstractFileSystem, out: AnyPathT, meta: dict[str, Any]) -> None:
    body = (
        f"# {meta.get('name')}\n\n"
        f"Exported from `alp_data` with `alp_data.export.to_hf`. "
        f'Load with `datasets.load_dataset("parquet", data_dir=<this directory>)`.\n\n'
        "## Provenance\n\n```yaml\n" + yaml.safe_dump(meta, sort_keys=False) + "```\n"
    )
    with fs.open(join(out, README_FILE), "w") as f:
        f.write(body)


@dataclass
class _HFShard:
    fs: AbstractFileSystem
    out: AnyPathT
    job: ExportJob
    columns: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.columns[self.job.audio_key] = []
        self.name = file_name(_split_name(self.job.config), self.job.shard, self.job.num_shards)

    def add(self, source_index: int, row: dict[str, Any], audio: bytes) -> None:
        self.columns[self.job.audio_key].append(
            {"bytes": audio, "path": member_name(source_index, self.job.audio_format)}
        )
        for key, value in row.items():
            self.columns.setdefault(key, []).append(value)

    def close(self) -> dict[str, Any]:
        table = _to_arrow(self.columns, self.job.audio_key, self.job.declared_sample_rate)
        final = join(self.out, self.name)
        tmp = final + ".tmp"
        write_parquet(self.fs, tmp, table)
        self.fs.mv(tmp, final)
        return {"name": self.name, "size": self.fs.size(final)}

    def abort(self) -> None:
        pass


def _pack_to_hf(pack_path: Any, out_dir: str | AnyPathT, rows_per_file: int) -> AnyPathT:  # noqa: ANN401
    ds = PackedDataset(pack_path)
    out = anypath(str(out_dir))
    fs = filesystem_from_path(out)
    fs.makedirs(str(out), exist_ok=True)

    ext = ds.pack_config["audio_format"]
    sample_rate = ds.pack_config.get("sample_rate") or _first_sample_rate(ds)
    num_rows = len(ds._data)
    num_files = math.ceil(num_rows / rows_per_file)
    files = []
    for file_idx in range(num_files):
        start, stop = file_idx * rows_per_file, min((file_idx + 1) * rows_per_file, num_rows)
        columns: dict[str, list[Any]] = {ds.audio_key: []}
        for row_idx in range(start, stop):
            row = ds._data[row_idx]
            data = ds._store.read(int(row[SHARD_COL]), int(row[OFFSET_COL]), int(row[SIZE_COL]))
            columns[ds.audio_key].append(
                {"bytes": data, "path": member_name(int(row[SOURCE_INDEX_COL]), ext)}
            )
            for key, value in row.items():
                if key not in BOOKKEEPING_COLS:
                    columns.setdefault(key, []).append(value)
        name = file_name(ds.split, file_idx, num_files)
        write_parquet(fs, join(out, name), _to_arrow(columns, ds.audio_key, sample_rate))
        files.append({"name": name, "size": fs.size(join(out, name))})

    meta = dict(ds.pack_config)
    meta.pop("shards", None)
    meta["files"] = files
    meta["converted_from_pack"] = str(ds.path)
    _write_readme(fs, out, meta)
    return out


def _first_sample_rate(ds: PackedDataset) -> int:
    """Sample rate of the first row: from its table value, else by decoding its blob.

    Parameters
    ----------
    ds : PackedDataset
        The pack being converted.

    Returns
    -------
    int
        Sample rate in Hz.
    """
    row = ds._data[0]
    if ds.sample_rate_key is not None:
        return int(row[ds.sample_rate_key])
    data = ds._store.read(int(row[SHARD_COL]), int(row[OFFSET_COL]), int(row[SIZE_COL]))
    return int(decode_audio(data)[1])
