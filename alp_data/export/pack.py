"""Freeze a configured dataset into tar shards plus a parquet table.

`pack` builds the dataset from its config, calls `ds[i]` for every row, and
stores what comes back: the audio array as one encoded blob per row inside a
tar shard, every other key as a column of `table.parquet`. Nothing about the
source dataset class is touched, so whatever its `_process` does (windowing,
label parsing, `output_take_and_give`) is frozen into the pack.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import logging
import math
import multiprocessing as mp
import os
import subprocess
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from alp_data.dataset import (
    ChainedDatasetConfig,
    ConcatConfig,
    Dataset,
    DatasetConfig,
    dataset_from_config,
)
from alp_data.export.columns import (
    OFFSET_COL,
    SHA256_COL,
    SHARD_COL,
    SIZE_COL,
    SOURCE_INDEX_COL,
)
from alp_data.export.serializers import AudioFormat, encode_audio, encode_value
from alp_data.export.shards import ShardWriter
from alp_data.io import AnyPathT, anypath, filesystem_from_path

logger = logging.getLogger("alp_data")

PackConfig = DatasetConfig | ConcatConfig
OnError = Literal["raise", "skip"]

CONFIG_FILE = "config.yaml"
TABLE_FILE = "table.parquet"
ERRORS_FILE = "pack_errors.parquet"
MEDIA_DIR = "media"
PARTS_DIR = "parts"

_RETRIES = 3
_RETRY_BACKOFF_S = 0.5
_LARGE_SHARD_BYTES = 4 * 1024**3


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


@dataclass
class _ShardJob:
    config: PackConfig
    out: str
    shard: int
    start: int
    stop: int
    audio_format: AudioFormat
    audio_key: str
    sample_rate_key: str | None
    fallback_sample_rate: int | None
    on_error: OnError


@dataclass
class _ShardResult:
    shard: int
    size: int
    sha256: str
    num_rows: int
    num_skipped: int
    opaque_columns: dict[str, str] = field(default_factory=dict)


def pack(
    config: PackConfig | ChainedDatasetConfig,
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
        Output key holding the audio array. Defaults to `"audio"`.
    sample_rate_key : str | None
        Output key holding the sample rate. Defaults to `"sample_rate"` when
        present, else the config's `sample_rate` is used for every row.

    Returns
    -------
    AnyPathT
        `out_path`, as an `anypath`.

    Raises
    ------
    TypeError
        If `config` is not a dataset config (for example a live dataset).
    ValueError
        If the dataset is empty, the audio key cannot be resolved, or no
        sample rate is available for encoding.

    Notes
    -----
    Rerunning with the same arguments resumes: shards that already have both
    their tar and their table slice are skipped, and a pack whose
    `config.yaml` exists is left untouched.
    """
    if isinstance(config, Dataset):
        raise TypeError(
            "pack takes a dataset config, not a live dataset object, because workers "
            "rebuild the dataset from the config and the config is what gets frozen."
        )
    if not isinstance(config, (DatasetConfig, ConcatConfig, ChainedDatasetConfig)):
        raise TypeError(f"pack takes a DatasetConfig or ConcatConfig, got {type(config).__name__}")
    config = _normalise_config(config)
    out = anypath(str(out_path))
    fs = filesystem_from_path(out)

    if fs.exists(_join(out, CONFIG_FILE)):
        logger.info("Pack at %s is already complete; nothing to do.", out)
        return out

    ds, _ = dataset_from_config(config)
    num_rows = len(ds)
    if num_rows == 0:
        raise ValueError("Cannot pack an empty dataset")

    first = ds[0]
    mapping = getattr(config, "output_take_and_give", None) or {}
    audio_key = _resolve_audio_key(first, audio_key or mapping.get("audio"))
    sample_rate_key, fallback_sr = _resolve_sample_rate_key(
        first, sample_rate_key or mapping.get("sample_rate"), config
    )

    num_shards = math.ceil(num_rows / samples_per_shard)
    fs.makedirs(_join(out, MEDIA_DIR), exist_ok=True)
    fs.makedirs(_join(out, PARTS_DIR), exist_ok=True)

    jobs = []
    for shard in range(num_shards):
        if _shard_is_finished(fs, out, shard):
            logger.info("Shard %d already finished; skipping.", shard)
            continue
        jobs.append(
            _ShardJob(
                config=config,
                out=str(out),
                shard=shard,
                start=shard * samples_per_shard,
                stop=min((shard + 1) * samples_per_shard, num_rows),
                audio_format=audio_format,
                audio_key=audio_key,
                sample_rate_key=sample_rate_key,
                fallback_sample_rate=fallback_sr,
                on_error=on_error,
            )
        )

    if num_workers > 1 and jobs:
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_workers, initializer=_init_worker, initargs=(config,)) as pool:
            results = pool.map(_run_shard_job, jobs)
    else:
        results = [_pack_shard(ds, job) for job in jobs]

    _finalise(
        fs,
        out,
        ds,
        config,
        results,
        num_shards,
        samples_per_shard,
        audio_format,
        audio_key,
        sample_rate_key,
    )
    return out


# --- config handling -------------------------------------------------------


def _normalise_config(config: PackConfig | ChainedDatasetConfig) -> PackConfig:
    if isinstance(config, ChainedDatasetConfig):
        warnings.warn(
            "A ChainedDatasetConfig is packed as a concatenation: a pack is one table. "
            "Load several packs into a ChainedDataset instead if you need a chain.",
            UserWarning,
            stacklevel=3,
        )
        return ConcatConfig(datasets=config.datasets)
    return config


def _resolve_audio_key(item: dict[str, Any], audio_key: str | None) -> str:
    key = audio_key or "audio"
    if key not in item:
        raise ValueError(
            f"Audio key {key!r} not found in the first sample; keys are {sorted(item)}. "
            "Pass audio_key= to name the output key that holds the audio array."
        )
    return key


def _resolve_sample_rate_key(
    item: dict[str, Any], sample_rate_key: str | None, config: PackConfig
) -> tuple[str | None, int | None]:
    key = sample_rate_key or ("sample_rate" if "sample_rate" in item else None)
    if key is not None and key not in item:
        raise ValueError(f"Sample rate key {key!r} not found in the first sample")
    fallback = getattr(config, "sample_rate", None)
    if key is None and fallback is None:
        raise ValueError(
            "No sample rate available: the first sample has no 'sample_rate' key and the "
            "config sets none. Pass sample_rate_key= or set sample_rate in the config."
        )
    return key, fallback


# --- shard work ------------------------------------------------------------


def _join(base: AnyPathT, *parts: str) -> str:
    return str(base.joinpath(*parts))


def _shard_is_finished(fs: Any, out: AnyPathT, shard: int) -> bool:  # noqa: ANN401
    return fs.exists(_join(out, MEDIA_DIR, shard_name(shard))) and fs.exists(
        _join(out, PARTS_DIR, _part_name(shard))
    )


def _part_name(shard: int) -> str:
    return f"table-{shard:05d}.parquet"


def _errors_name(shard: int) -> str:
    return f"errors-{shard:05d}.parquet"


_WORKER_DS: Dataset | None = None


def _init_worker(config: PackConfig) -> None:
    global _WORKER_DS
    _WORKER_DS, _ = dataset_from_config(config)


def _run_shard_job(job: _ShardJob) -> _ShardResult:
    assert _WORKER_DS is not None, "worker not initialised"
    return _pack_shard(_WORKER_DS, job)


def _get_item_with_retries(ds: Dataset, idx: int) -> dict[str, Any]:
    for attempt in range(_RETRIES):
        try:
            return ds[idx]
        except (OSError, TimeoutError):
            if attempt == _RETRIES - 1:
                raise
            time.sleep(_RETRY_BACKOFF_S * 2**attempt)
    raise AssertionError("unreachable")


def _pack_shard(ds: Dataset, job: _ShardJob) -> _ShardResult:
    out = anypath(job.out)
    fs = filesystem_from_path(out)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    opaque: dict[str, str] = {}
    ext = job.audio_format

    logger.info("Packing shard %d: rows %d to %d", job.shard, job.start, job.stop)
    with ShardWriter(_join(out, MEDIA_DIR, shard_name(job.shard))) as writer:
        for idx in range(job.start, job.stop):
            try:
                item = _get_item_with_retries(ds, idx)
                row = _encode_row(item, job, opaque)
                audio = item[job.audio_key]
                sr = item[job.sample_rate_key] if job.sample_rate_key else job.fallback_sample_rate
                entry = writer.add(idx, encode_audio(audio, int(sr), job.audio_format), ext)
            except Exception as exc:
                if job.on_error == "raise":
                    raise
                logger.warning("Skipping row %d: %r", idx, exc)
                errors.append(
                    {
                        SOURCE_INDEX_COL: idx,
                        "error": "".join(traceback.format_exception_only(exc)).strip(),
                        "worker": os.getpid(),
                    }
                )
                continue
            row[SOURCE_INDEX_COL] = idx
            row[SHARD_COL] = job.shard
            row[OFFSET_COL] = entry.offset
            row[SIZE_COL] = entry.size
            row[SHA256_COL] = entry.sha256
            rows.append(row)

    if writer.size > _LARGE_SHARD_BYTES:
        logger.warning(
            "Shard %d is %.1f GB; consider a smaller samples_per_shard.",
            job.shard,
            writer.size / 1024**3,
        )

    _write_parquet(fs, _join(out, PARTS_DIR, _part_name(job.shard)), _rows_to_table(rows))
    if errors:
        _write_parquet(
            fs, _join(out, PARTS_DIR, _errors_name(job.shard)), pa.Table.from_pylist(errors)
        )

    return _ShardResult(
        shard=job.shard,
        size=writer.size,
        sha256=writer.sha256,
        num_rows=len(rows),
        num_skipped=len(errors),
        opaque_columns=opaque,
    )


def _encode_row(item: dict[str, Any], job: _ShardJob, opaque: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in item.items():
        if key == job.audio_key:
            continue
        encoded, kind = encode_value(value)
        if kind is not None:
            previous = opaque.setdefault(key, kind)
            if previous != kind:
                raise TypeError(f"Column {key!r} mixes opaque kinds {previous!r} and {kind!r}")
        row[key] = encoded
    return row


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
        }
    )


def _write_parquet(fs: Any, path: str, table: pa.Table) -> None:  # noqa: ANN401
    with fs.open(path, "wb") as f:
        pq.write_table(table, f)


def _read_parquet(fs: Any, path: str) -> pl.DataFrame:  # noqa: ANN401
    with fs.open(path, "rb") as f:
        return pl.read_parquet(f)


# --- finalisation ----------------------------------------------------------


def _finalise(
    fs: Any,  # noqa: ANN401
    out: AnyPathT,
    ds: Dataset,
    config: PackConfig,
    results: list[_ShardResult],
    num_shards: int,
    samples_per_shard: int,
    audio_format: AudioFormat,
    audio_key: str,
    sample_rate_key: str | None,
) -> None:
    parts = [_join(out, PARTS_DIR, _part_name(s)) for s in range(num_shards)]
    table = pl.concat([_read_parquet(fs, p) for p in parts], how="diagonal_relaxed")
    with fs.open(_join(out, TABLE_FILE), "wb") as f:
        table.write_parquet(f)

    error_parts = [
        _join(out, PARTS_DIR, _errors_name(s))
        for s in range(num_shards)
        if fs.exists(_join(out, PARTS_DIR, _errors_name(s)))
    ]
    num_skipped = 0
    if error_parts:
        errors = pl.concat([_read_parquet(fs, p) for p in error_parts])
        num_skipped = errors.height
        with fs.open(_join(out, ERRORS_FILE), "wb") as f:
            errors.write_parquet(f)

    opaque: dict[str, str] = {}
    for result in results:
        opaque.update(result.opaque_columns)
    # Shards finished by an earlier run carry no in-memory result; measure them.
    by_shard = {r.shard: r for r in results}
    shards = []
    for s in range(num_shards):
        path = _join(out, MEDIA_DIR, shard_name(s))
        if s in by_shard:
            shards.append(
                {"name": shard_name(s), "size": by_shard[s].size, "sha256": by_shard[s].sha256}
            )
        else:
            shards.append(
                {"name": shard_name(s), "size": fs.size(path), "sha256": _sha256_of(fs, path)}
            )
    if len(by_shard) < num_shards:
        opaque.update(_infer_opaque_columns(table, opaque))

    info = getattr(ds, "info", None)
    commit, dirty = _git_state()
    meta = {
        "source": config.model_dump(mode="json"),
        "name": getattr(info, "name", config.dataset_name),
        "version": getattr(info, "version", None),
        "split": getattr(config, "split", None),
        "sample_rate": getattr(config, "sample_rate", None),
        "audio_format": audio_format,
        "audio_key": audio_key,
        "sample_rate_key": sample_rate_key,
        "opaque_columns": opaque,
        "samples_per_shard": samples_per_shard,
        "num_rows": table.height,
        "num_skipped": num_skipped,
        "shards": shards,
        "alp_data_version": importlib.metadata.version("alp_data"),
        "alp_data_commit": commit,
        "alp_data_dirty": dirty,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with fs.open(_join(out, CONFIG_FILE), "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    fs.rm(_join(out, PARTS_DIR), recursive=True)
    logger.info("Packed %d rows into %d shards at %s", table.height, num_shards, out)


def _infer_opaque_columns(table: pl.DataFrame, known: dict[str, str]) -> dict[str, str]:
    """Recover opaque kinds for columns packed by an earlier, interrupted run.

    Parameters
    ----------
    table : pl.DataFrame
        The merged pack table.
    known : dict[str, str]
        Kinds already reported by shards packed in this run.

    Returns
    -------
    dict[str, str]
        Column name to opaque kind for columns not in `known`.
    """
    inferred: dict[str, str] = {}
    for name, dtype in table.schema.items():
        if name in known:
            continue
        if isinstance(dtype, pl.Struct) and {f.name for f in dtype.fields} == {
            "data",
            "dtype",
            "shape",
        }:
            inferred[name] = "ndarray"
        elif dtype == pl.Utf8 and table.height:
            sample = table[name].drop_nulls()
            if sample.len() and all("\t" in v or "\n" in v for v in sample.head(5).to_list()):
                inferred[name] = "dataframe_tsv"
    return inferred


def _sha256_of(fs: Any, path: str) -> str:  # noqa: ANN401
    import hashlib

    hasher = hashlib.sha256()
    with fs.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        return commit, bool(status.strip())
    except Exception:
        return None, None
