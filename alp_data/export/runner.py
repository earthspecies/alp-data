"""The loop every export format shares.

An export builds the dataset from its config, calls `ds[i]` for every row in
a shard, encodes the audio and the other values, and hands each row to a
sink. The sink decides how rows are written (tar shards plus a table for a
pack, parquet with embedded bytes for Hugging Face). This module owns
everything else: config validation, shard planning, worker processes,
retries, error records, resume, and provenance.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
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
from typing import Any, Literal, Protocol

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from pydantic import BaseModel

from alp_data.dataset import (
    ChainedDatasetConfig,
    ConcatConfig,
    Dataset,
    DatasetConfig,
    dataset_from_config,
)
from alp_data.export.columns import SOURCE_INDEX_COL
from alp_data.export.serializers import AudioFormat, encode_audio_lossless_if_needed, encode_value
from alp_data.io import AnyPathT, anypath, filesystem_from_path

logger = logging.getLogger("alp_data")

ExportConfig = DatasetConfig | ConcatConfig
OnError = Literal["raise", "skip"]

PARTS_DIR = "parts"
PART_METADATA_KEY = b"alp_data_export"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]

_RETRIES = 3
_RETRY_BACKOFF_S = 0.5


@dataclass
class ExportJob:
    """One shard's worth of work, sent to a worker.

    Attributes
    ----------
    config : ExportConfig
        Config the worker rebuilds the dataset from.
    out : str
        Export root.
    shard, num_shards : int
        This shard's number and the total.
    start, stop : int
        Half-open range of source rows in this shard.
    audio_format : AudioFormat
        Blob encoding.
    audio_key, sample_rate_key : str | None
        Output keys holding the audio and its sample rate.
    fallback_sample_rate : int | None
        Used when `sample_rate_key` is None.
    declared_sample_rate : int | None
        The rate the export advertises: the config's, else the first row's.
    on_error : OnError
        Whether a failing row stops the export or is skipped and recorded.
    sink : ExportSink
        The writer for this export format.
    """

    config: ExportConfig
    out: str
    shard: int
    num_shards: int
    start: int
    stop: int
    audio_format: AudioFormat
    audio_key: str
    sample_rate_key: str | None
    fallback_sample_rate: int | None
    declared_sample_rate: int | None
    on_error: OnError
    sink: ExportSink


@dataclass
class ShardResult:
    """What a worker reports back for one shard."""

    shard: int
    num_rows: int
    num_skipped: int
    opaque_columns: dict[str, str] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinaliseContext:
    """Everything a sink needs to write its final files."""

    ds: Dataset
    config: ExportConfig
    results: list[ShardResult]
    num_shards: int
    samples_per_shard: int
    audio_format: AudioFormat
    audio_key: str
    sample_rate_key: str | None
    declared_sample_rate: int | None
    num_skipped: int

    def provenance(
        self, opaque_columns: dict[str, str], num_rows: int, num_lossless_fallback: int = 0
    ) -> dict[str, Any]:
        """Build the provenance block every export format records.

        Parameters
        ----------
        opaque_columns : dict[str, str]
            Column name to opaque kind.
        num_rows : int
            Rows actually written.
        num_lossless_fallback : int
            Rows stored as float32 WAV because FLAC would have clipped them.

        Returns
        -------
        dict[str, Any]
            YAML-safe provenance fields.
        """
        info = getattr(self.ds, "info", None)
        commit, dirty = git_state(_PACKAGE_ROOT)
        return {
            # serialize_as_any keeps the fields of registered custom configs
            # nested inside a ConcatConfig, which pydantic would otherwise
            # serialise as the base DatasetConfig.
            "source": self.config.model_dump(mode="json", serialize_as_any=True),
            "name": getattr(info, "name", self.config.dataset_name),
            "version": getattr(info, "version", None),
            "split": getattr(self.config, "split", None),
            "sample_rate": self.declared_sample_rate,
            "audio_format": self.audio_format,
            "audio_key": self.audio_key,
            "sample_rate_key": self.sample_rate_key,
            "opaque_columns": opaque_columns,
            "samples_per_shard": self.samples_per_shard,
            "num_rows": num_rows,
            "num_skipped": self.num_skipped,
            "num_rows_lossless_fallback": num_lossless_fallback,
            "alp_data_version": importlib.metadata.version("alp_data"),
            "alp_data_commit": commit,
            "alp_data_dirty": dirty,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }


class ShardOutput(Protocol):
    """An open shard, as handed out by a sink."""

    def add(self, source_index: int, row: dict[str, Any], audio: bytes, ext: str) -> None:
        """Write one row whose values are already encoded; `ext` is the blob's format."""

    def close(self, opaque_columns: dict[str, str], stats: dict[str, Any]) -> dict[str, Any]:
        """Finish the shard and return sink-specific info about it.

        The sink stores `opaque_columns` and `stats` with the shard so that a
        later run that resumes past this shard can recover them without
        guessing. `stats` is merged into the returned info.
        """

    def abort(self) -> None:
        """Discard the shard after a failure."""


class ExportSink(Protocol):
    """What an export format has to provide.

    Attributes
    ----------
    errors_file : str
        Name of the file skipped rows are recorded in, under the export root.
        A `.jsonl` name is written as newline-delimited JSON, anything else as
        parquet.
    """

    errors_file: str

    def is_complete(self, fs: AbstractFileSystem, out: AnyPathT) -> bool:
        """Whether a finished export already exists at `out`."""

    def prepare(self, fs: AbstractFileSystem, out: AnyPathT) -> None:
        """Create whatever directories the format needs."""

    def shard_is_finished(self, fs: AbstractFileSystem, out: AnyPathT, job: ExportJob) -> bool:
        """Whether the shard described by `job` was completed by an earlier run."""

    def open_shard(self, fs: AbstractFileSystem, out: AnyPathT, job: ExportJob) -> ShardOutput:
        """Start writing shard `job.shard`."""

    def finalise(self, fs: AbstractFileSystem, out: AnyPathT, ctx: FinaliseContext) -> None:
        """Write the final files once every shard is done."""


def join(base: AnyPathT, *parts: str) -> str:
    """Join path parts under `base` as a string for fsspec.

    Parameters
    ----------
    base : AnyPathT
        Root path.
    *parts : str
        Components to append.

    Returns
    -------
    str
        The joined path.
    """
    return str(base.joinpath(*parts))


def run_export(
    config: ExportConfig | ChainedDatasetConfig,
    out_path: str | AnyPathT,
    sink: ExportSink,
    *,
    samples_per_shard: int,
    audio_format: AudioFormat,
    num_workers: int,
    on_error: OnError,
    audio_key: str | None,
    sample_rate_key: str | None,
) -> AnyPathT:
    """Run an export end to end.

    Parameters
    ----------
    config : DatasetConfig | ConcatConfig | ChainedDatasetConfig
        What to export. A chained config is exported as a concatenation,
        with a warning.
    out_path : str | AnyPathT
        Destination directory, any `anypath` target.
    sink : ExportSink
        The format writer.
    samples_per_shard : int
        Rows per shard.
    audio_format : AudioFormat
        Blob encoding.
    num_workers : int
        Spawned worker processes; `1` runs in the calling process.
    on_error : OnError
        `"raise"` or `"skip"`.
    audio_key, sample_rate_key : str | None
        Output keys; resolved from the config's `output_take_and_give` when None.

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
    """
    if isinstance(config, Dataset) or not isinstance(config, BaseModel):
        raise TypeError(
            "Exports take a dataset config, not a live dataset object, because workers "
            "rebuild the dataset from the config and the config is what gets frozen."
        )
    if not isinstance(config, (DatasetConfig, ConcatConfig, ChainedDatasetConfig)):
        raise TypeError(f"Expected a DatasetConfig or ConcatConfig, got {type(config).__name__}")
    config = _normalise_config(config)
    out = anypath(str(out_path))
    fs = filesystem_from_path(out)

    if sink.is_complete(fs, out):
        logger.info("Export at %s is already complete; nothing to do.", out)
        return out

    ds, _ = dataset_from_config(config)
    num_rows = len(ds)
    if num_rows == 0:
        raise ValueError("Cannot export an empty dataset")

    first = _first_readable_item(ds, on_error)
    mapping = getattr(config, "output_take_and_give", None) or {}
    audio_key = _resolve_audio_key(first, audio_key or mapping.get("audio"))
    sample_rate_key, fallback_sr = _resolve_sample_rate_key(
        first, sample_rate_key or mapping.get("sample_rate"), config
    )
    declared_sr = getattr(config, "sample_rate", None)
    if declared_sr is None:
        declared_sr = int(first[sample_rate_key]) if sample_rate_key else fallback_sr

    num_shards = math.ceil(num_rows / samples_per_shard)
    sink.prepare(fs, out)
    fs.makedirs(join(out, PARTS_DIR), exist_ok=True)

    jobs = []
    for shard in range(num_shards):
        job = ExportJob(
            config=config,
            out=str(out),
            shard=shard,
            num_shards=num_shards,
            start=shard * samples_per_shard,
            stop=min((shard + 1) * samples_per_shard, num_rows),
            audio_format=audio_format,
            audio_key=audio_key,
            sample_rate_key=sample_rate_key,
            fallback_sample_rate=fallback_sr,
            declared_sample_rate=declared_sr,
            on_error=on_error,
            sink=sink,
        )
        if sink.shard_is_finished(fs, out, job):
            logger.info("Shard %d already finished; skipping.", shard)
            continue
        jobs.append(job)

    if num_workers > 1 and jobs:
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_workers, initializer=_init_worker, initargs=(config,)) as pool:
            results = pool.map(_run_job, jobs)
    else:
        results = [_export_shard(ds, job) for job in jobs]

    num_skipped = _merge_errors(fs, out, sink.errors_file, num_shards)
    sink.finalise(
        fs,
        out,
        FinaliseContext(
            ds=ds,
            config=config,
            results=results,
            num_shards=num_shards,
            samples_per_shard=samples_per_shard,
            audio_format=audio_format,
            audio_key=audio_key,
            sample_rate_key=sample_rate_key,
            declared_sample_rate=declared_sr,
            num_skipped=num_skipped,
        ),
    )
    # Object stores have no empty directories: with nothing written under
    # parts/ the prefix does not exist and rm would raise.
    if fs.exists(join(out, PARTS_DIR)):
        fs.rm(join(out, PARTS_DIR), recursive=True)
    logger.info("Exported %d shards to %s", num_shards, out)
    return out


# --- config handling -------------------------------------------------------


def _normalise_config(config: ExportConfig | ChainedDatasetConfig) -> ExportConfig:
    if isinstance(config, ChainedDatasetConfig):
        warnings.warn(
            "A ChainedDatasetConfig is exported as a concatenation: an export is one table. "
            "Load several exports into a ChainedDataset instead if you need a chain.",
            UserWarning,
            stacklevel=4,
        )
        return ConcatConfig(datasets=config.datasets)
    return config


def _first_readable_item(ds: Dataset, on_error: OnError) -> dict[str, Any]:
    """Return the first item that loads, so a corrupt row 0 does not abort a skip run.

    Parameters
    ----------
    ds : Dataset
        The dataset being exported.
    on_error : OnError
        With `"raise"`, the first failure propagates.

    Returns
    -------
    dict[str, Any]
        The first item that could be loaded.

    Raises
    ------
    ValueError
        If no row loads.
    """
    last: Exception | None = None
    for idx in range(len(ds)):
        try:
            return ds[idx]
        except Exception as exc:
            if on_error == "raise":
                raise
            last = exc
    raise ValueError("No row of the dataset could be loaded") from last


def _resolve_audio_key(item: dict[str, Any], audio_key: str | None) -> str:
    key = audio_key or "audio"
    if key not in item:
        raise ValueError(
            f"Audio key {key!r} not found in the first sample; keys are {sorted(item)}. "
            "Pass audio_key= to name the output key that holds the audio array."
        )
    return key


def _resolve_sample_rate_key(
    item: dict[str, Any], sample_rate_key: str | None, config: ExportConfig
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


_WORKER_DS: Dataset | None = None


def _init_worker(config: ExportConfig) -> None:
    global _WORKER_DS
    _WORKER_DS, _ = dataset_from_config(config)


def _run_job(job: ExportJob) -> ShardResult:
    assert _WORKER_DS is not None, "worker not initialised"
    return _export_shard(_WORKER_DS, job)


def _get_item_with_retries(ds: Dataset, idx: int) -> dict[str, Any]:
    for attempt in range(_RETRIES):
        try:
            return ds[idx]
        except (OSError, TimeoutError):
            if attempt == _RETRIES - 1:
                raise
            time.sleep(_RETRY_BACKOFF_S * 2**attempt)
    raise AssertionError("unreachable")


def _errors_name(shard: int) -> str:
    return f"errors-{shard:05d}.parquet"


def _export_shard(ds: Dataset, job: ExportJob) -> ShardResult:
    out = anypath(job.out)
    fs = filesystem_from_path(out)
    errors: list[dict[str, Any]] = []
    opaque: dict[str, str] = {}
    num_rows = 0
    num_fallback = 0

    logger.info("Exporting shard %d: rows %d to %d", job.shard, job.start, job.stop)
    output = job.sink.open_shard(fs, out, job)
    try:
        for idx in range(job.start, job.stop):
            try:
                item = _get_item_with_retries(ds, idx)
                sr = item[job.sample_rate_key] if job.sample_rate_key else job.fallback_sample_rate
                audio, ext = encode_audio_lossless_if_needed(
                    item[job.audio_key], int(sr), job.audio_format
                )
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
            # A value that cannot be encoded is a schema problem with the
            # dataset, not a bad row: it is never skipped.
            row = _encode_row(item, job.audio_key, opaque)
            output.add(idx, row, audio, ext)
            num_rows += 1
            num_fallback += ext != job.audio_format
    except BaseException:
        output.abort()
        raise
    info = output.close(opaque, {"num_lossless_fallback": num_fallback})

    if errors:
        with fs.open(join(out, PARTS_DIR, _errors_name(job.shard)), "wb") as f:
            pq.write_table(pa.Table.from_pylist(errors), f)

    return ShardResult(
        shard=job.shard,
        num_rows=num_rows,
        num_skipped=len(errors),
        opaque_columns=opaque,
        info=info,
    )


def _encode_row(item: dict[str, Any], audio_key: str, opaque: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in item.items():
        if key == audio_key:
            continue
        encoded, kind = encode_value(value)
        if kind is not None:
            previous = opaque.setdefault(key, kind)
            if previous != kind:
                raise TypeError(f"Column {key!r} mixes opaque kinds {previous!r} and {kind!r}")
        row[key] = encoded
    return row


# --- finalisation helpers --------------------------------------------------


def _merge_errors(fs: AbstractFileSystem, out: AnyPathT, errors_file: str, num_shards: int) -> int:
    parts = [
        join(out, PARTS_DIR, _errors_name(s))
        for s in range(num_shards)
        if fs.exists(join(out, PARTS_DIR, _errors_name(s)))
    ]
    if not parts:
        return 0
    errors = pl.concat([read_parquet(fs, p) for p in parts])
    with fs.open(join(out, errors_file), "wb") as f:
        if errors_file.endswith(".jsonl"):
            errors.write_ndjson(f)
        else:
            errors.write_parquet(f)
    return errors.height


def read_parquet(fs: AbstractFileSystem, path: str) -> pl.DataFrame:
    """Read a parquet file through fsspec into polars.

    Parameters
    ----------
    fs : AbstractFileSystem
        Filesystem for `path`.
    path : str
        File path.

    Returns
    -------
    pl.DataFrame
        The table.
    """
    with fs.open(path, "rb") as f:
        return pl.read_parquet(f)


def write_parquet(fs: AbstractFileSystem, path: str, table: pa.Table) -> None:
    """Write an arrow table as parquet through fsspec.

    Parameters
    ----------
    fs : AbstractFileSystem
        Filesystem for `path`.
    path : str
        File path.
    table : pa.Table
        The table.
    """
    with fs.open(path, "wb") as f:
        pq.write_table(table, f)


def git_state(repo: Path) -> tuple[str | None, bool | None]:
    """Commit and dirty flag of an alp_data source checkout, best effort.

    Parameters
    ----------
    repo : Path
        Directory expected to be the alp_data repository root.

    Returns
    -------
    tuple[str | None, bool | None]
        `(commit, dirty)`, or `(None, None)` when `repo` is not an alp_data
        checkout (for example a wheel install under `site-packages`, whose
        enclosing git repository, if any, would be the user's own) or git
        is unavailable.
    """
    if not (repo / "pyproject.toml").is_file() or not (repo / "alp_data").is_dir():
        return None, None
    try:
        top = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        if Path(top).resolve() != repo.resolve():
            return None, None
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


def shard_metadata(opaque_columns: dict[str, str], info: dict[str, Any]) -> dict[bytes, bytes]:
    """Schema metadata a sink attaches to a shard's parquet file.

    Parameters
    ----------
    opaque_columns : dict[str, str]
        Column name to opaque kind seen in this shard.
    info : dict[str, Any]
        Sink-specific shard info, such as name, size, and digest.

    Returns
    -------
    dict[bytes, bytes]
        A single-entry mapping for `replace_schema_metadata`.
    """
    import json

    return {
        PART_METADATA_KEY: json.dumps({"opaque_columns": opaque_columns, "info": info}).encode()
    }


def read_shard_metadata(fs: AbstractFileSystem, path: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Inverse of `shard_metadata`, read from a parquet file's footer.

    Parameters
    ----------
    fs : AbstractFileSystem
        Filesystem for `path`.
    path : str
        Parquet file written with `shard_metadata` attached.

    Returns
    -------
    tuple[dict[str, str], dict[str, Any]]
        The opaque columns and the shard info.

    Raises
    ------
    KeyError
        If the file carries no export metadata.
    """
    with fs.open(path, "rb") as f:
        meta = pq.read_metadata(f).metadata
    if not meta or PART_METADATA_KEY not in meta:
        raise KeyError(f"{path} carries no {PART_METADATA_KEY!r} metadata")
    payload = json.loads(meta[PART_METADATA_KEY])
    return payload["opaque_columns"], payload["info"]
