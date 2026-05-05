"""Migrate ESP datasets between cloud locations with validation.

Copies metadata files and audio files from a source location to a destination
bucket, preserving the full directory structure (only the bucket/scheme
changes). Audio originals are partially decoded and validated; pre-resampled
variants are copied without validation. Produces a per-version manifest of
deterministic content hashes.

Uses ``esp_data.io`` (fsspec-backed) for filesystem access, so the script works
for ``gs://``, ``s3://``, and ``r2://`` sources/destinations.

Usage
-----
    python scripts/migrate_datasets_v2.py \\
        --datasets barkley_canyon birdset xeno-canto \\
        --new-bucket my-new-bucket \\
        --new-protocol gs \\
        --report-path validation_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from migrate_registry import (  # noqa: E402
    DATASET_REGISTRY,
    derive_data_root,
    get_dataset_class,
    get_version_configs,
    resolve_audio_paths,
)

from esp_data.io import (  # noqa: E402
    PureGSPath,
    PureR2Path,
    anypath,
    filesystem_from_path,
    get_audio_info,
    read_audio,
)
from esp_data.io.paths import PureCloudPath  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("migrate_datasets_v2")

# ---------------------------------------------------------------------------
# Concurrency limits
# ---------------------------------------------------------------------------
MAX_DATASET_WORKERS = 5
MAX_ROW_WORKERS = 32
MAX_MANIFEST_WORKERS = 32

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
# Abs local path (/, /home, /tmp, /Users). Mutually exclusive with cloud URLs.
_LOCAL_PATH = re.compile(r"^(/home/|/tmp/|/Users/|/)")
_CLOUD_PREFIXES = re.compile(r"^(gs|s3|r2)://")
_CHUNK = 8 * 1024 * 1024
_VALIDATE_FRAMES = 16000 * 30  # up to 30 s @ 16 kHz; enough for std + decode check
_LOW_STD_THRESHOLD = 1e-4
_NEW_PROTOCOL_TO_CLS: dict[str, type[PureCloudPath]] = {
    "gs": PureGSPath,
    "s3": PureR2Path,
    "r2": PureR2Path,
}

# issue_types that should cause a non-zero exit (fix 9)
_FAIL_EXIT_ISSUE_TYPES = frozenset({"copy_failed", "empty_audio", "missing_blob"})

# Log level per issue_type so operators see problems as they happen rather
# than only in the end-of-run report.
_ISSUE_LOG_LEVEL: dict[str, int] = {
    "copy_failed": logging.ERROR,
    "empty_audio": logging.ERROR,
    "missing_blob": logging.ERROR,
    "low_std": logging.WARNING,
    "absolute_cloud_path_in_cell": logging.WARNING,
    "local_path_in_cell": logging.WARNING,
}


def _log_issue(issue: "ValidationIssue") -> "ValidationIssue":
    """Log a ValidationIssue at the level configured for its issue_type.

    Parameters
    ----------
    issue : ValidationIssue
        Issue to log and return.

    Returns
    -------
    ValidationIssue
        The same issue, unchanged — allows inline use at append sites.
    """
    level = _ISSUE_LOG_LEVEL.get(issue.issue_type, logging.WARNING)
    logger.log(
        level,
        "[%s/%s] row=%d col=%s %s: %s (%s)",
        issue.dataset_name,
        issue.split or "-",
        issue.row_index,
        issue.path_column or "-",
        issue.issue_type,
        issue.detail,
        issue.file_path or "-",
    )
    return issue


@dataclass
class ValidationIssue:
    dataset_name: str
    split: str
    row_index: int
    path_column: str
    file_path: str
    issue_type: str  # empty_audio | low_std | local_path_in_cell
    #                  absolute_cloud_path_in_cell | copy_failed | missing_blob
    detail: str


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def swap_root(src: str, new_bucket: str, new_protocol: str) -> str:
    """Replace scheme+bucket in ``src`` with ``new_protocol://new_bucket``.

    Parameters
    ----------
    src : str
        Source URL (e.g. ``gs://old-bucket/path/to/file``).
    new_bucket : str
        Destination bucket name (no scheme).
    new_protocol : str
        Destination scheme (``gs``, ``s3``, or ``r2``).

    Returns
    -------
    str
        Rewritten URL pointing at the same blob path under the new bucket.

    Raises
    ------
    ValueError
        If ``src`` does not start with a supported protocol scheme or
        ``new_protocol`` is not one of ``gs``/``s3``/``r2``.
    """
    p = anypath(src)
    if not isinstance(p, PureCloudPath):
        raise ValueError(f"Unsupported source path scheme: {src}")
    if new_protocol not in _NEW_PROTOCOL_TO_CLS:
        raise ValueError(f"Unsupported destination protocol: {new_protocol}")

    # Everything after "<scheme>://<bucket>" in the source, including the
    # leading '/'. Joining an absolute-within-bucket path onto a new cloud
    # path replaces the bucket's body wholesale.
    blob_path = str(p)[len(p.drive) :]
    new_cls = _NEW_PROTOCOL_TO_CLS[new_protocol]
    return str(new_cls(f"{new_cls.cloud_prefix}{new_bucket}") / blob_path)


def read_metadata(path: str) -> pl.DataFrame:
    """Load CSV/JSONL/Parquet metadata into a Polars DataFrame.

    Parameters
    ----------
    path : str
        Local or cloud path to the metadata file. Format is inferred from
        the suffix (``.csv``, ``.jsonl``, ``.parquet``).

    Returns
    -------
    pl.DataFrame
        The loaded table. CSV/JSONL are read with ``infer_schema_length=0``
        so all columns come in as strings; parquet preserves stored types.

    Raises
    ------
    ValueError
        If the file suffix is not one of the supported formats.
    """
    fs = filesystem_from_path(path)
    with fs.open(path, "rb") as f:
        raw = f.read()
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(io.BytesIO(raw), infer_schema_length=0)
    if suffix == ".jsonl":
        return pl.read_ndjson(io.BytesIO(raw), infer_schema_length=50)
    if suffix == ".parquet":
        return pl.read_parquet(io.BytesIO(raw))
    if suffix == ".txt":  # Case of SuperbStarling
        # Read tab separated text file
        return pl.read_csv(io.BytesIO(raw), infer_schema_length=0, separator="\t")
    raise ValueError(f"Unsupported metadata format: {suffix} ({path})")


def _is_retryable(exc: Exception) -> bool:
    """Return True if ``exc`` looks like a transient server/network error.

    Parameters
    ----------
    exc : Exception
        The exception raised by a cloud operation.

    Returns
    -------
    bool
        True when the stringified error contains a token associated with
        rate-limiting (``429``), server errors (``500/502/503/504``), or
        timeouts; False otherwise.
    """
    msg = str(exc).lower()
    return any(tok in msg for tok in ("429", "rate", "500", "502", "503", "504", "timeout"))


def copy_file(src: str, dst: str, max_retries: int = 5) -> str | None:
    """Copy ``src`` → ``dst``.

    Server-side copy when src/dst share a filesystem instance; otherwise streams.
    Skips when ``dst`` already exists. Retries transient errors with
    exponential backoff + jitter.

    Parameters
    ----------
    src : str
        Source path (local or cloud).
    dst : str
        Destination path (local or cloud).
    max_retries : int, optional
        Maximum number of attempts for transient errors (default: 5).

    Returns
    -------
    str or None
        ``None`` if the copy succeeded or was skipped because ``dst``
        already exists; otherwise a short string describing the final
        error.
    """
    src_fs = filesystem_from_path(src)
    dst_fs = filesystem_from_path(dst)

    # Use info() rather than exists(): gcsfs returns a synthetic
    # ``type=directory, size=0`` entry for blob names containing characters
    # like ``?`` even when no real object is present, so exists() can falsely
    # report True and cause us to skip a needed copy.
    try:
        info = dst_fs.info(dst)
        if info.get("type") == "file" and (info.get("size") or 0) > 0:
            return None
    except FileNotFoundError:
        pass
    except Exception as exc:  # non-fatal
        logger.debug("info(%s) failed: %s", dst, exc)

    same_fs = src_fs is dst_fs
    for attempt in range(max_retries):
        try:
            if same_fs:
                # cp_file: single-file copy, no glob expansion. fs.copy() treats
                # '[' / ']' / '*' in src as glob chars, which breaks filenames
                # like 'XC...[AudioTrimmer.com].wav'.
                src_fs.cp_file(src, dst)
            else:
                with src_fs.open(src, "rb") as fr, dst_fs.open(dst, "wb") as fw:
                    while True:
                        buf = fr.read(_CHUNK)
                        if not buf:
                            break
                        fw.write(buf)
            return None
        except Exception as exc:
            last = attempt == max_retries - 1
            if last or not _is_retryable(exc):
                return f"{type(exc).__name__}: {exc}"
            delay = min((2**attempt) + random.random(), 30.0)
            logger.warning(
                "Retry %d/%d copy %s -> %s after %.1fs: %s",
                attempt + 1,
                max_retries - 1,
                src,
                dst,
                delay,
                exc,
            )
            time.sleep(delay)
    # Loop always returns inside — no unreachable return (fix 4).


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_audio_partial(path: str) -> list[tuple[str, str]]:
    """Validate audio via partial decode.

    Reads up to ``_VALIDATE_FRAMES`` frames to keep bandwidth bounded for
    very large files (avoids full downloads for validation).

    Parameters
    ----------
    path : str
        Path to the audio file to validate (local or cloud).

    Returns
    -------
    list of tuple of (str, str)
        Zero or more ``(issue_type, detail)`` pairs. ``issue_type`` is one
        of:

        - ``"empty_audio"``: audio has zero samples or cannot be decoded.
        - ``"low_std"``: decoded signal has std ≤ ``_LOW_STD_THRESHOLD``
          (potentially silent).
    """
    issues: list[tuple[str, str]] = []
    try:
        info = get_audio_info(path)
    except Exception as exc:
        return [("empty_audio", f"Cannot decode audio: {exc}")]

    if info["num_frames"] == 0:
        return [("empty_audio", "Audio array has 0 samples")]

    to_read = min(info["num_frames"], _VALIDATE_FRAMES)
    try:
        audio, _ = read_audio(path, frames=to_read)
    except Exception as exc:
        return [("empty_audio", f"Cannot decode audio: {exc}")]

    if audio.size == 0:
        return [("empty_audio", "Decoded zero samples")]

    std = float(np.std(audio.astype(np.float32, copy=False)))
    if std <= _LOW_STD_THRESHOLD or np.isnan(std):
        issues.append(("low_std", f"std={std:.6e}"))
    return issues


def validate_row_cells(
    row: dict,
    dataset_name: str,
    split: str,
    row_idx: int,
    audio_columns: set[str],
) -> list[ValidationIssue]:
    """Check non-audio string cells for absolute cloud/local paths.

    Skips configured audio columns (those are expected to be relative).
    Uses ``elif`` to avoid double-counting ``/`` + ``/Users/`` matches.

    Parameters
    ----------
    row : dict
        A single row of the metadata DataFrame as a ``{column: value}``
        mapping.
    dataset_name : str
        Dataset name, recorded on any issues emitted.
    split : str
        Split name, recorded on any issues emitted.
    row_idx : int
        Row index within the split, recorded on any issues emitted.
    audio_columns : set of str
        Columns the registry treats as audio paths. These are skipped
        since they are expected to contain relative paths.

    Returns
    -------
    list of ValidationIssue
        One issue per offending cell, with ``issue_type`` either
        ``"absolute_cloud_path_in_cell"`` or ``"local_path_in_cell"``.
    """
    issues: list[ValidationIssue] = []
    for col, val in row.items():
        if col in audio_columns:
            continue
        if not isinstance(val, str):
            continue
        if _CLOUD_PREFIXES.search(val):
            issues.append(
                _log_issue(
                    ValidationIssue(
                        dataset_name=dataset_name,
                        split=split,
                        row_index=row_idx,
                        path_column=col,
                        file_path=val,
                        issue_type="absolute_cloud_path_in_cell",
                        detail=f"Cell contains cloud prefix in column '{col}'",
                    )
                )
            )
        elif _LOCAL_PATH.search(val):
            issues.append(
                _log_issue(
                    ValidationIssue(
                        dataset_name=dataset_name,
                        split=split,
                        row_index=row_idx,
                        path_column=col,
                        file_path=val,
                        issue_type="local_path_in_cell",
                        detail=f"Cell contains local path in column '{col}'",
                    )
                )
            )
    return issues


def _audio_column_names(registry_entry: dict) -> set[str]:
    """Collect column names the registry treats as audio paths.

    Parameters
    ----------
    registry_entry : dict
        A value from ``DATASET_REGISTRY`` describing originals / presampled
        / derived audio columns for a single dataset.

    Returns
    -------
    set of str
        Union of ``originals_column``, ``presampled_columns``, and the
        source ``column`` of each ``derived_paths`` entry.
    """
    cols: set[str] = set()
    if "originals_column" in registry_entry:
        cols.add(registry_entry["originals_column"])
    cols.update(registry_entry.get("presampled_columns", []))
    for derived in registry_entry.get("derived_paths", []):
        cols.add(derived["column"])
    return cols


# ---------------------------------------------------------------------------
# Hash / manifest
# ---------------------------------------------------------------------------
def _blob_hash(path: str) -> tuple[str, str | None]:
    """Return ``(path, hash_token)`` for a single blob.

    Prefers md5 (GCS blobs uploaded in a single chunk), falling back to
    crc32c, then etag, then ``size:<N>``. The returned hash always carries
    a ``kind:`` prefix so mixing types across blobs is detectable.

    Parameters
    ----------
    path : str
        Blob URL (local or cloud).

    Returns
    -------
    tuple of (str, str or None)
        The original ``path`` and a deterministic hash string of the form
        ``"<kind>:<value>"``, or ``None`` if the blob is missing or its
        metadata could not be retrieved.
    """
    fs = filesystem_from_path(path)
    try:
        info = fs.info(path)
    except FileNotFoundError:
        return path, None
    except Exception as exc:
        logger.warning("info(%s) failed: %s", path, exc)
        return path, None

    # gcsfs synthesises a ``type=directory`` entry for blob names containing
    # ``?`` etc. even when no real object exists; treat as missing.
    if info.get("type") == "directory" or info.get("storageClass") == "DIRECTORY":
        return path, None

    for key in ("md5Hash", "md5_hash", "crc32c", "etag", "ETag"):
        val = info.get(key)
        if val:
            return path, f"{key}:{val}"
    size = info.get("size")
    if size is None:
        size = info.get("Size")
    if size is not None:
        return path, f"size:{size}"
    return path, None


def _audio_info(path: str) -> dict | None:
    """Read audio header and return basic metadata.

    Thin wrapper around :func:`esp_data.io.get_audio_info` that remaps the
    keys to the names used in this script's manifest.

    Parameters
    ----------
    path : str
        Audio file URL (local or cloud).

    Returns
    -------
    dict or None
        Mapping with keys ``sample_rate`` (int), ``channels`` (int),
        ``frames`` (int), and ``duration_seconds`` (float). Returns
        ``None`` if the header could not be read.
    """
    try:
        info = get_audio_info(path)
    except Exception as exc:
        logger.warning("audio info(%s) failed: %s", path, exc)
        return None
    return {
        "sample_rate": int(info["sr"]),
        "channels": int(info["num_channels"]),
        "frames": int(info["num_frames"]),
        "duration_seconds": float(info["duration"]),
    }


def _audio_entry(path: str) -> tuple[str, str | None, dict | None]:
    """Worker returning hash and audio metadata for one audio file.

    Parameters
    ----------
    path : str
        Audio file URL (local or cloud).

    Returns
    -------
    tuple of (str, str or None, dict or None)
        The original ``path``, the hash token from :func:`_blob_hash`
        (or ``None`` if missing), and the audio metadata dict from
        :func:`_audio_info` (or ``None`` if the header could not be
        read).
    """
    _, h = _blob_hash(path)
    meta = _audio_info(path) if h is not None else None
    return path, h, meta


def compute_manifest(
    dataset_name: str,
    version: str,
    split_paths_abs: list[str],
    audio_paths_abs: list[str],
) -> tuple[dict, list[ValidationIssue]]:
    """Build a deterministic hash manifest for one dataset-version.

    Queries destination-side hashes for every copied split file and audio
    file, producing a combined ``dataset_hash`` plus per-file entries.
    For audio files, also records sample rate, channel count, frame
    count, and duration (seconds) read from the file header. Blobs that
    cannot be located are recorded as ``missing_blob``
    ValidationIssues and excluded from the hash.

    Parameters
    ----------
    dataset_name : str
        Dataset name, written to the manifest and to any issues raised.
    version : str
        Dataset version, written to the manifest.
    split_paths_abs : list of str
        Absolute destination URLs of the split/metadata files.
    audio_paths_abs : list of str
        Absolute destination URLs of the audio files.

    Returns
    -------
    tuple of (dict, list of ValidationIssue)
        The manifest payload (``dataset``, ``version``, ``dataset_hash``,
        ``split_hashes``, ``audio_hashes``, ``audio_stats``) and a list of
        issues for any blobs whose metadata could not be read. Each
        ``audio_hashes`` entry includes ``path``, ``hash``, and — when
        the header could be read — ``sample_rate``, ``channels``,
        ``frames``, and ``duration_seconds``. ``audio_stats`` summarises
        total/mean duration and sample-rate distribution.
    """
    issues: list[ValidationIssue] = []

    split_hashes: dict[str, str] = {}
    for bp in split_paths_abs:
        _, h = _blob_hash(bp)
        if h is None:
            issues.append(
                _log_issue(
                    ValidationIssue(
                        dataset_name=dataset_name,
                        split="",
                        row_index=-1,
                        path_column="split_path",
                        file_path=bp,
                        issue_type="missing_blob",
                        detail="split file missing in destination",
                    )
                )
            )
            continue
        split_hashes[bp] = h

    audio_hashes: list[dict] = []
    workers = max(1, min(MAX_MANIFEST_WORKERS, len(audio_paths_abs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_audio_entry, bp): bp for bp in audio_paths_abs}
        for fut in as_completed(futures):
            bp, h, meta = fut.result()
            if h is None:
                issues.append(
                    _log_issue(
                        ValidationIssue(
                            dataset_name=dataset_name,
                            split="",
                            row_index=-1,
                            path_column="audio_path",
                            file_path=bp,
                            issue_type="missing_blob",
                            detail="audio file missing in destination",
                        )
                    )
                )
                continue
            entry: dict = {"path": bp, "hash": h}
            if meta is not None:
                entry.update(meta)
            audio_hashes.append(entry)

    audio_hashes.sort(key=lambda x: x["path"])
    all_tokens = sorted(
        [f"{e['path']}:{e['hash']}" for e in audio_hashes]
        + [f"{k}:{v}" for k, v in split_hashes.items()]
    )
    dataset_hash = hashlib.sha256("".join(all_tokens).encode()).hexdigest()

    durations = [e["duration_seconds"] for e in audio_hashes if "duration_seconds" in e]
    sr_counts: dict[int, int] = {}
    for e in audio_hashes:
        sr = e.get("sample_rate")
        if sr is not None:
            sr_counts[sr] = sr_counts.get(sr, 0) + 1
    audio_stats = {
        "num_files": len(audio_hashes),
        "num_files_with_info": len(durations),
        "total_duration_seconds": float(sum(durations)) if durations else 0.0,
        "mean_duration_seconds": float(sum(durations) / len(durations)) if durations else 0.0,
        "min_duration_seconds": float(min(durations)) if durations else 0.0,
        "max_duration_seconds": float(max(durations)) if durations else 0.0,
        "sample_rate_counts": dict(sorted(sr_counts.items())),
    }

    manifest = {
        "dataset": dataset_name,
        "version": version,
        "dataset_hash": dataset_hash,
        "split_hashes": {k: {"path": k, "hash": v} for k, v in split_hashes.items()},
        "audio_hashes": audio_hashes,
        "audio_stats": audio_stats,
    }
    return manifest, issues


def upload_manifest(dst_data_root: str, manifest: dict) -> str:
    """Upload ``manifest.json`` under ``dst_data_root``.

    Parameters
    ----------
    dst_data_root : str
        Destination prefix URL (a trailing ``/`` is added if missing).
    manifest : dict
        Manifest payload produced by :func:`compute_manifest`.

    Returns
    -------
    str
        The full URL of the written ``manifest.json``.
    """
    if not dst_data_root.endswith("/"):
        dst_data_root = dst_data_root + "/"
    manifest_path = dst_data_root + "manifest.json"
    fs = filesystem_from_path(manifest_path)
    payload = json.dumps(manifest, indent=2).encode()
    with fs.open(manifest_path, "wb") as f:
        f.write(payload)
    logger.info("Uploaded manifest to %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Split-level migration
# ---------------------------------------------------------------------------
def migrate_split(
    dataset_name: str,
    split: str,
    split_path: str,
    new_bucket: str,
    new_protocol: str,
    data_root: str,
    registry_entry: dict,
    already_copied: set[str],
    already_validated: set[str],
    copied_lock: threading.Lock,
) -> tuple[list[ValidationIssue], list[str], str]:
    """Migrate one split: copy metadata, validate cells, copy audio rows.

    Parameters
    ----------
    dataset_name : str
        Dataset name, recorded on emitted issues.
    split : str
        Split name (``"train"``, ``"test"``, …), recorded on issues.
    split_path : str
        Source URL of the split/metadata file.
    new_bucket : str
        Destination bucket name (no scheme).
    new_protocol : str
        Destination scheme (``gs``, ``s3``, ``r2``).
    data_root : str
        Source prefix under which audio paths in this split are rooted.
    registry_entry : dict
        ``DATASET_REGISTRY`` entry describing the audio columns.
    already_copied : set of str
        Shared set of source paths already successfully copied within the
        current dataset-version. Mutated: successful copies are added.
    already_validated : set of str
        Shared set of source originals already validated within the
        current dataset-version. Mutated: each originals path is added
        exactly once.
    copied_lock : threading.Lock
        Lock guarding ``already_copied`` and ``already_validated``.

    Returns
    -------
    tuple of (list of ValidationIssue, list of str, str)
        Issues collected for this split, destination URLs of successfully
        copied audio files, and the destination URL of the split file.
    """
    issues: list[ValidationIssue] = []
    copied_audio_dst: list[str] = []
    split_dst = swap_root(split_path, new_bucket, new_protocol)

    # ── 1. Copy metadata ──
    err = copy_file(split_path, split_dst)
    if err:
        issues.append(
            _log_issue(
                ValidationIssue(
                    dataset_name=dataset_name,
                    split=split,
                    row_index=-1,
                    path_column="split_path",
                    file_path=split_path,
                    issue_type="copy_failed",
                    detail=err,
                )
            )
        )
        return issues, copied_audio_dst, split_dst

    # ── 2. Read metadata ──
    try:
        df = read_metadata(split_path)
    except Exception as exc:
        issues.append(
            _log_issue(
                ValidationIssue(
                    dataset_name=dataset_name,
                    split=split,
                    row_index=-1,
                    path_column="split_path",
                    file_path=split_path,
                    issue_type="copy_failed",
                    detail=f"read failure: {exc}",
                )
            )
        )
        return issues, copied_audio_dst, split_dst

    logger.info("  [%s] split=%s rows=%d", dataset_name, split, len(df))
    audio_cols = _audio_column_names(registry_entry)

    def _process_row(row_idx: int) -> tuple[list[ValidationIssue], list[str]]:
        row = df.row(row_idx, named=True)
        originals_src, all_audio_src = resolve_audio_paths(row, registry_entry, data_root)

        row_issues = validate_row_cells(row, dataset_name, split, row_idx, audio_cols)
        copied: list[str] = []

        # Claim validation slot atomically so shared originals aren't revalidated.
        should_validate = False
        if originals_src:
            with copied_lock:
                if originals_src not in already_validated:
                    already_validated.add(originals_src)
                    should_validate = True

        if should_validate:
            for issue_type, detail in validate_audio_partial(originals_src):
                row_issues.append(
                    _log_issue(
                        ValidationIssue(
                            dataset_name=dataset_name,
                            split=split,
                            row_index=row_idx,
                            path_column=registry_entry["originals_column"],
                            file_path=originals_src,
                            issue_type=issue_type,
                            detail=detail,
                        )
                    )
                )

        for col, src_path in all_audio_src:
            # Claim copy slot. If another thread is actively copying this src
            # (in_flight) we skip. If a previous copy failed, the path will
            # NOT be in already_copied (we add on success, fix 7), so a later
            # row can retry.
            with copied_lock:
                if src_path in already_copied:
                    continue

            dst = swap_root(src_path, new_bucket, new_protocol)
            err = copy_file(src_path, dst)
            if err:
                row_issues.append(
                    _log_issue(
                        ValidationIssue(
                            dataset_name=dataset_name,
                            split=split,
                            row_index=row_idx,
                            path_column=col,
                            file_path=src_path,
                            issue_type="copy_failed",
                            detail=err,
                        )
                    )
                )
            else:
                with copied_lock:
                    already_copied.add(src_path)
                copied.append(dst)

        return row_issues, copied

    with ThreadPoolExecutor(max_workers=MAX_ROW_WORKERS) as pool:
        futures = {pool.submit(_process_row, i): i for i in range(len(df))}
        for fut in as_completed(futures):
            row_idx = futures[fut]
            try:
                row_issues, row_copied = fut.result()
                issues.extend(row_issues)
                copied_audio_dst.extend(row_copied)
            except Exception as exc:
                issues.append(
                    _log_issue(
                        ValidationIssue(
                            dataset_name=dataset_name,
                            split=split,
                            row_index=row_idx,
                            path_column="",
                            file_path="",
                            issue_type="copy_failed",
                            detail=f"Unhandled exception: {exc}",
                        )
                    )
                )

    return issues, copied_audio_dst, split_dst


# ---------------------------------------------------------------------------
# Dataset-level migration
# ---------------------------------------------------------------------------
def migrate_dataset(
    dataset_name: str,
    new_bucket: str,
    new_protocol: str,
) -> tuple[list[ValidationIssue], list[str]]:
    """Migrate all versions and splits of a dataset.

    For each version of ``dataset_name``, migrates every split, then
    computes and uploads a per-version manifest. The ``already_copied`` /
    ``already_validated`` sets are reset per version so that blobs shared
    across versions still appear in each version's manifest.

    Parameters
    ----------
    dataset_name : str
        Registered ESP dataset name (must appear in ``DATASET_REGISTRY``).
    new_bucket : str
        Destination bucket name (no scheme).
    new_protocol : str
        Destination scheme (``gs``, ``s3``, or ``r2``).

    Returns
    -------
    tuple of (list of ValidationIssue, list of str)
        All issues raised across versions/splits, and the URLs of the
        manifest files written. If the dataset is not in
        ``DATASET_REGISTRY``, both lists are empty.
    """
    all_issues: list[ValidationIssue] = []
    manifest_paths: list[str] = []

    if dataset_name not in DATASET_REGISTRY:
        logger.error("Dataset '%s' not in DATASET_REGISTRY, skipping", dataset_name)
        return all_issues, manifest_paths

    registry_entry = DATASET_REGISTRY[dataset_name]
    dataset_class = get_dataset_class(dataset_name)
    version_configs = get_version_configs(dataset_class, registry_entry)

    for version, vcfg in version_configs.items():
        logger.info("[%s] version=%s", dataset_name, version)
        split_paths = vcfg["split_paths"]

        # Fix 1: reset per version so shared blobs still appear in each
        # version's manifest.
        already_copied: set[str] = set()
        already_validated: set[str] = set()
        copied_lock = threading.Lock()

        all_split_dst: list[str] = []
        all_audio_dst: list[str] = []

        for split, split_path in split_paths.items():
            data_root = derive_data_root(split_path, vcfg, registry_entry)
            split_issues, audio_dst, split_dst = migrate_split(
                dataset_name=dataset_name,
                split=split,
                split_path=split_path,
                new_bucket=new_bucket,
                new_protocol=new_protocol,
                data_root=data_root,
                registry_entry=registry_entry,
                already_copied=already_copied,
                already_validated=already_validated,
                copied_lock=copied_lock,
            )
            all_issues.extend(split_issues)
            all_audio_dst.extend(audio_dst)
            all_split_dst.append(split_dst)

        # Manifest — uploaded to swapped-bucket data_root
        src_data_root = derive_data_root(
            next(iter(split_paths.values())),
            vcfg,
            registry_entry,
        )
        dst_data_root = swap_root(src_data_root, new_bucket, new_protocol)
        try:
            manifest, manifest_issues = compute_manifest(
                dataset_name=dataset_name,
                version=version,
                split_paths_abs=all_split_dst,
                audio_paths_abs=all_audio_dst,
            )
            all_issues.extend(manifest_issues)
            mpath = upload_manifest(dst_data_root, manifest)
            manifest_paths.append(mpath)
        except Exception as exc:
            logger.error("[%s] v%s manifest failed: %s", dataset_name, version, exc)

    return all_issues, manifest_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Command-line entry point.

    Parses CLI arguments, migrates each dataset in parallel, writes a
    combined JSON validation report, and exits with a non-zero status if
    any hard failures (``copy_failed``, ``empty_audio``, ``missing_blob``)
    occurred. ``low_std`` warnings do not affect the exit status.

    Parameters
    ----------
    None
        Arguments are read from ``sys.argv`` via argparse.

    Raises
    ------
    SystemExit
        Exit status ``1`` if any hard failures were recorded.
    """  # noqa: DOC502
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names to migrate (e.g. 'birdset', 'xeno-canto').",
    )
    parser.add_argument(
        "--new-bucket",
        required=True,
        help="Destination bucket name (no scheme).",
    )
    parser.add_argument(
        "--new-protocol",
        default="gs",
        choices=("gs", "r2", "s3"),
        help="Destination scheme (default: gs).",
    )
    parser.add_argument(
        "--report-path",
        required=True,
        help="Path to write the JSON validation report.",
    )
    args = parser.parse_args()

    new_bucket = (
        args.new_bucket.removeprefix("gs://").removeprefix("r2://").removeprefix("s3://").strip("/")
    )

    logger.info(
        "Migrating %d dataset(s) -> %s://%s",
        len(args.datasets),
        args.new_protocol,
        new_bucket,
    )

    all_issues: list[ValidationIssue] = []
    all_manifests: list[str] = []

    with ThreadPoolExecutor(max_workers=MAX_DATASET_WORKERS) as pool:
        futures = {
            pool.submit(migrate_dataset, name, new_bucket, args.new_protocol): name
            for name in args.datasets
        }
        for fut in as_completed(futures):
            ds_name = futures[fut]
            try:
                ds_issues, ds_manifests = fut.result()
                all_issues.extend(ds_issues)
                all_manifests.extend(ds_manifests)
                logger.info(
                    "[%s] done — %d issues, %d manifest(s)",
                    ds_name,
                    len(ds_issues),
                    len(ds_manifests),
                )
            except Exception as exc:
                logger.error("[%s] failed: %s", ds_name, exc, exc_info=True)

    report = {
        "total_issues": len(all_issues),
        "manifests": all_manifests,
        "issues": [asdict(i) for i in all_issues],
    }
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Fix 9: only exit non-zero on hard failures, not low_std warnings.
    hard_failures = [i for i in all_issues if i.issue_type in _FAIL_EXIT_ISSUE_TYPES]
    logger.info(
        "Done. %d total issues (%d hard failures). Report: %s",
        len(all_issues),
        len(hard_failures),
        args.report_path,
    )
    if hard_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
