"""Convert unreadable iNaturalist originals to .wav and publish cleaned split CSVs.

Some ``originals_path`` files cannot be loaded by ``alp_data.io.read_audio``
whenever the requested sample rate is not one of the pre-resampled rates
(16 kHz / 32 kHz), so the dataset class falls back to the originals (issue
#345):

- ``.m4a`` (AAC), which soundfile cannot decode from a stream;
- ``.mp3`` that libsndfile/libmpg123 cannot parse, or that were deleted after
  an earlier conversion while the CSV still points at them;
- broken or missing ``.wav``.

v0.1.0's CSVs were patched earlier for m4a and mp3 (by ``inat_mp3_to_wav.py``,
now removed; see git history); v0.2.0's never were. This script replaces it.

The script reads and rewrites the split CSVs of *both* dataset versions. Its
*targets* are every ``.m4a`` in either version (v0.1.0 has none; ``scan``
raises if it finds any) plus every ``originals_path`` listed in the failure
CSVs written by ``scripts/iterate_inaturalist.py`` (``--extra-targets``).

Changes made to each version (see ``VERSIONS`` for the input/output CSV names):

v0.1.0 (``*_20260201_v3.csv`` -> ``*_20260201_v4.csv``)
    - ``relative_path`` dropped (the v3 CSVs have no ``gcs_path``), so the
      columns match v0.2.0 minus its 7 extra GBIF columns.
    - 1 row repointed to a newly converted ``.wav``:
      ``audio/113/inat_113534110.mp3``, an mp3 that libsndfile cannot decode
      and that ``inat_mp3_to_wav.py`` missed.
    - No rows dropped; all other cells unchanged.
v0.2.0 (``*_20260616_v1.csv`` -> ``*_20260616_v2.csv``)
    - ``gcs_path`` and ``relative_path`` dropped.
    - 132,008 rows in ``all`` repointed to ``.wav`` (126,004 m4a + 6,004
      mp3). Of the unique files behind them, 41,998 were converted here and
      89,358 already had a ``.wav`` from the earlier v0.1.0 conversion.
    - 88 rows in ``all`` dropped (13 missing m4a, 30 missing mp3, 45 broken
      or missing wav).
    - All other cells unchanged.

Both versions share one audio tree and CSV directory, ``DATA_ROOT``
(``inaturalist/v0.1.0/raw``); v0.2.0 is a newer set of split CSVs over the
same audio, so there is no separate v0.2.0 bucket path.

The work is split into stages that share a work directory, so each can be
inspected before the next runs and nothing is written to GCS until ``upload``:

``scan``
    Download all split CSVs of both versions into ``<work-dir>/csv``, collect
    the targets and check (on GCS) whether each target and its sibling
    ``.wav`` exist. Status per target: ``broken_wav`` (a ``.wav`` target;
    nothing to convert from, rows are dropped), ``wav_exists`` (only the CSV
    path changes), ``to_convert`` or ``missing`` (rows are dropped). Writes
    ``scan.parquet``.
``convert``
    For every ``to_convert`` target, download it to
    ``<work-dir>/audio/<originals_path>``, decode it with ``librosa.load`` on
    audioread's ffmpeg backend (pinned, so the result does not depend on the
    OS) and write a PCM_16 ``.wav`` sibling at the native sample rate and
    channel count. Writes ``convert.parquet``.
``verify``
    Re-decode both local files and require equal sample rate, equal shape and
    bit-exact samples (audioread decodes to int16, so a PCM_16 round trip is
    lossless). Optionally cross-check a fraction of files against the existing
    ``16khz_path`` resample by correlation. Writes ``verify.parquet``.
``upload``
    Upload verified ``.wav`` files to their sibling path on GCS, never
    overwriting an existing blob. Writes ``upload.parquet``.
``rewrite``
    Write new split CSVs to ``<work-dir>/out`` for both versions: drop the
    ``gcs_path`` and ``relative_path`` columns where present (they point at
    the ingestion bucket and are not used by the dataset class), point every resolved target's
    ``originals_path`` at its ``.wav`` sibling, and drop rows whose target is
    unresolved (broken wav, missing, or failed decode/verify). Writes
    ``rewrite_report.json``.
``publish``
    Upload the rewritten CSVs to ``DATA_ROOT``. Refuses to overwrite unless
    ``--overwrite`` is given.

Note: the first (m4a-only) run decoded on macOS, where audioread picked its
CoreAudio backend rather than ffmpeg. Those WAVs are bit-exact to the
CoreAudio decode and correlate with the independently made 16 kHz files
(min r = 0.976 on a 2% sample); later runs pin ffmpeg.

Usage
-----
    uv run --with tqdm python scripts/data_preprocessing_scripts/inat_m4a_mp3_to_wav.py \\
        --work-dir ~/inat_m4a_work scan --extra-targets failures_0.2.0_all_srnative.csv
    ... convert --n-workers 8
    ... verify --n-workers 8 --crosscheck-frac 0.02
    ... upload
    ... rewrite
    ... publish [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import audioread.ffdec
import librosa
import numpy as np
import polars as pl
import soundfile as sf
from scipy.signal import correlate
from tqdm import tqdm

from alp_data.io import DATA_HOME, filesystem_from_path, read_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inat_to_wav")

# Shared by both dataset versions: v0.2.0's split CSVs and all audio live under
# the v0.1.0 path too (there is no v0.2.0 directory).
DATA_ROOT = f"{DATA_HOME}/inaturalist/v0.1.0/raw/"
ORIGINALS_COLUMN = "originals_path"
CROSSCHECK_COLUMN = "16khz_path"
CROSSCHECK_SR = 16000
CROSSCHECK_MIN_CORR = 0.9
MAX_LAG_S = 1.0
DROP_COLUMNS = ["gcs_path", "relative_path"]
SPLITS = ["train", "train_unseen", "val", "val_unseen", "all", "all_unseen"]
# version -> (input CSV suffix, output CSV suffix)
VERSIONS = {
    "0.1.0": ("_20260201_v3.csv", "_20260201_v4.csv"),
    "0.2.0": ("_20260616_v1.csv", "_20260616_v2.csv"),
}


# --------------------------------------------------------------------------- helpers


def _wav_sibling(rel: str) -> str:
    """Return the ``.wav`` sibling of a relative path.

    Parameters
    ----------
    rel : str
        Path with an audio file extension.

    Returns
    -------
    str
        Same path with the extension replaced by ``.wav``.
    """
    return rel.rsplit(".", 1)[0] + ".wav"


def _csv_names(version: str, output: bool = False) -> list[str]:
    """Return split CSV file names for a version.

    Parameters
    ----------
    version : str
        Key of `VERSIONS`.
    output : bool, default=False
        Return the rewritten (output) names instead of the input names.

    Returns
    -------
    list of str
        One file name per split.
    """
    suffix = VERSIONS[version][1 if output else 0]
    return [f"{split}{suffix}" for split in SPLITS]


def _read_csv(path: Path) -> pl.DataFrame:
    """Read a split CSV with every column as a string (no type inference).

    Parameters
    ----------
    path : Path
        Local CSV path.

    Returns
    -------
    pl.DataFrame
        The CSV contents.
    """
    return pl.read_csv(path, infer_schema_length=0)


def _pmap(fn: Callable, items: list, n_workers: int, desc: str, threads: bool = False) -> list:
    """Map `fn` over `items` in a pool with a progress bar, preserving order.

    Parameters
    ----------
    fn : Callable
        Picklable function of one argument.
    items : list
        Inputs.
    n_workers : int
        Pool size.
    desc : str
        Progress bar label.
    threads : bool, default=False
        Use threads (I/O bound work) instead of spawned processes.

    Returns
    -------
    list
        `fn(item)` for each item, in input order.
    """
    if not items:
        return []
    if threads:
        pool = ThreadPoolExecutor(max_workers=n_workers)
    else:
        # gcsfs is not fork-safe, so every worker gets a fresh interpreter.
        pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"))
    with pool:
        return list(tqdm(pool.map(fn, items, chunksize=1), total=len(items), desc=desc))


def _decode(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode an audio file with `librosa.load` at its native rate and channels.

    ``.wav`` files are read by librosa through soundfile. Everything else is
    decoded through audioread's ffmpeg backend explicitly, because audioread's
    default backend order is platform dependent (CoreAudio on macOS).

    Parameters
    ----------
    path : str | Path
        Local audio file.

    Returns
    -------
    tuple of (np.ndarray, int)
        Audio as ``(channels, frames)`` or ``(frames,)`` and the sample rate.

    Raises
    ------
    ValueError
        If the decoded audio is empty or contains non-finite samples.
    """
    with warnings.catch_warnings():
        # librosa warns that the audioread fallback is deprecated; it's the
        # only path that can decode AAC here.
        warnings.simplefilter("ignore")
        if str(path).endswith(".wav"):
            y, sr = librosa.load(str(path), sr=None, mono=False)
        else:
            with audioread.ffdec.FFmpegAudioFile(str(path)) as src:
                y, sr = librosa.load(src, sr=None, mono=False)
    if y.size == 0:
        raise ValueError("decoded audio is empty")
    if not np.all(np.isfinite(y)):
        raise ValueError("decoded audio contains non-finite samples")
    return y, int(sr)


def _load_stage(work_dir: Path, name: str) -> pl.DataFrame:
    """Load a stage's parquet output.

    Parameters
    ----------
    work_dir : Path
        Work directory.
    name : str
        Stage name (``scan``, ``convert``, ...).

    Returns
    -------
    pl.DataFrame
        The stage table.

    Raises
    ------
    FileNotFoundError
        If the stage has not been run yet.
    """
    path = work_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the `{name}` stage first")
    return pl.read_parquet(path)


def _save_stage(work_dir: Path, name: str, df: pl.DataFrame) -> None:
    """Write a stage's parquet output and log a status summary.

    Parameters
    ----------
    work_dir : Path
        Work directory.
    name : str
        Stage name.
    df : pl.DataFrame
        Stage table; a ``status`` column is summarised if present.
    """
    path = work_dir / f"{name}.parquet"
    df.write_parquet(path)
    if "status" in df.columns:
        logger.info("%s: %s", name, dict(df["status"].value_counts().iter_rows()))
    logger.info("wrote %s (%d rows)", path, df.height)


# --------------------------------------------------------------------------- scan


def _download_csvs(csv_dir: Path) -> None:
    """Download every input split CSV of every version into `csv_dir`.

    Existing local copies are kept, so re-running is cheap.

    Parameters
    ----------
    csv_dir : Path
        Local cache directory.
    """
    csv_dir.mkdir(parents=True, exist_ok=True)
    fs = filesystem_from_path(DATA_ROOT)
    for version in VERSIONS:
        for name in _csv_names(version):
            dst = csv_dir / name
            if dst.exists():
                continue
            logger.info("downloading %s", name)
            fs.get(DATA_ROOT + name, str(dst))


def _remote_exists(rel: str) -> bool:
    """Return whether `rel` exists under `DATA_ROOT`.

    Parameters
    ----------
    rel : str
        Path relative to `DATA_ROOT`.

    Returns
    -------
    bool
        Whether the blob exists.
    """
    return filesystem_from_path(DATA_ROOT).exists(DATA_ROOT + rel)


def _collect_targets(csv_dir: Path, extra_targets: list[Path]) -> pl.DataFrame:
    """Return the unique target paths with a ``16khz_path`` for cross-checks.

    Parameters
    ----------
    csv_dir : Path
        Local cache of the input split CSVs.
    extra_targets : list of Path
        Failure CSVs (``originals_path`` column) from ``iterate_inaturalist.py``.

    Returns
    -------
    pl.DataFrame
        Columns ``src`` (target ``originals_path``) and `CROSSCHECK_COLUMN`.

    Raises
    ------
    ValueError
        If a v0.1.0 CSV still references an ``.m4a`` original (it was
        expected to be fully converted already).
    """
    targets = set()
    for path in extra_targets:
        targets |= set(_read_csv(path)[ORIGINALS_COLUMN].to_list())
    logger.info("targets from failure CSVs: %d", len(targets))

    frames = []
    for version in VERSIONS:
        for name in _csv_names(version):
            df = _read_csv(csv_dir / name).select(ORIGINALS_COLUMN, CROSSCHECK_COLUMN)
            is_m4a = pl.col(ORIGINALS_COLUMN).str.ends_with(".m4a")
            if version == "0.1.0" and df.select(is_m4a.any()).item():
                raise ValueError(f"{name} has .m4a originals; expected 0")
            frames.append(df.filter(is_m4a | pl.col(ORIGINALS_COLUMN).is_in(list(targets))))
    out = pl.concat(frames).unique(ORIGINALS_COLUMN, keep="first").sort(ORIGINALS_COLUMN)
    missing = targets - set(out[ORIGINALS_COLUMN].to_list())
    if missing:
        logger.warning("%d failure-CSV targets are not in any split CSV", len(missing))
    return out.rename({ORIGINALS_COLUMN: "src"})


def scan(work_dir: Path, n_workers: int, extra_targets: list[Path]) -> None:
    """Find all target originals and their GCS status.

    Parameters
    ----------
    work_dir : Path
        Work directory.
    n_workers : int
        Number of threads for GCS existence checks.
    extra_targets : list of Path
        Failure CSVs whose ``originals_path`` values are targets in addition
        to every ``.m4a``.
    """
    csv_dir = work_dir / "csv"
    _download_csvs(csv_dir)
    df = _collect_targets(csv_dir, extra_targets).with_columns(
        wav=pl.col("src").str.replace(r"\.[^./]+$", ".wav"),
        ext=pl.col("src").str.extract(r"\.([^./]+)$"),
    )
    logger.info("targets by extension: %s", dict(df["ext"].value_counts().iter_rows()))

    src_exists = _pmap(_remote_exists, df["src"].to_list(), n_workers, "src exists", True)
    wav_exists = _pmap(_remote_exists, df["wav"].to_list(), n_workers, "wav exists", True)
    df = df.with_columns(src_exists=pl.Series(src_exists), wav_exists=pl.Series(wav_exists))
    df = df.with_columns(
        status=pl.when(pl.col("ext") == "wav")
        .then(pl.lit("broken_wav"))
        .when("wav_exists")
        .then(pl.lit("wav_exists"))
        .when("src_exists")
        .then(pl.lit("to_convert"))
        .otherwise(pl.lit("missing"))
    )
    _save_stage(work_dir, "scan", df)


# --------------------------------------------------------------------------- convert


def _convert_one(args: tuple[str, str]) -> dict:
    """Download one target and write its PCM_16 ``.wav`` sibling locally.

    Parameters
    ----------
    args : tuple of (str, str)
        ``(target relative path, local audio root)``.

    Returns
    -------
    dict
        ``{"src", "status", "message", "sr", "channels", "frames"}`` with
        status ``success``, ``skipped`` (local ``.wav`` already present) or
        ``decode_error``/``download_error``.
    """
    rel, root = args
    src_local = Path(root) / rel
    dst_local = Path(root) / _wav_sibling(rel)
    out = {"src": rel, "status": "skipped", "message": "", "sr": None, "channels": None,
           "frames": None}  # fmt: skip
    if dst_local.exists():
        return out
    try:
        if not src_local.exists():
            src_local.parent.mkdir(parents=True, exist_ok=True)
            filesystem_from_path(DATA_ROOT).get(DATA_ROOT + rel, str(src_local))
    except Exception as exc:  # noqa: BLE001 - recorded in the report
        return out | {"status": "download_error", "message": f"{type(exc).__name__}: {exc}"}
    try:
        y, sr = _decode(src_local)
        data = y.T if y.ndim == 2 else y  # soundfile wants (frames, channels)
        tmp = dst_local.with_suffix(".wav.part")
        sf.write(tmp, data, sr, format="WAV", subtype="PCM_16")
        tmp.rename(dst_local)
    except Exception as exc:  # noqa: BLE001 - recorded in the report
        return out | {"status": "decode_error", "message": f"{type(exc).__name__}: {exc}"}
    channels = 1 if y.ndim == 1 else y.shape[0]
    return out | {"status": "success", "sr": sr, "channels": channels, "frames": y.shape[-1]}


def convert(work_dir: Path, n_workers: int, limit: int | None = None) -> None:
    """Convert every ``to_convert`` file from the scan into a local ``.wav``.

    Parameters
    ----------
    work_dir : Path
        Work directory.
    n_workers : int
        Number of worker processes.
    limit : int, optional
        Only convert the first `limit` files (for smoke tests). Downstream
        stages then only see these files, so re-run without it before
        `upload`.
    """
    todo = _load_stage(work_dir, "scan").filter(pl.col("status") == "to_convert")["src"]
    if limit is not None:
        todo = todo.head(limit)
    root = str(work_dir / "audio")
    results = _pmap(_convert_one, [(r, root) for r in todo.to_list()], n_workers, "->wav")
    _save_stage(work_dir, "convert", pl.DataFrame(results, infer_schema_length=None))


# --------------------------------------------------------------------------- verify


def _crosscheck_16k(y: np.ndarray, sr: int, rel_16k: str) -> tuple[float, int]:
    """Correlate decoded audio with the existing 16 kHz resample on GCS.

    A high correlation shows that the target is the recording the
    ``16khz_path`` file was made from (i.e. the pairing is right). Decoders
    handle AAC priming differently, so the two can be offset by a few tens of
    milliseconds; the correlation is taken at the best lag within
    `MAX_LAG_S`.

    Parameters
    ----------
    y : np.ndarray
        Decoded original audio, ``(channels, frames)`` or ``(frames,)``.
    sr : int
        Sample rate of `y`.
    rel_16k : str
        ``16khz_path`` value relative to `DATA_ROOT`.

    Returns
    -------
    tuple of (float, int)
        Pearson correlation at the best lag, and that lag in 16 kHz samples
        (positive means our decode starts later than the 16 kHz file).
    """
    mono = y.mean(axis=0) if y.ndim == 2 else y
    ours = librosa.resample(mono, orig_sr=sr, target_sr=CROSSCHECK_SR, res_type="kaiser_best")
    theirs, _ = read_audio(DATA_ROOT + rel_16k)
    theirs = theirs.mean(axis=1) if theirs.ndim == 2 else theirs
    n = min(len(ours), len(theirs))
    xc = correlate(ours[:n], theirs[:n], mode="full", method="fft")
    max_lag = int(MAX_LAG_S * CROSSCHECK_SR)
    centre = n - 1
    window = xc[max(0, centre - max_lag) : centre + max_lag + 1]
    lag = int(np.argmax(np.abs(window))) + max(0, centre - max_lag) - centre
    a, b = (ours[lag:], theirs) if lag >= 0 else (ours, theirs[-lag:])
    m = min(len(a), len(b))
    return float(np.corrcoef(a[:m], b[:m])[0, 1]), lag


def _verify_one(args: tuple[str, str, str | None]) -> dict:
    """Check that a local ``.wav`` is a bit-exact copy of its decoded target.

    Parameters
    ----------
    args : tuple of (str, str, str or None)
        ``(target relative path, local audio root, 16khz_path or None)``. If the
        third element is given, also run `_crosscheck_16k`.

    Returns
    -------
    dict
        ``{"src", "status", "message", "max_abs_diff", "corr_16k", "lag_16k"}`` with
        status ``ok`` or ``mismatch``/``error``.
    """
    rel, root, rel_16k = args
    out = {"src": rel, "status": "ok", "message": "", "max_abs_diff": None, "corr_16k": None,
           "lag_16k": None}  # fmt: skip
    try:
        y_src, sr_src = _decode(Path(root) / rel)
        y_wav, sr_wav = _decode(Path(root) / _wav_sibling(rel))
    except Exception as exc:  # noqa: BLE001 - recorded in the report
        return out | {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
    if sr_src != sr_wav or y_src.shape != y_wav.shape:
        msg = f"sr {sr_src} vs {sr_wav}, shape {y_src.shape} vs {y_wav.shape}"
        return out | {"status": "mismatch", "message": msg}
    out["max_abs_diff"] = float(np.max(np.abs(y_src - y_wav)))
    if not np.array_equal(y_src, y_wav):
        out |= {"status": "mismatch", "message": "samples differ"}
    if rel_16k is not None:
        try:
            out["corr_16k"], out["lag_16k"] = _crosscheck_16k(y_src, sr_src, rel_16k)
        except Exception as exc:  # noqa: BLE001 - informational only
            out["message"] += f" crosscheck failed: {type(exc).__name__}: {exc}"
    return out


def verify(work_dir: Path, n_workers: int, crosscheck_frac: float, seed: int) -> None:
    """Verify every successfully converted file.

    Parameters
    ----------
    work_dir : Path
        Work directory.
    n_workers : int
        Number of worker processes.
    crosscheck_frac : float
        Fraction of files to also cross-check against ``16khz_path``.
    seed : int
        Seed for choosing the cross-check subset.
    """
    conv = _load_stage(work_dir, "convert").filter(pl.col("status").is_in(["success", "skipped"]))
    paths_16k = dict(_load_stage(work_dir, "scan").select("src", CROSSCHECK_COLUMN).iter_rows())
    rng = random.Random(seed)
    root = str(work_dir / "audio")
    items = [
        (rel, root, paths_16k[rel] if rng.random() < crosscheck_frac else None)
        for rel in conv["src"].to_list()
    ]
    df = pl.DataFrame(_pmap(_verify_one, items, n_workers, "verify"), infer_schema_length=None)
    _save_stage(work_dir, "verify", df)
    corr = df["corr_16k"].drop_nulls()
    if corr.len():
        logger.info(
            "16k cross-check on %d files: min corr %.4f, %d below %.2f (inspect those)",
            corr.len(),
            corr.min(),
            (corr < CROSSCHECK_MIN_CORR).sum(),
            CROSSCHECK_MIN_CORR,
        )


# --------------------------------------------------------------------------- upload


def _upload_one(args: tuple[str, str]) -> dict:
    """Upload one local ``.wav`` to its sibling path on GCS without overwriting.

    Parameters
    ----------
    args : tuple of (str, str)
        ``(target relative path, local audio root)``.

    Returns
    -------
    dict
        ``{"src", "status", "message"}`` with status ``uploaded``,
        ``exists`` or ``error``.
    """
    rel, root = args
    wav = _wav_sibling(rel)
    fs = filesystem_from_path(DATA_ROOT)
    try:
        if fs.exists(DATA_ROOT + wav):
            return {"src": rel, "status": "exists", "message": ""}
        fs.put(str(Path(root) / wav), DATA_ROOT + wav)
    except Exception as exc:  # noqa: BLE001 - recorded in the report
        return {"src": rel, "status": "error", "message": f"{type(exc).__name__}: {exc}"}
    return {"src": rel, "status": "uploaded", "message": ""}


def upload(work_dir: Path, n_workers: int) -> None:
    """Upload every verified ``.wav`` to GCS.

    Parameters
    ----------
    work_dir : Path
        Work directory.
    n_workers : int
        Number of upload threads.
    """
    ok = _load_stage(work_dir, "verify").filter(pl.col("status") == "ok")["src"].to_list()
    root = str(work_dir / "audio")
    results = _pmap(_upload_one, [(r, root) for r in ok], n_workers, "upload", True)
    _save_stage(work_dir, "upload", pl.DataFrame(results, infer_schema_length=None))


# --------------------------------------------------------------------------- rewrite


def _targets_and_resolved(work_dir: Path) -> tuple[set[str], set[str]]:
    """Return all target paths and those whose ``.wav`` sibling is on GCS.

    Parameters
    ----------
    work_dir : Path
        Work directory.

    Returns
    -------
    tuple of (set of str, set of str)
        ``(targets, resolved)``: every scanned target, and the subset that
        already had a ``.wav`` at scan time or was uploaded (or found already
        present) by the `upload` stage.
    """
    scan_df = _load_stage(work_dir, "scan")
    targets = set(scan_df["src"].to_list())
    resolved = set(scan_df.filter(pl.col("status") == "wav_exists")["src"].to_list())
    if scan_df.filter(pl.col("status") == "to_convert").height:
        up = _load_stage(work_dir, "upload").filter(pl.col("status").is_in(["uploaded", "exists"]))
        resolved |= set(up["src"].to_list())
    return targets, resolved


def _rewrite_one(
    df: pl.DataFrame, targets: set[str], resolved: set[str]
) -> tuple[pl.DataFrame, dict]:
    """Apply the column drop and target -> ``.wav`` rewrite to one split.

    Parameters
    ----------
    df : pl.DataFrame
        Input split.
    targets : set of str
        All target ``originals_path`` values.
    resolved : set of str
        Targets with a ``.wav`` sibling on GCS.

    Returns
    -------
    tuple of (pl.DataFrame, dict)
        The rewritten split and counts of rewritten / dropped rows.
    """
    col = pl.col(ORIGINALS_COLUMN)
    is_target = col.is_in(list(targets))
    is_resolved = col.is_in(list(resolved))
    counts = {
        "rows_in": df.height,
        "rewritten_to_wav": int(df.select((is_target & is_resolved).sum()).item()),
        "dropped_unresolved": int(df.select((is_target & ~is_resolved).sum()).item()),
    }
    out = (
        df.filter(~is_target | is_resolved)
        .with_columns(
            pl.when(is_resolved).then(col.str.replace(r"\.[^./]+$", ".wav")).otherwise(col)
        )
        .drop(DROP_COLUMNS, strict=False)
    )
    counts["rows_out"] = out.height
    return out, counts


def rewrite(work_dir: Path) -> None:
    """Write the cleaned split CSVs of both versions to ``<work-dir>/out``.

    Parameters
    ----------
    work_dir : Path
        Work directory.

    Raises
    ------
    ValueError
        If an output still contains ``.m4a`` originals, an unresolved target
        or a dropped column.
    """
    targets, resolved = _targets_and_resolved(work_dir)
    unresolved = list(targets - resolved)
    out_dir = work_dir / "out"
    out_dir.mkdir(exist_ok=True)
    report: dict[str, dict] = {}
    for version in VERSIONS:
        for src, dst in zip(_csv_names(version), _csv_names(version, output=True), strict=True):
            df, counts = _rewrite_one(_read_csv(work_dir / "csv" / src), targets, resolved)
            if df[ORIGINALS_COLUMN].str.ends_with(".m4a").any():
                raise ValueError(f"{dst} still has .m4a originals")
            if df[ORIGINALS_COLUMN].is_in(unresolved).any():
                raise ValueError(f"{dst} still has unresolved targets")
            if set(DROP_COLUMNS) & set(df.columns):
                raise ValueError(f"{dst} still has one of {DROP_COLUMNS}")
            df.write_csv(out_dir / dst)
            report[dst] = counts | {"columns": df.width}
            logger.info("%s: %s", dst, report[dst])
    (work_dir / "rewrite_report.json").write_text(json.dumps(report, indent=2))


# --------------------------------------------------------------------------- publish


def publish(work_dir: Path, overwrite: bool = False) -> None:
    """Upload the rewritten CSVs from ``<work-dir>/out`` to `DATA_ROOT`.

    Parameters
    ----------
    work_dir : Path
        Work directory.
    overwrite : bool, default=False
        Replace destination CSVs that already exist on GCS.

    Raises
    ------
    FileExistsError
        If any destination CSV already exists on GCS and `overwrite` is
        False (nothing is uploaded).
    """
    fs = filesystem_from_path(DATA_ROOT)
    names = [n for v in VERSIONS for n in _csv_names(v, output=True)]
    clash = [n for n in names if fs.exists(DATA_ROOT + n)]
    if clash and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing CSVs: {clash}")
    if clash:
        logger.warning("overwriting existing CSVs: %s", clash)
    for name in names:
        logger.info("uploading %s", name)
        fs.put(str(work_dir / "out" / name), DATA_ROOT + name)


# --------------------------------------------------------------------------- CLI


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Iterable of str, optional
        Arguments; defaults to `sys.argv`.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--n-workers", type=int, default=8)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ["upload", "rewrite"]:
        sub.add_parser(name)
    p_scan = sub.add_parser("scan")
    p_scan.add_argument(
        "--extra-targets",
        type=Path,
        nargs="*",
        default=[],
        help="Failure CSVs from iterate_inaturalist.py; their originals_path are targets.",
    )
    p_publish = sub.add_parser("publish")
    p_publish.add_argument("--overwrite", action="store_true")
    p_convert = sub.add_parser("convert")
    p_convert.add_argument("--limit", type=int, default=None)
    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--crosscheck-frac", type=float, default=0.02)
    p_verify.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    """Command-line entry point."""
    args = _parse_args()
    work_dir = args.work_dir.expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "scan":
        scan(work_dir, max(args.n_workers, 32), args.extra_targets)
    elif args.stage == "convert":
        convert(work_dir, args.n_workers, args.limit)
    elif args.stage == "verify":
        verify(work_dir, args.n_workers, args.crosscheck_frac, args.seed)
    elif args.stage == "upload":
        upload(work_dir, max(args.n_workers, 16))
    elif args.stage == "rewrite":
        rewrite(work_dir)
    else:
        publish(work_dir, args.overwrite)


if __name__ == "__main__":
    main()
