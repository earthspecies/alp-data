"""Convert iNaturalist .m4a originals to .wav and update split CSVs.

The ``originals_path`` column in iNaturalist split CSVs contains some files
encoded as ``.m4a``. ``esp_data.io.read_audio`` is backed by ``soundfile``
(libsndfile) which cannot decode ``.m4a``, so these rows fail at load /
migration time.

This script:

1. Scans every iNaturalist split CSV for rows whose ``originals_path`` ends
   in ``.m4a``.
2. For each unique m4a blob, downloads it, decodes via ``librosa.load`` (which
   uses ``audioread``/``ffmpeg`` for AAC), and writes a sibling ``.wav`` blob
   at the original sample rate, preserving channels.
3. Rewrites every split CSV in place, replacing trailing ``.m4a`` in
   ``originals_path`` with ``.wav`` for entries that converted successfully.

Existing ``.wav`` siblings are not overwritten (idempotent re-runs).

Usage
-----
    uv run python scripts/data_preprocessing_scripts/inat_m4a_to_wav.py \\
        --n-workers 8 \\
        --report-path inat_m4a_to_wav_report.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import multiprocessing as mp
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import polars as pl
import soundfile as sf
from tqdm import tqdm

from esp_data.io import exists, filesystem_from_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inat_m4a_to_wav")

DATA_ROOT = "gs://esp-ml-datasets/inaturalist/v0.1.0/raw/"
ORIGINALS_COLUMN = "originals_path"
SPLIT_FILES = [
    "train_20260201_v2.csv",
    "train_unseen_20260201_v2.csv",
    "val_20260201_v2.csv",
    "val_unseen_20260201_v2.csv",
    "all_20260201_v2.csv",
    "all_unseen_20260201_v2.csv",
]


def _src_dst(rel_path: str) -> tuple[str, str, str]:
    """Return source/destination URLs and the new relative path.

    Parameters
    ----------
    rel_path : str
        Path of the source ``.m4a`` file relative to ``DATA_ROOT``.

    Returns
    -------
    tuple of (str, str, str)
        ``(src_url, dst_url, dst_rel)``: full source URL, full destination
        URL with ``.wav`` extension, and the destination path relative to
        ``DATA_ROOT``.
    """
    src = DATA_ROOT + rel_path
    dst_rel = rel_path[: -len(".m4a")] + ".wav"
    dst = DATA_ROOT + dst_rel
    return src, dst, dst_rel


def convert_one(rel_path: str) -> dict:
    """Convert a single ``.m4a`` blob to a sibling ``.wav`` blob.

    Parameters
    ----------
    rel_path : str
        Path of the source file relative to ``DATA_ROOT``.

    Returns
    -------
    dict
        ``{"rel": rel_path, "status": "success"|"skipped"|"error",
        "dst_rel": new relative path, "message": optional str}``.
        ``dst_rel`` is the ``.wav`` sibling regardless of status, so callers
        can rewrite the CSV column on success/skipped alike.
    """
    src, dst, dst_rel = _src_dst(rel_path)

    try:
        if exists(dst):
            return {"rel": rel_path, "status": "skipped", "dst_rel": dst_rel}
    except Exception as exc:
        return {
            "rel": rel_path,
            "status": "error",
            "dst_rel": dst_rel,
            "message": f"exists check failed: {exc}",
        }

    try:
        src_fs = filesystem_from_path(src)
        with src_fs.open(src, "rb") as f:
            buf = f.read()

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=True) as tmp:
            tmp.write(buf)
            tmp.flush()
            data, sr = librosa.load(tmp.name, sr=None, mono=False)

        # librosa returns (channels, frames) for multi-channel; soundfile wants (frames, channels).
        if data.ndim == 2:
            data = data.T

        out = io.BytesIO()
        sf.write(out, data, int(sr), format="WAV", subtype="PCM_16")
        payload = out.getvalue()

        dst_fs = filesystem_from_path(dst)
        with dst_fs.open(dst, "wb") as f:
            f.write(payload)

        return {"rel": rel_path, "status": "success", "dst_rel": dst_rel}
    except Exception as exc:
        return {
            "rel": rel_path,
            "status": "error",
            "dst_rel": dst_rel,
            "message": f"{type(exc).__name__}: {exc}",
        }


def collect_m4a_paths(split_files: list[str]) -> tuple[set[str], dict[str, pl.DataFrame]]:
    """Return all unique ``.m4a`` relative paths and the loaded split DataFrames.

    Parameters
    ----------
    split_files : list of str
        File names (relative to ``DATA_ROOT``) of split CSVs to scan.

    Returns
    -------
    tuple of (set of str, dict)
        Unique ``originals_path`` values ending in ``.m4a``, and a mapping
        from split file name to its loaded polars DataFrame (strings, no
        type inference) so we don't re-download for the rewrite pass.
    """
    paths: set[str] = set()
    dfs: dict[str, pl.DataFrame] = {}
    for fname in split_files:
        url = DATA_ROOT + fname
        fs = filesystem_from_path(url)
        with fs.open(url, "rb") as f:
            df = pl.read_csv(f, infer_schema_length=0)
        dfs[fname] = df
        if ORIGINALS_COLUMN not in df.columns:
            logger.warning("[%s] missing column %s", fname, ORIGINALS_COLUMN)
            continue
        col = df[ORIGINALS_COLUMN].drop_nulls()
        m4a = col.filter(col.str.ends_with(".m4a"))
        paths.update(m4a.to_list())
        logger.info("[%s] rows=%d m4a=%d", fname, df.height, m4a.len())
    return paths, dfs


def rewrite_splits(
    dfs: dict[str, pl.DataFrame],
    converted: set[str],
    output_suffix: str,
) -> dict[str, dict]:
    """Write rewritten split CSVs alongside the originals at ``DATA_ROOT``.

    Each output file lives at ``DATA_ROOT + {stem}{output_suffix}.csv``,
    so the cloud prefix is preserved and only the file name changes.

    Parameters
    ----------
    dfs : dict
        Map of split file name → loaded DataFrame.
    converted : set of str
        Relative ``.m4a`` paths that now have a corresponding ``.wav`` blob
        (success or skipped). Only these get rewritten.
    output_suffix : str
        String inserted between the source file's stem and ``.csv``
        (e.g. ``"_wav"`` turns ``train_20260201_v2.csv`` into
        ``train_20260201_v2_wav.csv``). Use ``""`` to overwrite in place.

    Returns
    -------
    dict
        Map of source split file name → ``{"rows_updated": int,
        "output_path": str}``.
    """
    counts: dict[str, dict] = {}
    for fname, df in dfs.items():
        stem = fname[: -len(".csv")] if fname.endswith(".csv") else fname
        out_name = f"{stem}{output_suffix}.csv"
        out_url = DATA_ROOT + out_name

        if ORIGINALS_COLUMN not in df.columns:
            new_df = df
            n = 0
        else:
            col = df[ORIGINALS_COLUMN]
            mask = col.is_in(list(converted))
            n = int(mask.sum())
            if n > 0:
                new_col = (
                    pl.when(mask)
                    .then(col.str.slice(0, col.str.len_chars() - 4) + ".wav")
                    .otherwise(col)
                    .alias(ORIGINALS_COLUMN)
                )
                new_df = df.with_columns(new_col)
            else:
                new_df = df

        buf = io.BytesIO()
        new_df.write_csv(buf)
        fs = filesystem_from_path(out_url)
        with fs.open(out_url, "wb") as f:
            f.write(buf.getvalue())
        counts[fname] = {"rows_updated": n, "output_path": out_url}
        logger.info("[%s] wrote %s (%d rows updated)", fname, out_url, n)
    return counts


def main() -> None:
    """Command-line entry point.

    Raises
    ------
    SystemExit
        Exit status ``1`` if any file conversions failed (the ``.wav``
        siblings of those rows do not exist, so the CSVs are not rewritten
        for them).
    """  # noqa: DOC502
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument(
        "--report-path",
        type=str,
        default="inat_m4a_to_wav_report.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report m4a counts only; do not convert or rewrite.",
    )
    parser.add_argument(
        "--skip-rewrite",
        action="store_true",
        help="Convert files but do not write any rewritten split CSVs.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_wav",
        help=(
            "Suffix inserted before .csv in the output filename "
            "(default: '_wav'). The cloud prefix (DATA_ROOT) is preserved. "
            "Pass '' to overwrite the source CSVs in place."
        ),
    )
    args = parser.parse_args()

    logger.info("Scanning splits at %s", DATA_ROOT)
    m4a_paths, dfs = collect_m4a_paths(SPLIT_FILES)
    logger.info("Found %d unique m4a originals across all splits", len(m4a_paths))

    if args.dry_run:
        Path(args.report_path).write_text(
            json.dumps({"unique_m4a": len(m4a_paths), "paths": sorted(m4a_paths)}, indent=2)
        )
        return

    successes: list[str] = []
    skipped: list[str] = []
    errors: list[dict] = []
    converted: set[str] = set()

    paths = sorted(m4a_paths)
    # gcsfs is not fork-safe — use 'spawn' so each worker initializes its
    # own filesystem instance instead of inheriting the parent's cached one.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(convert_one, p): p for p in paths}
        with tqdm(total=len(futures), desc="m4a→wav", unit="file") as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                if res["status"] == "success":
                    successes.append(res["rel"])
                    converted.add(res["rel"])
                elif res["status"] == "skipped":
                    skipped.append(res["rel"])
                    converted.add(res["rel"])
                else:
                    errors.append({"rel": res["rel"], "message": res.get("message", "")})
                pbar.set_postfix({"ok": len(successes), "skip": len(skipped), "err": len(errors)})
                pbar.update(1)

    logger.info(
        "Conversion done: %d ok, %d skipped, %d errors",
        len(successes),
        len(skipped),
        len(errors),
    )

    if args.skip_rewrite:
        rewrite_counts: dict[str, dict] = {}
    else:
        rewrite_counts = rewrite_splits(dfs, converted, args.output_suffix)

    report = {
        "unique_m4a": len(m4a_paths),
        "successes": len(successes),
        "skipped": len(skipped),
        "errors": errors,
        "rewrite_counts": rewrite_counts,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", args.report_path)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
