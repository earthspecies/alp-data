"""Repair corrupt WAV files flagged as ``empty_audio`` by migrate_datasets.

Reads a migration report, extracts ``empty_audio`` issues, downloads each
source blob, re-encodes to PCM16 WAV via ffmpeg (fixes mp3-in-wav containers
and truncated/garbled PCM), validates the result with soundfile, and writes
to a local output directory mirroring the source blob path.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
import soundfile as sf


def filesystem_from_path(url: str) -> fsspec.AbstractFileSystem:
    if url.startswith("gs://"):
        return fsspec.filesystem("gs")
    if url.startswith("s3://"):
        return fsspec.filesystem("s3")
    if url.startswith("r2://"):
        return fsspec.filesystem("r2")
    return fsspec.filesystem("file")


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("repair_bad_wavs")

MAX_WORKERS = 8


def _strip_scheme(url: str) -> str:
    for s in ("gs://", "s3://", "r2://"):
        if url.startswith(s):
            return url[len(s) :]
    return url


def _local_dst(out_dir: Path, src_url: str) -> Path:
    return out_dir / _strip_scheme(src_url)


def repair_one(src_url: str, out_dir: Path) -> tuple[str, str | None]:
    """Download, re-encode, validate.

    Returns
    -------
    tuple[str, str | None]
        `(src_url, error)` where `error` is None on success.
    """
    dst = _local_dst(out_dir, src_url)
    if dst.exists():
        return src_url, None
    dst.parent.mkdir(parents=True, exist_ok=True)

    fs = filesystem_from_path(src_url)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "in.wav"
        fixed = Path(td) / "out.wav"
        try:
            with fs.open(src_url, "rb") as fr, raw.open("wb") as fw:
                shutil.copyfileobj(fr, fw)
        except Exception as exc:
            return src_url, f"download_failed: {exc}"

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw),
            "-c:a",
            "pcm_s16le",
            str(fixed),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not fixed.exists() or fixed.stat().st_size == 0:
            return src_url, f"ffmpeg_failed: {proc.stderr.strip()[:200]}"

        try:
            info = sf.info(str(fixed))
            if info.frames == 0:
                return src_url, "validate_failed: 0 frames after repair"
        except Exception as exc:
            return src_url, f"validate_failed: {exc}"

        shutil.move(str(fixed), str(dst))
    return src_url, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Path to migrate_datasets report JSON")
    parser.add_argument("--out-dir", required=True, help="Local output directory")
    parser.add_argument(
        "--issue-types",
        nargs="+",
        default=["empty_audio"],
        help="Issue types to repair (default: empty_audio)",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = json.load(open(args.report))
    paths = sorted(
        {i["file_path"] for i in report["issues"] if i["issue_type"] in args.issue_types}
    )
    logger.info("Repairing %d files -> %s", len(paths), out_dir)

    failures: list[tuple[str, str]] = []
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(repair_one, p, out_dir): p for p in paths}
        for fut in as_completed(futs):
            src, err = fut.result()
            if err:
                failures.append((src, err))
                logger.error("FAIL %s: %s", src, err)
            else:
                ok += 1
                logger.info("OK   [%d/%d] %s", ok + len(failures), len(paths), src)

    summary = {
        "total": len(paths),
        "repaired": ok,
        "failed": len(failures),
        "failures": [{"path": p, "error": e} for p, e in failures],
    }
    summary_path = out_dir / "repair_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Done. %d/%d repaired. Summary: %s", ok, len(paths), summary_path)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
