# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "soundfile",
#     "google-cloud-storage",
# ]
# ///
"""Validate that the pre-resampled audio for the completed AudioSkillsXL
``wavcaps`` split actually exists and decodes.

For every row in the wavcaps CSV, downloads the audio at the chosen rate column
(default ``32khz_path``, resolved under the AudioSet root) and decodes it with
soundfile, checking: present, readable, expected sample rate, non-empty, finite,
not all-zero. Prints a status breakdown and writes any failures to a CSV.

Heavy IO (downloads every clip once); run on a Slurm cpu node, never the dev VM.
"""

from __future__ import annotations

import argparse
import io
import multiprocessing
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
from google.cloud import storage

CSV = "gs://esp-data-ingestion/AudioSkillsXL/v0.1.0/raw/wavcaps.csv"
AUDIOSET_BUCKET = "esp-ml-datasets"
AUDIOSET_PREFIX = "audioset/v0.2.0/raw"  # rate-column paths are relative to this


def _split_gs(uri: str) -> tuple[str, str]:
    rest = uri[len("gs://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def validate_batch(args: tuple) -> list[tuple[str, str, str]]:
    rows, bucket_name, prefix, rate_col, expect_sr, project = args
    client = storage.Client(project=project) if project else storage.Client()
    bucket = client.bucket(bucket_name)
    out: list[tuple[str, str, str]] = []
    for r in rows:
        cid = r["id"]
        rel = str(r[rate_col]).strip()
        if not rel or rel.lower() == "nan":
            out.append((cid, "no_path", rate_col))
            continue
        blob = bucket.blob(f"{prefix}/{rel}")
        try:
            raw = blob.download_as_bytes()
        except Exception as exc:
            out.append((cid, "missing", type(exc).__name__))
            continue
        try:
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception as exc:
            out.append((cid, "decode_error", str(exc)[:60]))
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != expect_sr:
            out.append((cid, "bad_sr", str(sr)))
        elif audio.size < expect_sr // 10:  # < 0.1 s
            out.append((cid, "too_short", str(audio.size)))
        elif not np.all(np.isfinite(audio)):
            out.append((cid, "nonfinite", ""))
        elif not np.any(audio):
            out.append((cid, "all_zero", ""))
        else:
            out.append((cid, "ok", ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--rate-col", default="32khz_path")
    ap.add_argument("--expect-sr", type=int, default=32000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="wavcaps_audio_validation_failures.csv")
    args = ap.parse_args()

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or None
    cb, ck = _split_gs(args.csv)
    raw = storage.Client(project=project).bucket(cb).blob(ck).download_as_bytes()
    df = pd.read_csv(io.BytesIO(raw), usecols=["id", args.rate_col], keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)
    records = df.to_dict("records")
    print(f"validating {len(records)} clips, rate_col={args.rate_col}, expect_sr={args.expect_sr}")

    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or os.cpu_count() or 8
    batches = [
        (records[i : i + args.batch_size], AUDIOSET_BUCKET, AUDIOSET_PREFIX,
         args.rate_col, args.expect_sr, project)
        for i in range(0, len(records), args.batch_size)
    ]
    print(f"{len(batches)} batches, {workers} workers")

    results: list[tuple[str, str, str]] = []
    t0 = time.time()
    with multiprocessing.Pool(processes=workers) as pool:
        for i, part in enumerate(pool.imap_unordered(validate_batch, batches)):
            results.extend(part)
            if (i + 1) % 50 == 0 or (i + 1) == len(batches):
                done = len(results)
                ok = sum(1 for _, s, _ in results if s == "ok")
                print(f"  [{i + 1}/{len(batches)}] checked={done} ok={ok} "
                      f"elapsed={(time.time() - t0) / 60:.1f}min", flush=True)

    status = pd.Series([s for _, s, _ in results])
    print("\n=== status breakdown ===")
    print(status.value_counts().to_string())
    n_ok = int((status == "ok").sum())
    print(f"\nTOTAL {len(results)} | ok {n_ok} ({100 * n_ok / len(results):.2f}%) | "
          f"failed {len(results) - n_ok}")

    fails = pd.DataFrame(
        [(c, s, d) for c, s, d in results if s != "ok"], columns=["id", "status", "detail"]
    )
    if len(fails):
        fails.to_csv(args.out, index=False)
        print(f"failures -> {args.out} ({len(fails)})")
        print(fails["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
