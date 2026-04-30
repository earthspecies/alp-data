# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "librosa",
#     "soxr",
#     "soundfile",
#     "google-cloud-storage",
# ]
# ///
"""Resample Silent Cities v1+avex files from 48kHz FLAC → 16kHz WAV.

Reads a manifest of basenames, downloads each source FLAC from
gs://fewshot/data_large_clean/silentcities_sampled_big/, resamples to
16kHz with librosa kaiser_best, and uploads as WAV to
gs://fewshot/data_large_clean/silentcities_sampled_big_filtered_cropped_16kHz_v1_plus_avex/.

Uses multiprocessing for CPU-bound resampling and threading within each
worker for overlapping I/O with compute.
"""

from __future__ import annotations

import argparse
import io
import multiprocessing
import os
import time

import librosa
import numpy as np
import soundfile as sf
from google.cloud import storage

TARGET_SR = 16000
MAX_DURATION_S = 10.0
BUCKET_NAME = "fewshot"
SOURCE_PREFIX = "data_large_clean/silentcities_sampled_big/"
DEST_PREFIX = "data_large_clean/silentcities_sampled_big_filtered_cropped_16kHz_v1_plus_avex/"


def process_batch(batch: list[str]) -> tuple[int, int]:
    """Process a batch of basenames in a single worker process."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    dest_prefix = DEST_PREFIX.rstrip("/") + "/"
    done = 0
    errors = 0

    for basename in batch:
        try:
            src_blob = bucket.blob(SOURCE_PREFIX + basename + ".flac")
            flac_bytes = src_blob.download_as_bytes()

            audio, sr = sf.read(io.BytesIO(flac_bytes), dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]

            max_samples = int(MAX_DURATION_S * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            if sr != TARGET_SR:
                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=TARGET_SR,
                    res_type="kaiser_best", scale=True, fix=True,
                )

            max_out = int(MAX_DURATION_S * TARGET_SR)
            if len(audio) > max_out:
                audio = audio[:max_out]

            buf = io.BytesIO()
            sf.write(buf, audio, TARGET_SR, format="WAV", subtype="PCM_16")
            buf.seek(0)

            dest_blob = bucket.blob(dest_prefix + basename + ".wav")
            dest_blob.upload_from_file(buf, content_type="audio/wav")
            done += 1
        except Exception as exc:
            errors += 1
            print(f"  ERROR {basename}: {exc}", flush=True)

    return done, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Text file with one basename per line (no extension)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Files per worker batch (default: 50)")
    args = parser.parse_args()

    workers = args.workers or os.cpu_count() or 8

    with open(args.manifest) as f:
        all_names = [line.strip() for line in f if line.strip()]

    print(f"Manifest: {len(all_names)} files")
    print(f"Workers: {workers}, batch size: {args.batch_size}")
    print(f"Dest: gs://{BUCKET_NAME}/{DEST_PREFIX}")

    batches = [
        all_names[i : i + args.batch_size]
        for i in range(0, len(all_names), args.batch_size)
    ]
    print(f"Split into {len(batches)} batches")

    t0 = time.time()
    total_done = 0
    total_errors = 0

    with multiprocessing.Pool(processes=workers) as pool:
        for i, (done, errs) in enumerate(pool.imap_unordered(process_batch, batches)):
            total_done += done
            total_errors += errs
            elapsed = time.time() - t0
            rate = total_done / max(elapsed, 1)
            remaining = (len(all_names) - total_done - total_errors) / max(rate, 0.01)
            if (i + 1) % 10 == 0 or (i + 1) == len(batches):
                print(
                    f"  [{total_done + total_errors}/{len(all_names)}] "
                    f"done={total_done} err={total_errors} "
                    f"rate={rate:.1f}/s ETA={remaining/60:.0f}min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed/60:.1f} min. processed={total_done} errors={total_errors}")


if __name__ == "__main__":
    main()
