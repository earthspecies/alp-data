# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "librosa",
#     "resampy",
#     "soundfile",
#     "google-cloud-storage",
# ]
# ///
"""Resample a flat GCS audio directory to 32kHz mono WAV.

Lists every audio object under ``--source-prefix`` (a ``gs://bucket/path/``
URI), downloads each blob, resamples to 32kHz with librosa kaiser_best,
and uploads the result as a 16-bit PCM WAV to ``--dest-prefix``.

Heavy I/O and listing happen inside the job (the script is intended to be
launched on a Slurm CPU node), never on the login VM.
"""

from __future__ import annotations

import argparse
import io
import multiprocessing
import os
import time
from pathlib import PurePosixPath

import librosa
import soundfile as sf
from google.cloud import storage

TARGET_SR = 32000
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".aac", ".m4a", ".aiff", ".aif")

_CLIENT: storage.Client | None = None


def _resolve_project() -> str | None:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        return project
    try:
        import google.auth

        _, detected = google.auth.default()
        if detected:
            os.environ["GOOGLE_CLOUD_PROJECT"] = detected
            return detected
    except Exception:
        pass
    return None


def _client() -> storage.Client:
    global _CLIENT
    if _CLIENT is None:
        project = _resolve_project()
        _CLIENT = storage.Client(project=project) if project else storage.Client()
    return _CLIENT


def _split_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    rest = uri[len("gs://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def list_source_blobs(source_uri: str) -> list[str]:
    """List object names (keys) under ``source_uri`` that look like audio.

    Parameters
    ----------
    source_uri : str
        A ``gs://bucket/path/`` prefix to enumerate.

    Returns
    -------
    list[str]
        Object keys (without the ``gs://bucket/`` prefix) whose names
        end with a supported audio extension.
    """
    bucket_name, prefix = _split_gs(source_uri)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    bucket = _client().bucket(bucket_name)
    keys: list[str] = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        if blob.name.lower().endswith(AUDIO_EXTS):
            keys.append(blob.name)
    return keys


def existing_dest_stems(dest_uri: str) -> set[str]:
    bucket_name, prefix = _split_gs(dest_uri)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    bucket = _client().bucket(bucket_name)
    stems: set[str] = set()
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix) :]
        stems.add(str(PurePosixPath(rel).with_suffix("")))
    return stems


def _dest_key(src_key: str, src_prefix: str, dst_prefix: str) -> str:
    rel = src_key[len(src_prefix) :]
    rel_wav = str(PurePosixPath(rel).with_suffix(".wav"))
    return dst_prefix + rel_wav


def process_batch(args: tuple[list[str], str, str, str, str, str | None]) -> tuple[int, int]:
    src_keys, src_bucket_name, src_prefix, dst_bucket_name, dst_prefix, project = args
    client = storage.Client(project=project) if project else storage.Client()
    src_bucket = client.bucket(src_bucket_name)
    dst_bucket = client.bucket(dst_bucket_name)
    done = 0
    errors = 0

    for key in src_keys:
        try:
            raw = src_bucket.blob(key).download_as_bytes()
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sr != TARGET_SR:
                audio = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=TARGET_SR,
                    res_type="kaiser_best",
                    scale=True,
                    fix=True,
                )

            buf = io.BytesIO()
            sf.write(buf, audio, TARGET_SR, format="WAV", subtype="PCM_16")
            buf.seek(0)

            dst_key = _dest_key(key, src_prefix, dst_prefix)
            dst_bucket.blob(dst_key).upload_from_file(buf, content_type="audio/wav")
            done += 1
        except Exception as exc:
            errors += 1
            print(f"  ERROR {key}: {exc}", flush=True)

    return done, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-prefix",
        required=True,
        help="gs://bucket/path/ source directory of audio files (flat or nested)",
    )
    parser.add_argument(
        "--dest-prefix",
        required=True,
        help="gs://bucket/path/ destination directory; mirrors source layout",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files already present at dest (by relative stem)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes (default: SLURM_CPUS_PER_TASK or cpu_count)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Files per worker batch (default: 50)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files (smoke testing)",
    )
    args = parser.parse_args()

    src_uri = args.source_prefix if args.source_prefix.endswith("/") else args.source_prefix + "/"
    dst_uri = args.dest_prefix if args.dest_prefix.endswith("/") else args.dest_prefix + "/"

    src_bucket_name, src_prefix = _split_gs(src_uri)
    dst_bucket_name, dst_prefix = _split_gs(dst_uri)
    project = _resolve_project()
    print(f"GCP project: {project}", flush=True)

    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or os.cpu_count() or 8

    print(f"Source: {src_uri}", flush=True)
    print(f"Dest:   {dst_uri}", flush=True)
    print(f"Workers: {workers}, batch size: {args.batch_size}", flush=True)

    t0 = time.time()
    print("Listing source ...", flush=True)
    src_keys = list_source_blobs(src_uri)
    print(f"  found {len(src_keys)} audio files in {time.time() - t0:.1f}s", flush=True)

    if args.skip_existing:
        t1 = time.time()
        print("Listing dest (for --skip-existing) ...", flush=True)
        existing = existing_dest_stems(dst_uri)
        print(f"  found {len(existing)} existing in {time.time() - t1:.1f}s", flush=True)

        def is_new(key: str) -> bool:
            rel = key[len(src_prefix) :]
            return str(PurePosixPath(rel).with_suffix("")) not in existing

        before = len(src_keys)
        src_keys = [k for k in src_keys if is_new(k)]
        print(
            f"  remaining after skip: {len(src_keys)} (dropped {before - len(src_keys)})",
            flush=True,
        )

    if args.limit is not None:
        src_keys = src_keys[: args.limit]
        print(f"  limited to {len(src_keys)} files", flush=True)

    if not src_keys:
        print("Nothing to do.", flush=True)
        return

    batches = [
        (
            src_keys[i : i + args.batch_size],
            src_bucket_name,
            src_prefix,
            dst_bucket_name,
            dst_prefix,
            project,
        )
        for i in range(0, len(src_keys), args.batch_size)
    ]
    print(f"Split into {len(batches)} batches", flush=True)

    total_done = 0
    total_errors = 0
    t0 = time.time()
    with multiprocessing.Pool(processes=workers) as pool:
        for i, (done, errs) in enumerate(pool.imap_unordered(process_batch, batches)):
            total_done += done
            total_errors += errs
            if (i + 1) % 10 == 0 or (i + 1) == len(batches):
                elapsed = time.time() - t0
                rate = total_done / max(elapsed, 1)
                remaining = (len(src_keys) - total_done - total_errors) / max(rate, 0.01)
                print(
                    f"  [{total_done + total_errors}/{len(src_keys)}] "
                    f"done={total_done} err={total_errors} "
                    f"rate={rate:.1f}/s ETA={remaining / 60:.0f}min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(
        f"\nFinished in {elapsed / 60:.1f} min. processed={total_done} errors={total_errors}",
        flush=True,
    )


if __name__ == "__main__":
    main()
