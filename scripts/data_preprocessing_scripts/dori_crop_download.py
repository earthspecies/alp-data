# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "soundfile",
#     "librosa",
#     "resampy",
#     "huggingface_hub",
#     "google-cloud-storage",
# ]
# ///
"""Crop-on-download the DORI Phase-1 audio windows and upload to GCS.

Reads the manifest (one row per labeled segment / negative), and for each clip:
streams the source recording from its HF repo, reads only the labeled window
(``segment_start..segment_end``) via a soundfile seek (no full decode), and
writes three artifacts to gs://esp-data-ingestion/dori/v0.1.0/:

    recordings/<source>/<clip_id>.flac   (window at source sample rate)
    audio_16k/<source>/<clip_id>.wav     (16 kHz PCM16)
    audio_32k/<source>/<clip_id>.wav     (32 kHz PCM16)

The full source file (often ~32 MB for ONC) is never persisted — only the
~15 s window is kept. Runs on a Slurm node (needs HF internet + GCS).
"""

from __future__ import annotations

import argparse
import io
import multiprocessing
import os
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from google.cloud import storage
from huggingface_hub import HfFileSystem

GCS_BUCKET = "esp-data-ingestion"
GCS_ROOT = "dori/v0.1.0"
MIN_WIN = 1.0  # seconds


def _crop_window(raw: bytes, start: float, end: float) -> tuple[np.ndarray, int]:
    with sf.SoundFile(io.BytesIO(raw)) as f:
        sr = f.samplerate
        dur = len(f) / sr
        s = max(0.0, min(start, dur))
        e = max(s + MIN_WIN, min(end if end > start else dur, dur))
        if e - s < MIN_WIN:  # tiny/degenerate window -> take whole file
            s, e = 0.0, dur
        f.seek(int(s * sr))
        win = f.read(int((e - s) * sr), dtype="float32", always_2d=False)
    if win.ndim > 1:
        win = win.mean(axis=1)
    return win.astype(np.float32), sr


def _wav_bytes(audio: np.ndarray, sr: int) -> io.BytesIO:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


def process_batch(args: tuple) -> list[tuple[str, str]]:
    rows, project, skip = args
    fs = HfFileSystem()
    client = storage.Client(project=project) if project else storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    out: list[tuple[str, str]] = []
    for r in rows:
        cid, src = r["clip_id"], r["source"]
        k32 = f"{GCS_ROOT}/audio_32k/{src}/{cid}.wav"
        if skip:
            try:
                if bucket.blob(k32).exists():
                    out.append((cid, "exists"))
                    continue
            except Exception:
                pass  # transient GCS/cert blip — just reprocess this clip
        hf_path = f"datasets/{r['repo_id']}/{r['repo_path']}"
        raw = None
        for attempt in range(5):
            try:
                raw = fs.read_bytes(hf_path)
                break
            except Exception as exc:
                if attempt == 4:
                    out.append((cid, f"download_fail:{type(exc).__name__}"))
                time.sleep(2 * (attempt + 1))
        if raw is None:
            continue
        try:
            win, sr = _crop_window(raw, float(r["segment_start"]), float(r["segment_end"]))
        except Exception as exc:
            out.append((cid, f"decode_fail:{str(exc)[:40]}"))
            continue
        if win.size < int(MIN_WIN * sr):
            out.append((cid, "too_short"))
            continue
        try:
            # original-sr window as FLAC
            fbuf = io.BytesIO()
            sf.write(fbuf, win, sr, format="FLAC")
            fbuf.seek(0)
            bucket.blob(f"{GCS_ROOT}/recordings/{src}/{cid}.flac").upload_from_file(
                fbuf, content_type="audio/flac"
            )
            a16 = librosa.resample(win, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
            a32 = librosa.resample(win, orig_sr=sr, target_sr=32000, res_type="kaiser_best")
            bucket.blob(f"{GCS_ROOT}/audio_16k/{src}/{cid}.wav").upload_from_file(
                _wav_bytes(a16, 16000), content_type="audio/wav"
            )
            bucket.blob(k32).upload_from_file(_wav_bytes(a32, 32000), content_type="audio/wav")
            out.append((cid, "ok"))
        except Exception as exc:
            out.append((cid, f"upload_fail:{str(exc)[:40]}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    _default_manifest = os.path.expanduser("~/dori_staging/dori_phase1_manifest.csv")
    ap.add_argument("--manifest", default=_default_manifest)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or None
    df = pd.read_csv(args.manifest, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)
    records = df.to_dict("records")
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or os.cpu_count() or 8
    batches = [
        (records[i : i + args.batch_size], project, args.skip_existing)
        for i in range(0, len(records), args.batch_size)
    ]
    print(f"{len(records)} clips, {len(batches)} batches, {workers} workers", flush=True)

    results: list[tuple[str, str]] = []
    t0 = time.time()
    with multiprocessing.Pool(processes=workers) as pool:
        for i, part in enumerate(pool.imap_unordered(process_batch, batches)):
            results.extend(part)
            if (i + 1) % 25 == 0 or (i + 1) == len(batches):
                ok = sum(1 for _, s in results if s in ("ok", "exists"))
                print(
                    f"  [{i + 1}/{len(batches)}] done={len(results)} ok={ok} "
                    f"elapsed={(time.time() - t0) / 60:.1f}min",
                    flush=True,
                )

    st = pd.Series([s.split(":")[0] for _, s in results])
    print("\n=== status ===")
    print(st.value_counts().to_string())
    fails = [(c, s) for c, s in results if s not in ("ok", "exists")]
    if fails:
        # stdlib write wrapped in try/except: the /scratch uv env can be
        # evicted by end of a long run, breaking pandas' lazy imports.
        try:
            outp = os.path.expanduser("~/dori_staging/crop_failures.csv")
            with open(outp, "w") as fh:
                fh.write("clip_id,status\n")
                for c, s in fails:
                    fh.write(f"{c},{str(s).replace(',', ';')}\n")
            print(f"failures -> {outp} ({len(fails)})")
        except Exception as e:  # noqa: BLE001
            print(f"failures-write skipped: {e} ({len(fails)} failures)")


if __name__ == "__main__":
    main()
