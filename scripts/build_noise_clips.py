# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "soundfile",
#     "google-cloud-storage",
# ]
# ///
"""Extract inter-call "noise" windows (no annotated vocalisations) from an
esp-data detection dataset and upload them as 32 kHz WAV clips.

For each recording in the dataset's ``all.csv``, the embedded selection table
gives the time intervals of every annotated event. We take the complement
(gaps with no events), shrink each gap by a buffer on both sides (to stay clear
of call onsets/tails that may extend past the annotation box), and tile each
remaining gap into non-overlapping windows of ``--min-dur``..``--max-dur`` s.
Each window is sliced from the pre-resampled 32 kHz audio and uploaded to
``gs://foundation-model-data/audio_32k/noise/<dataset>/``.

NOTE: the source annotations are not exhaustive, so a clip is guaranteed free
of *annotated* calls only. ``build_noise_clips_review.py`` renders spectrograms
of a random sample for visual verification. RMS/peak per clip are recorded in
the manifest for transparency (no energy filtering — ambient noise is kept).

Heavy IO (downloads every 32 kHz recording once); run on a Slurm cpu node.
"""

from __future__ import annotations

import argparse
import io
import multiprocessing
import os
import time
from io import StringIO

import numpy as np
import pandas as pd
import soundfile as sf
from google.cloud import storage

DEST_BASE = "gs://foundation-model-data/audio_32k/noise"
_ESP = "gs://esp-data-ingestion"
DATASETS = {
    "dartmouth_avian_soundscapes": {
        "csv": f"{_ESP}/dartmouth-avian-soundscapes/v0.1.0/all.csv",
        "root": f"{_ESP}/dartmouth-avian-soundscapes/v0.1.0",
    },
    "pteroset": {
        "csv": f"{_ESP}/pteroset/v0.1.0/all.csv",
        "root": f"{_ESP}/pteroset/v0.1.0",
    },
}

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
    rest = uri[len("gs://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/touching [begin, end] intervals.

    Returns
    -------
    list[tuple[float, float]]
        Sorted, non-overlapping merged intervals.
    """
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for b, e in intervals[1:]:
        if b <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([b, e])
    return [(b, e) for b, e in merged]


def noise_windows(
    events: list[tuple[float, float]],
    duration: float,
    buffer: float,
    min_dur: float,
    max_dur: float,
) -> list[tuple[float, float]]:
    """Tile inter-event gaps into non-overlapping [min_dur, max_dur] windows.

    Each gap (complement of merged events within [0, duration]) is shrunk by
    ``buffer`` on both sides before tiling.

    Returns
    -------
    list[tuple[float, float]]
        (start, end) windows in seconds, none overlapping any event +/- buffer.
    """
    merged = merge_intervals([(max(0.0, b), min(duration, e)) for b, e in events if e > b])
    windows: list[tuple[float, float]] = []
    prev_end = 0.0
    bounds = [*merged, (duration, duration)]
    for b, e in bounds:
        g0, g1 = prev_end + buffer, b - buffer
        t = g0
        while g1 - t >= min_dur:
            w = min(max_dur, g1 - t)
            windows.append((t, t + w))
            t += w
        prev_end = e
    return windows


def process_batch(args: tuple) -> list[dict]:
    rows, audio_root, dest, project, min_dur, max_dur, buffer, skip = args
    client = storage.Client(project=project) if project else storage.Client()
    root_bucket, root_prefix = _split_gs(audio_root)
    dst_bucket_name, dst_prefix = _split_gs(dest)
    src_bucket = client.bucket(root_bucket)
    dst_bucket = client.bucket(dst_bucket_name)

    manifest: list[dict] = []
    for r in rows:
        fn = r["fn"]
        rel = r["32khz_path"]
        key = f"{root_prefix}/{rel}" if root_prefix else rel
        try:
            raw = src_bucket.blob(key).download_as_bytes()
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception as exc:
            print(f"  ERROR read {fn}: {exc}", flush=True)
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / float(sr)

        events: list[tuple[float, float]] = []
        st_raw = r.get("selection_table", "")
        if isinstance(st_raw, str) and st_raw.strip():
            try:
                st = pd.read_csv(StringIO(st_raw), sep="\t")
                if {"Begin Time (s)", "End Time (s)"} <= set(st.columns):
                    events = list(
                        zip(
                            st["Begin Time (s)"].astype(float),
                            st["End Time (s)"].astype(float),
                            strict=False,
                        )
                    )
            except Exception as exc:
                print(f"  WARN st parse {fn}: {exc}", flush=True)

        for t0, t1 in noise_windows(events, dur, buffer, min_dur, max_dur):
            a, b = int(round(t0 * sr)), int(round(t1 * sr))
            clip = audio[a:b]
            if clip.size < int(min_dur * sr):
                continue
            name = f"{fn}_noise_{int(round(t0 * 1000))}_{int(round(t1 * 1000))}.wav"
            dst_key = f"{dst_prefix}/{name}" if dst_prefix else name
            if skip and dst_bucket.blob(dst_key).exists():
                continue
            buf = io.BytesIO()
            sf.write(buf, clip, sr, format="WAV", subtype="PCM_16")
            buf.seek(0)
            dst_bucket.blob(dst_key).upload_from_file(buf, content_type="audio/wav")
            rms = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2))) if clip.size else 0.0
            manifest.append(
                {
                    "clip": name,
                    "source_fn": fn,
                    "start_s": round(t0, 3),
                    "end_s": round(t1, 3),
                    "dur_s": round(t1 - t0, 3),
                    "sr": sr,
                    "rms": round(rms, 6),
                    "peak": round(float(np.max(np.abs(clip))) if clip.size else 0.0, 6),
                }
            )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--min-dur", type=float, default=2.0)
    parser.add_argument("--max-dur", type=float, default=10.0)
    parser.add_argument("--buffer", type=float, default=1.0, help="seconds clear of any event")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    dest = f"{DEST_BASE}/{args.dataset}"
    project = _resolve_project()
    print(f"dataset={args.dataset} dest={dest} project={project}", flush=True)

    csv_bucket, csv_key = _split_gs(cfg["csv"])
    raw = _client().bucket(csv_bucket).blob(csv_key).download_as_bytes()
    df = pd.read_csv(io.BytesIO(raw), keep_default_na=False, na_values=[""])
    if args.limit:
        df = df.head(args.limit)
    records = df[["fn", "32khz_path", "selection_table"]].to_dict("records")
    print(f"recordings: {len(records)}", flush=True)

    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or os.cpu_count() or 8
    batches = [
        (
            records[i : i + args.batch_size],
            cfg["root"],
            dest,
            project,
            args.min_dur,
            args.max_dur,
            args.buffer,
            args.skip_existing,
        )
        for i in range(0, len(records), args.batch_size)
    ]
    print(f"{len(batches)} batches, {workers} workers", flush=True)

    manifest: list[dict] = []
    t0 = time.time()
    with multiprocessing.Pool(processes=workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_batch, batches)):
            manifest.extend(rows)
            if (i + 1) % 10 == 0 or (i + 1) == len(batches):
                print(
                    f"  [{i + 1}/{len(batches)}] clips={len(manifest)} "
                    f"elapsed={ (time.time() - t0) / 60:.1f}min",
                    flush=True,
                )

    mdf = pd.DataFrame(manifest)
    total_dur = float(mdf["dur_s"].sum()) if len(mdf) else 0.0
    print(
        f"\nDONE {args.dataset}: {len(mdf)} clips from "
        f"{mdf['source_fn'].nunique() if len(mdf) else 0} recordings, "
        f"{total_dur / 3600:.2f} h total",
        flush=True,
    )
    out = f"noise_manifest_{args.dataset}.csv"
    mdf.to_csv(out, index=False)
    os.system(f"gsutil -q cp {out} {dest}/{out}")
    print(f"manifest -> {dest}/{out}", flush=True)


if __name__ == "__main__":
    main()
