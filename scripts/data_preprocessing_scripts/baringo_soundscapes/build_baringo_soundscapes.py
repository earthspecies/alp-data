"""Build the western-Kenya (Baringo) soundscape dataset (Zenodo 10943500).

Kahl, Reers, Cherutich, Jacot, Klinck (2024), CC-BY-4.0. 35 ~1-hour soundscape
recordings (32 h total) from Baringo County, Kenya, with **centerpoint**
annotations — 10,294 bird-call labels, each marking the *center time* of a call
(``Start Time (s) == End Time (s)``), for 176 species (eBird codes). Audio is
32 kHz FLAC (Nyquist 16 kHz — a native fit for the stack). Ingested WABAD-shaped
so it flows through ``window_annotations``; the centers are stored faithfully
(zero-width events), so any center->box expansion is left to the consumer.

Note: partly used as 2023 BirdCLEF test data — ingested here as a single ``all``
split (intended as a held-out soundscape SED eval, not training).

Two stages (mirrors build_ndege_zetu.py):

1. ``resample`` — decode every FLAC (soundfile, falling back to librosa/ffmpeg
   because ~2/3 of the source FLACs trip a libsndfile decoder bug) and re-encode
   to clean 16 kHz + 32 kHz WAV mirrors, recording per-file duration to
   ``durations.csv``. Run on Slurm.
2. ``manifests`` — from ``durations.csv`` + ``annotations.csv`` + ``species.csv``
   write a single-``all``-split WABAD manifest with an ``audio_duration``
   column and a centerpoint ``selection_table``.

GCS layout::

    gs://esp-data-ingestion/baringo-soundscapes/v0.1.0/
        baringo_soundscapes_all.csv
        baringo_soundscapes_labels.csv
        audio_16k/<stem>.wav
        audio_32k/<stem>.wav
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

LICENSE = "CC-BY-4.0"
SOURCE_DATASET = "baringo_soundscapes"
SOURCE_NYQUIST_HZ = 16000  # 32 kHz source
ST_COLUMNS = [
    "Selection",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
    "eBird_Code",
]


def _selection_tsv(events: list[tuple[float, str, str]]) -> str:
    """Serialise centerpoint events into a WABAD-shaped Raven TSV blob.

    Parameters
    ----------
    events : list[tuple[float, str, str]]
        ``(center_time_sec, scientific_name, ebird_code)`` per call.

    Returns
    -------
    str
        Tab-separated selection table (:data:`ST_COLUMNS`); Begin == End ==
        center (zero-width centerpoint).
    """
    events = sorted(events, key=lambda e: e[0])
    df = pd.DataFrame(
        {
            "Selection": range(1, len(events) + 1),
            "Begin Time (s)": [round(c, 3) for c, _, _ in events],
            "End Time (s)": [round(c, 3) for c, _, _ in events],
            "Low Freq (Hz)": [0] * len(events),
            "High Freq (Hz)": [SOURCE_NYQUIST_HZ] * len(events),
            "Species": [sci for _, sci, _ in events],
            "eBird_Code": [code for _, _, code in events],
        }
    )[ST_COLUMNS]
    return df.to_csv(sep="\t", index=False)


def build_manifests(anno_csv: Path, species_csv: Path, durations_csv: Path, out_dir: Path) -> None:
    """Build the single-``all`` WABAD manifest + labels csv.

    Parameters
    ----------
    anno_csv : Path
        ``annotations.csv`` (Filename, Start/End Time (s), Species eBird Code).
    species_csv : Path
        ``species.csv`` (eBird code -> Scientific / Common name).
    durations_csv : Path
        ``durations.csv`` from the resample stage.
    out_dir : Path
        Destination directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(species_csv)
    code2sci = {str(r["Species eBird Code"]): str(r["Scientific Name"]) for _, r in sp.iterrows()}
    anno = pd.read_csv(anno_csv)
    durs = pd.read_csv(durations_csv)
    dur_map = {str(r["basename"]): float(r["duration_sec"]) for _, r in durs.iterrows()}

    events_per_file: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    labels: set[str] = set()
    for _, r in anno.iterrows():
        fn = str(r["Filename"]).strip()
        code = str(r["Species eBird Code"]).strip()
        sci = code2sci.get(code, code)
        center = float(r["Start Time (s)"])
        events_per_file[fn].append((center, sci, code))
        labels.add(sci)

    rows = []
    n_missing_audio = 0
    for base, dur in dur_map.items():
        evs = events_per_file.get(base, [])
        stem = Path(base).stem
        rows.append(
            {
                "sound_name": base,
                "split": "all",
                "audio_duration": round(dur, 3),
                "audio_fp": f"audio_32k/{stem}.wav",  # clean re-encoded 32 kHz
                "16khz_path": f"audio_16k/{stem}.wav",
                "32khz_path": f"audio_32k/{stem}.wav",
                "n_events": len(evs),
                "n_species": len({e[2] for e in evs}),
                "source_dataset": SOURCE_DATASET,
                "license": LICENSE,
                "selection_table": _selection_tsv(evs),
            }
        )
    # Annotation filenames without matching audio (should be none).
    for fn in events_per_file:
        if fn not in dur_map:
            n_missing_audio += 1
            print(f"  WARNING: annotations for missing audio {fn}")

    df = pd.DataFrame(rows)
    head = ["sound_name", "split", "audio_duration", "audio_fp",
            "16khz_path", "32khz_path", "n_events", "n_species"]
    df = df[head + [c for c in df.columns if c not in head]]
    out = out_dir / "baringo_soundscapes_all.csv"
    df.to_csv(out, index=False)
    print(f"  all: {len(df)} recordings, {int(df['n_events'].sum())} centerpoints, "
          f"{len(labels)} species, {n_missing_audio} missing-audio -> {out.name}")

    sp_ren = sp.rename(columns={"Scientific Name": "Species"})[
        ["Species", "Species eBird Code", "Common Name"]
    ]
    lab = pd.merge(pd.DataFrame({"Species": sorted(labels)}), sp_ren, on="Species", how="left")
    lab.to_csv(out_dir / "baringo_soundscapes_labels.csv", index=False)
    print(f"  labels: {lab['Species'].nunique()} species -> baringo_soundscapes_labels.csv")


def _resample_one(args: tuple[str, str]) -> tuple[str, float, str]:
    """Re-encode one recording to clean 16 kHz + 32 kHz WAV mirrors.

    The source FLACs trip a libsndfile decoder bug (``LibsndfileError`` on
    ``sf.read`` for ~2/3 of files), and ``esp_data.read_audio`` uses soundfile
    too — so we decode with a soundfile->librosa(ffmpeg) fallback and re-encode
    to WAV (which libsndfile reads reliably) rather than serving the raw FLAC.

    Parameters
    ----------
    args : tuple[str, str]
        ``(src_flac, out_root)``.

    Returns
    -------
    tuple[str, float, str]
        ``(basename, duration_sec, status)``.
    """
    import librosa
    import numpy as np
    import soundfile as sf

    src, out_root = args
    stem = Path(src).stem
    try:
        try:
            audio, sr = sf.read(src, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except Exception:
            # libsndfile FLAC decode failure -> ffmpeg/audioread via librosa.
            audio, sr = librosa.load(src, sr=None, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        dur = len(audio) / float(sr)
        for tgt in (16000, 32000):
            y = audio if sr == tgt else librosa.resample(
                y=audio, orig_sr=sr, target_sr=tgt, scale=True, res_type="kaiser_best"
            )
            out = Path(out_root) / f"audio_{tgt // 1000}k" / f"{stem}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, np.clip(y, -1.0, 1.0), tgt, subtype="PCM_16")
        return Path(src).name, dur, "ok"
    except Exception as exc:  # noqa: BLE001
        return Path(src).name, 0.0, f"ERROR: {exc}"


def resample(audio_root: Path, out_root: Path, workers: int) -> None:
    """Write 16 kHz FLAC mirrors for every FLAC and emit ``durations.csv``.

    Parameters
    ----------
    audio_root : Path
        Directory tree of source 32 kHz FLACs.
    out_root : Path
        Destination root for ``audio/`` ``audio_16k/`` and ``durations.csv``.
    workers : int
        Process-pool size.

    Raises
    ------
    SystemExit
        If any file fails to decode/resample.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    flacs = sorted(audio_root.rglob("*.flac"))
    print(f"resampling {len(flacs)} flacs with {workers} workers ...", flush=True)
    durations: list[tuple[str, float]] = []
    errors = done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_resample_one, (str(f), str(out_root))) for f in flacs]):
            base, dur, status = fut.result()
            done += 1
            if status != "ok":
                errors += 1
                print(f"  {status}  ({base})", flush=True)
            else:
                durations.append((base, dur))
            if done % 10 == 0:
                print(f"  {done}/{len(flacs)} ...", flush=True)
    with open(out_root / "durations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["basename", "duration_sec"])
        w.writerows(sorted(durations))
    print(f"done: {done} written, {errors} errors, durations.csv={len(durations)}", flush=True)
    if errors:
        raise SystemExit(f"{errors} files failed")


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)

    pr = sub.add_parser("resample")
    pr.add_argument("--audio-root", type=Path, required=True)
    pr.add_argument("--out-root", type=Path, required=True)
    pr.add_argument("--workers", type=int, default=16)

    pm = sub.add_parser("manifests")
    pm.add_argument("--anno-csv", type=Path, required=True)
    pm.add_argument("--species-csv", type=Path, required=True)
    pm.add_argument("--durations-csv", type=Path, required=True)
    pm.add_argument("--out-dir", type=Path, required=True)

    args = p.parse_args()
    if args.stage == "resample":
        resample(args.audio_root, args.out_root, args.workers)
    else:
        build_manifests(args.anno_csv, args.species_csv, args.durations_csv, args.out_dir)


if __name__ == "__main__":
    main()
