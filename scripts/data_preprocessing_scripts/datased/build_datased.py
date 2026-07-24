"""Build DataSED (Zenodo 15346092) into esp-data WABAD-shaped form.

DataSED — Dataset for Sound Event Detection of environmental noise
(Fredianelli et al. 2025, CC-BY-NC-SA-4.0). 717 mono WAV recordings at
44.1 kHz with strong (time-localised) event annotations in two schemes:

- ``Monophonic_sound_detection.csv`` — 4,309 events / 717 recordings /
  22 classes (non-overlapping).
- ``Polyphonic_sound_detection.csv`` — 4,034 events / 703 recordings /
  21 classes (overlapping events; excludes ``Wind turbine``).

Each source row is one event: ``sound_name, class_name, start_perc,
end_perc, start_time, end_time, event_length``. ``start_time`` /
``end_time`` are absolute seconds and map directly to Raven begin / end.

Two independent stages:

1. ``manifests`` (metadata only, light — runs anywhere):
   Pivot each scheme's CSV into a WABAD-shaped manifest with one row per
   recording and an inline ``selection_table`` TSV (``Selection``,
   ``Begin Time (s)``, ``End Time (s)``, ``Label``). Emits per-scheme
   ``all`` / ``train`` / ``val`` manifests (recording-level 90/10 split,
   shared across schemes so a recording never crosses splits) plus
   ``datased_labels.csv``.

2. ``resample`` (heavy — run on Slurm, NOT the dev VM):
   Read every original 44.1 kHz WAV and write pre-resampled 16 kHz and
   32 kHz mono mirrors (librosa ``kaiser_best``) into ``audio_16k/`` and
   ``audio_32k/`` for direct loading at train time.

GCS layout produced (uploads handled by the companion ``.sh``)::

    gs://esp-data-ingestion/datased/v0.1.0/
        datased_mono_all.csv  datased_mono_train.csv  datased_mono_val.csv
        datased_poly_all.csv  datased_poly_train.csv  datased_poly_val.csv
        datased_labels.csv
        audio/S-0001.wav ...          (original 44.1 kHz)
        audio_16k/S-0001.wav ...      (pre-resampled 16 kHz)
        audio_32k/S-0001.wav ...      (pre-resampled 32 kHz)

Usage::

    # metadata (fast)
    uv run --with pandas python .../build_datased.py manifests \
        --csv-dir <work>/csv --out-dir <stage>/manifests
    # audio (heavy, Slurm)
    uv run python .../build_datased.py resample \
        --audio-dir <work>/audio --out-root <work> --workers 16
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SCHEMES = {"mono": "Monophonic_sound_detection.csv", "poly": "Polyphonic_sound_detection.csv"}
SPLIT_SEED = 42
VAL_FRACTION = 0.10
LICENSE = "CC-BY-NC-SA-4.0"
SOURCE_DATASET = "datased"

# Raven-style selection-table columns (matches audioset_strong; DataSED has
# no frequency bounds so only the time + label columns are emitted).
ST_COLUMNS = ["Selection", "Begin Time (s)", "End Time (s)", "Label"]


def _selection_tsv(group: pd.DataFrame) -> str:
    """Serialise one recording's events into a Raven-style TSV blob.

    Parameters
    ----------
    group : pd.DataFrame
        Event rows for a single recording (source columns).

    Returns
    -------
    str
        Tab-separated selection table with :data:`ST_COLUMNS`.
    """
    g = group.sort_values("start_time").reset_index(drop=True)
    st = pd.DataFrame(
        {
            "Selection": range(1, len(g) + 1),
            "Begin Time (s)": g["start_time"].astype(float).round(4),
            "End Time (s)": g["end_time"].astype(float).round(4),
            "Label": g["class_name"].astype(str),
        }
    )[ST_COLUMNS]
    return st.to_csv(sep="\t", index=False)


def _recording_splits(sound_names: list[str]) -> dict[str, str]:
    """Assign each recording deterministically to ``train`` or ``val``.

    Parameters
    ----------
    sound_names : list[str]
        Every recording basename across all schemes.

    Returns
    -------
    dict[str, str]
        Mapping ``sound_name -> {"train","val"}`` (shared across schemes
        so a recording never appears in both a mono and poly split of
        different kinds).
    """
    names = sorted(set(sound_names))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(names)
    n_val = round(len(names) * VAL_FRACTION)
    val = set(names[:n_val])
    return {n: ("val" if n in val else "train") for n in names}


def _build_scheme_manifest(csv_path: Path, split_map: dict[str, str]) -> pd.DataFrame:
    """Pivot one scheme's event CSV into a per-recording manifest.

    Parameters
    ----------
    csv_path : Path
        Path to ``Monophonic``/``Polyphonic``_sound_detection.csv.
    split_map : dict[str, str]
        Recording-level train/val assignment.

    Returns
    -------
    pd.DataFrame
        One row per recording with an inline ``selection_table`` blob.
    """
    events = pd.read_csv(csv_path)
    rows = []
    for sound_name, group in events.groupby("sound_name", sort=True):
        stem = str(sound_name)
        rows.append(
            {
                "sound_name": stem,
                "audio_fp": f"audio/{stem}",
                "16khz_path": f"audio_16k/{stem}",
                "32khz_path": f"audio_32k/{stem}",
                "n_events": len(group),
                "split": split_map[stem],
                "source_dataset": SOURCE_DATASET,
                "license": LICENSE,
                "selection_table": _selection_tsv(group),
            }
        )
    head = ["sound_name", "audio_fp", "16khz_path", "32khz_path", "split", "n_events"]
    df = pd.DataFrame(rows)
    return df[head + [c for c in df.columns if c not in head]]


def build_manifests(csv_dir: Path, out_dir: Path) -> None:
    """Build mono/poly all/train/val manifests + labels csv.

    Parameters
    ----------
    csv_dir : Path
        Directory holding the two source CSVs.
    out_dir : Path
        Directory to write the manifests into.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Union of recordings across schemes for a single shared split.
    all_names: list[str] = []
    for fname in SCHEMES.values():
        all_names += pd.read_csv(csv_dir / fname)["sound_name"].astype(str).tolist()
    split_map = _recording_splits(all_names)

    label_set: set[str] = set()
    for scheme, fname in SCHEMES.items():
        df = _build_scheme_manifest(csv_dir / fname, split_map)
        label_set.update(pd.read_csv(csv_dir / fname)["class_name"].astype(str).unique())
        for split in ("all", "train", "val"):
            sub = df if split == "all" else df[df["split"] == split]
            out = out_dir / f"datased_{scheme}_{split}.csv"
            sub.to_csv(out, index=False)
            print(
                f"  {scheme}/{split}: {len(sub):,} recordings, "
                f"{int(sub['n_events'].sum()):,} events -> {out.name}"
            )

    labels = pd.DataFrame({"Label": sorted(label_set)})
    labels["label_snake"] = (
        labels["Label"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )
    labels_out = out_dir / "datased_labels.csv"
    labels.to_csv(labels_out, index=False)
    print(f"  labels: {len(labels)} classes -> {labels_out.name}")


def _resample_one(args: tuple[str, str, str, int]) -> tuple[str, str]:
    """Resample one WAV to a target rate and write it as mono PCM_16.

    Parameters
    ----------
    args : tuple[str, str, str, int]
        ``(src_path, dst_path, _, target_sr)``.

    Returns
    -------
    tuple[str, str]
        ``(dst_path, status)`` where status is ``"ok"`` or an error string.
    """
    import librosa
    import numpy as np
    import soundfile as sf

    src, dst, _, target_sr = args
    try:
        audio, sr = sf.read(src, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(
                y=audio, orig_sr=sr, target_sr=target_sr, scale=True, res_type="kaiser_best"
            )
        audio = np.clip(audio, -1.0, 1.0)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, audio, target_sr, subtype="PCM_16")
        return dst, "ok"
    except Exception as exc:  # noqa: BLE001 - report and continue
        return dst, f"ERROR: {exc}"


def resample(audio_dir: Path, out_root: Path, workers: int) -> None:
    """Write 16 kHz and 32 kHz mirrors of every WAV in ``audio_dir``.

    Parameters
    ----------
    audio_dir : Path
        Directory of original 44.1 kHz WAVs (flat ``S-*.wav``).
    out_root : Path
        Root under which ``audio_16k/`` and ``audio_32k/`` are written.
    workers : int
        Process-pool size.
    """
    wavs = sorted(audio_dir.glob("*.wav"))
    print(f"resampling {len(wavs)} files with {workers} workers ...", flush=True)
    jobs: list[tuple[str, str, str, int]] = []
    for sr in (16000, 32000):
        out_dir = out_root / f"audio_{sr // 1000}k"
        out_dir.mkdir(parents=True, exist_ok=True)
        for w in wavs:
            jobs.append((str(w), str(out_dir / w.name), "", sr))

    errors = 0
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_resample_one, j) for j in jobs]
        for fut in as_completed(futures):
            dst, status = fut.result()
            done += 1
            if status != "ok":
                errors += 1
                print(f"  {status}  ({dst})", flush=True)
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} ...", flush=True)
    print(f"done: {done} written, {errors} errors", flush=True)
    if errors:
        raise SystemExit(f"{errors} files failed to resample")


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)

    pm = sub.add_parser("manifests", help="build per-scheme manifests + labels csv")
    pm.add_argument("--csv-dir", type=Path, required=True, help="dir with the two source CSVs")
    pm.add_argument("--out-dir", type=Path, required=True, help="dir to write manifests into")

    pr = sub.add_parser("resample", help="write 16k + 32k mirrors (heavy; run on Slurm)")
    pr.add_argument("--audio-dir", type=Path, required=True, help="dir of original 44.1 kHz WAVs")
    pr.add_argument("--out-root", type=Path, required=True, help="root for audio_16k/ audio_32k/")
    pr.add_argument("--workers", type=int, default=16)

    args = p.parse_args()
    if args.stage == "manifests":
        build_manifests(args.csv_dir, args.out_dir)
    elif args.stage == "resample":
        resample(args.audio_dir, args.out_root, args.workers)


if __name__ == "__main__":
    main()
