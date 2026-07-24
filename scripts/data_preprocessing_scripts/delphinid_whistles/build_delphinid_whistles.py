"""Build the delphinid-whistle detection dataset (Dryad z34tmpgq6) into esp-data.

Ferguson et al. 2025 "Bounding-box detection data for delphinid whistles"
(CC0-1.0). Bottlenose dolphin (*Tursiops truncatus*) whistle detection with
time+frequency bounding boxes in Raven selection tables (``.txt``). Four sites:

  aquarium_imms          IMMS Gulfport      — merged multi-hour wav(s)
  aquarium_oceanografic  Oceanogràfic OF    — merged multi-hour wav(s)
  openocean_dclde2011    DCLDE 2011         — many native wavs, one .txt
  openocean_swfsc        NOAA SWFSC towed   — many native wavs, one .txt

All audio is 48 kHz mono; whistle boxes were annotated on a 0–24 kHz view.
The recording granularity is one **audio file** (a merged aquarium wav, or one
native open-ocean wav). Open-ocean selection tables are multi-file: each event
is assigned to its ``Begin File`` and timed by ``File Offset (s)`` (within-file
begin) + event duration. Aquarium tables have no ``Begin File`` — the merged
wav is the single recording and ``Begin/End Time (s)`` are used directly.
Open-ocean wavs with zero whistles are kept as pure-negative recordings.

Two stages (mirrors build_datased.py):

1. ``manifests`` — parse the extracted train/test trees into WABAD-shaped
   per-recording manifests (one row per wav + inline ``selection_table`` TSV of
   ``Selection, Begin Time (s), End Time (s), Low Freq (Hz), High Freq (Hz),
   Species``, Species="Tursiops truncatus"). Native train/test split preserved;
   a recording-level val holdout is carved from open-ocean training files.
2. ``resample`` — write 16 kHz + 32 kHz mono mirrors of every wav (+ copy the
   48 kHz original), flattened to ``<split_of_audio>/<site>/<basename>``.

GCS layout::

    gs://esp-data-ingestion/delphinid-whistles/v0.1.0/
        delphinid_whistles_{all,train,val,test}.csv
        delphinid_whistles_labels.csv
        audio/<site>/<basename>.wav        (48 kHz original)
        audio_16k/<site>/<basename>.wav
        audio_32k/<site>/<basename>.wav
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SPECIES = "Tursiops truncatus"
LICENSE = "CC0-1.0"
SOURCE_DATASET = "delphinid_whistles"
VAL_FRACTION = 0.15
SPLIT_SEED = 42
ST_COLUMNS = [
    "Selection",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
]


def site_slug(path: str) -> str:
    """Map any zip path/component to a canonical site slug.

    Returns
    -------
    str
        One of the four site slugs, or ``"unknown"``.
    """
    p = path.lower()
    if "imms" in p:
        return "aquarium_imms"
    if "oceanogr" in p or "valencia" in p:
        return "aquarium_oceanografic"
    if "dclde" in p:
        return "openocean_dclde2011"
    if "swfsc" in p or "towedarray" in p or "piceas" in p or "hiceas" in p:
        return "openocean_swfsc"
    return "unknown"


def _index_wavs(root: Path) -> dict[str, Path]:
    """Map wav basename -> absolute path for every wav under ``root``.

    Returns
    -------
    dict[str, Path]
        Basename (lower-cased) to path.
    """
    return {w.name.lower(): w for w in root.rglob("*.wav")}


def _read_raven(txt: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a Raven selection table (latin-1, tab-separated).

    Returns
    -------
    tuple[list[str], list[dict]]
        Column names and row dicts.
    """
    with open(txt, newline="", encoding="latin-1") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def _events_by_recording(
    txt: Path, wav_index: dict[str, Path]
) -> dict[str, list[dict[str, float]]]:
    """Assign each annotation row to its recording wav (by basename).

    Aquarium tables (no ``Begin File``) attach every event to the merged wav
    that shares the table's site+index. Open-ocean tables group by ``Begin
    File`` and use ``File Offset (s)`` for within-file timing.

    Returns
    -------
    dict[str, list[dict]]
        wav basename (lower) -> list of within-file event dicts
        (``begin, end, low, high``).

    Raises
    ------
    RuntimeError
        If a multi-file table lacks ``File Offset (s)`` (within-file times
        unresolvable), or a merged aquarium table cannot be mapped to a wav.
    """
    cols, rows = _read_raven(txt)
    has_begin_file = "Begin File" in cols and any(r.get("Begin File") for r in rows)
    out: dict[str, list[dict[str, float]]] = defaultdict(list)

    def _num(row: dict[str, str], key: str) -> float | None:
        v = row.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    if has_begin_file:
        # Within-file begin = File Offset (s) when present; otherwise fall back
        # to Begin Time (s), which is file-relative only for single-file tables
        # (e.g. a merged aquarium wav referenced via Begin File without an
        # offset column). Guard against multi-file tables missing offsets,
        # where Begin Time would be cumulative across files.
        has_offset = "File Offset (s)" in cols
        distinct_bf = {
            (r.get("Begin File") or "").strip().lower() for r in rows if r.get("Begin File")
        }
        if not has_offset and len(distinct_bf) > 1:
            raise RuntimeError(
                f"{txt.name}: multi-file table ({len(distinct_bf)} files) without "
                "'File Offset (s)' — cannot resolve within-file times"
            )
        for r in rows:
            bf = (r.get("Begin File") or "").strip().lower()
            if not bf or bf not in wav_index:
                continue
            bt, et = _num(r, "Begin Time (s)"), _num(r, "End Time (s)")
            lo, hi = _num(r, "Low Freq (Hz)"), _num(r, "High Freq (Hz)")
            if None in (bt, et, lo, hi):
                continue
            fo = _num(r, "File Offset (s)") if has_offset else None
            begin = fo if fo is not None else bt
            out[bf].append(
                {"begin": begin, "end": begin + max(0.0, et - bt), "low": lo, "high": hi}
            )
    else:
        # Merged aquarium recording: resolve the single/indexed wav in the
        # table's site by the two-digit index in the table filename.
        slug = site_slug(txt.name) if site_slug(txt.name) != "unknown" else site_slug(str(txt))
        cand = [n for n, p in wav_index.items() if site_slug(str(p)) == slug]
        wav = None
        if len(cand) == 1:
            wav = cand[0]
        else:
            m = re.search(r"(\d{2})", txt.name)
            if m:
                idx = m.group(1)
                wav = next((n for n in cand if re.search(rf"[_.]{idx}[_.]", n)
                            or re.search(rf"{idx}[_.]", n)), None)
        if wav is None:
            raise RuntimeError(f"could not map merged table {txt.name} to a wav (cands={cand})")
        for r in rows:
            bt, et = _num(r, "Begin Time (s)"), _num(r, "End Time (s)")
            lo, hi = _num(r, "Low Freq (Hz)"), _num(r, "High Freq (Hz)")
            if None in (bt, et, lo, hi):
                continue
            out[wav].append({"begin": bt, "end": et, "low": lo, "high": hi})
    return out


def _selection_tsv(events: list[dict[str, float]]) -> str:
    """Serialise within-file events into a WABAD-shaped Raven TSV blob.

    Returns
    -------
    str
        Tab-separated selection table (:data:`ST_COLUMNS`); header-only when
        the recording has no whistles (a pure-negative recording).
    """
    events = sorted(events, key=lambda e: e["begin"])
    df = pd.DataFrame(
        {
            "Selection": range(1, len(events) + 1),
            "Begin Time (s)": [round(e["begin"], 4) for e in events],
            "End Time (s)": [round(e["end"], 4) for e in events],
            "Low Freq (Hz)": [round(e["low"], 1) for e in events],
            "High Freq (Hz)": [round(e["high"], 1) for e in events],
            "Species": [SPECIES] * len(events),
        }
    )[ST_COLUMNS]
    return df.to_csv(sep="\t", index=False)


def _collect_recordings(root: Path, split_of_audio: str) -> dict[str, dict]:
    """Build per-recording rows from one extracted tree (train or test).

    Every wav becomes a recording (0-event wavs kept as negatives).

    Returns
    -------
    dict[str, dict]
        wav basename (lower) -> partial manifest row.
    """
    wav_index = _index_wavs(root)
    events_per_wav: dict[str, list[dict[str, float]]] = defaultdict(list)
    for txt in root.rglob("*.txt"):
        for wav_base, evs in _events_by_recording(txt, wav_index).items():
            events_per_wav[wav_base].extend(evs)

    recs: dict[str, dict] = {}
    for wav_base, wav_path in wav_index.items():
        slug = site_slug(str(wav_path))
        base = wav_path.name
        rel = f"{slug}/{base}"
        evs = events_per_wav.get(wav_base, [])
        recs[wav_base] = {
            "sound_name": base,
            "site": slug,
            "audio_fp": f"audio/{rel}",
            "16khz_path": f"audio_16k/{rel}",
            "32khz_path": f"audio_32k/{rel}",
            "n_events": len(evs),
            "audio_split": split_of_audio,
            "source_dataset": SOURCE_DATASET,
            "license": LICENSE,
            "selection_table": _selection_tsv(evs),
        }
    return recs


def build_manifests(train_root: Path, test_root: Path, out_dir: Path) -> None:
    """Parse both trees and write all/train/val/test manifests + labels.

    Parameters
    ----------
    train_root, test_root : Path
        Extracted ``Training_*`` and ``Testing_*`` directories.
    out_dir : Path
        Destination directory for the manifest CSVs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    train_recs = _collect_recordings(train_root, "train")
    test_recs = _collect_recordings(test_root, "test")

    # Recording-level val holdout from OPEN-OCEAN training files only
    # (aquarium has too few merged recordings to split).
    oo_train = sorted(k for k, r in train_recs.items() if r["site"].startswith("openocean"))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(oo_train)
    n_val = round(len(oo_train) * VAL_FRACTION)
    val_keys = set(oo_train[:n_val])

    rows = []
    for k, r in train_recs.items():
        r = dict(r)
        r["split"] = "val" if k in val_keys else "train"
        rows.append(r)
    for r in test_recs.values():
        r = dict(r)
        r["split"] = "test"
        rows.append(r)

    df = pd.DataFrame(rows)
    head = ["sound_name", "site", "split", "audio_fp", "16khz_path", "32khz_path", "n_events"]
    df = df[head + [c for c in df.columns if c not in head]]

    for split in ("all", "train", "val", "test"):
        sub = df if split == "all" else df[df["split"] == split]
        out = out_dir / f"delphinid_whistles_{split}.csv"
        sub.to_csv(out, index=False)
        by_site = sub.groupby("site").size().to_dict()
        n_empty = int((sub["n_events"] == 0).sum())
        print(
            f"  {split}: {len(sub)} recordings, {int(sub['n_events'].sum())} events "
            f"({n_empty} empty) -> {out.name}  {by_site}"
        )

    labels_out = out_dir / "delphinid_whistles_labels.csv"
    pd.DataFrame({"Species": [SPECIES]}).to_csv(labels_out, index=False)
    print("  labels: 1 class -> delphinid_whistles_labels.csv")


def _resample_one(args: tuple[str, str, str, str]) -> tuple[str, str]:
    """Write 16k + 32k mirrors and copy the original for one wav.

    Parameters
    ----------
    args : tuple[str, str, str, str]
        ``(src, out_root, site, basename)``.

    Returns
    -------
    tuple[str, str]
        ``(basename, status)``.
    """
    import shutil

    import librosa
    import numpy as np
    import soundfile as sf

    src, out_root, site, base = args
    try:
        audio, sr = sf.read(src, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        for tgt in (16000, 32000):
            y = audio if sr == tgt else librosa.resample(
                y=audio, orig_sr=sr, target_sr=tgt, scale=True, res_type="kaiser_best"
            )
            out = Path(out_root) / f"audio_{tgt // 1000}k" / site / base
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, np.clip(y, -1.0, 1.0), tgt, subtype="PCM_16")
        orig = Path(out_root) / "audio" / site / base
        orig.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, orig)
        return base, "ok"
    except Exception as exc:  # noqa: BLE001
        return base, f"ERROR: {exc}"


def resample(audio_root: Path, out_root: Path, workers: int) -> None:
    """Resample every wav under ``audio_root`` to 16k + 32k (+ copy original).

    Parameters
    ----------
    audio_root : Path
        Root of the extracted train/test trees.
    out_root : Path
        Destination root for ``audio/`` ``audio_16k/`` ``audio_32k/``.
    workers : int
        Process-pool size.

    Raises
    ------
    SystemExit
        If any wav fails to resample.
    """
    wavs = sorted(audio_root.rglob("*.wav"))
    jobs = [(str(w), str(out_root), site_slug(str(w)), w.name) for w in wavs]
    print(f"resampling {len(jobs)} wavs with {workers} workers ...", flush=True)
    errors = done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_resample_one, j) for j in jobs]):
            base, status = fut.result()
            done += 1
            if status != "ok":
                errors += 1
                print(f"  {status}  ({base})", flush=True)
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)} ...", flush=True)
    print(f"done: {done} written, {errors} errors", flush=True)
    if errors:
        raise SystemExit(f"{errors} wavs failed")


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)

    pm = sub.add_parser("manifests")
    pm.add_argument("--train-root", type=Path, required=True)
    pm.add_argument("--test-root", type=Path, required=True)
    pm.add_argument("--out-dir", type=Path, required=True)

    pr = sub.add_parser("resample")
    pr.add_argument("--audio-root", type=Path, required=True)
    pr.add_argument("--out-root", type=Path, required=True)
    pr.add_argument("--workers", type=int, default=12)

    args = p.parse_args()
    if args.stage == "manifests":
        build_manifests(args.train_root, args.test_root, args.out_dir)
    else:
        resample(args.audio_root, args.out_root, args.workers)


if __name__ == "__main__":
    main()
