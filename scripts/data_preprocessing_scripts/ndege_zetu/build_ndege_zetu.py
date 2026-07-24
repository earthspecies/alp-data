"""Build the Ndege Zetu Mt Kenya bird soundscape dataset (Dryad d51c5b0c7).

wa Maina / DeKUT-DSAIL (2025) "Ndege Zetu" (CC0-1.0). ~1-minute ARU soundscape
recordings from two Mt Kenya sites (DeKUWC, MKNP) with **weak, clip-level
multi-label** annotations (foreground + background species per recording — NO
time localization). Ingested WABAD-shaped so it can flow through the weak
species templates (and, with the caveat that windows inherit the whole-clip
label, through ``window_annotations``).

Audio is 16 kHz mono MP3 (~60 s; Nyquist 8 kHz — high-pitched birds are capped).
Annotations use common names; we map them to scientific names via the dataset's
own ``Kenya-Species-List.csv`` (covers all 100 observed names). The weak label is
stored three ways per recording:
  * ``foreground_species`` / ``background_species`` — ";"-joined scientific names
  * a full-clip ``selection_table`` (one row per species, ``Begin 0 → End =
    audio_duration_sec``, Low/High Freq 0/8000, ``Species`` + ``Presence``)

Two stages (mirrors build_delphinid_whistles.py):

1. ``resample`` — decode every MP3, write 16 kHz (native) + 32 kHz (upsampled)
   WAV mirrors (+ copy the MP3 original), and record per-file duration to
   ``durations.csv``. Run on Slurm.
2. ``manifests`` — from ``durations.csv`` (authoritative audio list + durations)
   + the ARU CSVs, write WABAD-shaped weak manifests with an
   ``audio_duration_sec`` column. Site-stratified seeded train/val/test.

GCS layout::

    gs://esp-data-ingestion/ndege-zetu/v0.1.0/
        ndege_zetu_{all,train,val,test}.csv
        ndege_zetu_labels.csv
        audio/<basename>.mp3          (16 kHz original)
        audio_16k/<stem>.wav
        audio_32k/<stem>.wav
"""

from __future__ import annotations

import argparse
import csv
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

LICENSE = "CC0-1.0"
SOURCE_DATASET = "ndege_zetu"
SOURCE_NYQUIST_HZ = 8000  # 16 kHz source sample rate
SPLIT_SEED = 42
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10
ARU_FILES = {
    "dekuwc_2016": "dekuwc-aru-2016.csv",
    "dekuwc_2017": "dekuwc-aru-2017.csv",
    "mknp_2017_2018": "mknp-aru-2017-2018.csv",
}
ST_COLUMNS = [
    "Selection",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
    "Presence",
]


def _norm(s: object) -> str:
    """Normalise a common name for case/whitespace-insensitive matching.

    Returns
    -------
    str
        Lower-cased, single-spaced string.
    """
    return " ".join(str(s).strip().lower().split())


def _split_species(cell: object) -> list[str]:
    """Split a ``;``/``,``-joined species cell into a clean list.

    Returns
    -------
    list[str]
        Stripped species names ([] for null/empty).
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    return [s.strip() for s in str(cell).replace(",", ";").split(";") if s.strip()]


def _site_of(basename: str) -> str:
    """Return the site slug from a recording filename.

    Returns
    -------
    str
        ``"dekuwc"`` / ``"mknp"`` / ``"unknown"``.
    """
    b = basename.lower()
    if b.startswith("dekuwc"):
        return "dekuwc"
    if b.startswith("mknp"):
        return "mknp"
    return "unknown"


def _common_to_scientific(anno_dir: Path) -> dict[str, str]:
    """Build a normalised common-name -> scientific-name map.

    Reads ``Kenya-Species-List.csv`` (primary) plus, if present, the eBird
    files staged alongside it, for completeness.

    Returns
    -------
    dict[str, str]
        Normalised common name -> scientific name.
    """
    m: dict[str, str] = {}
    ksl = anno_dir / "Kenya-Species-List.csv"
    df = pd.read_csv(ksl)
    for _, r in df.iterrows():
        m[_norm(r["Common Name"])] = str(r["Scientific Name"]).strip()
    for extra, com, sci in (
        ("ebird_ke.csv", "Common Name", "Scientific Name"),
        ("ebird_taxonomy_v2023.csv", "PRIMARY_COM_NAME", "SCI_NAME"),
    ):
        p = anno_dir / extra
        if p.exists():
            e = pd.read_csv(p)
            for _, r in e.iterrows():
                m.setdefault(_norm(r[com]), str(r[sci]).strip())
    return m


def _selection_tsv(fg: list[str], bg: list[str], dur: float) -> str:
    """Serialise weak full-clip boxes (one row per species) into a TSV blob.

    Returns
    -------
    str
        Tab-separated selection table (:data:`ST_COLUMNS`); header-only when the
        recording has no annotated species (a negative recording).
    """
    rows = [(s, "foreground") for s in fg] + [(s, "background") for s in bg]
    df = pd.DataFrame(
        {
            "Selection": range(1, len(rows) + 1),
            "Begin Time (s)": [0.0] * len(rows),
            "End Time (s)": [round(dur, 3)] * len(rows),
            "Low Freq (Hz)": [0] * len(rows),
            "High Freq (Hz)": [SOURCE_NYQUIST_HZ] * len(rows),
            "Species": [s for s, _ in rows],
            "Presence": [p for _, p in rows],
        }
    )[ST_COLUMNS]
    return df.to_csv(sep="\t", index=False)


def _annotation_index(anno_dir: Path, cmap: dict[str, str]) -> dict[str, dict]:
    """Accumulate per-recording foreground/background species (scientific).

    Returns
    -------
    dict[str, dict]
        filename -> {fg, bg, batch, remarks, unmapped} with de-duplicated,
        mapped scientific names.
    """
    idx: dict[str, dict] = {}
    unmapped: set[str] = set()

    def _map(names: list[str]) -> list[str]:
        out = []
        for n in names:
            sci = cmap.get(_norm(n))
            if sci:
                out.append(sci)
            else:
                unmapped.add(n)
        return out

    for batch, fname in ARU_FILES.items():
        df = pd.read_csv(anno_dir / fname)
        for _, r in df.iterrows():
            fn = str(r["Filename"]).strip()
            rec = idx.setdefault(fn, {"fg": set(), "bg": set(), "batch": batch, "remarks": []})
            rec["fg"].update(_map(_split_species(r.get("Foreground Species"))))
            rec["bg"].update(_map(_split_species(r.get("Background Species"))))
            rem = r.get("Remarks")
            if isinstance(rem, str) and rem.strip():
                rec["remarks"].append(rem.strip())
    if unmapped:
        print(f"  WARNING: {len(unmapped)} unmapped common names: {sorted(unmapped)[:10]}")
    return idx


def _assign_split(basename: str) -> str:
    """Deterministic per-recording train/val/test assignment.

    Returns
    -------
    str
        ``"train"`` / ``"val"`` / ``"test"`` (hashed on the filename + seed).
    """
    h = random.Random(f"{SPLIT_SEED}:{basename}").random()
    if h < TEST_FRACTION:
        return "test"
    if h < TEST_FRACTION + VAL_FRACTION:
        return "val"
    return "train"


def build_manifests(anno_dir: Path, durations_csv: Path, out_dir: Path) -> None:
    """Build all/train/val/test weak manifests + labels csv.

    Parameters
    ----------
    anno_dir : Path
        Extracted ``annotations/`` directory.
    durations_csv : Path
        ``durations.csv`` from the resample stage (basename, duration_sec).
    out_dir : Path
        Destination for manifest CSVs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = _common_to_scientific(anno_dir)
    anno = _annotation_index(anno_dir, cmap)
    durs = pd.read_csv(durations_csv)
    dur_map = {str(r["basename"]): float(r["duration_sec"]) for _, r in durs.iterrows()}

    rows = []
    labels: set[str] = set()
    n_no_anno = 0
    for base, dur in dur_map.items():
        rec = anno.get(base)
        if rec is None:
            n_no_anno += 1
            fg, bg, batch, remarks = [], [], "", ""
        else:
            fg = sorted(rec["fg"])
            # A species annotated foreground for this recording takes precedence
            # over a background listing (avoids a duplicate fg+bg row).
            bg = sorted(rec["bg"] - rec["fg"])
            batch, remarks = rec["batch"], " | ".join(rec["remarks"])
        labels.update(fg)
        labels.update(bg)
        stem = Path(base).stem
        rows.append(
            {
                "sound_name": base,
                "site": _site_of(base),
                "aru_batch": batch,
                "split": _assign_split(base),
                "audio_duration_sec": round(dur, 3),
                "audio_fp": f"audio/{base}",
                "16khz_path": f"audio_16k/{stem}.wav",
                "32khz_path": f"audio_32k/{stem}.wav",
                "foreground_species": ";".join(fg),
                "background_species": ";".join(bg),
                "n_species": len(fg) + len(bg),
                "remarks": remarks,
                "source_dataset": SOURCE_DATASET,
                "license": LICENSE,
                "selection_table": _selection_tsv(fg, bg, dur),
            }
        )
    df = pd.DataFrame(rows)
    head = ["sound_name", "site", "aru_batch", "split", "audio_duration_sec",
            "audio_fp", "16khz_path", "32khz_path",
            "foreground_species", "background_species", "n_species"]
    df = df[head + [c for c in df.columns if c not in head]]
    print(f"  {len(df)} recordings ({n_no_anno} without an annotation row -> empty)")

    for split in ("all", "train", "val", "test"):
        sub = df if split == "all" else df[df["split"] == split]
        out = out_dir / f"ndege_zetu_{split}.csv"
        sub.to_csv(out, index=False)
        n_pos = int((sub["n_species"] > 0).sum())
        by_site = sub.groupby("site").size().to_dict()
        print(f"  {split}: {len(sub)} recs, {n_pos} with >=1 species -> {out.name}  {by_site}")

    lab = pd.DataFrame({"Species": sorted(labels)})
    lab.to_csv(out_dir / "ndege_zetu_labels.csv", index=False)
    print(f"  labels: {len(lab)} species -> ndege_zetu_labels.csv")


def _resample_one(args: tuple[str, str]) -> tuple[str, float, str]:
    """Write 16k + 32k WAV mirrors and copy the MP3 original for one file.

    Parameters
    ----------
    args : tuple[str, str]
        ``(src_mp3, out_root)``.

    Returns
    -------
    tuple[str, float, str]
        ``(basename, duration_sec, status)``.
    """
    import shutil

    import librosa
    import numpy as np
    import soundfile as sf

    src, out_root = args
    base = Path(src).name
    stem = Path(src).stem
    try:
        audio, sr = sf.read(src, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / float(sr)
        for tgt in (16000, 32000):
            y = audio if sr == tgt else librosa.resample(
                y=audio, orig_sr=sr, target_sr=tgt, scale=True, res_type="kaiser_best"
            )
            out = Path(out_root) / f"audio_{tgt // 1000}k" / f"{stem}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, np.clip(y, -1.0, 1.0), tgt, subtype="PCM_16")
        orig = Path(out_root) / "audio" / base
        orig.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, orig)
        return base, dur, "ok"
    except Exception as exc:  # noqa: BLE001
        return base, 0.0, f"ERROR: {exc}"


def resample(audio_dir: Path, out_root: Path, workers: int) -> None:
    """Write 16k + 32k mirrors for every MP3 and emit ``durations.csv``.

    Parameters
    ----------
    audio_dir : Path
        Directory of source MP3s.
    out_root : Path
        Destination root for ``audio/`` ``audio_16k/`` ``audio_32k/`` and
        ``durations.csv``.
    workers : int
        Process-pool size.

    Raises
    ------
    SystemExit
        If any file fails to decode/resample.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    mp3s = sorted(audio_dir.glob("*.mp3"))
    print(f"resampling {len(mp3s)} mp3s with {workers} workers ...", flush=True)
    durations: list[tuple[str, float]] = []
    errors = done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_resample_one, (str(m), str(out_root))) for m in mp3s]):
            base, dur, status = fut.result()
            done += 1
            if status != "ok":
                errors += 1
                print(f"  {status}  ({base})", flush=True)
            else:
                durations.append((base, dur))
            if done % 500 == 0:
                print(f"  {done}/{len(mp3s)} ...", flush=True)
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
    pr.add_argument("--audio-dir", type=Path, required=True)
    pr.add_argument("--out-root", type=Path, required=True)
    pr.add_argument("--workers", type=int, default=16)

    pm = sub.add_parser("manifests")
    pm.add_argument("--anno-dir", type=Path, required=True)
    pm.add_argument("--durations-csv", type=Path, required=True)
    pm.add_argument("--out-dir", type=Path, required=True)

    args = p.parse_args()
    if args.stage == "resample":
        resample(args.audio_dir, args.out_root, args.workers)
    else:
        build_manifests(args.anno_dir, args.durations_csv, args.out_dir)


if __name__ == "__main__":
    main()
