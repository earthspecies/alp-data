"""Build the CEB dataset (Zenodo 20762099) into esp-data form.

Two independent stages:

1. ``--manifests`` (metadata only, fast, runs anywhere):
   Pivot each raw per-event CSV into a WABAD-shaped manifest with one row per
   audio file and an inline ``selection_table`` TSV holding every event for that
   file. Multi-species soundscape events (pipe-joined) are exploded to one
   selection row per species, sharing the event's time window. Vocalization type
   is parsed from the part after ``#`` in ``ebird#voc_type``. Writes
   ``ceb_<subset>_with_selection_table.csv`` and (``--upload``) copies to GCS.

2. ``--extract <subset>`` (heavy, run on Slurm):
   Stream the subset's ``.tar.gz`` straight from GCS and batch-upload the
   extracted FLACs to ``audio/<subset>/`` (node/NFS disk is too small to hold the
   46.5 GB train_soundscape tar, so members are extracted to a small rolling
   batch dir and ``gsutil -m rsync``-ed, then deleted). Idempotent (rsync skips
   files already present), so it is safely resumable.

Layout produced on GCS::

    gs://esp-data-ingestion/ceb/v0.1.0/
        raw/            <- original Zenodo files (already uploaded)
        ceb_<subset>_with_selection_table.csv
        audio/<subset>/<internal-tar-path>.flac

Usage::

    uv run python scripts/data_preprocessing_scripts/ceb/build_ceb.py --manifests --upload
    uv run python scripts/data_preprocessing_scripts/ceb/build_ceb.py --extract test_soundscape
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

ROOT = "gs://esp-data-ingestion/ceb/v0.1.0"
RAW = f"{ROOT}/raw"
AUDIO_ROOT = f"{ROOT}/audio"
SUBSETS = ("train_xenocanto", "train_soundscape", "test_soundscape")

# WABAD-style selection-table columns (superset; freq empty -> 0 for weak rows).
ST_COLUMNS = [
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
    "common_name",
    "ebird_code",
    "sound_type",
    "sex",
]
# Per-file metadata carried onto the manifest (first non-null value per file).
FILE_META = ["dataset_name", "label_quality", "lat", "long", "license"]
XC_META = [
    "xc_id",
    "xc_url",
    "xc_recordist",
    "xc_original_scientific_name",
    "xc_original_common_name",
]


def _split_pipe(cell: object) -> list[str]:
    """Split a pipe-joined multi-label cell into a list (``[""]`` when empty).

    Returns
    -------
    list[str]
        Stripped, pipe-separated tokens; ``[""]`` for null/empty input.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return [""]
    parts = str(cell).split("|")
    return [p.strip() for p in parts] if parts else [""]


def _voc_after_hash(part: str) -> str:
    """Vocalization type = text after '#' in an 'ebird#voc_type' token.

    Returns
    -------
    str
        The voc-type substring, or ``""`` when the token has no ``#``.
    """
    if "#" in part:
        return part.split("#", 1)[1].strip()
    return ""


def _explode_events(df: pd.DataFrame) -> pd.DataFrame:
    """Explode multi-species events to one row per (event, species).

    Returns
    -------
    pd.DataFrame
        One row per (event, species) with selection-table columns.
    """
    for col in ("start_time", "end_time", "low_freq", "high_freq"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    out: list[dict] = []
    for r in df.to_dict("records"):
        codes = _split_pipe(r.get("ebird_code_multilabel"))
        names = _split_pipe(r.get("scientific_name"))
        commons = _split_pipe(r.get("common_name"))
        vocs = [_voc_after_hash(p) for p in _split_pipe(r.get("ebird#voc_type"))]
        sexes = _split_pipe(r.get("sex"))
        n = max(len(codes), len(names), len(vocs))

        def _at(lst: list[str], i: int) -> str:
            return lst[i] if i < len(lst) else (lst[0] if lst else "")

        for i in range(n):
            out.append(
                {
                    "filepath": r["filepath"],
                    "Begin Time (s)": r["start_time"],
                    "End Time (s)": r["end_time"],
                    "Low Freq (Hz)": r["low_freq"],
                    "High Freq (Hz)": r["high_freq"],
                    "Species": _at(names, i),
                    "common_name": _at(commons, i),
                    "ebird_code": _at(codes, i),
                    "sound_type": _at(vocs, i),
                    "sex": _at(sexes, i),
                }
            )
    return pd.DataFrame(out)


def _to_tsv(group: pd.DataFrame) -> str:
    df = pd.DataFrame(
        {
            "Begin Time (s)": group["Begin Time (s)"].round(4),
            "End Time (s)": group["End Time (s)"].round(4),
            "Low Freq (Hz)": group["Low Freq (Hz)"].fillna(0).round().astype(int),
            "High Freq (Hz)": group["High Freq (Hz)"].fillna(0).round().astype(int),
            "Species": group["Species"].astype(str),
            "common_name": group["common_name"].astype(str),
            "ebird_code": group["ebird_code"].astype(str),
            "sound_type": group["sound_type"].astype(str),
            "sex": group["sex"].astype(str),
        }
    )[ST_COLUMNS]
    return df.to_csv(sep="\t", index=False)


def build_manifest(subset: str, out_dir: Path, upload: bool) -> Path:
    """Pivot one subset's raw CSV into a WABAD-shaped manifest CSV.

    Returns
    -------
    Path
        The local manifest CSV written.
    """
    src = f"{RAW}/{subset}.csv"
    print(f"[{subset}] reading {src} ...", flush=True)
    raw = pd.read_csv(
        io.StringIO(subprocess.run(["gsutil", "cat", src], check=True,
                                   capture_output=True, text=True, timeout=1200).stdout),
        keep_default_na=False,
        na_values=[""],
    )
    print(f"  {len(raw):,} event rows over {raw['filepath'].nunique():,} files")

    events = _explode_events(raw)
    # Keep events with usable time bounds for the selection table.
    events = events[events["Begin Time (s)"].notna() & events["End Time (s)"].notna()]
    events = events.sort_values(["filepath", "Begin Time (s)"]).reset_index(drop=True)

    pivot = (
        events.groupby("filepath", sort=False)
        .apply(lambda g: pd.Series({"selection_table": _to_tsv(g), "n_events": len(g)}))
        .reset_index()
    )

    # First-per-file metadata.
    meta_cols = [c for c in FILE_META if c in raw.columns]
    if subset == "train_xenocanto":
        meta_cols += [c for c in XC_META if c in raw.columns]
    meta = raw.groupby("filepath", sort=False)[meta_cols].first().reset_index()

    df = pivot.merge(meta, on="filepath", how="left")
    df["audio_fp"] = subset + "/" + df["filepath"].astype(str)
    df["subset"] = subset
    df["source_dataset"] = "ceb"

    head = ["filepath", "audio_fp", "subset", "label_quality", "n_events", "selection_table"]
    df = df[head + [c for c in df.columns if c not in head]]

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ceb_{subset}_with_selection_table.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {len(df):,} files -> {out} ({out.stat().st_size / 1e6:.1f} MB)")

    if upload:
        dest = f"{ROOT}/{out.name}"
        subprocess.run(["gsutil", "-q", "cp", str(out), dest], check=True)
        print(f"  uploaded -> {dest}")
    return out


def extract_audio(subset: str, workdir: Path, batch: int = 1500) -> None:
    """Stream the subset tar from GCS and batch-upload FLACs to audio/<subset>/."""
    src = f"{RAW}/{subset}.tar.gz"
    dest = f"{AUDIO_ROOT}/{subset}"
    stage = workdir / f"ceb_extract_{subset}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    print(f"[{subset}] streaming {src} -> {dest} (batch={batch})", flush=True)

    proc = subprocess.Popen(["gsutil", "cat", src], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    n = 0
    pending = 0

    def flush() -> None:
        nonlocal pending
        if pending == 0:
            return
        subprocess.run(["gsutil", "-m", "-q", "rsync", "-r", str(stage), dest], check=True)
        for child in stage.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        pending = 0

    with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".flac"):
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            target = stage / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as fh:
                shutil.copyfileobj(fobj, fh)
            n += 1
            pending += 1
            if pending >= batch:
                flush()
                print(f"  uploaded {n:,} files ...", flush=True)
    flush()
    proc.wait()
    shutil.rmtree(stage, ignore_errors=True)
    print(f"[{subset}] done: {n:,} FLACs uploaded -> {dest}", flush=True)


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests", action="store_true", help="build per-subset manifest CSVs")
    p.add_argument("--upload", action="store_true", help="upload manifests to GCS")
    p.add_argument("--extract", choices=SUBSETS, help="stream+upload FLACs for one subset")
    p.add_argument("--out-dir", default="/mnt/home/esp-data-dev/ceb_staging")
    p.add_argument("--workdir", default="/tmp")
    p.add_argument("--batch", type=int, default=1500)
    args = p.parse_args()

    if args.manifests:
        for subset in SUBSETS:
            build_manifest(subset, Path(args.out_dir), upload=args.upload)
    if args.extract:
        extract_audio(args.extract, Path(args.workdir), batch=args.batch)
    if not args.manifests and not args.extract:
        p.error("nothing to do: pass --manifests and/or --extract")


if __name__ == "__main__":
    main()
