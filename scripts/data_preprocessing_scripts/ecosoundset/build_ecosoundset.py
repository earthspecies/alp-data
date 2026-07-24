"""Build the ECOSoundSet dataset (Zenodo 18636037 v3) into esp-data form.

Two stages:

1. ``--manifests`` (metadata only): pivot ``annotated_audio_segments.csv`` into a
   WABAD-shaped manifest with one row per annotated 4 s clip and an inline
   ``selection_table`` TSV over ALL events (clip-relative times). Also emits two
   classification columns: ``target_species_list`` (Orthoptera + Cicadidae) and
   ``all_species_list`` (all biotic species incl. background), both as
   ", "-joined GBIF canonical binomials (from ``ecosoundset_species_taxonomy.csv``).
   One CSV per ``subset`` -> uploaded to GCS.

2. ``--extract <tar>`` (heavy, Slurm): stream a Split-recording tar from GCS and
   batch-upload the WAV/FLAC clips to ``audio/`` (disk-safe streaming; resumable).

Usage::

    uv run python .../build_ecosoundset.py --manifests --upload
    uv run python .../build_ecosoundset.py --extract split_recordings1.tar.gz
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

ROOT = "gs://esp-data-ingestion/ecosoundset/v0.1.0"
RAW = f"{ROOT}/raw"
AUDIO_ROOT = f"{ROOT}/audio"
ABIOTIC = {"Anthropophony", "Geophony"}
TARGET = {"Orthoptera", "Hemiptera"}
SPLIT_TARS = ("split_recordings1.tar.gz", "split_recordings2.tar.gz")
SEG_DUR = 4.0

ST_COLUMNS = [
    "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)",
    "Species", "subspecies", "label_category",
]
_HEADER = "\t".join(ST_COLUMNS)


def _read_csv(path: str) -> pd.DataFrame:
    if path.startswith("gs://"):
        txt = subprocess.run(
            ["gsutil", "cat", path], check=True, capture_output=True, text=True
        ).stdout
        return pd.read_csv(io.StringIO(txt))
    return pd.read_csv(path)


def _join_species(s: pd.Series) -> str:
    return ", ".join(sorted(set(s)))


def build_manifests(annotated: str, taxonomy: str, out_dir: Path, upload: bool) -> None:
    """Pivot the annotations into per-subset WABAD manifests (vectorized)."""
    df = _read_csv(annotated)
    tax = _read_csv(taxonomy)
    canon = dict(zip(tax["label_verbatim"], tax["canonical_name"], strict=False))

    for c in ("audio_segment_initial_time", "annotation_initial_time",
              "annotation_final_time", "annotation_min_freq", "annotation_max_freq"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    seg0 = df["audio_segment_initial_time"]
    begin = (df["annotation_initial_time"] - seg0).clip(lower=0, upper=SEG_DUR).round(4)
    end = (df["annotation_final_time"] - seg0).clip(lower=0, upper=SEG_DUR).round(4)
    lo = df["annotation_min_freq"].fillna(0).round().astype(int)
    hi = df["annotation_max_freq"].fillna(0).round().astype(int)
    species = df["label"].map(lambda x: canon.get(x, " ".join(str(x).split()[:2])))
    df["Species"] = species
    df["is_biotic"] = ~df["label_category"].isin(ABIOTIC)
    df["is_target"] = df["label_category"].isin(TARGET)
    # one tab-joined selection-table line per event (vectorized string build)
    df["_line"] = (
        begin.astype(str) + "\t" + end.astype(str) + "\t"
        + lo.astype(str) + "\t" + hi.astype(str) + "\t"
        + species.astype(str) + "\t" + df["label"].astype(str) + "\t"
        + df["label_category"].astype(str)
    )
    df["_begin"] = begin
    df = df.sort_values(["subset", "audio_segment_file_name", "_begin"])

    out_dir.mkdir(parents=True, exist_ok=True)
    key = "audio_segment_file_name"
    for subset, sub in df.groupby("subset", sort=False):
        g = sub.groupby(key, sort=False)
        st = g["_line"].agg(lambda s: f"{_HEADER}\n" + "\n".join(s) + "\n")
        rid = g["recording_id"].first()
        nev = g.size()
        tgt = sub[sub.is_target].groupby(key)["Species"].agg(_join_species)
        allb = sub[sub.is_biotic].groupby(key)["Species"].agg(_join_species)

        man = pd.DataFrame({"audio_segment_file_name": st.index})
        man["audio_fp"] = man["audio_segment_file_name"]
        man["subset"] = subset
        man["recording_id"] = rid.to_numpy()
        man["n_events"] = nev.to_numpy()
        man["selection_table"] = st.to_numpy()
        man["target_species_list"] = man["audio_segment_file_name"].map(tgt).fillna("")
        man["all_species_list"] = man["audio_segment_file_name"].map(allb).fillna("")
        man["n_target_species"] = man["target_species_list"].map(
            lambda s: len(s.split(", ")) if s else 0
        )
        man["n_all_species"] = man["all_species_list"].map(
            lambda s: len(s.split(", ")) if s else 0
        )

        out = out_dir / f"ecosoundset_{subset}_with_selection_table.csv"
        man.to_csv(out, index=False)
        mb = out.stat().st_size / 1e6
        print(f"[{subset}] {len(man)} clips -> {out} ({mb:.1f} MB)", flush=True)
        if upload:
            subprocess.run(["gsutil", "-q", "cp", str(out), f"{ROOT}/{out.name}"], check=True)
            print(f"  uploaded -> {ROOT}/{out.name}")


def extract_audio(tar_name: str, workdir: Path, batch: int = 2000) -> None:
    """Stream a Split tar from GCS and batch-upload its clips to audio/."""
    src = f"{RAW}/{tar_name}"
    stage = workdir / f"eco_extract_{tar_name}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    print(f"streaming {src} -> {AUDIO_ROOT} (batch={batch})", flush=True)
    proc = subprocess.Popen(["gsutil", "cat", src], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    n = 0
    pending = 0

    def flush() -> None:
        nonlocal pending
        if pending == 0:
            return
        subprocess.run(["gsutil", "-m", "-q", "rsync", "-r", str(stage), AUDIO_ROOT], check=True)
        for child in stage.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        pending = 0

    with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith((".wav", ".flac")):
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            with open(stage / Path(member.name).name, "wb") as fh:
                shutil.copyfileobj(fobj, fh)
            n += 1
            pending += 1
            if pending >= batch:
                flush()
                print(f"  uploaded {n:,} clips ...", flush=True)
    flush()
    proc.wait()
    shutil.rmtree(stage, ignore_errors=True)
    print(f"done: {n:,} clips uploaded from {tar_name}", flush=True)


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests", action="store_true")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--extract", help="Split tar basename, or 'all'")
    p.add_argument("--annotated", default=f"{RAW}/annotated_audio_segments.csv")
    p.add_argument("--taxonomy", default=f"{ROOT}/metadata/ecosoundset_species_taxonomy.csv")
    p.add_argument("--out-dir", default="/mnt/home/ecosoundset_staging")
    p.add_argument("--workdir", default="/tmp")
    p.add_argument("--batch", type=int, default=2000)
    args = p.parse_args()

    if args.manifests:
        build_manifests(args.annotated, args.taxonomy, Path(args.out_dir), args.upload)
    if args.extract:
        tars = SPLIT_TARS if args.extract == "all" else (args.extract,)
        for t in tars:
            extract_audio(t, Path(args.workdir), batch=args.batch)
    if not args.manifests and not args.extract:
        p.error("nothing to do: pass --manifests and/or --extract")


if __name__ == "__main__":
    main()
