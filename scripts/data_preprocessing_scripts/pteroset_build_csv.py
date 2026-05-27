"""
Build PteroSet split CSVs (WABAD-style) from staged data.

Each recording becomes one CSV row with an embedded tab-separated
``selection_table`` string. Writes one CSV per project
(``map1``/``ppa1``/``ppa2``/``ppa3``/``ppa4``) plus ``all.csv`` and
``species_labels.csv``, and a QC report, then uploads them to the GCS root.

Selection-table columns (per annotated event):
    Begin Time (s), End Time (s), Low Freq (Hz), High Freq (Hz),
    Species (GBIF canonical), Species Code (raw Determination),
    Identification (Raven ID, e.g. AVEVOC), Type (Raven Tipo, e.g. BIO)

The Raven ``Determination`` column holds a species code resolved to a GBIF
canonical name; empty or ``INDETE`` (indeterminate) -> ``Unknown``.

Inputs (local, from pteroset_stage.sh):
    meta/metadata.csv               recording<->annotation join + project/site
    pteroset_species_taxonomy.csv   species code -> GBIF canonical
    extract_lab/**/*.txt            Raven Pro selection tables
    extract_audio/**/*.wav          used only to read audio_duration (header)

Usage:
    uv run python scripts/data_preprocessing_scripts/pteroset_build_csv.py
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd
import soundfile as sf

UNKNOWN_LABEL = "Unknown"
INDETERMINATE = "INDETE"
GCS_ROOT = "gs://esp-data-ingestion/pteroset/v0.1.0"

PROJECTS = {"MAP1": "map1", "PPA1": "ppa1", "PPA2": "ppa2", "PPA3": "ppa3", "PPA4": "ppa4"}

ST_COLUMNS = [
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
    "Species Code",
    "Identification",
    "Type",
]

_ANN_SUFFIX = re.compile(r"\.Table\.\d+\.selections\.txt$")


def _ann_stem(basename: str) -> str:
    s = _ANN_SUFFIX.sub("", basename)
    return s if s != basename else os.path.splitext(basename)[0]


def build_selection_table(ann_path: str, code_to_species: dict[str, str], qc: list, fn: str) -> str:
    """Parse one Raven .txt and return a cleaned tab-separated selection table.

    Returns
    -------
    str
        The cleaned selection table serialised as a tab-separated string.
    """
    if not ann_path or not os.path.exists(ann_path):
        qc.append({"fn": fn, "issue": "annotation_file_missing"})
        return pd.DataFrame(columns=ST_COLUMNS).to_csv(sep="\t", index=False)

    raw = pd.read_csv(ann_path, sep="\t")
    raw = raw.rename(
        columns={"Determination": "Species Code", "ID": "Identification", "Tipo": "Type"}
    )
    for col in ("Species Code", "Identification", "Type"):
        if col not in raw.columns:
            raw[col] = ""
        raw[col] = raw[col].fillna("").astype(str).str.strip()

    def to_species(code: str) -> str:
        if code == "" or code == INDETERMINATE:
            return UNKNOWN_LABEL
        return code_to_species.get(code, UNKNOWN_LABEL)

    raw["Species"] = raw["Species Code"].map(to_species)

    unmapped = sorted(
        {c for c in raw["Species Code"] if c and c != INDETERMINATE and c not in code_to_species}
    )
    if unmapped:
        qc.append({"fn": fn, "issue": f"codes_not_in_species_csv: {unmapped}"})

    st = raw[[c for c in ST_COLUMNS if c in raw.columns]].copy()
    return st.to_csv(sep="\t", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/mnt/home/pteroset_staging")
    p.add_argument("--taxonomy", default="pteroset_species_taxonomy.csv")
    p.add_argument("--gcs-root", default=GCS_ROOT)
    p.add_argument("--out-dir", default="/mnt/home/pteroset_staging/csv")
    args = p.parse_args()

    meta_dir = os.path.join(args.work, "meta")
    ann_dir = os.path.join(args.work, "extract_lab")
    rec_dir = os.path.join(args.work, "extract_audio")
    os.makedirs(args.out_dir, exist_ok=True)

    wav_paths = glob.glob(os.path.join(rec_dir, "**", "*.wav"), recursive=True)
    txt_paths = glob.glob(os.path.join(ann_dir, "**", "*.txt"), recursive=True)
    # index by recording stem (annotation files share the stem before ".Table")
    rec_index = {os.path.splitext(os.path.basename(p))[0]: p for p in wav_paths}
    ann_index = {_ann_stem(os.path.basename(p)): p for p in txt_paths}
    print(f"indexed {len(rec_index)} wav / {len(ann_index)} txt by stem")

    recs = pd.read_csv(os.path.join(meta_dir, "metadata.csv"), dtype=str).fillna("")
    tax = pd.read_csv(args.taxonomy, dtype=str).fillna("")
    code_to_species = dict(zip(tax["code"], tax["canonical_name"], strict=False))

    qc: list = []
    rows: list = []
    for i, r in recs.iterrows():
        if i % 100 == 0:
            print(f"{i} / {len(recs)}")
        proj = r["project_name"]
        sub = PROJECTS.get(proj)
        if sub is None:
            qc.append({"fn": r.get("audio_file", ""), "issue": f"unknown_project:{proj}"})
            continue

        audio_file = r["audio_file"]
        stem = os.path.splitext(audio_file)[0]
        wav_path = rec_index.get(stem, "")
        ann_path = ann_index.get(stem, "")
        # use the actual on-disk filename for GCS paths
        rec_fn = os.path.basename(wav_path) if wav_path else audio_file
        stem = os.path.splitext(rec_fn)[0]

        duration = ""
        sr = ""
        if wav_path and os.path.exists(wav_path):
            info = sf.info(wav_path)
            duration = round(float(info.frames) / float(info.samplerate), 6)
            sr = int(info.samplerate)
        else:
            qc.append({"fn": stem, "issue": "recording_file_missing"})

        st_str = build_selection_table(ann_path, code_to_species, qc, stem)
        n_events = max(st_str.count("\n") - 1, 0)

        rows.append(
            {
                "fn": stem,
                "site": r.get("event_indicator", ""),
                "project": sub,
                "audio_fp": f"recordings/{rec_fn}",
                "16khz_path": f"audio_16k/{stem}.wav",
                "32khz_path": f"audio_32k/{stem}.wav",
                "audio_duration": duration,
                "sample_rate": sr,
                "date": r.get("date_recorded", ""),
                "country": r.get("country", ""),
                "department": r.get("department", ""),
                "municipality": r.get("municipality", ""),
                "latitude": r.get("latitude", ""),
                "longitude": r.get("longitude", ""),
                "zone": r.get("zone", ""),
                "land_cover": r.get("land_cover", ""),
                "n_events": n_events,
                "selection_table": st_str,
            }
        )

    all_df = pd.DataFrame(rows)

    def _save(df: pd.DataFrame, name: str) -> None:
        local = os.path.join(args.out_dir, name)
        df.to_csv(local, index=False)
        os.system(f"gsutil -q cp {local} {args.gcs_root}/{name}")
        print(f"  wrote {name}: {len(df)} rows -> {args.gcs_root}/{name}")

    _save(all_df, "all.csv")
    for sub in PROJECTS.values():
        _save(all_df[all_df["project"] == sub].reset_index(drop=True), f"{sub}.csv")

    labels = sorted({c for c in code_to_species.values() if c and c != UNKNOWN_LABEL})
    _save(pd.DataFrame({"Species": labels}), "species_labels.csv")

    qc_df = pd.DataFrame(qc)
    qc_local = os.path.join(args.out_dir, "qc_report.csv")
    qc_df.to_csv(qc_local, index=False)
    print(f"\nTotal recordings: {len(all_df)}  total events: {int(all_df['n_events'].sum())}")
    print(f"QC issues: {len(qc_df)} -> {qc_local}")
    if len(qc_df):
        print(qc_df["issue"].str.replace(r":.*", "", regex=True).value_counts().to_string())


if __name__ == "__main__":
    main()
