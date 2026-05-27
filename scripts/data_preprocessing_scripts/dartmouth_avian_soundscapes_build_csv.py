"""
Build Dartmouth Avian Soundscapes split CSVs (WABAD-style) from staged data.

Each 10-minute recording becomes one CSV row with an embedded tab-separated
``selection_table`` string. Writes ``acad.csv`` / ``mabi.csv`` / ``simr.csv`` /
``all.csv`` (one per split) plus ``species_labels.csv``, and a QC report, then
uploads them to the GCS dataset root.

Selection-table columns (per annotated event):
    Begin Time (s), End Time (s), Low Freq (Hz), High Freq (Hz),
    Species (GBIF canonical), Species Code (raw AOU), Common Name, Background

Inputs (all local, from the staging dir produced by *_stage.sh):
    meta/recording_metadata.csv   recording <-> annotation join + site/date
    meta/site_metadata.csv        lat/lon per site
    species_taxonomy.csv          AOU code -> GBIF canonical + common name
    extract_ann/<Dataset>/*.txt   Raven Pro selection tables
    extract_rec/<Dataset>/*.flac  used only to read audio_duration (header)

Usage:
    uv run python scripts/data_preprocessing_scripts/dartmouth_avian_soundscapes_build_csv.py
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import soundfile as sf

UNKNOWN_LABEL = "Unknown"
GCS_ROOT = "gs://esp-data-ingestion/dartmouth-avian-soundscapes/v0.1.0"

# Dataset_ID -> short split key
SUBDATASET = {"DatasetACAD": "acad", "DatasetMABI": "mabi", "DatasetSIMR": "simr"}

# Final order of columns kept in each embedded selection table.
ST_COLUMNS = [
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
    "Species Code",
    "Common Name",
    "Background",
]


def build_selection_table(
    ann_path: str,
    code_to_species: dict[str, str],
    code_to_common: dict[str, str],
    qc: list[dict],
    fn: str,
) -> str:
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
    # Raw Raven column "Species" holds the 4-letter AOU code.
    raw = raw.rename(columns={"Species": "Species Code"})
    raw["Species Code"] = raw["Species Code"].astype(str).str.strip()

    unmapped = sorted(set(raw["Species Code"]) - set(code_to_species))
    if unmapped:
        qc.append({"fn": fn, "issue": f"codes_not_in_metadata: {unmapped}"})

    raw["Species"] = raw["Species Code"].map(lambda c: code_to_species.get(c, UNKNOWN_LABEL))
    raw["Common Name"] = raw["Species Code"].map(lambda c: code_to_common.get(c, UNKNOWN_LABEL))
    if "Background" not in raw.columns:
        raw["Background"] = ""
    raw["Background"] = raw["Background"].fillna("").astype(str).str.strip()

    st = raw[[c for c in ST_COLUMNS if c in raw.columns]].copy()
    return st.to_csv(sep="\t", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work", default=os.path.expanduser("~/dartmouth_staging"))
    p.add_argument("--taxonomy", default="species_taxonomy.csv")
    p.add_argument("--gcs-root", default=GCS_ROOT)
    p.add_argument("--out-dir", default=os.path.expanduser("~/dartmouth_staging/csv"))
    args = p.parse_args()

    meta_dir = os.path.join(args.work, "meta")
    ann_dir = os.path.join(args.work, "extract_ann")
    rec_dir = os.path.join(args.work, "extract_rec")
    os.makedirs(args.out_dir, exist_ok=True)

    # Index local files by basename (robust to the differing zip folder names:
    # recordings extract to <ds>_Recordings/ while annotations extract to <ds>/).
    flac_paths = glob.glob(os.path.join(rec_dir, "**", "*.flac"), recursive=True)
    ann_paths = glob.glob(os.path.join(ann_dir, "**", "*.txt"), recursive=True)
    # Index by ItemID (filename prefix before the first "."). recording_metadata
    # occasionally disagrees with the actual filename on the volatile middle
    # field / time for a handful of rows, but the ItemID prefix is stable+unique.
    flac_index = {os.path.basename(p).split(".")[0]: p for p in flac_paths}
    ann_index = {os.path.basename(p).split(".")[0]: p for p in ann_paths}
    print(
        f"indexed {len(flac_index)} flac and {len(ann_index)} annotation files by ItemID "
        f"({len(flac_paths)} flac / {len(ann_paths)} txt on disk)"
    )

    recs = pd.read_csv(os.path.join(meta_dir, "recording_metadata.csv"), dtype=str).fillna("")
    sites = pd.read_csv(os.path.join(meta_dir, "site_metadata.csv"), dtype=str).fillna("")
    site_ll = {
        r["SiteID"]: (r.get("Latitude", ""), r.get("Longitude", "")) for _, r in sites.iterrows()
    }

    tax = pd.read_csv(args.taxonomy, dtype=str).fillna("")
    code_to_species = dict(zip(tax["AOU_Code"], tax["canonical_name"], strict=False))
    code_to_common = dict(zip(tax["AOU_Code"], tax["Common_Name"], strict=False))

    qc: list[dict] = []
    rows: list[dict] = []

    for i, r in recs.iterrows():
        if i % 200 == 0:
            print(f"{i} / {len(recs)}")
        ds_id = r["Dataset_ID"]
        sub = SUBDATASET.get(ds_id)
        if sub is None:
            qc.append({"fn": r.get("Recording_FileName", ""), "issue": f"unknown_dataset:{ds_id}"})
            continue

        item_id = r.get("ItemID", "")
        flac_path = flac_index.get(item_id, "")
        ann_path = ann_index.get(item_id, "")
        # use the actual on-disk filename for GCS paths (metadata can be stale)
        rec_fn = os.path.basename(flac_path) if flac_path else r["Recording_FileName"]
        stem = os.path.splitext(rec_fn)[0]
        fn = stem

        # audio_duration from FLAC header (fast; no decode)
        duration = ""
        sr = ""
        if flac_path and os.path.exists(flac_path):
            info = sf.info(flac_path)
            duration = round(float(info.frames) / float(info.samplerate), 6)
            sr = int(info.samplerate)
        else:
            qc.append({"fn": fn, "issue": "recording_file_missing"})

        st_str = build_selection_table(ann_path, code_to_species, code_to_common, qc, fn)
        n_events = max(st_str.count("\n") - 1, 0)  # minus header line
        lat, lon = site_ll.get(r.get("SiteID", ""), ("", ""))

        rows.append(
            {
                "fn": fn,
                "item_id": r.get("ItemID", ""),
                "subdataset": sub,
                "audio_fp": f"recordings/{ds_id}/{rec_fn}",
                "16khz_path": f"audio_16k/{ds_id}/{stem}.wav",
                "32khz_path": f"audio_32k/{ds_id}/{stem}.wav",
                "audio_duration": duration,
                "sample_rate": sr,
                "site_id": r.get("SiteID", ""),
                "park_id": r.get("ParkID", ""),
                "year": r.get("Year", ""),
                "date": r.get("Date", ""),
                "time": r.get("Time", ""),
                "latitude": lat,
                "longitude": lon,
                "n_events": n_events,
                "selection_table": st_str,
            }
        )

    all_df = pd.DataFrame(rows)

    # Write splits + all
    def _save(df: pd.DataFrame, name: str) -> None:
        local = os.path.join(args.out_dir, name)
        df.to_csv(local, index=False)
        os.system(f"gsutil -q cp {local} {args.gcs_root}/{name}")
        print(f"  wrote {name}: {len(df)} rows -> {args.gcs_root}/{name}")

    _save(all_df, "all.csv")
    for sub in SUBDATASET.values():
        _save(all_df[all_df["subdataset"] == sub].reset_index(drop=True), f"{sub}.csv")

    # species label vocabulary (resolved canonical names, excludes Unknown)
    labels = sorted({s for s in code_to_species.values() if s and s != UNKNOWN_LABEL})
    labels_df = pd.DataFrame({"Species": labels})
    _save(labels_df, "species_labels.csv")

    # QC report
    qc_df = pd.DataFrame(qc)
    qc_local = os.path.join(args.out_dir, "qc_report.csv")
    qc_df.to_csv(qc_local, index=False)
    print(f"\nTotal recordings: {len(all_df)}  total events: {int(all_df['n_events'].sum())}")
    print(f"QC issues: {len(qc_df)} -> {qc_local}")
    if len(qc_df):
        print(qc_df["issue"].str.replace(r":.*", "", regex=True).value_counts().to_string())


if __name__ == "__main__":
    main()
