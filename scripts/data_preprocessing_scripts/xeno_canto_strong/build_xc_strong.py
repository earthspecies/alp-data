"""Pivot xeno-canto strong-annotation segments into a WABAD-shaped manifest.

Input (from a prior pipeline run already on GCS). We read the ``_unseen``
variants — pre-filtered to drop xc_ids whose recording-level OR per-event
species appear in the BEANS-Zero held-out list. The YAML chain layers a
second ``drop_where_text_mentions_taxa`` filter on top as defense in depth.

- ``gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/xc_annotation_segments_unseen.csv``
    One row per annotation event (68,744 events on ~21,291 files; 811
    unique species via segment-level ``scientific_name``). 11 events / 9
    xc_ids pre-filtered vs. the full segments table.
- ``gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/xc_annotated_extras_unseen.csv``
    File-level GBIF + paths (21,300 files; the recording-level focal
    species is mostly ``Sonus naturalis`` so the unseen filter drops 0
    rows here, but we read this variant for symmetry / future-proofing).
- ``gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/xc_annotated_recordings.csv``
    XC's raw recording metadata (42,770 rows; not held-out-filtered, but
    we only join the ``length`` field for duration parsing).

Output (one row per file with inline selection_table TSV — WABAD shape):
- ``gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/xc_strong_with_selection_table.csv``

Schema (matches WABAD's expectations exactly so the existing
``window_annotations`` + ``annotation_features`` chain works as-is):
    audio_fp                  - relative path to original (e.g. ``audio/XC65654.mp3``)
    16khz_path / 32khz_path   - relative paths to pre-resampled mirrors
    audio_duration            - float seconds (parsed from XC ``length``)
    selection_table           - TSV string, columns:
                                ``Begin Time (s)``, ``End Time (s)``,
                                ``Low Freq (Hz)``, ``High Freq (Hz)``,
                                ``Species`` (= segment ``scientific_name``),
                                ``sound_type``, ``sex``, ``life_stage``,
                                ``annotator``
    xc_id                     - XC recording id (numeric, no XC prefix)
    recording_canonical_name  - file-level focal species (mostly Sonus naturalis)
    recording_species_common  - file-level common name
    annotator_set             - top-level annotation set name
    license                   - license URL
    media_license             - media-specific license URL
    rightsHolder, recordedBy  - attribution
    country_code, locality    - geo
    latitudeDecimal, longitudeDecimal - geo
    source_dataset            - constant "xeno_canto_strong"
    n_events                  - count of events in selection_table

Pipeline is metadata-only (~50 MB CSV in, ~30 MB CSV out); runs in seconds
locally, no Slurm needed.

Usage:
    uv run python scripts/data_preprocessing_scripts/xeno_canto_strong/build_xc_strong.py \\
        [--out-dir DIR] [--upload]
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
from pathlib import Path

import pandas as pd

INPUT_ROOT = "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw"
SEGMENTS_CSV = f"{INPUT_ROOT}/xc_annotation_segments_unseen.csv"
EXTRAS_CSV = f"{INPUT_ROOT}/xc_annotated_extras_unseen.csv"
RECORDINGS_CSV = f"{INPUT_ROOT}/xc_annotated_recordings.csv"
OUT_CSV_NAME = "xc_strong_with_selection_table.csv"

_DURATION_RE = re.compile(r"^\s*(?:(\d+):)?(\d+):(\d+)\s*$")


def parse_duration(s: str) -> float | None:
    """Parse the XC ``length`` field into seconds.

    Accepts ``M:SS`` or ``H:MM:SS``. Returns ``None`` if unparseable.

    Parameters
    ----------
    s : str
        Duration string from xc_annotated_recordings.csv ``length`` column.

    Returns
    -------
    float | None
        Total seconds, or ``None`` if input is empty / malformed.
    """
    if not isinstance(s, str):
        return None
    m = _DURATION_RE.match(s)
    if not m:
        return None
    h = int(m.group(1) or 0)
    mm = int(m.group(2))
    ss = int(m.group(3))
    return float(h * 3600 + mm * 60 + ss)


def gsutil_cat(uri: str) -> str:
    """Stream a file via gsutil cat; return decoded text."""
    out = subprocess.run(
        ["gsutil", "cat", uri],
        check=True, capture_output=True, text=True, timeout=600,
    )
    return out.stdout


def build(out_dir: Path, upload: bool) -> Path:
    """Build the WABAD-shaped CSV and optionally upload.

    Parameters
    ----------
    out_dir : Path
        Local staging directory for the CSV.
    upload : bool
        If True, gsutil cp the result to the GCS output path.

    Returns
    -------
    Path
        Local CSV path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {SEGMENTS_CSV} ...", flush=True)
    seg = pd.read_csv(io.StringIO(gsutil_cat(SEGMENTS_CSV)),
                      keep_default_na=False, na_values=[""])
    print(f"  loaded {len(seg):,} segments")

    print(f"Fetching {EXTRAS_CSV} ...", flush=True)
    extras = pd.read_csv(io.StringIO(gsutil_cat(EXTRAS_CSV)),
                         keep_default_na=False, na_values=[""])
    print(f"  loaded {len(extras):,} extras")

    print(f"Fetching {RECORDINGS_CSV} ...", flush=True)
    rec = pd.read_csv(io.StringIO(gsutil_cat(RECORDINGS_CSV)),
                      keep_default_na=False, na_values=[""])
    print(f"  loaded {len(rec):,} recordings")
    rec["audio_duration"] = rec["length"].map(parse_duration)
    rec = rec[["xc_id", "audio_duration"]].dropna()
    print(f"  parsed durations for {len(rec):,} recordings")

    # --- Pivot: group segments by xc_id, write selection_table TSV ---
    print("Building per-file selection_tables ...", flush=True)
    seg["start_time_s"] = pd.to_numeric(seg["start_time_s"], errors="coerce")
    seg["end_time_s"] = pd.to_numeric(seg["end_time_s"], errors="coerce")
    seg["frequency_low_hz"] = pd.to_numeric(seg["frequency_low_hz"], errors="coerce")
    seg["frequency_high_hz"] = pd.to_numeric(seg["frequency_high_hz"], errors="coerce")
    seg = seg.dropna(subset=["start_time_s", "end_time_s"])
    seg = seg[seg["scientific_name"].astype(str).str.strip().ne("")]
    seg = seg.sort_values(["xc_id", "start_time_s"]).reset_index(drop=True)

    st_columns = ["Begin Time (s)", "End Time (s)", "Low Freq (Hz)",
                  "High Freq (Hz)", "Species", "sound_type", "sex",
                  "life_stage", "annotator"]

    def _to_tsv(group: pd.DataFrame) -> str:
        df = pd.DataFrame({
            "Begin Time (s)": group["start_time_s"].round(4),
            "End Time (s)": group["end_time_s"].round(4),
            "Low Freq (Hz)": group["frequency_low_hz"].fillna(0).astype(int),
            "High Freq (Hz)": group["frequency_high_hz"].fillna(0).astype(int),
            "Species": group["scientific_name"].astype(str),
            "sound_type": group["sound_type"].fillna("").astype(str),
            "sex": group["sex"].fillna("").astype(str),
            "life_stage": group["life_stage"].fillna("").astype(str),
            "annotator": group["annotator"].fillna("").astype(str),
        })[st_columns]
        return df.to_csv(sep="\t", index=False)

    pivot = (
        seg.groupby("xc_id", sort=False)
        .apply(lambda g: pd.Series({
            "selection_table": _to_tsv(g),
            "n_events": len(g),
        }))
        .reset_index()
    )
    print(f"  pivoted to {len(pivot):,} files (mean {seg.shape[0] / len(pivot):.1f} events/file)")

    # --- Join with extras for file-level metadata + paths ---
    extras_keep = [
        "xc_id", "gcs_path", "relative_path", "16khz_path", "32khz_path",
        "scientific_name_unified", "vernacularName", "species_common",
        "kingdom", "phylum", "class", "order", "family", "genus",
        "gbifID", "taxonKey", "speciesKey",
        "latitudeDecimal", "longitudeDecimal", "country_code", "locality",
        "continent", "eventDate", "year", "month", "day",
        "recordedBy", "rightsHolder",
    ]
    # `license` and `media_license` columns vary in the extras file — pull
    # what's present.
    for col in ("license", "media_license", "license_url", "media_license_url"):
        if col in extras.columns:
            extras_keep.append(col)
    extras_keep = [c for c in extras_keep if c in extras.columns]
    extras_slim = extras[extras_keep].copy()
    extras_slim["xc_id"] = extras_slim["xc_id"].astype(int)
    pivot["xc_id"] = pivot["xc_id"].astype(int)
    rec["xc_id"] = rec["xc_id"].astype(int)

    df = pivot.merge(extras_slim, on="xc_id", how="left")
    df = df.merge(rec, on="xc_id", how="left")

    # --- Derive WABAD-shaped path columns (relative-to-data_root) ---
    # data_root will be gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/, so the
    # relative paths need the audio/ + audio_16k/ + audio_32k/ subdirs.
    def _prefix(col: str, sub: str) -> pd.Series:
        return sub + "/" + df[col].astype(str)

    df["audio_fp"] = _prefix("relative_path", "audio")
    df["16khz_path"] = _prefix("16khz_path", "audio_16k")
    df["32khz_path"] = _prefix("32khz_path", "audio_32k")

    # Rename file-level focal columns so they don't shadow the per-event
    # species name embedded in selection_table.
    df = df.rename(columns={
        "scientific_name_unified": "recording_scientific_name",
        "species_common": "recording_species_common",
    })

    df["source_dataset"] = "xeno_canto_strong"

    # Drop missing-duration rows (loader needs it for window bounds).
    n_pre = len(df)
    df = df.dropna(subset=["audio_duration"])
    if n_pre - len(df):
        print(f"  WARN dropped {n_pre - len(df)} rows with no audio_duration")

    # Drop columns we no longer need to keep the CSV lean.
    df = df.drop(columns=["gcs_path", "relative_path"], errors="ignore")

    # Final column order (small headers first for readability).
    head_cols = [
        "xc_id", "audio_fp", "16khz_path", "32khz_path", "audio_duration",
        "n_events", "selection_table", "recording_scientific_name",
        "recording_species_common",
    ]
    tail_cols = [c for c in df.columns if c not in head_cols]
    df = df[head_cols + tail_cols]

    out = out_dir / OUT_CSV_NAME
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")

    if upload:
        dest = f"{INPUT_ROOT}/{OUT_CSV_NAME}"
        print(f"Uploading to {dest} ...", flush=True)
        subprocess.run(["gsutil", "-q", "cp", str(out), dest], check=True)
        print("Upload done.")

    return out


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="/mnt/home/esp-data-dev/xc_strong_staging")
    p.add_argument("--upload", action="store_true")
    args = p.parse_args()
    build(Path(args.out_dir), upload=args.upload)


if __name__ == "__main__":
    main()
