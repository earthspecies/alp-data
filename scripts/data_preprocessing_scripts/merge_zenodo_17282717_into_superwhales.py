"""
Merge the staged Zenodo 17282717 (Mediterranean Cetacean PAM) manifest CSV
into the SuperWhales merged detection CSV.

This is simpler than the DCLDE 2018 HF merge because
``build_zenodo_17282717_mediterranean_csv.py`` already produced rows in the
full SuperWhales schema (GBIF-linked, with top-level rolled-up species /
taxonomy columns and per-event selection_table TSV).

Phase 3's resample job populates ``audio_16k/zenodo_17282717/.../*.wav`` and
``audio_32k/zenodo_17282717/.../*.wav``. This script verifies the existence
of each pre-resampled file via cached gsutil listings and sets
``16khz_path`` / ``32khz_path`` accordingly (blank when missing).

Output: new merged ``superwhale_detection.csv`` uploaded back to GCS. The
pre-merge backup lives under ``_backups/`` on the same bucket.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

GCS_ROOT = "gs://esp-data-ingestion/superwhale/v0.1.0/raw"
EXISTING = f"{GCS_ROOT}/superwhale_detection.csv"
NEW = f"{GCS_ROOT}/zenodo_17282717_mediterranean_cetacean_clips.csv"

SOURCE_DATASET = "zenodo_17282717_mediterranean_cetacean_clips"


def _load_resampled_set(listing_path: str, prefix: str) -> set[str]:
    """Load a cached ``gsutil ls -r`` listing of pre-resampled files.

    Returns
    -------
    set[str]
        Set of GCS-relative paths (with the ``raw/`` ``prefix`` stripped).
        Empty set when ``listing_path`` does not exist.
    """
    if not os.path.exists(listing_path):
        return set()
    out: set[str] = set()
    with open(listing_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if prefix in line:
                rel = line.split(prefix, 1)[1]
                out.add(rel)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--existing", default=EXISTING)
    p.add_argument("--new", default=NEW)
    p.add_argument(
        "--out-local",
        default="/mnt/home/esp-data-dev/scripts/data_preprocessing_scripts/_out/"
        "superwhale_detection_merged_zenodo_17282717.csv",
    )
    p.add_argument("--out-gcs", default=f"{GCS_ROOT}/superwhale_detection.csv")
    p.add_argument(
        "--listing-16k",
        default="/tmp/zenodo_17282717_audio_16k_files.txt",
        help="Cached `gsutil ls -r audio_16k/zenodo_17282717/.../*.wav` listing.",
    )
    p.add_argument(
        "--listing-32k",
        default="/tmp/zenodo_17282717_audio_32k_files.txt",
        help="Cached `gsutil ls -r audio_32k/zenodo_17282717/.../*.wav` listing.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out_local), exist_ok=True)

    bucket_prefix = f"{GCS_ROOT.split('//', 1)[1].rstrip('/')}/"
    set_16k = _load_resampled_set(args.listing_16k, bucket_prefix)
    set_32k = _load_resampled_set(args.listing_32k, bucket_prefix)
    print(f"[load] pre-resampled file listings: {len(set_16k)} @ 16k, {len(set_32k)} @ 32k")

    print(f"[load] existing merged CSV: {args.existing}")
    existing_df = pd.read_csv(args.existing, dtype=str, keep_default_na=False)
    n_exist = len(existing_df)
    print(f"  rows={n_exist}  source_datasets={existing_df['source_dataset'].nunique()}")

    # Don't double-merge: short-circuit if this source_dataset is already present.
    if SOURCE_DATASET in set(existing_df["source_dataset"]):
        n_existing_for_ds = (existing_df["source_dataset"] == SOURCE_DATASET).sum()
        raise RuntimeError(
            f"{SOURCE_DATASET} already present in merged CSV ({n_existing_for_ds} rows). "
            "Remove those rows before re-merging, or this would create duplicates."
        )

    print(f"[load] new CSV: {args.new}")
    new_df = pd.read_csv(args.new, dtype=str, keep_default_na=False)
    n_new = len(new_df)
    print(f"  rows={n_new}  source_datasets={sorted(new_df['source_dataset'].unique())}")
    assert (new_df["source_dataset"] == SOURCE_DATASET).all(), (
        "all new rows must have source_dataset = " + SOURCE_DATASET
    )

    # Set 16khz_path / 32khz_path from cached listings
    audio_path_col = new_df["audio_path"]
    new_df["16khz_path"] = audio_path_col.apply(
        lambda ap: f"audio_16k/{ap}" if f"audio_16k/{ap}" in set_16k else ""
    )
    new_df["32khz_path"] = audio_path_col.apply(
        lambda ap: f"audio_32k/{ap}" if f"audio_32k/{ap}" in set_32k else ""
    )
    n_pre_16k = (new_df["16khz_path"] != "").sum()
    n_pre_32k = (new_df["32khz_path"] != "").sum()
    print(f"  pre-resampled coverage: 16k={n_pre_16k}/{n_new}  32k={n_pre_32k}/{n_new}")

    # Schema alignment
    for col in existing_df.columns:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[existing_df.columns]

    print(f"[concat] {n_exist} existing + {n_new} new = {n_exist + n_new} rows")
    merged = pd.concat([existing_df, new_df], ignore_index=True)

    print(f"[write] local: {args.out_local}")
    merged.to_csv(args.out_local, index=False)
    print(f"  size: {os.path.getsize(args.out_local) / 1e6:.1f} MB")

    sd_counts = merged["source_dataset"].value_counts()
    print("\n[verify] source_dataset row counts:")
    for k, v in sd_counts.items():
        print(f"  {v:>6}  {k}")

    if args.dry_run:
        print("\n[dry-run] not uploading. Inspect:", args.out_local)
        return

    print(f"\n[upload] {args.out_gcs}")
    rc = os.system(f"gsutil -q cp {args.out_local} {args.out_gcs}")
    if rc != 0:
        raise RuntimeError(f"gsutil cp failed rc={rc}")
    print("Done.")


if __name__ == "__main__":
    main()
