"""
Build Clotho v2.1 manifest CSVs (development / validation / evaluation + all).

Inputs (GCS):
  gs://esp-data-ingestion/clotho/v0.1.0/raw/audio/{development,validation,evaluation}/*.wav
  gs://esp-data-ingestion/clotho/v0.1.0/raw/metadata/clotho_captions_{split}.csv
  gs://esp-data-ingestion/clotho/v0.1.0/raw/metadata/clotho_metadata_{split}.csv

Schema (one row per audio clip, five captions per row):
  file_name, audio_path, 16khz_path, 32khz_path, duration_s, sample_rate_hz,
  caption_1, caption_2, caption_3, caption_4, caption_5,
  keywords, sound_id, sound_link, freesound_license, manufacturer,
  split, source_dataset, source_url, license, source_paper_doi

Outputs: per-split CSVs + ``all.csv`` uploaded to gs://...clotho/v0.1.0/.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

GCS_ROOT = "gs://esp-data-ingestion/clotho/v0.1.0"
GCS_RAW = f"{GCS_ROOT}/raw"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--listing-16k",
        default="/tmp/clotho_audio_16k.txt",
        help="Cached `gsutil ls -r audio_16k/...` listing (one path per line).",
    )
    p.add_argument(
        "--listing-32k",
        default="/tmp/clotho_audio_32k.txt",
        help="Cached `gsutil ls -r audio_32k/...` listing.",
    )
    p.add_argument("--out-dir", default="/mnt/home/clotho_staging/out")
    p.add_argument("--out-gcs-prefix", default=GCS_ROOT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Pre-resampled path lookups ---------------------------------------
    def load_listing(path: str) -> set[str]:
        s: set[str] = set()
        if not os.path.exists(path):
            return s
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln.endswith(".wav"):
                    # ".../audio_16k/<split>/<file>.wav" -> "audio_16k/<split>/<file>.wav"
                    idx = ln.find("clotho/v0.1.0/")
                    if idx >= 0:
                        s.add(ln[idx + len("clotho/v0.1.0/") :])
        return s

    set_16k = load_listing(args.listing_16k)
    set_32k = load_listing(args.listing_32k)
    print(f"pre-resampled coverage: 16k={len(set_16k)}, 32k={len(set_32k)}")

    SPLITS = {"development": "train", "validation": "val", "evaluation": "test"}
    all_dfs = []
    for raw_split, esp_split in SPLITS.items():
        cap_csv = f"{GCS_RAW}/metadata/clotho_captions_{raw_split}.csv"
        meta_csv = f"{GCS_RAW}/metadata/clotho_metadata_{raw_split}.csv"
        cap = pd.read_csv(cap_csv, keep_default_na=False)
        meta = pd.read_csv(meta_csv, keep_default_na=False, encoding_errors="replace")
        # Some metadata CSVs may have encoding quirks for non-ASCII filenames;
        # fall back to a more permissive read if needed.
        if "file_name" not in meta.columns:
            meta = pd.read_csv(meta_csv, keep_default_na=False, encoding="latin-1")
        df = cap.merge(meta, on="file_name", how="left")
        df["split"] = esp_split
        df["audio_path"] = "raw/audio/" + raw_split + "/" + df["file_name"]
        df["16khz_path"] = df["file_name"].apply(
            lambda f, rs=raw_split: f"raw/audio_16k/{rs}/{f}"
            if f"raw/audio_16k/{rs}/{f}" in set_16k
            else ""
        )
        df["32khz_path"] = df["file_name"].apply(
            lambda f, rs=raw_split: f"raw/audio_32k/{rs}/{f}"
            if f"raw/audio_32k/{rs}/{f}" in set_32k
            else ""
        )
        # Source audio rate is 44.1 kHz mono (Freesound originals re-encoded by Clotho)
        df["sample_rate_hz"] = 44100
        # Clotho clips are 15-30 s; precise duration not given without probing audio.
        # Leave duration_s blank (downstream code can probe via soundfile.info).
        df["duration_s"] = ""

        # Rename + align columns
        if "license" in df.columns:
            df["freesound_license"] = df["license"]
        df["source_dataset"] = "clotho_v2_1"
        df["source_url"] = "https://zenodo.org/records/4783391"
        df["license"] = "CC-BY (audio) + Tampere caption license"
        df["source_paper_doi"] = "10.1109/ICASSP40776.2020.9052990"

        cols = [
            "file_name",
            "audio_path",
            "16khz_path",
            "32khz_path",
            "sample_rate_hz",
            "duration_s",
            "caption_1",
            "caption_2",
            "caption_3",
            "caption_4",
            "caption_5",
            "keywords",
            "sound_id",
            "sound_link",
            "freesound_license",
            "manufacturer",
            "split",
            "source_dataset",
            "source_url",
            "license",
            "source_paper_doi",
        ]
        # Some metadata cols may be absent for some splits; create empty
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]

        n16 = (df["16khz_path"] != "").sum()
        n32 = (df["32khz_path"] != "").sum()
        print(f"  {raw_split} (-> {esp_split}): {len(df)} rows; resampled: 16k={n16} 32k={n32}")
        all_dfs.append(df)
        local = os.path.join(args.out_dir, f"{esp_split}.csv")
        df.to_csv(local, index=False)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(args.out_dir, "all.csv"), index=False)
    print(f"\ntotal: {len(combined)} rows; splits {combined['split'].value_counts().to_dict()}")

    if args.dry_run:
        print("[dry-run] not uploading.")
        return

    for csv in os.listdir(args.out_dir):
        if csv.endswith(".csv"):
            local = os.path.join(args.out_dir, csv)
            rc = os.system(f"gsutil -q cp {local} {args.out_gcs_prefix}/{csv}")
            if rc != 0:
                raise RuntimeError(f"gsutil cp failed for {csv}")
            print(f"  uploaded {csv} -> {args.out_gcs_prefix}/{csv}")


if __name__ == "__main__":
    main()
