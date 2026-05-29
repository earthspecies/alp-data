"""
Merge the staged DCLDE 2018 HF Odontocete annotation CSV into the SuperWhales
merged detection CSV.

Inputs (on GCS):
  gs://esp-data-ingestion/superwhale/v0.1.0/raw/superwhale_detection.csv
  gs://esp-data-ingestion/superwhale/v0.1.0/raw/dclde_2018_hf_odontocete_annotations.csv

For each new row, this script:
  1. Parses the selection_table TSV.
  2. GBIF-links each non-empty Species via ``esp_data.discover.gbif_taxonomy.GBIFConverter``
     (cached per species), adding ``canonical_name, genus, family, species_common, gbifID``
     columns to the TSV.
  3. Re-serialises the selection_table.
  4. Builds top-level rolled-up columns:
       species  / scientific_name_unified_original  / canonical_name
         -> pipe-joined unique species names present
       kingdom..genus / species_common / gbifID
         -> taken from the first identified species (matches existing convention;
            empty if no species were identified)
  5. For each row, checks the sibling pre-resampled file's actual presence on
     GCS (via cached listings of ``audio_16k/dclde/2018/.../*.x.wav`` and
     ``audio_32k/dclde/2018/.../*.x.wav``). DCLDE 2018 HF originals are
     ``.x.flac`` at 200 kHz; resampled versions are ``.x.wav``. Sets
     ``16khz_path`` / ``32khz_path`` only when the corresponding ``.x.wav``
     exists; otherwise leaves them blank (dataset class falls back to the
     ``.x.flac`` original and resamples on the fly).

Output: new merged ``superwhale_detection.csv`` uploaded back to GCS.
The pre-merge backup lives under ``_backups/`` on the same bucket.
"""

from __future__ import annotations

import argparse
import os
from io import StringIO

import pandas as pd

from esp_data.discover.gbif_taxonomy import GBIFConverter

GCS_ROOT = "gs://esp-data-ingestion/superwhale/v0.1.0/raw"
EXISTING = f"{GCS_ROOT}/superwhale_detection.csv"
NEW = f"{GCS_ROOT}/dclde_2018_hf_odontocete_annotations.csv"

TAXONOMY_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]

# Selection-table columns that exist on existing merged rows (post-GBIF-link).
# DCLDE 2018 HF lacks the GBIF columns — we append them.
ST_GBIF_COLS = ["canonical_name", "genus", "family", "species_common", "gbifID"]


def _link_species(converter: GBIFConverter, sci: str, cache: dict) -> dict:
    """GBIF-link a single scientific name, with memoization.

    Returns
    -------
    dict
        canonical_name / genus / family / species_common / gbifID / kingdom /
        phylum / class / order (empty strings when ``sci`` is empty or
        unresolved). Cached in-place via the ``cache`` dict.
    """
    key = sci.strip()
    if key in cache:
        return cache[key]
    out = {
        "canonical_name": "",
        "genus": "",
        "family": "",
        "species_common": "",
        "gbifID": "",
        "kingdom": "",
        "phylum": "",
        "class": "",
        "order": "",
    }
    if key:
        info, ok = converter(key)
        if ok:
            out["canonical_name"] = info.get("canonicalName", "")
            out["genus"] = info.get("genus", "")
            out["family"] = info.get("family", "")
            out["species_common"] = info.get("vernacularName", "") or ""
            out["gbifID"] = str(info.get("taxonID", ""))
            for r in ("kingdom", "phylum", "class", "order"):
                out[r] = info.get(r, "")
    cache[key] = out
    return out


def _enrich_selection_table(
    st_tsv: str, converter: GBIFConverter, cache: dict
) -> tuple[str, list[str], list[str], dict]:
    """Parse a TSV selection-table, GBIF-enrich each row's species, and
    re-serialise.

    Returns
    -------
    new_tsv : str
        Re-serialised TSV with the 5 GBIF columns appended.
    unique_species : list[str]
        Distinct non-empty Species names present (preserves first-appearance order).
    unique_canonical : list[str]
        Corresponding canonical names (preserves order).
    first_taxonomy : dict
        Taxonomy dict for the first identified species (or all-empty dict).
    """
    if not st_tsv or not st_tsv.strip():
        return st_tsv, [], [], {}

    st = pd.read_csv(StringIO(st_tsv), sep="\t", keep_default_na=False)

    # Initialise empty columns
    for c in ST_GBIF_COLS:
        if c not in st.columns:
            st[c] = ""

    seen_sci: list[str] = []
    seen_canon: list[str] = []
    first_tax: dict = {}

    for i, sci in enumerate(st["species"].astype(str)):
        if not sci.strip():
            continue
        info = _link_species(converter, sci, cache)
        st.at[i, "canonical_name"] = info["canonical_name"]
        st.at[i, "genus"] = info["genus"]
        st.at[i, "family"] = info["family"]
        st.at[i, "species_common"] = info["species_common"]
        st.at[i, "gbifID"] = info["gbifID"]
        if sci not in seen_sci:
            seen_sci.append(sci)
            seen_canon.append(info["canonical_name"])
        if not first_tax and info["canonical_name"]:
            first_tax = {r: info[r] for r in TAXONOMY_RANKS} | {
                "species_common": info["species_common"],
                "gbifID": info["gbifID"],
            }

    new_tsv = st.to_csv(sep="\t", index=False)
    return new_tsv, seen_sci, seen_canon, first_tax


def _load_resampled_set(listing_path: str, prefix: str) -> set[str]:
    """Load a cached ``gsutil ls -r`` listing of pre-resampled files.

    Returns
    -------
    set[str]
        Set of GCS-relative paths (with the ``raw/`` ``prefix`` stripped) for
        downstream relative-path matching against the merged CSV's
        ``audio_path`` column. Empty set when ``listing_path`` does not exist.
    """
    if not os.path.exists(listing_path):
        return set()
    out: set[str] = set()
    with open(listing_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # gs://esp-data-ingestion/superwhale/v0.1.0/raw/<rel>
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
        default="/mnt/home/esp-data-dev/scripts/data_preprocessing_scripts/_out/superwhale_detection_merged.csv",
    )
    p.add_argument("--out-gcs", default=f"{GCS_ROOT}/superwhale_detection.csv")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--gbif-cache",
        default="/mnt/home/superwhale_merge/gbif_animals.tsv",
        help="Local GBIF animals TSV cache (downloaded on first use).",
    )
    p.add_argument(
        "--listing-16k",
        default="/tmp/dclde2018_audio_16k_files.txt",
        help="Cached `gsutil ls -r audio_16k/dclde/2018/.../*.x.wav` listing.",
    )
    p.add_argument(
        "--listing-32k",
        default="/tmp/dclde2018_audio_32k_files.txt",
        help="Cached `gsutil ls -r audio_32k/dclde/2018/.../*.x.wav` listing.",
    )
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out_local), exist_ok=True)
    os.makedirs(os.path.dirname(args.gbif_cache), exist_ok=True)

    bucket_prefix = f"{GCS_ROOT.split('//', 1)[1].rstrip('/')}/"
    set_16k = _load_resampled_set(args.listing_16k, bucket_prefix)
    set_32k = _load_resampled_set(args.listing_32k, bucket_prefix)
    print(f"[load] pre-resampled file listings: {len(set_16k)} @ 16k, {len(set_32k)} @ 32k")

    print(f"[load] existing merged CSV: {args.existing}")
    existing_df = pd.read_csv(args.existing, dtype=str, keep_default_na=False)
    print(f"  rows={len(existing_df)}  source_datasets={existing_df['source_dataset'].nunique()}")

    print(f"[load] new CSV: {args.new}")
    new_df = pd.read_csv(args.new, dtype=str, keep_default_na=False)
    print(f"  rows={len(new_df)}  source_datasets={sorted(new_df['source_dataset'].unique())}")

    converter = GBIFConverter(cache_path=args.gbif_cache)
    cache: dict = {}

    print("[enrich] GBIF-linking selection tables ...")
    new_rows: list[dict] = []
    for i, r in new_df.iterrows():
        new_tsv, seen_sci, seen_canon, first_tax = _enrich_selection_table(
            r["selection_table"], converter, cache
        )

        row = dict(r)
        row["selection_table"] = new_tsv

        # Top-level rolled-up columns (mirror existing convention)
        row["species"] = "|".join(seen_sci)
        row["scientific_name_unified_original"] = "|".join(seen_sci)
        row["canonical_name"] = "|".join(seen_canon)
        row["gbif_link_ok"] = "True" if seen_canon and any(seen_canon) else "False"
        row["gbifID"] = first_tax.get("gbifID", "")
        for rank in TAXONOMY_RANKS:
            row[rank] = first_tax.get(rank, "")
        row["species_common"] = first_tax.get("species_common", "")

        # Pre-resampled paths (audio already staged on GCS at sibling paths).
        # DCLDE 2018 HF originals are *.x.flac (200 kHz); resampled versions are
        # *.x.wav at 16/32 kHz. Only assign when the file actually exists on
        # GCS; otherwise leave blank and let the dataset class fall back to the
        # original .flac path and resample on the fly.
        ap = r["audio_path"]
        wav_rel = ap[: -len(".flac")] + ".wav" if ap.endswith(".flac") else ap
        cand_16k = f"audio_16k/{wav_rel}"
        cand_32k = f"audio_32k/{wav_rel}"
        row["16khz_path"] = cand_16k if cand_16k in set_16k else ""
        row["32khz_path"] = cand_32k if cand_32k in set_32k else ""

        new_rows.append(row)

        if (i + 1) % 200 == 0:
            print(f"  enriched {i + 1}/{len(new_df)}")

    new_enriched = pd.DataFrame(new_rows)
    n_new = len(new_enriched)
    n_exist = len(existing_df)
    print(f"  enriched {n_new} rows; unique species across all selection tables: {len(cache)}")

    # Schema alignment: ensure new_enriched has every column the existing merged CSV has
    for col in existing_df.columns:
        if col not in new_enriched.columns:
            new_enriched[col] = ""
    new_enriched = new_enriched[existing_df.columns]

    print(f"[concat] {n_exist} existing + {n_new} new = {n_exist + n_new} rows")
    merged = pd.concat([existing_df, new_enriched], ignore_index=True)

    print(f"[write] local: {args.out_local}")
    merged.to_csv(args.out_local, index=False)
    print(f"  size: {os.path.getsize(args.out_local) / 1e6:.1f} MB")

    # Sanity check
    sd_counts = merged["source_dataset"].value_counts()
    print("\n[verify] source_dataset row counts in merged CSV:")
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
