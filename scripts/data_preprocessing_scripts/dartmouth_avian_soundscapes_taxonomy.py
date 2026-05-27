"""
Resolve Dartmouth Avian Soundscapes AOU species codes -> GBIF canonical names.

Reads ``species_metadata.csv`` (AOU_Code, IsBird, Common_Name, Scientific_Name,
Family, N_*) and resolves each ``Scientific_Name`` (eBird checklist 2021)
against the GBIF animals backbone via
``esp_data.discover.gbif_taxonomy.GBIFConverter``.

Writes ``species_taxonomy.csv`` with columns:

    AOU_Code, IsBird, Common_Name, Scientific_Name, canonical_name,
    gbifID, kingdom, phylum, class, order, family, genus, resolved

Codes with no scientific name (e.g. ``????``, ``UNMA``, ``UNWO``) map to
``canonical_name = "Unknown"``. The ``canonical_name`` column is what gets
written into the per-event ``Species`` field of the selection tables, while the
raw ``AOU_Code`` is retained for traceability.

Usage:
    uv run python scripts/data_preprocessing_scripts/dartmouth_avian_soundscapes_taxonomy.py
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from esp_data.discover.gbif_taxonomy import GBIFConverter

UNKNOWN_LABEL = "Unknown"
TAXONOMY_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]

# eBird-2021 scientific names that are not directly present in the GBIF animals
# backbone (filled in after inspecting the first run's unresolved list).
SCI_NAME_FIX: dict[str, str] = {
    # Cooper's Hawk: moved Accipiter -> Astur (AOU/IOC 2023); GBIF backbone
    # predates the split and lists the accepted usage under Accipiter.
    "Astur cooperii": "Accipiter cooperii",
}


def _is_unknown(sci_name: str) -> bool:
    s = str(sci_name).strip()
    return s == "" or s.upper() == "NA" or s.lower() == "nan"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--species-csv",
        default=os.path.expanduser("~/dartmouth_staging/meta/species_metadata.csv"),
    )
    p.add_argument("--out", default="species_taxonomy.csv")
    p.add_argument(
        "--gcs",
        default="gs://esp-data-ingestion/dartmouth-avian-soundscapes/v0.1.0/metadata/species_taxonomy.csv",
        help="Optional GCS destination to upload the result to ('' to skip).",
    )
    p.add_argument(
        "--gbif-cache",
        default=os.path.expanduser("~/dartmouth_staging/gbif_animals.tsv"),
        help="Local cache path for the GBIF animals TSV (kept outside the repo).",
    )
    args = p.parse_args()

    df = pd.read_csv(args.species_csv, keep_default_na=False, na_values=[])
    converter = GBIFConverter(cache_path=args.gbif_cache)

    rows = []
    unresolved = []
    for _, r in df.iterrows():
        code = str(r["AOU_Code"]).strip()
        sci = str(r["Scientific_Name"]).strip()
        out = {
            "AOU_Code": code,
            "IsBird": r.get("IsBird", ""),
            "Common_Name": r.get("Common_Name", ""),
            "Scientific_Name": sci,
            "canonical_name": UNKNOWN_LABEL,
            "gbifID": "",
            "resolved": False,
        }
        for rank in TAXONOMY_RANKS:
            out[rank] = ""

        if _is_unknown(sci):
            rows.append(out)
            continue

        lookup = SCI_NAME_FIX.get(sci, sci)
        info, ok = converter(lookup)
        if ok:
            out["canonical_name"] = info["canonicalName"]
            out["gbifID"] = int(info["taxonID"])
            out["resolved"] = True
            for rank in TAXONOMY_RANKS:
                out[rank] = info.get(rank, "")
        else:
            unresolved.append((code, sci))
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out_df)} species rows")
    print(
        f"Resolved {int(out_df['resolved'].sum())} / {len(out_df)} "
        f"({(out_df['canonical_name'] == UNKNOWN_LABEL).sum()} mapped to Unknown)"
    )

    if unresolved:
        print("\nUNRESOLVED (need SCI_NAME_FIX entries):")
        for code, sci in unresolved:
            print(f"  {code}\t{sci}")

    if args.gcs:
        os.system(f"gsutil cp {args.out} {args.gcs}")
        print(f"\nUploaded to {args.gcs}")


if __name__ == "__main__":
    main()
