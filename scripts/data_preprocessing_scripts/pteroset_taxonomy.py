"""
Resolve PteroSet species codes -> GBIF canonical names.

Reads ``species.csv`` (columns: code, species, identification, type) and
resolves each ``species`` (scientific name) against the GBIF animals backbone
via ``esp_data.discover.gbif_taxonomy.GBIFConverter``.

Writes ``pteroset_species_taxonomy.csv`` with columns:

    code, species, identification, type, canonical_name, gbifID,
    kingdom, phylum, class, order, family, genus, resolved

The ``canonical_name`` populates the per-event ``Species`` field of the
selection tables; the raw ``code`` is retained for traceability. Annotation
rows whose ``Determination`` code is empty or ``INDETE`` (indeterminate) are
mapped to ``Unknown`` at CSV-build time, not here.

Usage:
    uv run python scripts/data_preprocessing_scripts/pteroset_taxonomy.py
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from esp_data.discover.gbif_taxonomy import GBIFConverter

UNKNOWN_LABEL = "Unknown"
TAXONOMY_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]

# Source scientific-name typos / synonyms corrected to GBIF-accepted binomials
# (verified against the GBIF animals backbone). Entries left unresolved after
# this (higher-taxon / indeterminate labels like "Picidae" or "-") fall back to
# canonical_name = "Unknown".
SCI_NAME_FIX: dict[str, str] = {
    "Icterus cayaensis": "Icterus cayanensis",
    "Megarynchus pitagua": "Megarynchus pitangua",
    "Melanerpes cruentats": "Melanerpes cruentatus",
    "Ramphoceaenus melanurus": "Ramphocaenus melanurus",
    "Daptrius chimachima": "Milvago chimachima",
    "Hypnellus ruficollis": "Hypnelus ruficollis",
    "Orthopsittaca manilatus": "Orthopsittaca manilata",
    "Aramides cajaneus": "Aramides cajanea",
    "Mesembrinibis cayanensis": "Mesembrinibis cayennensis",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--species-csv",
        default="/mnt/home/pteroset_staging/meta/species.csv",
    )
    p.add_argument("--out", default="pteroset_species_taxonomy.csv")
    p.add_argument(
        "--gcs",
        default="gs://esp-data-ingestion/pteroset/v0.1.0/metadata/species_taxonomy.csv",
        help="Optional GCS destination ('' to skip).",
    )
    p.add_argument(
        "--gbif-cache",
        default="/mnt/home/pteroset_staging/gbif_animals.tsv",
        help="Local cache path for the GBIF animals TSV (kept outside the repo).",
    )
    args = p.parse_args()

    df = pd.read_csv(args.species_csv, keep_default_na=False, na_values=[])
    converter = GBIFConverter(cache_path=args.gbif_cache)

    rows = []
    unresolved = []
    for _, r in df.iterrows():
        code = str(r["code"]).strip()
        sci = " ".join(str(r["species"]).split())  # collapse stray double spaces
        out = {
            "code": code,
            "species": sci,
            "identification": r.get("identification", ""),
            "type": r.get("type", ""),
            "canonical_name": UNKNOWN_LABEL,
            "gbifID": "",
            "resolved": False,
        }
        for rank in TAXONOMY_RANKS:
            out[rank] = ""

        info, ok = converter(SCI_NAME_FIX.get(sci, sci))
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
    print(f"Resolved {int(out_df['resolved'].sum())} / {len(out_df)}")

    if unresolved:
        print("\nUNRESOLVED (need SCI_NAME_FIX entries):")
        for code, sci in unresolved:
            print(f"  {code}\t{sci}")

    if args.gcs:
        os.system(f"gsutil cp {args.out} {args.gcs}")
        print(f"\nUploaded to {args.gcs}")


if __name__ == "__main__":
    main()
