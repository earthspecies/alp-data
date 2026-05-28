"""
Resolve Weldy NW dawn-chorus species (eBird codes -> GBIF canonical names).

Reads ``annotation_metadata.tsv`` (label vocabulary; columns include
``eBird_2021``, ``scientific_name``, ``common_name``). For each unique eBird
code with a real binomial scientific name, resolves it against the GBIF animals
backbone via ``esp_data.discover.gbif_taxonomy.GBIFConverter``.

Aggregates (``Aves``, ``Insecta``, ``... spp.``) and non-species sounds (rain,
engine, etc.) are kept as common-name only with empty ``canonical_name``.

Writes ``weldy_species_taxonomy.csv`` and uploads to GCS metadata/.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from esp_data.discover.gbif_taxonomy import GBIFConverter

TAXONOMY_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
UNKNOWN_LABEL = "Unknown"

# Manual scientific-name corrections for the GBIF animals backbone.
SCI_NAME_FIX: dict[str, str] = {
    # filled in after the first run surfaces any unresolved cases.
}


def _is_real_species(sci: str) -> bool:
    s = sci.strip()
    if not s or s.lower() == "nan":
        return False
    low = s.lower()
    if "spp" in low or s in ("Aves", "Insecta"):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="/mnt/home/weldy_staging/meta/annotation_metadata.tsv")
    ap.add_argument("--out", default="weldy_species_taxonomy.csv")
    ap.add_argument(
        "--gcs",
        default="gs://esp-data-ingestion/weldy_dawn_chorus/v0.1.0/metadata/weldy_species_taxonomy.csv",
    )
    ap.add_argument(
        "--gbif-cache",
        default="/mnt/home/weldy_staging/gbif_animals.tsv",
        help="Local cache path for the GBIF animals TSV (kept outside the repo).",
    )
    args = ap.parse_args()

    m = pd.read_csv(args.metadata, sep="\t", keep_default_na=False, na_values=[])
    # one row per (eBird_2021, scientific_name, common_name) with non-empty common_name
    sp = (
        m[["eBird_2021", "scientific_name", "common_name"]]
        .dropna()
        .drop_duplicates(subset=["eBird_2021"])
        .reset_index(drop=True)
    )

    converter = GBIFConverter(cache_path=args.gbif_cache)
    rows = []
    unresolved = []
    for _, r in sp.iterrows():
        code = str(r["eBird_2021"]).strip()
        sci = " ".join(str(r["scientific_name"]).split())
        common = str(r["common_name"]).strip()
        out = {
            "eBird_2021": code,
            "scientific_name": sci,
            "common_name": common,
            "canonical_name": "",
            "gbifID": "",
            "resolved": False,
        }
        for rank in TAXONOMY_RANKS:
            out[rank] = ""
        if _is_real_species(sci):
            info, ok = converter(SCI_NAME_FIX.get(sci, sci))
            if ok:
                out["canonical_name"] = info["canonicalName"]
                out["gbifID"] = int(info["taxonID"])
                out["resolved"] = True
                for rank in TAXONOMY_RANKS:
                    out[rank] = info.get(rank, "")
            else:
                unresolved.append((code, sci, common))
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    n_res = int(out_df["resolved"].sum())
    print(
        f"wrote {args.out} | {len(out_df)} eBird codes | "
        f"resolved {n_res} | aggregates/non-sp {len(out_df) - n_res}"
    )
    if unresolved:
        print("\nUNRESOLVED (need SCI_NAME_FIX):")
        for code, sci, common in unresolved:
            print(f"  {code}\t{sci}\t({common})")
    if args.gcs:
        os.system(f"gsutil -q cp {args.out} {args.gcs}")
        print(f"uploaded -> {args.gcs}")


if __name__ == "__main__":
    main()
