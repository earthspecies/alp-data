"""Resolve ECOSoundSet species labels to GBIF-linked canonical binomials.

Reads ``annotated_audio_segments.csv`` and, for every unique BIOTIC ``label``
(target Orthoptera/Cicadidae + background animals; abiotic wind/car/... skipped),
resolves the binomial (first two tokens of the label) against the GBIF animals
backbone via ``esp_data.discover.gbif_taxonomy.GBIFConverter``. Writes
``ecosoundset_species_taxonomy.csv`` mapping each verbatim label to its GBIF
canonical name + higher ranks, keeping the full (possibly trinomial) label as
``label_verbatim``. Falls back to the raw binomial when GBIF has no match.

Usage:
    uv run python scripts/data_preprocessing_scripts/ecosoundset/ecosoundset_taxonomy.py \\
        --annotated <local-or-gs csv> [--upload]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from esp_data.discover.gbif_taxonomy import GBIFConverter

ROOT = "gs://esp-data-ingestion/ecosoundset/v0.1.0"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
ABIOTIC = {"Anthropophony", "Geophony"}
TARGET = {"Orthoptera", "Hemiptera"}
DEFAULT_GBIF = "/mnt/home/superwhale_merge/gbif_animals.tsv"


def _binomial(label: str) -> str:
    """Return the genus+species binomial (first two whitespace tokens).

    Returns
    -------
    str
        The first two whitespace-separated tokens of ``label``.
    """
    return " ".join(str(label).split()[:2])


def build(annotated: str, out: Path, gbif_tsv: str, upload: bool) -> Path:
    """Build the label -> GBIF taxonomy CSV.

    Parameters
    ----------
    annotated : str
        Path (local or ``gs://``) to ``annotated_audio_segments.csv``.
    out : Path
        Local output CSV path.
    gbif_tsv : str
        Path to the GBIF animals backbone TSV.
    upload : bool
        Upload the result to the GCS metadata directory when True.

    Returns
    -------
    Path
        The local CSV written.
    """
    read = annotated
    if str(annotated).startswith("gs://"):
        read = subprocess.run(["gsutil", "cat", annotated], check=True,
                              capture_output=True, text=True).stdout
        from io import StringIO
        df = pd.read_csv(StringIO(read))
    else:
        df = pd.read_csv(annotated)

    biotic = df[~df["label_category"].isin(ABIOTIC)]
    labels = (
        biotic[["label", "label_category"]]
        .drop_duplicates()
        .sort_values("label")
        .reset_index(drop=True)
    )
    print(f"unique biotic labels: {len(labels)} "
          f"(target {labels.label_category.isin(TARGET).sum()}, "
          f"background {(~labels.label_category.isin(TARGET)).sum()})", flush=True)

    conv = GBIFConverter(gbif_animals_tsv_fp=gbif_tsv, cache_path=None)
    rows = []
    n_ok = 0
    for r in labels.itertuples(index=False):
        binom = _binomial(r.label)
        info, ok = conv(binom)
        n_ok += int(ok)
        rec = {
            "label_verbatim": r.label,
            "label_category": r.label_category,
            "is_target": r.label_category in TARGET,
            "binomial_query": binom,
            "canonical_name": (info.get("canonicalName") or binom) if ok else binom,
            "gbif_ok": ok,
        }
        for rank in RANKS:
            rec[rank] = info.get(rank, "") if ok else ""
        rows.append(rec)

    tax = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    tax.to_csv(out, index=False)
    print(f"resolved {n_ok}/{len(labels)} via GBIF; wrote {out} ({len(tax)} rows)", flush=True)
    unresolved = tax.loc[~tax.gbif_ok, "label_verbatim"].tolist()
    if unresolved:
        print(f"  {len(unresolved)} unresolved (binomial fallback), e.g. {unresolved[:8]}")

    if upload:
        dest = f"{ROOT}/metadata/{out.name}"
        subprocess.run(["gsutil", "-q", "cp", str(out), dest], check=True)
        print(f"uploaded -> {dest}")
    return out


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotated", default=f"{ROOT}/raw/annotated_audio_segments.csv")
    p.add_argument(
        "--out", default="/mnt/home/ecosoundset_staging/ecosoundset_species_taxonomy.csv"
    )
    p.add_argument("--gbif-tsv", default=DEFAULT_GBIF)
    p.add_argument("--upload", action="store_true")
    args = p.parse_args()
    build(args.annotated, Path(args.out), args.gbif_tsv, args.upload)


if __name__ == "__main__":
    main()
