"""Build the NABirds species-classification manifests (GBIF-linked).

Given a locally-extracted NABirds release (``--src`` containing the standard
``images.txt`` / ``image_class_labels.txt`` / ``classes.txt`` /
``hierarchy.txt`` / ``train_test_split.txt`` tables and the ``images/``
tree), this script:

1. Rolls each of the 555 visual categories up to its **species** node. In
   NABirds the plumage / age / sex variants carry a parenthetical qualifier
   in their class name (e.g. ``"American Kestrel (Female/juvenile)"``) whose
   parent is the plain-named species node (``"American Kestrel"``); walking
   up to the first ancestor whose name has no parenthesis yields the species.
2. Resolves each species' English common name to a GBIF canonical scientific
   name — via a manual ``COMMON_TO_SCI`` crosswalk first, else the GBIF
   vernacular search API — and then canonicalises it (and pulls higher-rank
   taxonomy) through ``esp_data.discover.gbif_taxonomy.GBIFConverter``,
   exactly as the SSW60 builder does. Fails loudly on any unresolved species
   so the manifest never ships partial taxonomy.
3. Builds per-split (train/test) + unified manifest CSVs with absolute
   ``gs://`` image-path columns and writes ``taxa_gbif.csv`` for auditing.

The image upload + manifest upload is done by ``jobs/build_nabirds.sh`` via
``gsutil -m rsync`` after this script finishes.

NABirds is gated (its terms require agreeing to a usage agreement), so the
tarball must be pre-staged on scratch by the job wrapper — this script only
processes an already-extracted tree.

Usage (see jobs/build_nabirds.sh):
    uv run python scripts/data_preprocessing_scripts/nabirds/build_nabirds.py \
        --src /scratch/$USER/nabirds/nabirds \
        --out /scratch/$USER/nabirds/staging
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from esp_data.discover.gbif_taxonomy import GBIFConverter

GCS_ROOT_DEFAULT = "gs://esp-data-ingestion/nabirds/v0.1.0"
TAXONOMY_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
# GBIF backbone taxonKey for class Aves — restricts vernacular search to birds.
_AVES_HIGHER_TAXON_KEY = 212

# Manual common-name -> scientific-name crosswalk for NABirds species whose
# vernacular does not resolve cleanly against GBIF (filled after the first
# dry-run reports unresolved names, mirroring SSW60's SCI_NAME_FIX).
COMMON_TO_SCI: dict[str, str] = {}


def _parse_id_value(fp: Path) -> dict[str, str]:
    """Parse a NABirds ``<id> <value>`` table into a dict.

    Parameters
    ----------
    fp : Path
        Path to a whitespace-delimited two-column NABirds table.

    Returns
    -------
    dict[str, str]
        Mapping from the first column (id) to the remainder of the line.
    """
    out: dict[str, str] = {}
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, value = line.partition(" ")
            out[key] = value.strip()
    return out


def load_species_map(src: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Roll every NABirds class up to its species node.

    Parameters
    ----------
    src : Path
        Extracted NABirds root (containing ``classes.txt`` / ``hierarchy.txt``).

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        ``(class_to_species_id, species_id_to_common)`` where the first maps
        every class id to its species-node class id and the second maps each
        species-node class id to its common name.
    """
    class_names = _parse_id_value(src / "classes.txt")
    parent = _parse_id_value(src / "hierarchy.txt")  # child_id -> parent_id

    def species_of(class_id: str) -> str:
        current = class_id
        seen: set[str] = set()
        while current not in seen:
            seen.add(current)
            name = class_names.get(current, "")
            if "(" not in name:  # plain-named node == species level
                return current
            nxt = parent.get(current)
            if nxt is None or nxt == current:
                return current
            current = nxt
        return current

    class_to_species: dict[str, str] = {c: species_of(c) for c in class_names}
    species_ids = sorted(set(class_to_species.values()))
    species_to_common = {sid: class_names[sid] for sid in species_ids}
    return class_to_species, species_to_common


def _gbif_vernacular_to_scientific(common: str, cache: dict[str, str | None]) -> str | None:
    """Resolve a bird common name to a scientific name via the GBIF API.

    Parameters
    ----------
    common : str
        English common name (e.g. ``"American Kestrel"``).
    cache : dict[str, str | None]
        In-memory cache of resolved names (mutated in place).

    Returns
    -------
    str | None
        The scientific (species) name, or None if no accepted bird species
        match was found.
    """
    if common in cache:
        return cache[common]
    q = urllib.parse.urlencode(
        {
            "q": common,
            "rank": "SPECIES",
            "status": "ACCEPTED",
            "highertaxonKey": _AVES_HIGHER_TAXON_KEY,
            "limit": 20,
        }
    )
    url = f"https://api.gbif.org/v1/species/search?{q}"
    result: str | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            candidates = data.get("results", [])
            target = common.strip().lower()
            # Prefer a candidate that lists this exact vernacular name.
            for cand in candidates:
                names = {
                    v.get("vernacularName", "").strip().lower()
                    for v in cand.get("vernacularNames", [])
                }
                if target in names and cand.get("species"):
                    result = cand["species"]
                    break
            if result is None:
                for cand in candidates:
                    if cand.get("species"):
                        result = cand["species"]
                        break
            break
        except Exception:  # noqa: BLE001 - transient network errors: retry
            time.sleep(2 * (attempt + 1))
    cache[common] = result
    return result


def link_species(
    species_to_common: dict[str, str], out: Path, gbif_cache: str | None
) -> pd.DataFrame:
    """GBIF-link the NABirds species and write ``taxa_gbif.csv``.

    Parameters
    ----------
    species_to_common : dict[str, str]
        Species-node class id -> common name.
    out : Path
        Staging output directory for ``taxa_gbif.csv``.
    gbif_cache : str | None
        Local cache path for the GBIF animals TSV, or None for the default.

    Returns
    -------
    pd.DataFrame
        Species table with GBIF fields, indexed by ``species_code`` (the
        NABirds species-node class id) and carrying a 0-based ``label``
        assigned by sorted canonical name.

    Raises
    ------
    RuntimeError
        If any species fails to resolve to a GBIF canonical name.
    """
    converter = GBIFConverter(cache_path=gbif_cache) if gbif_cache else GBIFConverter()
    vern_cache: dict[str, str | None] = {}

    records = []
    unresolved = []
    for sid, common in sorted(species_to_common.items(), key=lambda kv: kv[1]):
        sci = COMMON_TO_SCI.get(common) or _gbif_vernacular_to_scientific(common, vern_cache)
        rec = {
            "species_code": sid,
            "species_common": common,
            "scientific_name": sci or "",
            "canonical_name": "",
            "gbifID": "",
            "taxonKey": "",
        }
        for rank in TAXONOMY_RANKS:
            rec[rank] = ""

        info, ok = converter(sci) if sci else ({}, False)
        if ok:
            rec["canonical_name"] = info["canonicalName"]
            rec["gbifID"] = int(info["taxonID"])
            rec["taxonKey"] = int(info["taxonID"])
            for rank in TAXONOMY_RANKS:
                rec[rank] = info.get(rank, "")
        else:
            unresolved.append((sid, common, sci))
        records.append(rec)

    df = pd.DataFrame(records).sort_values("canonical_name").reset_index(drop=True)
    df["label"] = range(len(df))
    df.to_csv(out / "taxa_gbif.csv", index=False)
    print(
        f"taxa_gbif.csv: {len(df)} species, "
        f"resolved {int((df['canonical_name'] != '').sum())}/{len(df)}"
    )
    if unresolved:
        print("UNRESOLVED (add to COMMON_TO_SCI):")
        for sid, common, sci in unresolved:
            print(f"  {sid}\t{common!r}\t-> {sci!r}")
        raise RuntimeError(
            f"{len(unresolved)} species failed GBIF resolution; add COMMON_TO_SCI entries."
        )
    return df.set_index("species_code", drop=False)


_OUT_COLUMNS = [
    "asset_id",
    "modality",
    "label",
    "split",
    "species_code",
    "canonical_name",
    "species_common",
    "family",
    "order",
    "kingdom",
    "phylum",
    "class",
    "genus",
    "gbifID",
    "taxonKey",
    "image_path",
]


def build_manifest(
    src: Path,
    class_to_species: dict[str, str],
    taxa: pd.DataFrame,
    gcs_root: str,
) -> pd.DataFrame:
    """Build the unified NABirds image manifest.

    Parameters
    ----------
    src : Path
        Extracted NABirds root.
    class_to_species : dict[str, str]
        Class id -> species-node class id.
    taxa : pd.DataFrame
        GBIF-linked species table indexed by ``species_code``.
    gcs_root : str
        GCS root for absolute ``image_path`` columns.

    Returns
    -------
    pd.DataFrame
        The manifest with one row per image, columns :data:`_OUT_COLUMNS`.

    Raises
    ------
    RuntimeError
        If any referenced image file is absent under ``src/images/``.
    """
    image_paths = _parse_id_value(src / "images.txt")  # image_id -> relpath
    image_labels = _parse_id_value(src / "image_class_labels.txt")  # image_id -> class_id
    split_flags = _parse_id_value(src / "train_test_split.txt")  # image_id -> "1"/"0"
    images_dir = src / "images"

    rows = []
    missing = 0
    for image_id, relpath in image_paths.items():
        class_id = image_labels[image_id]
        species_id = class_to_species[class_id]
        if not (images_dir / relpath).exists():
            missing += 1
            continue
        sp = taxa.loc[species_id]
        split = "train" if split_flags.get(image_id, "0") == "1" else "test"
        row = {
            "asset_id": image_id,
            "modality": "image",
            "label": int(sp["label"]),
            "split": split,
            "species_code": species_id,
            "canonical_name": sp["canonical_name"],
            "species_common": sp["species_common"],
            "gbifID": sp["gbifID"],
            "taxonKey": sp["taxonKey"],
            "image_path": f"{gcs_root}/images/{relpath}",
        }
        for rank in TAXONOMY_RANKS:
            row[rank] = sp[rank]
        rows.append(row)
    if missing:
        raise RuntimeError(
            f"{missing} NABirds rows reference an image file absent from "
            f"{images_dir}; refusing to ship a manifest with missing files."
        )
    return pd.DataFrame(rows, columns=_OUT_COLUMNS)


def write_splits(df: pd.DataFrame, out: Path) -> None:
    """Write ``nabirds_all`` + per-split (train/test) manifest CSVs.

    Parameters
    ----------
    df : pd.DataFrame
        The unified manifest (with a ``split`` column).
    out : Path
        Staging output directory.
    """
    df.to_csv(out / "nabirds_all.csv", index=False)
    print(f"nabirds_all.csv: {len(df)} rows, {df['canonical_name'].nunique()} species")
    for split in ("train", "test"):
        sub = df[df["split"] == split]
        sub.to_csv(out / f"nabirds_{split}.csv", index=False)
        print(f"nabirds_{split}.csv: {len(sub)} rows")


def main() -> None:
    """Run the full NABirds manifest build."""
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Extracted NABirds root dir.")
    p.add_argument("--out", required=True, help="Staging output dir.")
    p.add_argument("--gcs-root", default=GCS_ROOT_DEFAULT)
    p.add_argument("--gbif-cache", default=None, help="Local cache path for the GBIF animals TSV.")
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== 1. roll categories up to species ===", flush=True)
    class_to_species, species_to_common = load_species_map(src)
    print(f"{len(class_to_species)} categories -> {len(species_to_common)} species")

    print("\n=== 2. GBIF-link species ===", flush=True)
    taxa = link_species(species_to_common, out, args.gbif_cache)

    print("\n=== 3. build manifest ===", flush=True)
    manifest = build_manifest(src, class_to_species, taxa, args.gcs_root)
    write_splits(manifest, out)

    print("\nDONE.")


if __name__ == "__main__":
    main()
