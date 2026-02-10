"""
This script caches all possible outputs with ok==True of the Slow version of
GBIFConverter, to allow for faster lookups.

SlowGBIFConverter redirects synonyms to GBIF accepted taxonomy.

Uses a preprocessed TSV of GBIF backbone taxonomy for animals,
downloadable from Google Cloud Storage, but can be cached
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from esp_data.io import AnyPathT, exists, filesystem_from_path

logger = logging.getLogger("esp_data")

TAXONOMY_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
VERSION = "0.1.0"
DEFAULT_LOCATION = "gs://esp-ml-datasets/gbif_taxonomy/v0.1.0/gbif_animals.tsv"
# Use current script directory for cache
this_dir = Path(__file__).parent.resolve()
OUTPUT_CACHE_PATH = "gs://esp-ml-datasets/gbif_taxonomy/v0.1.0/gbif_animals_converter_cache.json"


class SlowGBIFConverter:
    """
    Utility for resolving GBIF taxonomic names to their accepted species-level
    usage using the GBIF backbone taxonomy.

    The underlying GBIF table is indexed by both ``taxonID`` (unique) and
    ``canonicalName`` (potentially non-unique) to support efficient lookups.

    Parameters
    ----------
    gbif_animals_tsv_fp : str, optional
        Path to a TSV file containing animal GBIF taxonomy records,
        preprocessed via scripts/v2_source_to_tsv.py

    cache_path : str | AnyPathT | None, optional
        Path to a local cached copy of the GBIF taxonomy table. If provided,
        this path will be used instead of ``gbif_animals_tsv_fp``.
    """

    def __init__(
        self,
        gbif_animals_tsv_fp: str | AnyPathT = DEFAULT_LOCATION,
        cache_path: str | AnyPathT | None = None,
    ) -> None:
        """
        Load the GBIF animals taxonomy table and construct lookup indices.

        Parameters
        ----------
        gbif_animals_tsv_fp : str, optional
            Path to a TSV file containing animal GBIF taxonomy records,
            preprocessed via scripts/v2_source_to_tsv.py
        cache_path : str | AnyPathT | None, optional
            Path to a local cached copy of the GBIF taxonomy table. If provided,
            this path will be used instead of ``gbif_animals_tsv_fp``.
        """
        if cache_path is not None:
            if exists(cache_path):
                gbif_animals_tsv_fp = cache_path

        fs = filesystem_from_path(gbif_animals_tsv_fp)
        with fs.open(gbif_animals_tsv_fp, "rb") as f:
            self.df = pd.read_csv(f, sep="\t")

        # Ensure unique integer taxonID index for O(1)-ish label lookups
        self.df["taxonID"] = self.df["taxonID"].astype(np.int64)
        self.df = self.df.set_index("taxonID", verify_integrity=True, drop=False)

        # Canonical name index may be non-unique but with low-dup rate
        self.df_by_canonical_name = self.df.set_index("canonicalName", drop=False)

    def __call__(self, lookup_name: str) -> tuple[dict[str, Any], bool]:
        """
        Resolve a scientific (canonical) name to its accepted species-level
        GBIF taxonomic record.

        The method:
        - Resolves duplicate canonical-name matches by preferring accepted usages.
        - Walks up the taxonomy if the matched record is below species rank.
        - Redirects unaccepted names to their accepted usage.
        - Detects and aborts on cyclic or inconsistent references.

        Parameters
        ----------
        lookup_name : str
            Canonical scientific name to resolve (e.g., ``"Corvus corax"``).

        Returns
        -------
        (dict, bool)
            A tuple ``(info, ok)`` where ``info`` is a dictionary containing the
            resolved GBIF taxonomic fields (empty on failure), and ``ok`` is a
            boolean indicating whether resolution succeeded.
        """
        visited: set[str] = set()

        while True:
            # Protect against pathological cycles / corrupted pointers
            if lookup_name in visited:
                return {}, False
            visited.add(lookup_name)

            try:
                looked_up = self.df_by_canonical_name.loc[lookup_name]
            except KeyError:
                return {}, False

            # Resolve duplicates: prefer an accepted usage if present; else take first.
            if isinstance(looked_up, pd.DataFrame):
                accepted_mask = looked_up["taxonomicStatus"].to_numpy() == "accepted"
                if accepted_mask.any():
                    looked_up = looked_up.iloc[int(accepted_mask.argmax())]
                else:
                    looked_up = looked_up.iloc[0]

            # Resolve lower taxa (walk up to species)
            if looked_up["taxonRank"] != "species":
                parent_id = looked_up["parentNameUsageID"]
                if pd.isna(parent_id):
                    return {}, False
                parent_id = int(parent_id)

                try:
                    lookup_name = self.df.loc[parent_id, "canonicalName"]
                except KeyError:
                    return {}, False

                continue

            # Resolve unaccepted names (walk to accepted/doubtful)
            # We allow doubtful names for completeness because they don't have synonyms.
            # e.g. Aegithalos caudatus is considered doubtful but is widely recognized.
            if looked_up["taxonomicStatus"] not in ["accepted", "doubtful"]:
                accepted_id = looked_up["acceptedNameUsageID"]
                if pd.isna(accepted_id):
                    return {}, False
                accepted_id = int(accepted_id)

                try:
                    lookup_name = self.df.loc[accepted_id, "canonicalName"]
                except KeyError:
                    return {}, False

                continue

            # Return info as dict
            out = looked_up.to_dict()
            out["canonicalName"] = lookup_name
            return out, True


def _write_gbif_tsv(tmp_path: Path, rows: List[Dict[str, Any]]) -> str:
    """
    Write a minimal GBIF-like TSV for testing and return its filepath.

    The SlowGBIFConverter implementation expects at least these columns:
    - taxonID
    - canonicalName
    - taxonomicStatus
    - taxonRank
    - parentNameUsageID
    - acceptedNameUsageID

    Raises
    ------
    AssertionError
        If required columns are missing

    Returns
    ------
    str filepath of cache
    """
    df = pd.DataFrame(rows)

    # Ensure column presence and ordering (useful for debugging)
    required_cols = [
        "taxonID",
        "canonicalName",
        "taxonomicStatus",
        "taxonRank",
        "parentNameUsageID",
        "acceptedNameUsageID",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing required columns for fixture TSV: {missing}")

    fp = tmp_path / "gbif_animals_minimal.tsv"
    df.to_csv(fp, sep="\t", index=False)
    return str(fp)


# SlowGBIFConverter Unit Tests


def test_get_accepted_species_info_no_match(tmp_path: Path) -> None:
    fp = _write_gbif_tsv(
        tmp_path,
        rows=[
            {
                "taxonID": 1,
                "canonicalName": "Corvus corax",
                "taxonomicStatus": "accepted",
                "taxonRank": "species",
                "parentNameUsageID": np.nan,
                "acceptedNameUsageID": np.nan,
            }
        ],
    )
    converter = SlowGBIFConverter(gbif_animals_tsv_fp=fp, cache_path=None)

    out, ok = converter("Does not exist")
    assert out == {}
    assert ok is False


def test_get_accepted_species_info_accepts_species(tmp_path: Path) -> None:
    fp = _write_gbif_tsv(
        tmp_path,
        rows=[
            {
                "taxonID": 10,
                "canonicalName": "Corvus corax",
                "taxonomicStatus": "accepted",
                "taxonRank": "species",
                "parentNameUsageID": np.nan,
                "acceptedNameUsageID": np.nan,
            }
        ],
    )
    converter = SlowGBIFConverter(gbif_animals_tsv_fp=fp, cache_path=None)

    out, ok = converter("Corvus corax")
    assert ok is True
    assert out["taxonID"] == 10
    assert out["taxonomicStatus"] == "accepted"
    assert out["taxonRank"] == "species"
    assert out["canonicalName"] == "Corvus corax"


def test_get_accepted_species_info_resolves_synonym_to_accepted(tmp_path: Path) -> None:
    fp = _write_gbif_tsv(
        tmp_path,
        rows=[
            # accepted usage
            {
                "taxonID": 100,
                "canonicalName": "Puma concolor",
                "taxonomicStatus": "accepted",
                "taxonRank": "species",
                "parentNameUsageID": np.nan,
                "acceptedNameUsageID": np.nan,
            },
            # synonym that points to accepted usage
            {
                "taxonID": 101,
                "canonicalName": "Felis concolor",
                "taxonomicStatus": "synonym",
                "taxonRank": "species",
                "parentNameUsageID": np.nan,
                "acceptedNameUsageID": 100.0,
            },
        ],
    )
    converter = SlowGBIFConverter(gbif_animals_tsv_fp=fp, cache_path=None)

    out, ok = converter("Felis concolor")
    assert ok is True
    assert out["taxonID"] == 100
    assert out["canonicalName"] == "Puma concolor"
    assert out["taxonomicStatus"] == "accepted"
    assert out["taxonRank"] == "species"


def test_get_accepted_species_info_walks_up_from_lower_rank(tmp_path: Path) -> None:
    fp = _write_gbif_tsv(
        tmp_path,
        rows=[
            # accepted species
            {
                "taxonID": 200,
                "canonicalName": "Canis lupus",
                "taxonomicStatus": "accepted",
                "taxonRank": "species",
                "parentNameUsageID": np.nan,
                "acceptedNameUsageID": np.nan,
            },
            # subspecies that points to parent species
            {
                "taxonID": 201,
                "canonicalName": "Canis lupus familiaris",
                "taxonomicStatus": "accepted",
                "taxonRank": "subspecies",
                "parentNameUsageID": 200.0,
                "acceptedNameUsageID": np.nan,
            },
        ],
    )
    converter = SlowGBIFConverter(gbif_animals_tsv_fp=fp, cache_path=None)

    out, ok = converter("Canis lupus familiaris")
    assert ok is True
    assert out["taxonID"] == 200
    assert out["canonicalName"] == "Canis lupus"
    assert out["taxonRank"] == "species"
    assert out["taxonomicStatus"] == "accepted"


def test_get_accepted_species_info_duplicate_canonical_prefers_accepted(tmp_path: Path) -> None:
    fp = _write_gbif_tsv(
        tmp_path,
        rows=[
            # accepted row for same canonicalName
            {
                "taxonID": 300,
                "canonicalName": "Dup name",
                "taxonomicStatus": "accepted",
                "taxonRank": "species",
                "parentNameUsageID": np.nan,
                "acceptedNameUsageID": np.nan,
            },
            # non-accepted duplicate
            {
                "taxonID": 301,
                "canonicalName": "Dup name",
                "taxonomicStatus": "synonym",
                "taxonRank": "species",
                "parentNameUsageID": np.nan,
                "acceptedNameUsageID": 300.0,
            },
        ],
    )
    converter = SlowGBIFConverter(gbif_animals_tsv_fp=fp, cache_path=None)

    out, ok = converter("Dup name")
    assert ok is True
    assert out["taxonID"] == 300
    assert out["taxonomicStatus"] == "accepted"
    assert out["taxonRank"] == "species"
    assert out["canonicalName"] == "Dup name"


def test_get_accepted_species_info_cycle_is_detected(tmp_path: Path) -> None:
    fp = _write_gbif_tsv(
        tmp_path,
        rows=[
            # non-species record whose parent points to itself (cycle)
            {
                "taxonID": 400,
                "canonicalName": "Cycle name",
                "taxonomicStatus": "accepted",
                "taxonRank": "subspecies",
                "parentNameUsageID": 400.0,
                "acceptedNameUsageID": np.nan,
            }
        ],
    )
    converter = SlowGBIFConverter(gbif_animals_tsv_fp=fp, cache_path=None)

    out, ok = converter("Cycle name")
    assert out == {}
    assert ok is False


def create_cache() -> None:
    cached = {}
    converter = SlowGBIFConverter()
    to_lookup = list(converter.df_by_canonical_name.index)

    issues = []

    for i, lookup in enumerate(to_lookup):
        if i % 100 == 0:
            print(f"{i} / {len(to_lookup)} completed")
        result, ok = converter(lookup)
        if not ok:
            issues.append(lookup)
            continue
        cached[lookup] = result

    print(f"Found {len(issues)} issues in {len(to_lookup)} records")

    pd.DataFrame.from_dict(cached, orient="index").to_json(OUTPUT_CACHE_PATH, indent=2)


if __name__ == "__main__":
    # run tests
    tmp_path = Path(__file__).parent.resolve()
    test_get_accepted_species_info_no_match(tmp_path)
    test_get_accepted_species_info_accepts_species(tmp_path)
    test_get_accepted_species_info_resolves_synonym_to_accepted(tmp_path)
    test_get_accepted_species_info_walks_up_from_lower_rank(tmp_path)
    test_get_accepted_species_info_duplicate_canonical_prefers_accepted(tmp_path)
    test_get_accepted_species_info_cycle_is_detected(tmp_path)
    print("passed tests")

    # create the cache
    create_cache()
