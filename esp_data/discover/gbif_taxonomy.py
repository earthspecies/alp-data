"""
V2 of taxonomy lookup service. Redirects synonyms to GBIF accepted taxonomy.

Uses a preprocessed TSV of GBIF backbone taxonomy for animals,
downloadable from Google Cloud Storage, but can be cached
"""

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from esp_data.backends import DataBackend
from esp_data.io import AnyPathT, exists, filesystem_from_path
from esp_data.transforms import register_transform

logger = logging.getLogger("esp_data")


TAXONOMY_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
# TODO: need a more managed location for this file
DEFAULT_LOCATION = "gs://sound-event-detection/taxonomy/gbif_animals.tsv"
CACHE_PATH = None  # TODO: what's the right place for a default cache?


class GBIFConverter:
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

    Examples
    --------
    >>> converter = GBIFConverter()
    >>> info, ok = converter("Corvus corax")
    >>> assert ok
    >>> assert info["canonicalName"] == "Corvus corax"
    >>> assert info["taxonRank"] == "species"
    """

    def __init__(
        self,
        gbif_animals_tsv_fp: str | AnyPathT = DEFAULT_LOCATION,
        cache_path: str | AnyPathT | None = CACHE_PATH,
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
        _save_tsv = False
        if cache_path is not None:
            if not exists(cache_path):
                logger.warning(
                    f"GBIFConverter: cache_path {cache_path} does not exist but has been set"
                    ", so we'll download and save the data to it."
                )
                _save_tsv = True
            else:
                gbif_animals_tsv_fp = cache_path

        fs = filesystem_from_path(gbif_animals_tsv_fp)
        with fs.open(gbif_animals_tsv_fp, "rb") as f:
            self.df = pd.read_csv(f, sep="\t")

        if _save_tsv:
            self.df.to_csv(cache_path, sep="\t", index=False)

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


class AddTaxonomyConfig(BaseModel):
    """Configuration for AddTaxonomyTransform."""

    type: Literal["add_taxonomy"] = "add_taxonomy"
    feature: str = Field(
        description="Column name containing scientific names to look up.",
        default="scientific_name",
    )
    gbif_taxonomy_path: str = Field(
        description="Path to GBIF taxonomy TSV file.",
        default=DEFAULT_LOCATION,
    )
    add_taxonomic_name: bool = Field(
        description="Whether to add a 'taxonomic_name' column with the full taxonomic name.",
        default=False,
    )

    @field_validator("gbif_taxonomy_path")
    def check_file_exists(cls, v: str) -> str:
        if not exists(v):
            raise ValueError(f"GBIF data file does not exist: {v}")
        return v


class AddTaxonomy:
    """
    Transform that adds resolved GBIF taxonomy info to each row.

    Uses GBIFConverter to resolve scientific names in a specified column
    to their accepted species-level taxonomic records. New columns are added
    for each taxonomy rank: 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'.
    An extra column 'taxonomic_name' is also added containing the resolved
    canonical species name.

    Parameters
    ----------
    feature : str
        Column name containing scientific names to look up.
    gbif_taxonomy_db : str | AnyPathT
        Path to GBIF taxonomy TSV file.

    Examples
    --------
    >>> from esp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> df = pd.DataFrame({"scientific_name": ["Corvus corax", "Passer domesticus"]})
    >>> backend = PandasBackend(df)
    >>> transform = AddTaxonomyTransform(feature="scientific_name")
    >>> transformed_backend, metadata = transform(backend)
    """

    def __init__(
        self,
        feature: str = "scientific_name",
        gbif_taxonomy_path: str | AnyPathT = DEFAULT_LOCATION,
        add_taxonomic_name: bool = False,
    ) -> None:
        self.feature = feature
        self.converter = GBIFConverter(gbif_taxonomy_path)
        self.add_taxonomic_name = add_taxonomic_name

    @classmethod
    def from_config(cls, cfg: AddTaxonomyConfig) -> "AddTaxonomy":
        return cls(**cfg.model_dump(exclude={"type"}))

    def _make_taxonomic_name(self, info: dict[str, str]) -> str | None:
        """Construct the full taxonomic name from GBIF info.

        Parameters
        ----------
        info : dict[str, str]
            GBIF taxonomic record fields.

        Returns
        -------
        str | None
            Full taxonomic name (including higher ranks) or None if unavailable.
        """
        taxonomic_name = ""
        for rank in TAXONOMY_RANKS[:-1]:  # Exclude genus
            rank_value = info.get(rank)
            if rank_value:
                if taxonomic_name:
                    taxonomic_name += " "
                taxonomic_name += rank_value
        # Add canonicalName
        canonical_name = info.get("canonicalName")
        if canonical_name:
            if taxonomic_name:
                taxonomic_name += " "
            taxonomic_name += canonical_name

        return taxonomic_name if len(taxonomic_name) > 0 else None

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """
        Apply the transform to add taxonomy columns.

        Parameters
        ----------
        backend : DataBackend
            The backend wrapping the DataFrame to transform.

        Returns
        -------
        tuple[DataBackend, dict]
            A tuple containing the transformed backend with taxonomy columns added,
            and metadata about the resolution (success/failure counts).

        Raises
        ------
        ValueError
            If the specified feature column is not found in the DataFrame.
        """
        if self.feature not in backend.columns:
            raise ValueError(f"Feature column '{self.feature}' not found in data.")

        # Get unique scientific names to look up (avoids redundant lookups)
        unique_names = backend.get_unique(self.feature)

        # Build a lookup cache: rank -> [{scientific_name -> value}]
        # e.g. {'kingdom': [{'Corvus corax': 'Animalia'}, ...], ...}
        EXTENDED_RANKS = TAXONOMY_RANKS
        if self.add_taxonomic_name:
            EXTENDED_RANKS = TAXONOMY_RANKS + ["taxonomic_name"]
        taxonomy_cache: dict[str, list[tuple[str, str]]] = {r: [] for r in EXTENDED_RANKS}
        success_count = 0
        failure_count = 0

        for name in unique_names:
            info, ok = self.converter(name)
            if ok:
                # Fill by rank
                for rank in TAXONOMY_RANKS:
                    taxonomy_cache[rank].append((name, info.get(rank)))
                if self.add_taxonomic_name:
                    taxonomy_cache["taxonomic_name"].append((name, self._make_taxonomic_name(info)))
                success_count += 1
            else:
                failure_count += 1
                logger.debug(f"Failed to resolve taxonomy for: {name}")

        if failure_count > 0:
            logger.warning(f"Failed to resolve {failure_count}/{len(unique_names)} unique names")

        # Map resolved taxonomy back to backend, adding new columns
        for rank in EXTENDED_RANKS:
            rank_mapping = {src: target for src, target in taxonomy_cache[rank]}
            backend = backend.map_column(self.feature, mapping=rank_mapping, output_column=rank)

        metadata = {
            "feature": self.feature,
            "unique_names": len(unique_names),
            "resolved": success_count,
            "failed": failure_count,
            "taxonomy_columns_added": EXTENDED_RANKS,
        }

        return backend, metadata


register_transform(AddTaxonomyConfig, AddTaxonomy)
