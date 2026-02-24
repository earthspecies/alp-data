"""
V2 of taxonomy lookup service. Redirects synonyms to GBIF accepted taxonomy.

Uses a preprocessed TSV of GBIF backbone taxonomy for animals,
downloadable from Google Cloud Storage, but can be cached
"""

import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from esp_data.backends import DataBackend
from esp_data.io import AnyPathT, exists, filesystem_from_path
from esp_data.transforms import register_transform

logger = logging.getLogger("esp_data")

TAXONOMY_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
# TODO: need a better versioning system
VERSION = "0.1.0"
# location of precomputed outputs
DEFAULT_PRECOMPUTED_LOCATION = (
    "gs://esp-ml-datasets/gbif_taxonomy/v0.1.0/gbif_animals_converter_cache.json"
)
# Use current script directory for cache
this_dir = Path(__file__).parent.resolve()
PRECOMPUTED_CACHE_PATH = str(this_dir / "gbif_animals_converter_cache_0.1.0.json")


class GBIFConverter:
    """
    Utility for resolving GBIF taxonomic names to their accepted species-level
    usage using the GBIF backbone taxonomy.

    The underlying GBIF table is indexed by both ``taxonID`` (unique) and
    ``canonicalName`` (potentially non-unique) to support efficient lookups.

    Parameters
    ----------
    precomputed_fp : str, optional
        Path to a json file containing animal GBIF taxonomy records,
        preprocessed via scripts/taxonomy_v2_source_to_tsv.py and
        scripts/data_preprocessing_scripts/cache_gbif_taxonomy_conversion.py

    precomputed_cache_path : str | AnyPathT | None, optional
        Path to a local cached copy of the GBIF taxonomy json. If provided,
        this path will be used instead of ``precomputed_fp``.
    """

    def __init__(
        self,
        precomputed_fp: str | AnyPathT = DEFAULT_PRECOMPUTED_LOCATION,
        precomputed_cache_path: str | AnyPathT | None = PRECOMPUTED_CACHE_PATH,
    ) -> None:
        """
        Load the GBIF animals taxonomy table and construct lookup indices.

        Parameters
        ----------
        precomputed_fp : str, optional
            Path to a json file containing precomputed outputs from
            scripts/cache_gbif_taxonomy_conversion.py, which shortcuts the
            search operations that are implemented in this class
        precomputed_cache_path : str | AnyPathT | None
            Path to a local cached copy of the GBIF taxonomy table. If provided,
            this path will be used instead of ``gbif_animals_converter_cache.json``.
        """

        _save_json = False
        if precomputed_cache_path is not None:
            if not exists(precomputed_cache_path):
                logger.warning(
                    f"GBIFConverter: precomputed_cache_path {precomputed_cache_path}"
                    "does not exist but has been set"
                    ", so we'll download and save the data to it."
                )
                _save_json = True
            else:
                precomputed_fp = precomputed_cache_path

        fs = filesystem_from_path(precomputed_fp)

        with fs.open(precomputed_fp, "rb") as f:
            self.lookupdict = pd.read_json(f).to_dict(orient="index")

        if _save_json:
            pd.DataFrame.from_dict(self.lookupdict, orient="index").to_json(
                precomputed_cache_path, indent=2
            )

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

        out = self.lookupdict.get(lookup_name, False)
        if not out:
            return {}, False
        else:
            return out, True


class AddTaxonomyConfig(BaseModel):
    """Configuration for AddTaxonomyTransform."""

    type: Literal["add_taxonomy"] = "add_taxonomy"
    feature: str = Field(
        description="Column name containing scientific names to look up.",
        default="scientific_name",
    )
    gbif_precomputed_taxonomy_path: str = Field(
        description="Path to GBIF taxonomy json file.",
        default=DEFAULT_PRECOMPUTED_LOCATION,
    )
    add_taxonomic_name: bool = Field(
        description="Whether to add a 'taxonomic_name' column with the full taxonomic name.",
        default=False,
    )

    @field_validator("gbif_precomputed_taxonomy_path")
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
    An extra column 'taxonomic_name' is also added, which concatenates
    the higher ranks with the canonical name e.g.
    "Animalia Chordata Aves Passeriformes Corvidae Corvus corax".

    Parameters
    ----------
    feature : str
        Column name containing scientific names to look up.
    precomputed_fp : str | AnyPathT
        Path to precomputed GBIF taxonomy json file.
    """

    def __init__(
        self,
        feature: str = "scientific_name",
        gbif_precomputed_taxonomy_path: str | AnyPathT = DEFAULT_PRECOMPUTED_LOCATION,
        add_taxonomic_name: bool = False,
    ) -> None:
        self.feature = feature
        self.converter = GBIFConverter(
            precomputed_cache_path=gbif_precomputed_taxonomy_path,
        )
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
        if not info:
            return None

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
            "resolved": success_count,
            "failed": failure_count,
            "taxonomy_columns_added": EXTENDED_RANKS,
        }

        return backend, metadata


register_transform(AddTaxonomyConfig, AddTaxonomy)
