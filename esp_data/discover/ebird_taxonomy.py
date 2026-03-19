"""
eBird taxonomy converter and transforms.

Uses bidirectional JSON caches produced by cache_ebird_conversions.py to map
between Clements scientific names and eBird species codes.
"""

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from esp_data.backends import DataBackend
from esp_data.io import AnyPathT, filesystem_from_path
from esp_data.transforms import register_transform

logger = logging.getLogger("esp_data")

CACHED_DIR = "gs://esp-data-ingestion/taxonomy/ebird_taxonomy_cached"


class EBirdConverter:
    """
    Utility for converting between Clements scientific names and eBird species codes.

    Loads bidirectional JSON caches produced by cache_ebird_conversions.py.
    """

    def __init__(
        self,
        year: int,
        precomputed_dir: str | AnyPathT = CACHED_DIR,
    ) -> None:
        """
        Parameters
        ----------
        year : int
            eBird taxonomy year (2021–2025).
        precomputed_dir : str | AnyPathT, optional
            Directory containing the cached JSON files. Defaults to the GCS location.
            Pass a local path to override (e.g. for testing).
        """
        precomputed_dir = str(precomputed_dir).rstrip("/")
        sci_path = f"{precomputed_dir}/sci_name_to_ebird_{year}.json"
        ebird_path = f"{precomputed_dir}/ebird_to_sci_name_{year}.json"

        fs = filesystem_from_path(sci_path)
        with fs.open(sci_path, "r") as f:
            self.sci_to_ebird: dict[str, str] = json.load(f)

        with fs.open(ebird_path, "r") as f:
            self.ebird_to_sci: dict[str, str] = json.load(f)

    def to_ebird_code(self, sci_name: str) -> tuple[dict[str, str], bool]:
        """Convert a Clements scientific name to an eBird species code.

        Returns
        -------
        tuple[dict[str, str], bool]
            ``({"ebird_code": code}, True)`` on success, ``({}, False)`` on failure.
        """
        code = self.sci_to_ebird.get(sci_name)
        if code is None:
            return {}, False
        return {"ebird_code": code}, True

    def to_scientific_name(self, ebird_code: str) -> tuple[dict[str, str], bool]:
        """Convert an eBird species code to a Clements scientific name.

        Returns
        -------
        tuple[dict[str, str], bool]
            ``({"scientific_name": name}, True)`` on success, ``({}, False)`` on failure.
        """
        name = self.ebird_to_sci.get(ebird_code)
        if name is None:
            return {}, False
        return {"scientific_name": name}, True


# ---------------------------------------------------------------------------
# EBirdToClements — convert ebird_code column → scientific_name column
# ---------------------------------------------------------------------------


class EBirdToClementsConfig(BaseModel):
    """Configuration for EBirdToClements transform."""

    type: Literal["ebird_to_clements"] = "ebird_to_clements"
    feature: str = Field(
        description="Column name containing eBird species codes.",
        default="ebird_code",
    )
    year: int = Field(
        description="eBird taxonomy year (2021–2025).",
        default=2021,
    )


class EBirdToClements:
    """Transform that adds a ``scientific_name`` column by resolving eBird codes."""

    def __init__(self, feature: str = "ebird_code", year: int = 2021) -> None:
        self.feature = feature
        self.converter = EBirdConverter(year=year)

    @classmethod
    def from_config(cls, cfg: EBirdToClementsConfig) -> "EBirdToClements":
        return cls(**cfg.model_dump(exclude={"type"}))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict[str, Any]]:
        if self.feature not in backend.columns:
            raise ValueError(f"Feature column '{self.feature}' not found in data.")

        unique_codes = backend.get_unique(self.feature)
        mapping: dict[str, str] = {}
        resolved = 0
        failed = 0

        for code in unique_codes:
            info, ok = self.converter.to_scientific_name(code)
            if ok:
                mapping[code] = info["scientific_name"]
                resolved += 1
            else:
                logger.debug(f"Failed to resolve scientific name for eBird code: {code}")
                failed += 1

        if failed > 0:
            logger.warning(f"Failed to resolve {failed}/{len(unique_codes)} unique eBird codes")

        backend = backend.map_column(self.feature, mapping=mapping, output_column="scientific_name")

        return backend, {
            "feature": self.feature,
            "resolved": resolved,
            "failed": failed,
            "columns_added": ["scientific_name"],
        }


# ---------------------------------------------------------------------------
# ClementsToEBird — convert scientific_name column → ebird_code column
# ---------------------------------------------------------------------------


class ClementsToEBirdConfig(BaseModel):
    """Configuration for ClementsToEBird transform."""

    type: Literal["clements_to_ebird"] = "clements_to_ebird"
    feature: str = Field(
        description="Column name containing Clements scientific names.",
        default="scientific_name",
    )
    year: int = Field(
        description="eBird taxonomy year (2021–2025).",
        default=2021,
    )


class ClementsToEBird:
    """Transform that adds an ``ebird_code`` column by resolving scientific names."""

    def __init__(self, feature: str = "scientific_name", year: int = 2021) -> None:
        self.feature = feature
        self.converter = EBirdConverter(year=year)

    @classmethod
    def from_config(cls, cfg: ClementsToEBirdConfig) -> "ClementsToEBird":
        return cls(**cfg.model_dump(exclude={"type"}))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict[str, Any]]:
        if self.feature not in backend.columns:
            raise ValueError(f"Feature column '{self.feature}' not found in data.")

        unique_names = backend.get_unique(self.feature)
        mapping: dict[str, str] = {}
        resolved = 0
        failed = 0

        for name in unique_names:
            info, ok = self.converter.to_ebird_code(name)
            if ok:
                mapping[name] = info["ebird_code"]
                resolved += 1
            else:
                logger.debug(f"Failed to resolve eBird code for scientific name: {name}")
                failed += 1

        if failed > 0:
            logger.warning(
                f"Failed to resolve {failed}/{len(unique_names)} unique scientific names"
            )

        backend = backend.map_column(self.feature, mapping=mapping, output_column="ebird_code")

        return backend, {
            "feature": self.feature,
            "resolved": resolved,
            "failed": failed,
            "columns_added": ["ebird_code"],
        }


register_transform(EBirdToClementsConfig, EBirdToClements)
register_transform(ClementsToEBirdConfig, ClementsToEBird)
