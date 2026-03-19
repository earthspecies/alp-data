from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from esp_data.backends import PandasBackend
from esp_data.discover import (
    ClementsToEBird,
    ClementsToEBirdConfig,
    EBirdConverter,
    EBirdToClements,
    EBirdToClementsConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCI_TO_EBIRD = {
    "Corvus corax": "corcor",
    "Passer domesticus": "houspa",
    "Canis lupus": "canlup1",
}
_EBIRD_TO_SCI = {v: k for k, v in _SCI_TO_EBIRD.items()}

YEAR = 2024


def _create_ebird_jsons(tmp_path: Path) -> Path:
    """Write both cached JSON files to tmp_path and return the directory."""
    (tmp_path / f"sci_name_to_ebird_{YEAR}.json").write_text(
        json.dumps(_SCI_TO_EBIRD, indent=2)
    )
    (tmp_path / f"ebird_to_sci_name_{YEAR}.json").write_text(
        json.dumps(_EBIRD_TO_SCI, indent=2)
    )
    return tmp_path


# ---------------------------------------------------------------------------
# EBirdConverter tests
# ---------------------------------------------------------------------------

def test_ebird_converter_to_ebird_code(tmp_path: Path) -> None:
    cache_dir = _create_ebird_jsons(tmp_path)
    converter = EBirdConverter(year=YEAR, precomputed_dir=cache_dir)

    info, ok = converter.to_ebird_code("Corvus corax")
    assert ok
    assert info == {"ebird_code": "corcor"}

    info, ok = converter.to_ebird_code("Fraudulus animalaticus")
    assert not ok
    assert info == {}


def test_ebird_converter_to_scientific_name(tmp_path: Path) -> None:
    cache_dir = _create_ebird_jsons(tmp_path)
    converter = EBirdConverter(year=YEAR, precomputed_dir=cache_dir)

    info, ok = converter.to_scientific_name("houspa")
    assert ok
    assert info == {"scientific_name": "Passer domesticus"}

    info, ok = converter.to_scientific_name("xxxxx")
    assert not ok
    assert info == {}


# ---------------------------------------------------------------------------
# EBirdToClements transform tests
# ---------------------------------------------------------------------------

def test_ebird_to_clements_basic(tmp_path: Path) -> None:
    cache_dir = _create_ebird_jsons(tmp_path)
    transform = EBirdToClements(feature="ebird_code", year=YEAR)
    transform.converter = EBirdConverter(year=YEAR, precomputed_dir=cache_dir)

    df = pd.DataFrame({"ebird_code": ["corcor", "houspa"]})
    backend = PandasBackend(df)

    result_backend, _ = transform(backend)
    result_df = result_backend.unwrap

    assert "scientific_name" in result_df.columns
    assert result_df.loc[0, "scientific_name"] == "Corvus corax"
    assert result_df.loc[1, "scientific_name"] == "Passer domesticus"


def test_ebird_to_clements_from_config(tmp_path: Path) -> None:
    cache_dir = _create_ebird_jsons(tmp_path)
    cfg = EBirdToClementsConfig(type="ebird_to_clements", feature="code", year=YEAR)
    transform = EBirdToClements.from_config(cfg)
    transform.converter = EBirdConverter(year=YEAR, precomputed_dir=cache_dir)

    df = pd.DataFrame({"code": ["canlup1"]})
    backend = PandasBackend(df)

    result_backend, metadata = transform(backend)
    assert result_backend[0]["scientific_name"] == "Canis lupus"
    assert metadata["feature"] == "code"


def test_ebird_to_clements_missing_column(tmp_path: Path) -> None:
    cache_dir = _create_ebird_jsons(tmp_path)
    transform = EBirdToClements(feature="ebird_code", year=YEAR)
    transform.converter = EBirdConverter(year=YEAR, precomputed_dir=cache_dir)

    df = pd.DataFrame({"wrong_column": ["corcor"]})
    backend = PandasBackend(df)

    with pytest.raises(ValueError, match="Feature column 'ebird_code' not found in data"):
        transform(backend)


def test_ebird_to_clements_metadata(tmp_path: Path) -> None:
    cache_dir = _create_ebird_jsons(tmp_path)
    transform = EBirdToClements(feature="ebird_code", year=YEAR)
    transform.converter = EBirdConverter(year=YEAR, precomputed_dir=cache_dir)

    df = pd.DataFrame({"ebird_code": ["corcor", "corcor", "houspa", "xxxxx"]})
    backend = PandasBackend(df)

    _, metadata = transform(backend)
    assert metadata["feature"] == "ebird_code"
    assert metadata["resolved"] == 2  # corcor and houspa are unique resolved
    assert metadata["failed"] == 1    # xxxxx fails
    assert metadata["columns_added"] == ["scientific_name"]


def test_ebird_to_clements_unresolvable(tmp_path: Path) -> None:
    cache_dir = _create_ebird_jsons(tmp_path)
    transform = EBirdToClements(feature="ebird_code", year=YEAR)
    transform.converter = EBirdConverter(year=YEAR, precomputed_dir=cache_dir)

    df = pd.DataFrame({"ebird_code": ["zzz1", "zzz2"]})
    backend = PandasBackend(df)

    result_backend, metadata = transform(backend)
    result_df = result_backend.unwrap

    assert pd.isna(result_df.loc[0, "scientific_name"])
    assert metadata["resolved"] == 0
    assert metadata["failed"] == 2


# ---------------------------------------------------------------------------
# ClementsToEBird transform tests
# ---------------------------------------------------------------------------

def test_clements_to_ebird_basic(tmp_path: Path) -> None:
    cache_dir = _create_ebird_jsons(tmp_path)
    transform = ClementsToEBird(feature="scientific_name", year=YEAR)
    transform.converter = EBirdConverter(year=YEAR, precomputed_dir=cache_dir)

    df = pd.DataFrame({"scientific_name": ["Corvus corax", "Passer domesticus"]})
    backend = PandasBackend(df)

    result_backend, _ = transform(backend)
    result_df = result_backend.unwrap

    assert "ebird_code" in result_df.columns
    assert result_df.loc[0, "ebird_code"] == "corcor"
    assert result_df.loc[1, "ebird_code"] == "houspa"


def test_clements_to_ebird_from_config(tmp_path: Path) -> None:
    cache_dir = _create_ebird_jsons(tmp_path)
    cfg = ClementsToEBirdConfig(type="clements_to_ebird", feature="sci", year=YEAR)
    transform = ClementsToEBird.from_config(cfg)
    transform.converter = EBirdConverter(year=YEAR, precomputed_dir=cache_dir)

    df = pd.DataFrame({"sci": ["Canis lupus"]})
    backend = PandasBackend(df)

    result_backend, metadata = transform(backend)
    assert result_backend[0]["ebird_code"] == "canlup1"
    assert metadata["feature"] == "sci"


def test_clements_to_ebird_missing_column(tmp_path: Path) -> None:
    cache_dir = _create_ebird_jsons(tmp_path)
    transform = ClementsToEBird(feature="scientific_name", year=YEAR)
    transform.converter = EBirdConverter(year=YEAR, precomputed_dir=cache_dir)

    df = pd.DataFrame({"wrong_column": ["Corvus corax"]})
    backend = PandasBackend(df)

    with pytest.raises(ValueError, match="Feature column 'scientific_name' not found in data"):
        transform(backend)


def test_clements_to_ebird_metadata(tmp_path: Path) -> None:
    cache_dir = _create_ebird_jsons(tmp_path)
    transform = ClementsToEBird(feature="scientific_name", year=YEAR)
    transform.converter = EBirdConverter(year=YEAR, precomputed_dir=cache_dir)

    df = pd.DataFrame({
        "scientific_name": ["Corvus corax", "Corvus corax", "Passer domesticus", "Unknown sp."]
    })
    backend = PandasBackend(df)

    _, metadata = transform(backend)
    assert metadata["feature"] == "scientific_name"
    assert metadata["resolved"] == 2  # corcor and houspa unique resolved
    assert metadata["failed"] == 1    # Unknown sp. fails
    assert metadata["columns_added"] == ["ebird_code"]


def test_clements_to_ebird_unresolvable(tmp_path: Path) -> None:
    cache_dir = _create_ebird_jsons(tmp_path)
    transform = ClementsToEBird(feature="scientific_name", year=YEAR)
    transform.converter = EBirdConverter(year=YEAR, precomputed_dir=cache_dir)

    df = pd.DataFrame({"scientific_name": ["Unknown sp.", "Another unknown"]})
    backend = PandasBackend(df)

    result_backend, metadata = transform(backend)
    result_df = result_backend.unwrap

    assert pd.isna(result_df.loc[0, "ebird_code"])
    assert metadata["resolved"] == 0
    assert metadata["failed"] == 2
