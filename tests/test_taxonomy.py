from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from esp_data.backends import PandasBackend
from esp_data.discover import AddTaxonomy, AddTaxonomyConfig, GBIFConverter


def _write_gbif_tsv(tmp_path: Path, rows: List[Dict[str, Any]]) -> str:
    """
    Write a minimal GBIF-like TSV for testing and return its filepath.

    The GBIFConverter implementation expects at least these columns:
    - taxonID
    - canonicalName
    - taxonomicStatus
    - taxonRank
    - parentNameUsageID
    - acceptedNameUsageID
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


# GBIFConverter Unit Tests

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
    converter = GBIFConverter(gbif_animals_tsv_fp=fp)

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
    converter = GBIFConverter(gbif_animals_tsv_fp=fp)

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
    converter = GBIFConverter(gbif_animals_tsv_fp=fp)

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
    converter = GBIFConverter(gbif_animals_tsv_fp=fp)

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
    converter = GBIFConverter(gbif_animals_tsv_fp=fp)

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
    converter = GBIFConverter(gbif_animals_tsv_fp=fp)

    out, ok = converter("Cycle name")
    assert out == {}
    assert ok is False


# AddTaxonomy Transform Unit Tests

def _create_gbif_tsv_with_taxonomy(tmp_path: Path) -> str:
    """Create a GBIF TSV with full taxonomy info for testing AddTaxonomy."""
    rows = [
        {
            "taxonID": 1,
            "canonicalName": "Corvus corax",
            "taxonomicStatus": "accepted",
            "taxonRank": "species",
            "parentNameUsageID": np.nan,
            "acceptedNameUsageID": np.nan,
            "kingdom": "Animalia",
            "phylum": "Chordata",
            "class": "Aves",
            "order": "Passeriformes",
            "family": "Corvidae",
            "genus": "Corvus",
        },
        {
            "taxonID": 2,
            "canonicalName": "Passer domesticus",
            "taxonomicStatus": "accepted",
            "taxonRank": "species",
            "parentNameUsageID": np.nan,
            "acceptedNameUsageID": np.nan,
            "kingdom": "Animalia",
            "phylum": "Chordata",
            "class": "Aves",
            "order": "Passeriformes",
            "family": "Passeridae",
            "genus": "Passer",
        },
        {
            "taxonID": 3,
            "canonicalName": "Canis lupus",
            "taxonomicStatus": "accepted",
            "taxonRank": "species",
            "parentNameUsageID": np.nan,
            "acceptedNameUsageID": np.nan,
            "kingdom": "Animalia",
            "phylum": "Chordata",
            "class": "Mammalia",
            "order": "Carnivora",
            "family": "Canidae",
            "genus": "Canis",
        },
        # add one synonym entry
        {
            "taxonID": 4,
            "canonicalName": "Felis concolor",
            "taxonomicStatus": "synonym",
            "taxonRank": "species",
            "parentNameUsageID": np.nan,
            "acceptedNameUsageID": 5.0,
            "kingdom": "Animalia",
            "phylum": "Chordata",
            "class": "Mammalia",
            "order": "Carnivora",
            "family": "Felidae",
            "genus": "Felis",
        },
        {
            "taxonID": 5,
            "canonicalName": "Puma concolor",
            "taxonomicStatus": "accepted",
            "taxonRank": "species",
            "parentNameUsageID": np.nan,
            "acceptedNameUsageID": np.nan,
            "kingdom": "Animalia",
            "phylum": "Chordata",
            "class": "Mammalia",
            "order": "Carnivora",
            "family": "Felidae",
            "genus": "Puma",
        },
    ]
    df = pd.DataFrame(rows)
    fp = tmp_path / "gbif_with_taxonomy.tsv"
    df.to_csv(fp, sep="\t", index=False)
    return str(fp)


def test_add_taxonomy_basic(tmp_path: Path) -> None:
    """Test basic AddTaxonomy transform functionality."""
    gbif_path = _create_gbif_tsv_with_taxonomy(tmp_path)

    # Create test data
    df = pd.DataFrame({"scientific_name": ["Corvus corax", "Passer domesticus"]})
    backend = PandasBackend(df)

    # Apply transform
    transform = AddTaxonomy(feature="scientific_name", gbif_taxonomy_path=gbif_path)
    result_backend, metadata = transform(backend)

    # Check that taxonomy columns were added
    result_df = result_backend.unwrap
    assert "kingdom" in result_df.columns
    assert "phylum" in result_df.columns
    assert "class" in result_df.columns
    assert "order" in result_df.columns
    assert "family" in result_df.columns
    assert "genus" in result_df.columns

    # Check values
    assert result_df.loc[0, "kingdom"] == "Animalia"
    assert result_df.loc[0, "class"] == "Aves"
    assert result_df.loc[0, "family"] == "Corvidae"
    assert result_df.loc[1, "family"] == "Passeridae"


def test_add_taxonomy_from_config(tmp_path: Path) -> None:
    """Test AddTaxonomy.from_config class method."""
    gbif_path = _create_gbif_tsv_with_taxonomy(tmp_path)

    config = AddTaxonomyConfig(
        type="add_taxonomy",
        feature="species",
        gbif_taxonomy_path=gbif_path,
    )
    transform = AddTaxonomy.from_config(config)

    # Create test data
    df = pd.DataFrame({"species": ["Canis lupus"]})
    backend = PandasBackend(df)

    result_backend, metadata = transform(backend)

    result_df = result_backend.unwrap
    assert result_df.loc[0, "class"] == "Mammalia"
    assert result_df.loc[0, "order"] == "Carnivora"
    assert result_df.loc[0, "family"] == "Canidae"


def test_add_taxonomy_missing_feature_column(tmp_path: Path) -> None:
    """Test that AddTaxonomy raises ValueError for missing feature column."""
    gbif_path = _create_gbif_tsv_with_taxonomy(tmp_path)

    df = pd.DataFrame({"wrong_column": ["Corvus corax"]})
    backend = PandasBackend(df)

    transform = AddTaxonomy(feature="scientific_name", gbif_taxonomy_path=gbif_path)

    with pytest.raises(ValueError, match="Feature column 'scientific_name' not found in data"):
        transform(backend)


def test_add_taxonomy_metadata(tmp_path: Path) -> None:
    """Test that AddTaxonomy returns correct metadata."""
    gbif_path = _create_gbif_tsv_with_taxonomy(tmp_path)

    # Create test data with some duplicates and one unresolvable name
    df = pd.DataFrame(
        {
            "scientific_name": [
                "Corvus corax",
                "Corvus corax",  # duplicate
                "Passer domesticus",
                "Unknown species",  # won't resolve
            ]
        }
    )
    backend = PandasBackend(df)

    transform = AddTaxonomy(feature="scientific_name", gbif_taxonomy_path=gbif_path)
    _, metadata = transform(backend)

    assert metadata["feature"] == "scientific_name"
    assert metadata["unique_names"] == 3  # 3 unique names
    assert metadata["resolved"] == 2  # 2 resolved
    assert metadata["failed"] == 1  # 1 failed


def test_add_taxonomy_unresolvable_names(tmp_path: Path) -> None:
    """Test AddTaxonomy with names that cannot be resolved."""
    gbif_path = _create_gbif_tsv_with_taxonomy(tmp_path)

    df = pd.DataFrame({"scientific_name": ["Unknown species", "Another unknown"]})
    backend = PandasBackend(df)

    transform = AddTaxonomy(feature="scientific_name", gbif_taxonomy_path=gbif_path)
    result_backend, metadata = transform(backend)

    result_df = result_backend.unwrap

    # Unresolved names should have NaN/None in taxonomy columns
    assert pd.isna(result_df.loc[0, "kingdom"])
    assert pd.isna(result_df.loc[0, "family"])

    assert metadata["resolved"] == 0
    assert metadata["failed"] == 2


def test_add_taxonomy_with_add_taxonomic_name(tmp_path: Path) -> None:
    """Test AddTaxonomy with add_taxonomic_name=True."""
    gbif_path = _create_gbif_tsv_with_taxonomy(tmp_path)

    config = AddTaxonomyConfig(
        type="add_taxonomy",
        feature="scientific_name",
        gbif_taxonomy_path=gbif_path,
        add_taxonomic_name=True,
    )
    transform = AddTaxonomy.from_config(config)

    df = pd.DataFrame({"scientific_name": ["Corvus corax"]})
    backend = PandasBackend(df)

    _, metadata = transform(backend)

    # Check that taxonomic_name is in the added columns
    assert "taxonomic_name" in metadata["taxonomy_columns_added"]


def test_add_taxonomy_make_taxonomic_name(tmp_path: Path) -> None:
    """Test the _make_taxonomic_name method."""
    gbif_path = _create_gbif_tsv_with_taxonomy(tmp_path)

    transform = AddTaxonomy(feature="scientific_name", gbif_taxonomy_path=gbif_path)

    info = {
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class": "Aves",
        "order": "Passeriformes",
        "family": "Corvidae",
        "genus": "Corvus",
        "canonicalName": "Corvus corax",
    }

    result = transform._make_taxonomic_name(info)

    # Should include kingdom, phylum, class, order, family (not genus) and canonicalName
    assert "Animalia" in result
    assert "Chordata" in result
    assert "Aves" in result
    assert "Passeriformes" in result
    assert "Corvidae" in result
    assert "Corvus corax" in result


def test_add_taxonomy_make_taxonomic_name_empty(tmp_path: Path) -> None:
    """Test _make_taxonomic_name with empty info."""
    gbif_path = _create_gbif_tsv_with_taxonomy(tmp_path)

    transform = AddTaxonomy(feature="scientific_name", gbif_taxonomy_path=gbif_path)

    result = transform._make_taxonomic_name({})

    assert result is None


def test_add_taxonomy_config_validation(tmp_path: Path) -> None:
    """Test that AddTaxonomyConfig validates file existence."""
    with pytest.raises(ValueError, match="GBIF data file does not exist"):
        AddTaxonomyConfig(
            type="add_taxonomy",
            feature="scientific_name",
            gbif_taxonomy_path="/nonexistent/path/to/file.tsv",
        )


def test_add_taxonomy_empty_dataframe(tmp_path: Path) -> None:
    """Test AddTaxonomy with an empty dataframe."""
    gbif_path = _create_gbif_tsv_with_taxonomy(tmp_path)

    df = pd.DataFrame({"scientific_name": []})
    backend = PandasBackend(df)

    transform = AddTaxonomy(feature="scientific_name", gbif_taxonomy_path=gbif_path)
    result_backend, metadata = transform(backend)

    assert len(result_backend) == 0
    assert metadata["unique_names"] == 0
    assert metadata["resolved"] == 0
    assert metadata["failed"] == 0


def test_add_taxonomy_integration_with_beanszero() -> None:
    """Integration test: Apply AddTaxonomy to a subset of BEANSZero dataset.

    This test uses the real GBIF taxonomy data from GCS and applies the
    AddTaxonomy transform to a sample from the BEANSZero dataset.
    """
    from esp_data.datasets import BeansZero

    # Load a small subset of iNaturalist (just enough for testing)
    dataset = BeansZero(split="unseen-species-sci", backend="pandas")

    # Get the first 100 rows for testing
    sample_backend = dataset._data[:100]

    # iNaturalist has 'canonical_name' column with scientific names
    transform = AddTaxonomy(
        feature="output",  # 'output' column has the canonical names in BeansZero
        add_taxonomic_name=True,
        # Uses default GCS path: gs://sound-event-detection/taxonomy/gbif_animals.tsv
    )

    result_backend, metadata = transform(sample_backend)

    result_df = result_backend.unwrap

    # Check that taxonomy columns were added
    expected_columns = ["kingdom", "phylum", "class", "order", "family", "genus"]
    for col in expected_columns:
        assert col in result_df.columns, f"Expected column '{col}' not found in result"

    # Check metadata
    assert metadata["feature"] == "output"
    assert metadata["unique_names"] > 0
    assert metadata["resolved"] >= 0

    # At least some names should have resolved successfully
    non_null_kingdoms = result_df["kingdom"].notna().sum()
    assert non_null_kingdoms > 0, "Expected at least some taxonomy resolutions"

    # Check that resolved rows have consistent taxonomy
    resolved_rows = result_df[result_df["kingdom"].notna()]
    if len(resolved_rows) > 0:
        # All resolved rows should have "Animalia" as kingdom (iNaturalist animal sounds)
        assert (resolved_rows["kingdom"] == "Animalia").all()
