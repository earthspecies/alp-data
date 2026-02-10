from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from esp_data.backends import PandasBackend
from esp_data.discover import AddTaxonomy, AddTaxonomyConfig, GBIFConverter

# AddTaxonomy Transform Unit Tests

def _create_gbif_json_with_taxonomy(tmp_path: Path) -> str:
    """Create a GBIF json with full taxonomy info for testing AddTaxonomy."""
    lookupdict = {
        "Corvus corax": {
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
        "Passer domesticus": {
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
        "Canis lupus" : {
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
        "Puma concolor" : {
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
    }
    fp = tmp_path / "gbif_with_taxonomy.json"
    pd.DataFrame.from_dict(lookupdict, orient="index").to_json(fp, indent=2)
    return str(fp)

def test_gbif_converter() -> None:
    converter = GBIFConverter()
    info, ok = converter("Puma concolor")
    assert ok

    info, ok = converter("Fraudulus animalaticus")
    assert not ok

def test_add_taxonomy_basic(tmp_path: Path) -> None:
    """Test basic AddTaxonomy transform functionality."""
    gbif_path = _create_gbif_json_with_taxonomy(tmp_path)

    # Create test data
    df = pd.DataFrame({"scientific_name": ["Corvus corax", "Passer domesticus"]})
    backend = PandasBackend(df)

    # Apply transform
    transform = AddTaxonomy(feature="scientific_name", gbif_precomputed_taxonomy_path=gbif_path)
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
    gbif_path = _create_gbif_json_with_taxonomy(tmp_path)

    config = AddTaxonomyConfig(
        type="add_taxonomy",
        feature="species",
        gbif_precomputed_taxonomy_path=gbif_path,

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
    gbif_path = _create_gbif_json_with_taxonomy(tmp_path)

    df = pd.DataFrame({"wrong_column": ["Corvus corax"]})
    backend = PandasBackend(df)

    transform = AddTaxonomy(feature="scientific_name", gbif_precomputed_taxonomy_path=gbif_path)

    with pytest.raises(ValueError, match="Feature column 'scientific_name' not found in data"):
        transform(backend)


def test_add_taxonomy_metadata(tmp_path: Path) -> None:
    """Test that AddTaxonomy returns correct metadata."""
    gbif_path = _create_gbif_json_with_taxonomy(tmp_path)

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

    transform = AddTaxonomy(feature="scientific_name", gbif_precomputed_taxonomy_path=gbif_path)
    _, metadata = transform(backend)

    assert metadata["feature"] == "scientific_name"
    assert metadata["resolved"] == 2  # 2 resolved
    assert metadata["failed"] == 1  # 1 failed


def test_add_taxonomy_unresolvable_names(tmp_path: Path) -> None:
    """Test AddTaxonomy with names that cannot be resolved."""
    gbif_path = _create_gbif_json_with_taxonomy(tmp_path)

    df = pd.DataFrame({"scientific_name": ["Unknown species", "Another unknown"]})
    backend = PandasBackend(df)

    transform = AddTaxonomy(feature="scientific_name", gbif_precomputed_taxonomy_path=gbif_path)
    result_backend, metadata = transform(backend)

    result_df = result_backend.unwrap

    # Unresolved names should have NaN/None in taxonomy columns
    assert pd.isna(result_df.loc[0, "kingdom"])
    assert pd.isna(result_df.loc[0, "family"])

    assert metadata["resolved"] == 0
    assert metadata["failed"] == 2


def test_add_taxonomy_with_add_taxonomic_name(tmp_path: Path) -> None:
    """Test AddTaxonomy with add_taxonomic_name=True."""
    gbif_path = _create_gbif_json_with_taxonomy(tmp_path)

    config = AddTaxonomyConfig(
        type="add_taxonomy",
        feature="scientific_name",
        gbif_precomputed_taxonomy_path=gbif_path,
        add_taxonomic_name=True,

    )
    transform = AddTaxonomy.from_config(config)

    df = pd.DataFrame({"scientific_name": ["Corvus corax"]})
    backend = PandasBackend(df)

    backend, metadata = transform(backend)

    # Check that taxonomic_name is in the added columns
    assert "taxonomic_name" in metadata["taxonomy_columns_added"]
    assert backend[0]["taxonomic_name"] == "Animalia Chordata Aves Passeriformes Corvidae Corvus corax"


def test_add_taxonomy_make_taxonomic_name(tmp_path: Path) -> None:
    """Test the _make_taxonomic_name method."""
    gbif_path = _create_gbif_json_with_taxonomy(tmp_path)

    transform = AddTaxonomy(feature="scientific_name", gbif_precomputed_taxonomy_path=gbif_path)

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
    gbif_path = _create_gbif_json_with_taxonomy(tmp_path)

    transform = AddTaxonomy(feature="scientific_name", gbif_precomputed_taxonomy_path=gbif_path)

    result = transform._make_taxonomic_name({})

    assert result is None


def test_add_taxonomy_config_validation(tmp_path: Path) -> None:
    """Test that AddTaxonomyConfig validates file existence."""
    with pytest.raises(ValueError, match="GBIF data file does not exist"):
        AddTaxonomyConfig(
            type="add_taxonomy",
            feature="scientific_name",
            gbif_precomputed_taxonomy_path="/nonexistent/path/to/file.tsv",

        )


def test_add_taxonomy_empty_dataframe(tmp_path: Path) -> None:
    """Test AddTaxonomy with an empty dataframe."""
    gbif_path = _create_gbif_json_with_taxonomy(tmp_path)

    df = pd.DataFrame({"scientific_name": []})
    backend = PandasBackend(df)

    transform = AddTaxonomy(feature="scientific_name", gbif_precomputed_taxonomy_path=gbif_path)
    result_backend, metadata = transform(backend)

    assert len(result_backend) == 0
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

    transform = AddTaxonomy(
        feature="output",  # 'output' column has the canonical names in BeansZero
        add_taxonomic_name=True,

        # Uses cache if present
    )

    result_backend, metadata = transform(sample_backend)
    result_df = result_backend.unwrap

    # Check that taxonomy columns were added
    expected_columns = ["kingdom", "phylum", "class", "order", "family", "genus"]
    for col in expected_columns:
        assert col in result_df.columns, f"Expected column '{col}' not found in result"

    # Check metadata
    assert metadata["feature"] == "output"
    assert metadata["resolved"] >= 0

    # At least some names should have resolved successfully
    non_null_kingdoms = result_df["kingdom"].notna().sum()
    assert non_null_kingdoms > 0, "Expected at least some taxonomy resolutions"

    # Check that resolved rows have consistent taxonomy
    resolved_rows = result_df[result_df["kingdom"].notna()]
    if len(resolved_rows) > 0:
        # All resolved rows should have "Animalia" as kingdom (iNaturalist animal sounds)
        assert (resolved_rows["kingdom"] == "Animalia").all()

    # Taxonomic names should be present
    assert "taxonomic_name" in result_df.columns
    for idx, row in resolved_rows.iterrows():
        taxonomic_name = row["taxonomic_name"]
        assert taxonomic_name is not None
        assert "Animalia" in taxonomic_name
