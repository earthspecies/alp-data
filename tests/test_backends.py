"""Tests for backend implementations."""

import pandas as pd
import polars as pl
import pytest

from esp_data.backends import PandasBackend, PolarsBackend, get_backend
from esp_data.backends.webdataset_backend import WebDatasetBackend
from esp_data.backends.webdataset_utils import json_decoder


class TestPandasBackend:
    """Tests for PandasBackend."""

    def test_init_and_unwrap(self) -> None:
        """Test initialization and unwrapping."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PandasBackend(df)
        assert isinstance(backend.unwrap, pd.DataFrame)
        assert len(backend) == 3

    def test_getitem_single_row(self) -> None:
        """Test getting a single row as dict."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PandasBackend(df)
        row = backend[0]
        assert isinstance(row, dict)
        assert row == {"a": 1, "b": "x"}

    def test_getitem_multiple_rows(self) -> None:
        """Test getting multiple rows."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PandasBackend(df)
        subset = backend[[0, 2]]
        assert isinstance(subset, PandasBackend)
        assert len(subset) == 2

    def test_getitem_slice(self) -> None:
        """Test getting rows by slice."""
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})
        backend = PandasBackend(df)
        subset = backend[1:3]
        assert isinstance(subset, PandasBackend)
        assert len(subset) == 2

    def test_iter(self) -> None:
        """Test iteration over rows."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PandasBackend(df)
        rows = list(backend)
        assert len(rows) == 3
        assert rows[0] == {"a": 1, "b": "x"}

    def test_filter_isin(self) -> None:
        """Test filtering with isin."""
        df = pd.DataFrame({"species": ["cat", "dog", "bird", "cat"]})
        backend = PandasBackend(df)
        filtered = backend.filter_isin("species", ["cat", "dog"])
        assert len(filtered) == 3

    def test_filter_isin_negate(self) -> None:
        """Test filtering with isin negation."""
        df = pd.DataFrame({"species": ["cat", "dog", "bird", "cat"]})
        backend = PandasBackend(df)
        filtered = backend.filter_isin("species", ["cat"], negate=True)
        assert len(filtered) == 2

    def test_drop_duplicates(self) -> None:
        """Test deduplication."""
        df = pd.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "y", "y", "z"]})
        backend = PandasBackend(df)
        deduped = backend.drop_duplicates()
        assert len(deduped) == 3

    def test_dropna(self) -> None:
        """Test dropping null values."""
        df = pd.DataFrame({"a": [1, 2, None, 4], "b": ["x", "y", "z", "w"]})
        backend = PandasBackend(df)
        cleaned = backend.dropna(subset=["a"])
        assert len(cleaned) == 3

    def test_get_unique(self) -> None:
        """Test getting unique values."""
        df = pd.DataFrame({"species": ["cat", "dog", "bird", "cat", "dog"]})
        backend = PandasBackend(df)
        uniques = backend.get_unique("species")
        assert set(uniques) == {"bird", "cat", "dog"}

    def test_histogram(self) -> None:
        """Test getting value counts (histogram)."""
        df = pd.DataFrame({"species": ["cat", "dog", "bird", "cat", "dog", "cat"]})
        backend = PandasBackend(df)
        histogram = backend.histogram("species")
        assert histogram == {"cat": 3, "dog": 2, "bird": 1}

    def test_histogram_with_nulls(self) -> None:
        """Test histogram excludes null values."""
        df = pd.DataFrame({"species": ["cat", "dog", None, "cat", None, "bird"]})
        backend = PandasBackend(df)
        histogram = backend.histogram("species")
        assert histogram == {"cat": 2, "dog": 1, "bird": 1}
        assert None not in histogram

    def test_map_column(self) -> None:
        """Test mapping column values."""
        df = pd.DataFrame({"species": ["cat", "dog", "bird"]})
        backend = PandasBackend(df)
        mapping = {"cat": 0, "dog": 1, "bird": 2}
        mapped = backend.map_column("species", mapping, "label")
        assert "label" in mapped.columns
        assert mapped[0]["label"] == 0

    def test_rename_columns(self) -> None:
        """Test renaming columns."""
        df = pd.DataFrame({"old_name": [1, 2, 3]})
        backend = PandasBackend(df)
        renamed = backend.rename_columns({"old_name": "new_name"})
        assert "new_name" in renamed.columns
        assert "old_name" not in renamed.columns

    def test_add_column(self) -> None:
        """Test adding a column."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        backend = PandasBackend(df)
        with_col = backend.add_column("b", 0)
        assert "b" in with_col.columns
        assert with_col[0]["b"] == 0

    def test_select_columns(self) -> None:
        """Test selecting columns."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4, 5, 6]})
        backend = PandasBackend(df)
        selected = backend.select_columns(["a", "c"])
        assert selected.columns == ["a", "c"]

    def test_concat(self) -> None:
        """Test concatenation."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        backend1 = PandasBackend(df1)
        backend2 = PandasBackend(df2)
        combined = PandasBackend.concat([backend1, backend2])
        assert len(combined) == 4

    def test_columns_property(self) -> None:
        """Test columns property."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PandasBackend(df)
        assert backend.columns == ["a", "b"]

    def test_column_exists(self) -> None:
        """Test column existence check."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        backend = PandasBackend(df)
        assert backend.column_exists("a")
        assert not backend.column_exists("b")

    def test_sample_rows(self) -> None:
        """Test random sampling."""
        df = pd.DataFrame({"a": range(100)})
        backend = PandasBackend(df)
        sampled = backend.sample_rows(10, seed=42)
        assert len(sampled) == 10

    def test_copy(self) -> None:
        """Test copying."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        backend = PandasBackend(df)
        copied = backend.copy()
        assert copied.unwrap is not backend.unwrap
        assert len(copied) == len(backend)

    def test_save_to_unsupported_format_raises(self, tmp_path) -> None:
        """Test that save_to raises for unsupported formats."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        backend = PandasBackend(df)
        with pytest.raises(ValueError, match="Unsupported format"):
            backend.save_to(str(tmp_path / "out.csv"), format="csv")

    def test_save_to_webdataset(self, tmp_path) -> None:
        """Test saving to webdataset format and reading back."""
        df = pd.DataFrame({"id": [0, 1, 2], "name": ["a", "b", "c"]})
        backend = PandasBackend(df)
        output_dir = tmp_path / "out"
        backend.save_to(str(output_dir), format="webdataset")
        reloaded = list(WebDatasetBackend.from_path(output_dir, data_processor=json_decoder))
        assert len(reloaded) == 3
        assert {s["id"] for s in reloaded} == {0, 1, 2}

    def test_save_to_streaming_raises(self, tmp_path) -> None:
        """Test that save_to raises in streaming mode."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(str(csv_path), index=False)
        backend = PandasBackend.from_csv(str(csv_path), streaming=True)
        with pytest.raises(RuntimeError):
            backend.save_to(str(tmp_path / "out"))


class TestPolarsBackend:
    """Tests for PolarsBackend."""

    def test_init_and_unwrap(self) -> None:
        """Test initialization and unwrapping."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PolarsBackend(df)
        assert isinstance(backend.unwrap, pl.DataFrame)
        assert len(backend) == 3

    def test_getitem_single_row(self) -> None:
        """Test getting a single row as dict."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PolarsBackend(df)
        row = backend[0]
        assert isinstance(row, dict)
        assert row == {"a": 1, "b": "x"}

    def test_getitem_multiple_rows(self) -> None:
        """Test getting multiple rows."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PolarsBackend(df)
        subset = backend[[0, 2]]
        assert isinstance(subset, PolarsBackend)
        assert len(subset) == 2

    def test_iter(self) -> None:
        """Test iteration over rows."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PolarsBackend(df)
        rows = list(backend)
        assert len(rows) == 3
        assert rows[0] == {"a": 1, "b": "x"}

    def test_filter_isin(self) -> None:
        """Test filtering with isin."""
        df = pl.DataFrame({"species": ["cat", "dog", "bird", "cat"]})
        backend = PolarsBackend(df)
        filtered = backend.filter_isin("species", ["cat", "dog"])
        assert len(filtered) == 3

    def test_drop_duplicates(self) -> None:
        """Test deduplication."""
        df = pl.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "y", "y", "z"]})
        backend = PolarsBackend(df)
        deduped = backend.drop_duplicates()
        assert len(deduped) == 3

    def test_get_unique(self) -> None:
        """Test getting unique values."""
        df = pl.DataFrame({"species": ["cat", "dog", "bird", "cat", "dog"]})
        backend = PolarsBackend(df)
        uniques = backend.get_unique("species")
        assert set(uniques) == {"bird", "cat", "dog"}

    def test_histogram(self) -> None:
        """Test getting value counts (histogram)."""
        df = pl.DataFrame({"species": ["cat", "dog", "bird", "cat", "dog", "cat"]})
        backend = PolarsBackend(df)
        histogram = backend.histogram("species")
        assert histogram == {"cat": 3, "dog": 2, "bird": 1}

    def test_histogram_with_nulls(self) -> None:
        """Test histogram excludes null values."""
        df = pl.DataFrame({"species": ["cat", "dog", None, "cat", None, "bird"]})
        backend = PolarsBackend(df)
        histogram = backend.histogram("species")
        assert histogram == {"cat": 2, "dog": 1, "bird": 1}
        assert None not in histogram

    def test_map_column(self) -> None:
        """Test mapping column values."""
        df = pl.DataFrame({"species": ["cat", "dog", "bird"]})
        backend = PolarsBackend(df)
        mapping = {"cat": 0, "dog": 1, "bird": 2}
        mapped = backend.map_column("species", mapping, "label")
        assert "label" in mapped.columns
        assert mapped[0]["label"] == 0

    def test_columns_property(self) -> None:
        """Test columns property."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PolarsBackend(df)
        assert backend.columns == ["a", "b"]

    def test_concat(self) -> None:
        """Test concatenation."""
        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"a": [3, 4]})
        backend1 = PolarsBackend(df1)
        backend2 = PolarsBackend(df2)
        combined = PolarsBackend.concat([backend1, backend2])
        assert len(combined) == 4

    def test_save_to_unsupported_format_raises(self, tmp_path) -> None:
        """Test that save_to raises for unsupported formats."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        backend = PolarsBackend(df)
        with pytest.raises(ValueError, match="Unsupported format"):
            backend.save_to(str(tmp_path / "out.csv"), format="csv")

    def test_save_to_webdataset(self, tmp_path) -> None:
        """Test saving to webdataset format and reading back."""
        df = pl.DataFrame({"id": [0, 1, 2], "name": ["a", "b", "c"]})
        backend = PolarsBackend(df)
        output_dir = tmp_path / "out"
        backend.save_to(str(output_dir), format="webdataset")
        reloaded = list(WebDatasetBackend.from_path(output_dir, data_processor=json_decoder))
        assert len(reloaded) == 3
        assert {s["id"] for s in reloaded} == {0, 1, 2}

    def test_save_to_streaming_warns_and_writes(self, tmp_path) -> None:
        """Test that save_to warns in streaming mode but still writes."""
        df = pl.LazyFrame({"id": [0, 1, 2], "name": ["a", "b", "c"]})
        backend = PolarsBackend(df, streaming=True)
        output_dir = tmp_path / "out"
        with pytest.warns(UserWarning, match="collection of LazyFrame"):
            backend.save_to(str(output_dir), format="webdataset")
        reloaded = list(WebDatasetBackend.from_path(output_dir, data_processor=json_decoder))
        assert len(reloaded) == 3


class TestBackendFactory:
    """Tests for backend factory functions."""

    def test_get_backend_pandas(self) -> None:
        """Test getting pandas backend."""
        backend_cls = get_backend("pandas")
        assert backend_cls == PandasBackend

    def test_get_backend_polars(self) -> None:
        """Test getting polars backend."""
        backend_cls = get_backend("polars")
        assert backend_cls == PolarsBackend

    def test_get_backend_invalid(self) -> None:
        """Test getting invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid")  # type: ignore
