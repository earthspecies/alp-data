"""Tests for backend implementations."""

import pandas as pd
import polars as pl
import pytest

from esp_data.backends import PandasBackend, PolarsBackend, get_backend


class TestPandasBackend:
    """Tests for PandasBackend."""

    def test_init_and_unwrap(self):
        """Test initialization and unwrapping."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PandasBackend(df)
        assert isinstance(backend.unwrap, pd.DataFrame)
        assert len(backend) == 3

    def test_getitem_single_row(self):
        """Test getting a single row as dict."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PandasBackend(df)
        row = backend[0]
        assert isinstance(row, dict)
        assert row == {"a": 1, "b": "x"}

    def test_getitem_multiple_rows(self):
        """Test getting multiple rows."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PandasBackend(df)
        subset = backend[[0, 2]]
        assert isinstance(subset, PandasBackend)
        assert len(subset) == 2

    def test_getitem_slice(self):
        """Test getting rows by slice."""
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})
        backend = PandasBackend(df)
        subset = backend[1:3]
        assert isinstance(subset, PandasBackend)
        assert len(subset) == 2

    def test_iter(self):
        """Test iteration over rows."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PandasBackend(df)
        rows = list(backend)
        assert len(rows) == 3
        assert rows[0] == {"a": 1, "b": "x"}

    def test_filter_isin(self):
        """Test filtering with isin."""
        df = pd.DataFrame({"species": ["cat", "dog", "bird", "cat"]})
        backend = PandasBackend(df)
        filtered = backend.filter_isin("species", ["cat", "dog"])
        assert len(filtered) == 3

    def test_filter_isin_negate(self):
        """Test filtering with isin negation."""
        df = pd.DataFrame({"species": ["cat", "dog", "bird", "cat"]})
        backend = PandasBackend(df)
        filtered = backend.filter_isin("species", ["cat"], negate=True)
        assert len(filtered) == 2

    def test_drop_duplicates(self):
        """Test deduplication."""
        df = pd.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "y", "y", "z"]})
        backend = PandasBackend(df)
        deduped = backend.drop_duplicates()
        assert len(deduped) == 3

    def test_dropna(self):
        """Test dropping null values."""
        df = pd.DataFrame({"a": [1, 2, None, 4], "b": ["x", "y", "z", "w"]})
        backend = PandasBackend(df)
        cleaned = backend.dropna(subset=["a"])
        assert len(cleaned) == 3

    def test_get_unique(self):
        """Test getting unique values."""
        df = pd.DataFrame({"species": ["cat", "dog", "bird", "cat", "dog"]})
        backend = PandasBackend(df)
        uniques = backend.get_unique("species")
        assert set(uniques) == {"bird", "cat", "dog"}

    def test_map_column(self):
        """Test mapping column values."""
        df = pd.DataFrame({"species": ["cat", "dog", "bird"]})
        backend = PandasBackend(df)
        mapping = {"cat": 0, "dog": 1, "bird": 2}
        mapped = backend.map_column("species", mapping, "label")
        assert "label" in mapped.columns
        assert mapped[0]["label"] == 0

    def test_rename_columns(self):
        """Test renaming columns."""
        df = pd.DataFrame({"old_name": [1, 2, 3]})
        backend = PandasBackend(df)
        renamed = backend.rename_columns({"old_name": "new_name"})
        assert "new_name" in renamed.columns
        assert "old_name" not in renamed.columns

    def test_add_column(self):
        """Test adding a column."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        backend = PandasBackend(df)
        with_col = backend.add_column("b", 0)
        assert "b" in with_col.columns
        assert with_col[0]["b"] == 0

    def test_select_columns(self):
        """Test selecting columns."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4, 5, 6]})
        backend = PandasBackend(df)
        selected = backend.select_columns(["a", "c"])
        assert selected.columns == ["a", "c"]

    def test_concat(self):
        """Test concatenation."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        backend1 = PandasBackend(df1)
        backend2 = PandasBackend(df2)
        combined = PandasBackend.concat([backend1, backend2])
        assert len(combined) == 4

    def test_columns_property(self):
        """Test columns property."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PandasBackend(df)
        assert backend.columns == ["a", "b"]

    def test_column_exists(self):
        """Test column existence check."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        backend = PandasBackend(df)
        assert backend.column_exists("a")
        assert not backend.column_exists("b")

    def test_sample_rows(self):
        """Test random sampling."""
        df = pd.DataFrame({"a": range(100)})
        backend = PandasBackend(df)
        sampled = backend.sample_rows(10, seed=42)
        assert len(sampled) == 10

    def test_copy(self):
        """Test copying."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        backend = PandasBackend(df)
        copied = backend.copy()
        assert copied.unwrap is not backend.unwrap
        assert len(copied) == len(backend)


class TestPolarsBackend:
    """Tests for PolarsBackend."""

    def test_init_and_unwrap(self):
        """Test initialization and unwrapping."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PolarsBackend(df)
        assert isinstance(backend.unwrap, pl.DataFrame)
        assert len(backend) == 3

    def test_getitem_single_row(self):
        """Test getting a single row as dict."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PolarsBackend(df)
        row = backend[0]
        assert isinstance(row, dict)
        assert row == {"a": 1, "b": "x"}

    def test_getitem_multiple_rows(self):
        """Test getting multiple rows."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PolarsBackend(df)
        subset = backend[[0, 2]]
        assert isinstance(subset, PolarsBackend)
        assert len(subset) == 2

    def test_iter(self):
        """Test iteration over rows."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PolarsBackend(df)
        rows = list(backend)
        assert len(rows) == 3
        assert rows[0] == {"a": 1, "b": "x"}

    def test_filter_isin(self):
        """Test filtering with isin."""
        df = pl.DataFrame({"species": ["cat", "dog", "bird", "cat"]})
        backend = PolarsBackend(df)
        filtered = backend.filter_isin("species", ["cat", "dog"])
        assert len(filtered) == 3

    def test_drop_duplicates(self):
        """Test deduplication."""
        df = pl.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "y", "y", "z"]})
        backend = PolarsBackend(df)
        deduped = backend.drop_duplicates()
        assert len(deduped) == 3

    def test_get_unique(self):
        """Test getting unique values."""
        df = pl.DataFrame({"species": ["cat", "dog", "bird", "cat", "dog"]})
        backend = PolarsBackend(df)
        uniques = backend.get_unique("species")
        assert set(uniques) == {"bird", "cat", "dog"}

    def test_map_column(self):
        """Test mapping column values."""
        df = pl.DataFrame({"species": ["cat", "dog", "bird"]})
        backend = PolarsBackend(df)
        mapping = {"cat": 0, "dog": 1, "bird": 2}
        mapped = backend.map_column("species", mapping, "label")
        assert "label" in mapped.columns
        assert mapped[0]["label"] == 0

    def test_columns_property(self):
        """Test columns property."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PolarsBackend(df)
        assert backend.columns == ["a", "b"]

    def test_concat(self):
        """Test concatenation."""
        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"a": [3, 4]})
        backend1 = PolarsBackend(df1)
        backend2 = PolarsBackend(df2)
        combined = PolarsBackend.concat([backend1, backend2])
        assert len(combined) == 4


class TestBackendFactory:
    """Tests for backend factory functions."""

    def test_get_backend_pandas(self):
        """Test getting pandas backend."""
        backend_cls = get_backend("pandas")
        assert backend_cls == PandasBackend

    def test_get_backend_polars(self):
        """Test getting polars backend."""
        backend_cls = get_backend("polars")
        assert backend_cls == PolarsBackend

    def test_get_backend_invalid(self):
        """Test getting invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid")  # type: ignore
