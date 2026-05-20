"""Tests for backend implementations."""

from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from esp_data.backends import PandasBackend, PolarsBackend, PyarrowBackend, get_backend


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


class TestPyarrowBackend:
    """Test for PyarrowBackend"""

    def test_init_and_unwrap(self) -> None:
        """Test initialization and unwrapping."""
        df = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PyarrowBackend(df)
        assert isinstance(backend.unwrap, pa.Table)
        assert len(backend) == 3

    def test_getitem_single_row(self) -> None:
        """Test getting a single row as dict."""
        df = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PyarrowBackend(df)
        row = backend[0]
        assert isinstance(row, dict)
        assert row == {"a": 1, "b": "x"}

    def test_getitem_multiple_rows(self) -> None:
        """Test getting multiple rows."""
        df = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PyarrowBackend(df)
        subset = backend[[0, 2]]
        assert isinstance(subset, PyarrowBackend)
        assert len(subset) == 2

    def test_getitem_slice(self) -> None:
        """Test getting rows by slice."""
        df = pa.table({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})
        backend = PyarrowBackend(df)
        subset = backend[1:3]
        assert isinstance(subset, PyarrowBackend)
        assert len(subset) == 2

    def test_iter(self) -> None:
        """Test iteration over rows."""
        df = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PyarrowBackend(df)
        rows = list(backend)
        assert len(rows) == 3
        assert rows[0] == {"a": 1, "b": "x"}

    def test_filter_isin(self) -> None:
        """Test filtering with isin."""
        df = pa.table({"species": ["cat", "dog", "bird", "cat"]})
        backend = PyarrowBackend(df)
        filtered = backend.filter_isin("species", ["cat", "dog"])
        assert len(filtered) == 3

    def test_drop_duplicates(self) -> None:
        """Test deduplication."""
        df = pa.table({"a": [1, 2, 2, 2, 3], "b": ["x", "y", "y", "x", "z"]})
        backend = PyarrowBackend(df)
        deduped = backend.drop_duplicates()
        assert len(deduped) == 4

    def test_get_unique(self) -> None:
        """Test getting unique values."""
        df = pa.table({"species": ["cat", "dog", "bird", "cat", "dog"]})
        backend = PyarrowBackend(df)
        uniques = backend.get_unique("species")
        assert set(uniques) == {"bird", "cat", "dog"}

    def test_histogram(self) -> None:
        """Test getting value counts (histogram)."""
        df = pa.table({"species": ["cat", "dog", "bird", "cat", "dog", "cat"]})
        backend = PyarrowBackend(df)
        histogram = backend.histogram("species")
        assert histogram == {"cat": 3, "dog": 2, "bird": 1}

    def test_histogram_with_nulls(self) -> None:
        """Test histogram excludes null values."""
        df = pa.table({"species": ["cat", "dog", None, "cat", None, "bird"]})
        backend = PyarrowBackend(df)
        histogram = backend.histogram("species")
        assert histogram == {"cat": 2, "dog": 1, "bird": 1}
        assert None not in histogram

    def test_map_column(self) -> None:
        """Test mapping column values."""
        df = pa.table({"species": ["cat", "dog", "bird"]})
        backend = PyarrowBackend(df)
        mapping = {"cat": 0, "dog": 1, "bird": 2}
        mapped = backend.map_column("species", mapping, "label")
        assert "label" in mapped.columns
        assert mapped[0]["label"] == 0

    def test_columns_property(self) -> None:
        """Test columns property."""
        df = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PyarrowBackend(df)
        assert backend.columns == ["a", "b"]

    def test_concat(self) -> None:
        """Test concatenation."""
        df1 = pa.table({"a": [1, 2]})
        df2 = pa.table({"a": [3, 4]})
        backend1 = PyarrowBackend(df1)
        backend2 = PyarrowBackend(df2)
        combined = PyarrowBackend.concat([backend1, backend2])
        assert len(combined) == 4

    def test_dropna_no_subset(self) -> None:
        """Test dropna removes rows with any null when no subset given."""
        df = pa.table({"a": [1, None, 3], "b": ["x", "y", None]})
        backend = PyarrowBackend(df)
        cleaned = backend.dropna()
        assert len(cleaned) == 1
        assert cleaned[0] == {"a": 1, "b": "x"}

    def test_dropna_with_subset(self) -> None:
        """Test dropna with subset only drops rows null in subset columns."""
        df = pa.table({"a": [1, None, 3], "b": ["x", "y", None]})
        backend = PyarrowBackend(df)
        cleaned = backend.dropna(subset=["a"])
        assert len(cleaned) == 2
        rows = list(cleaned)
        assert rows[0] == {"a": 1, "b": "x"}
        assert rows[1] == {"a": 3, "b": None}

    def test_dropna_subset_preserves_null_in_other_columns(self) -> None:
        """Test dropna subset keeps nulls in non-subset columns."""
        df = pa.table({"a": [1, 2, 3], "b": [None, None, "z"]})
        backend = PyarrowBackend(df)
        cleaned = backend.dropna(subset=["a"])
        assert len(cleaned) == 3

    def test_dropna_no_nulls(self) -> None:
        """Test dropna on table with no nulls returns all rows."""
        df = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PyarrowBackend(df)
        cleaned = backend.dropna()
        assert len(cleaned) == 3

    def test_dropna_all_null_rows(self) -> None:
        """Test dropna returns empty backend when all rows have nulls."""
        df = pa.table({"a": [None, None], "b": [None, None]})
        backend = PyarrowBackend(df)
        cleaned = backend.dropna()
        assert len(cleaned) == 0

    def test_dropna_returns_pyarrow_backend(self) -> None:
        """Test dropna returns PyarrowBackend instance."""
        df = pa.table({"a": [1, None, 3]})
        backend = PyarrowBackend(df)
        cleaned = backend.dropna()
        assert isinstance(cleaned, PyarrowBackend)

    # --- __getitem__ edge cases ---

    def test_getitem_out_of_bounds(self) -> None:
        """Test IndexError on out-of-bounds integer index."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        with pytest.raises(IndexError):
            backend[10]

    def test_getitem_unsupported_type(self) -> None:
        """Test TypeError on unsupported key type."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        with pytest.raises(TypeError):
            backend["bad_key"]  # type: ignore

    def test_getitem_slice_no_start(self) -> None:
        """Test slice with no start (backend[:2])."""
        df = pa.table({"a": [1, 2, 3, 4]})
        backend = PyarrowBackend(df)
        subset = backend[:2]
        assert isinstance(subset, PyarrowBackend)
        assert len(subset) == 2

    def test_getitem_slice_no_stop(self) -> None:
        """Test slice with no stop (backend[2:])."""
        df = pa.table({"a": [1, 2, 3, 4]})
        backend = PyarrowBackend(df)
        subset = backend[2:]
        assert isinstance(subset, PyarrowBackend)
        assert len(subset) == 2

    def test_getitem_slice_with_step_raises(self) -> None:
        """Test slice with step raises NotImplementedError."""
        df = pa.table({"a": [1, 2, 3, 4]})
        backend = PyarrowBackend(df)
        with pytest.raises(NotImplementedError):
            backend[::2]

    def test_getitem_streaming_raises(self) -> None:
        """Test __getitem__ raises RuntimeError in streaming mode."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df, streaming=True)
        with pytest.raises(RuntimeError):
            backend[0]

    # --- is_streaming ---

    def test_is_streaming_false_by_default(self) -> None:
        """Test is_streaming is False by default."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        assert backend.is_streaming is False

    def test_is_streaming_true_when_set(self) -> None:
        """Test is_streaming is True when streaming=True."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df, streaming=True)
        assert backend.is_streaming is True

    # --- filter_isin ---

    def test_filter_isin_negate(self) -> None:
        """Test filter_isin with negate=True keeps rows NOT in values."""
        df = pa.table({"species": ["cat", "dog", "bird", "cat"]})
        backend = PyarrowBackend(df)
        filtered = backend.filter_isin("species", ["cat", "dog"], negate=True)
        assert len(filtered) == 1
        assert list(filtered)[0]["species"] == "bird"

    # --- drop_duplicates ---

    def test_drop_duplicates_subset(self) -> None:
        """Test deduplication with subset of columns."""
        df = pa.table({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        backend = PyarrowBackend(df)
        deduped = backend.drop_duplicates(subset=["a"])
        assert len(deduped) == 2

    def test_drop_duplicates_keep_last(self) -> None:
        """Test keep='last' retains last duplicate row."""
        df = pa.table({"a": [1, 2, 1], "b": ["first", "x", "last"]})
        backend = PyarrowBackend(df)
        deduped = backend.drop_duplicates(subset=["a"], keep="last")
        rows = list(deduped)
        a1_row = next(r for r in rows if r["a"] == 1)
        assert a1_row["b"] == "last"

    # --- column_exists ---

    def test_column_exists_true(self) -> None:
        """Test column_exists returns True for existing column."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        assert backend.column_exists("a") is True

    def test_column_exists_false(self) -> None:
        """Test column_exists returns False for missing column."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        assert backend.column_exists("nonexistent") is False

    # --- map_column ---

    def test_map_column_with_default(self) -> None:
        """Test map_column uses default value for unmapped keys."""
        df = pa.table({"species": ["cat", "dog", "unknown"]})
        backend = PyarrowBackend(df)
        mapped = backend.map_column("species", {"cat": 0, "dog": 1}, "label", default=-1)
        assert list(mapped)[2]["label"] == -1

    # --- concat ---

    def test_concat_sort_columns(self) -> None:
        """Test concat with sort=True sorts columns alphabetically."""
        df1 = pa.table({"b": [1], "a": [2]})
        df2 = pa.table({"b": [3], "a": [4]})
        combined = PyarrowBackend.concat(
            [PyarrowBackend(df1), PyarrowBackend(df2)], sort=True
        )
        assert combined.columns == ["a", "b"]

    # --- from_csv / from_json / from_parquet ---

    def test_from_csv(self, tmp_path: Path) -> None:
        """Test loading a CSV file from local path."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,x\n2,y\n")
        backend = PyarrowBackend.from_csv(str(csv_file))
        assert isinstance(backend, PyarrowBackend)
        assert len(backend) == 2
        assert set(backend.columns) == {"a", "b"}

    def test_from_json(self, tmp_path: Path) -> None:
        """Test loading a JSON lines file from local path."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')
        backend = PyarrowBackend.from_json(str(json_file))
        assert isinstance(backend, PyarrowBackend)
        assert len(backend) == 2

    def test_from_parquet(self, tmp_path: Path) -> None:
        """Test loading a Parquet file from local path."""
        parquet_file = tmp_path / "test.parquet"
        pq.write_table(pa.table({"a": [1, 2], "b": ["x", "y"]}), str(parquet_file))
        backend = PyarrowBackend.from_parquet(str(parquet_file))
        assert isinstance(backend, PyarrowBackend)
        assert len(backend) == 2
        assert set(backend.columns) == {"a", "b"}

    # --- stub methods (tests define expected behavior for future implementation) ---

    def test_rename_columns(self) -> None:
        """Test renaming columns."""
        df = pa.table({"a": [1, 2], "b": ["x", "y"]})
        backend = PyarrowBackend(df)
        renamed = backend.rename_columns({"a": "id", "b": "label"})
        assert renamed.columns == ["id", "label"]
        assert list(renamed)[0] == {"id": 1, "label": "x"}

    def test_add_column(self) -> None:
        """Test adding a new column."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        updated = backend.add_column("b", [4, 5, 6])
        assert "b" in updated.columns
        assert list(updated)[0]["b"] == 4

    def test_select_columns(self) -> None:
        """Test selecting a subset of columns."""
        df = pa.table({"a": [1, 2], "b": ["x", "y"], "c": [True, False]})
        backend = PyarrowBackend(df)
        selected = backend.select_columns(["a", "c"])
        assert selected.columns == ["a", "c"]
        assert "b" not in selected.columns

    def test_sample_rows(self) -> None:
        """Test sampling a fixed number of rows."""
        df = pa.table({"a": list(range(100))})
        backend = PyarrowBackend(df)
        sampled = backend.sample_rows(10, seed=42)
        assert isinstance(sampled, PyarrowBackend)
        assert len(sampled) == 10

    def test_sample_rows_reproducible(self) -> None:
        """Test sample_rows gives same result for same seed."""
        df = pa.table({"a": list(range(100))})
        backend = PyarrowBackend(df)
        s1 = list(backend.sample_rows(10, seed=0))
        s2 = list(backend.sample_rows(10, seed=0))
        assert s1 == s2

    def test_copy(self) -> None:
        """Test copy returns an independent backend with same data."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        copied = backend.copy()
        assert isinstance(copied, PyarrowBackend)
        assert len(copied) == len(backend)
        assert list(copied) == list(backend)

    def test_apply_fn(self) -> None:
        """Test apply_fn passes underlying table to function."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        result = backend.apply_fn(lambda t: t.slice(0, 2))
        assert isinstance(result, PyarrowBackend)
        assert len(result) == 2

    # --- collect ---

    def test_collect_eager_returns_same_data(self) -> None:
        """Test collect on eager backend returns new backend with same data."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        collected = backend.collect()
        assert isinstance(collected, PyarrowBackend)
        assert collected.is_streaming is False
        assert list(collected) == list(backend)

    def test_collect_streaming_materializes(self) -> None:
        """Test collect on streaming backend materializes to eager."""
        df = pa.table({"a": [1, 2, 3]})
        reader = pa.RecordBatchReader.from_batches(df.schema, df.to_batches())
        backend = PyarrowBackend(reader, streaming=True)
        collected = backend.collect()
        assert collected.is_streaming is False
        assert len(collected) == 3

    # --- iter_batches ---

    def test_iter_batches_eager(self) -> None:
        """Test iter_batches yields correct-sized batches in eager mode."""
        df = pa.table({"a": list(range(10))})
        backend = PyarrowBackend(df)
        batches = list(backend.iter_batches(batch_size=3))
        assert len(batches) == 4  # 3+3+3+1
        assert [len(b) for b in batches] == [3, 3, 3, 1]

    def test_iter_batches_streaming_consistent_sizes(self) -> None:
        """Test iter_batches rechunks streaming batches to consistent batch_size."""
        df = pa.table({"a": list(range(10))})
        # Two unequal raw batches: 7 and 3 rows
        raw_batches = [df.slice(0, 7).to_batches()[0], df.slice(7).to_batches()[0]]
        reader = pa.RecordBatchReader.from_batches(df.schema, raw_batches)
        backend = PyarrowBackend(reader, streaming=True)
        batches = list(backend.iter_batches(batch_size=4))
        assert [len(b) for b in batches] == [4, 4, 2]

    def test_iter_batches_yields_pyarrow_backends(self) -> None:
        """Test iter_batches yields PyarrowBackend instances."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        for batch in backend.iter_batches(batch_size=2):
            assert isinstance(batch, PyarrowBackend)
            assert batch.is_streaming is False

    # --- negative index ---

    def test_getitem_negative_index(self) -> None:
        """Test negative index returns correct row."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        assert backend[-1] == {"a": 3}
        assert backend[-3] == {"a": 1}

    def test_getitem_negative_index_out_of_bounds(self) -> None:
        """Test negative index beyond length raises IndexError."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        with pytest.raises(IndexError):
            backend[-4]

    # --- streaming warnings ---

    def test_get_unique_streaming_warns(self) -> None:
        """Test get_unique emits UserWarning in streaming mode."""
        df = pa.table({"a": ["x", "y", "x"]})
        reader = pa.RecordBatchReader.from_batches(df.schema, df.to_batches())
        backend = PyarrowBackend(reader, streaming=True)
        with pytest.warns(UserWarning, match="get_unique"):
            result = backend.get_unique("a")
        assert set(result) == {"x", "y"}

    def test_histogram_streaming_warns(self) -> None:
        """Test histogram emits UserWarning in streaming mode."""
        df = pa.table({"a": ["x", "y", "x"]})
        reader = pa.RecordBatchReader.from_batches(df.schema, df.to_batches())
        backend = PyarrowBackend(reader, streaming=True)
        with pytest.warns(UserWarning, match="histogram"):
            result = backend.histogram("a")
        assert result == {"x": 2, "y": 1}

    # --- subsample_by_column ---

    def test_subsample_by_column_basic(self) -> None:
        """Test subsample keeps correct ratio of rows per category."""
        df = pa.table({"species": ["cat"] * 100 + ["dog"] * 100})
        backend = PyarrowBackend(df)
        result = backend.subsample_by_column("species", {"cat": 0.5, "dog": 0.1}, seed=0)
        hist = result.histogram("species")
        assert hist["cat"] == 50
        assert hist["dog"] == 10

    def test_subsample_by_column_other_key(self) -> None:
        """Test 'other' key subsamples all unlisted values pooled together."""
        df = pa.table({"species": ["cat"] * 10 + ["dog"] * 10 + ["bird"] * 10})
        backend = PyarrowBackend(df)
        result = backend.subsample_by_column("species", {"cat": 1.0, "other": 0.5}, seed=0)
        hist = result.histogram("species")
        assert hist["cat"] == 10
        assert hist.get("dog", 0) + hist.get("bird", 0) == 10

    def test_subsample_by_column_missing_column_raises(self) -> None:
        """Test KeyError on missing column."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        with pytest.raises(KeyError):
            backend.subsample_by_column("nonexistent", {"x": 0.5})

    def test_subsample_by_column_invalid_ratio_raises(self) -> None:
        """Test ValueError on ratio > 1."""
        df = pa.table({"a": ["x", "y"]})
        backend = PyarrowBackend(df)
        with pytest.raises(ValueError):
            backend.subsample_by_column("a", {"x": 1.5})

    # --- upsample_by_column ---

    def test_upsample_by_column_basic(self) -> None:
        """Test upsample reaches target counts per category."""
        df = pa.table({"species": ["cat"] * 5 + ["dog"] * 5})
        backend = PyarrowBackend(df)
        result = backend.upsample_by_column("species", {"cat": 20, "dog": 3}, seed=0)
        hist = result.histogram("species")
        assert hist["cat"] == 20
        assert hist["dog"] == 3

    def test_upsample_by_column_missing_column_raises(self) -> None:
        """Test KeyError on missing column."""
        df = pa.table({"a": [1, 2, 3]})
        backend = PyarrowBackend(df)
        with pytest.raises(KeyError):
            backend.upsample_by_column("nonexistent", {"x": 10})

    def test_upsample_by_column_negative_count_raises(self) -> None:
        """Test ValueError on negative target count."""
        df = pa.table({"a": ["x", "y"]})
        backend = PyarrowBackend(df)
        with pytest.raises(ValueError):
            backend.upsample_by_column("a", {"x": -1})

    def test_upsample_by_column_non_int_count_raises(self) -> None:
        """Test TypeError on non-integer target count."""
        df = pa.table({"a": ["x", "y"]})
        backend = PyarrowBackend(df)
        with pytest.raises(TypeError):
            backend.upsample_by_column("a", {"x": 1.5})  # type: ignore

    # --- multilabel_from_features ---

    def test_multilabel_from_features_basic(self) -> None:
        """Test multilabel creates correct integer label lists."""
        df = pa.table({"tags": [["a", "b"], ["b", "c"], ["a"]]})
        backend = PyarrowBackend(df)
        result, label_map = backend.multilabel_from_features(["tags"], "labels")
        assert set(label_map.keys()) == {"a", "b", "c"}
        rows = list(result)
        assert sorted(rows[0]["labels"]) == [label_map["a"], label_map["b"]]

    def test_multilabel_from_features_scalar_column(self) -> None:
        """Test multilabel wraps scalar column values into single-element lists."""
        df = pa.table({"species": ["cat", "dog", "cat"]})
        backend = PyarrowBackend(df)
        result, label_map = backend.multilabel_from_features(["species"], "labels")
        assert set(label_map.keys()) == {"cat", "dog"}
        assert len(result) == 3

    def test_multilabel_from_features_drops_empty_by_default(self) -> None:
        """Test rows with no labels are dropped when allow_missing_labels=False."""
        df = pa.table({"tags": [["a"], [], ["b"]]})
        backend = PyarrowBackend(df)
        result, _ = backend.multilabel_from_features(["tags"], "labels")
        assert len(result) == 2

    def test_multilabel_from_features_allow_missing_labels(self) -> None:
        """Test rows with no labels are kept when allow_missing_labels=True."""
        df = pa.table({"tags": [["a"], [], ["b"]]})
        backend = PyarrowBackend(df)
        result, _ = backend.multilabel_from_features(
            ["tags"], "labels", allow_missing_labels=True
        )
        assert len(result) == 3

    def test_multilabel_from_features_custom_label_map(self) -> None:
        """Test custom label_map is used instead of auto-generated."""
        df = pa.table({"tags": [["a", "b"]]})
        backend = PyarrowBackend(df)
        label_map = {"a": 10, "b": 20}
        result, returned_map = backend.multilabel_from_features(
            ["tags"], "labels", label_map=label_map
        )
        assert returned_map == label_map
        assert sorted(list(result)[0]["labels"]) == [10, 20]

    def test_multilabel_from_features_missing_feature_raises(self) -> None:
        """Test ValueError on missing input feature column."""
        df = pa.table({"a": [["x"]]})
        backend = PyarrowBackend(df)
        with pytest.raises(ValueError, match="does not exist"):
            backend.multilabel_from_features(["nonexistent"], "labels")


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

    def test_get_backend_pyarrow(self) -> None:
        """Test getting pyarrow backend."""
        backend_cls = get_backend("pyarrow")
        assert backend_cls == PyarrowBackend

    def test_get_backend_invalid(self) -> None:
        """Test getting invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid")  # type: ignore
