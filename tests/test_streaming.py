"""Tests for streaming mode in backends."""

import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from esp_data.backends import PandasBackend, PolarsBackend


class TestPandasStreaming:
    """Tests for PandasBackend streaming mode."""

    def test_streaming_property(self):
        """Test is_streaming property."""
        # Eager mode
        df = pd.DataFrame({"a": [1, 2, 3]})
        backend = PandasBackend(df, streaming=False)
        assert not backend.is_streaming

        # Streaming mode
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            backend_stream = PandasBackend.from_csv(f.name, streaming=True)
            assert backend_stream.is_streaming
            Path(f.name).unlink()

    def test_streaming_getitem_raises(self):
        """Test that __getitem__ raises error in streaming mode."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            backend = PandasBackend.from_csv(f.name, streaming=True)

            with pytest.raises(RuntimeError, match="Cannot use __getitem__ in streaming mode"):
                _ = backend[0]

            Path(f.name).unlink()

    def test_streaming_len_raises(self):
        """Test that __len__ raises error in streaming mode."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            backend = PandasBackend.from_csv(f.name, streaming=True)

            with pytest.raises(RuntimeError, match="Cannot get length in streaming mode"):
                _ = len(backend)

            Path(f.name).unlink()

    def test_streaming_iteration(self):
        """Test that iteration works in streaming mode."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            backend = PandasBackend.from_csv(f.name, streaming=True, streaming_chunk_size=2)

            # Use manual iteration instead of list() to avoid __len__ call
            rows = []
            for row in backend:
                rows.append(row)

            assert len(rows) == 3
            assert rows[0] == {"a": 1, "b": "x"}
            assert rows[2] == {"a": 3, "b": "z"}

            Path(f.name).unlink()

    def test_streaming_operations_raise(self):
        """Test that data operations raise errors in streaming mode."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            backend = PandasBackend.from_csv(f.name, streaming=True)

            with pytest.raises(RuntimeError, match="Cannot perform .* in streaming mode"):
                backend.filter_isin("a", [1, 2])

            with pytest.raises(RuntimeError, match="Cannot perform .* in streaming mode"):
                backend.drop_duplicates()

            with pytest.raises(RuntimeError, match="Cannot perform .* in streaming mode"):
                backend.dropna()

            with pytest.raises(RuntimeError, match="Cannot perform .* in streaming mode"):
                backend.get_unique("a")

            Path(f.name).unlink()

    def test_streaming_json_lines(self):
        """Test streaming with JSON lines."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            df.to_json(f.name, orient="records", lines=True)
            backend = PandasBackend.from_json(f.name, lines=True, streaming=True, streaming_chunk_size=2)

            # Use manual iteration instead of list() to avoid __len__ call
            rows = []
            for row in backend:
                rows.append(row)

            assert len(rows) == 5
            assert rows[0] == {"a": 1}

            Path(f.name).unlink()

    def test_eager_mode_still_works(self):
        """Test that eager mode still works correctly after adding streaming."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PandasBackend(df, streaming=False)

        assert not backend.is_streaming
        assert len(backend) == 3
        assert backend[0] == {"a": 1, "b": "x"}

        rows = list(backend)
        assert len(rows) == 3


class TestPolarsStreaming:
    """Tests for PolarsBackend streaming mode."""

    def test_streaming_property(self):
        """Test is_streaming property."""
        # Eager mode
        df = pl.DataFrame({"a": [1, 2, 3]})
        backend = PolarsBackend(df, streaming=False)
        assert not backend.is_streaming

        # Streaming mode
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            backend_stream = PolarsBackend.from_csv(f.name, streaming=True)
            assert backend_stream.is_streaming
            Path(f.name).unlink()

    def test_streaming_getitem_raises(self):
        """Test that __getitem__ raises error in streaming mode."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            backend = PolarsBackend.from_csv(f.name, streaming=True)

            with pytest.raises(RuntimeError, match="Cannot perform '__getitem__' in streaming mode"):
                _ = backend[0]

            Path(f.name).unlink()

    def test_streaming_len_raises(self):
        """Test that __len__ raises error in streaming mode."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            backend = PolarsBackend.from_csv(f.name, streaming=True)

            with pytest.raises(RuntimeError, match="Cannot perform '__len__' in streaming mode"):
                _ = len(backend)

            Path(f.name).unlink()

    def test_streaming_iteration(self):
        """Test that iteration works in streaming mode."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            backend = PolarsBackend.from_csv(f.name, streaming=True)

            # Use manual iteration instead of list() to avoid __len__ call
            rows = []
            for row in backend:
                rows.append(row)

            assert len(rows) == 3
            assert rows[0] == {"a": 1, "b": "x"}
            assert rows[2] == {"a": 3, "b": "z"}

            Path(f.name).unlink()

    def test_streaming_operations_work(self):
        """Test that LazyFrame operations work in streaming mode."""
        df = pl.DataFrame({"a": [1, 2, 2, 3]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            backend = PolarsBackend.from_csv(f.name, streaming=True)

            # These operations work with LazyFrames and preserve streaming mode
            filtered = backend.filter_isin("a", [1, 2])
            assert filtered.is_streaming
            # Use the backend's collect() method to materialize
            filtered_collected = filtered.collect()
            assert sorted(filtered_collected.unwrap["a"].to_list()) == [1, 2, 2]

            deduped = backend.drop_duplicates()
            assert deduped.is_streaming
            deduped_collected = deduped.collect()
            assert sorted(deduped_collected.unwrap["a"].to_list()) == [1, 2, 3]

            cleaned = backend.dropna()
            assert cleaned.is_streaming
            cleaned_collected = cleaned.collect()
            assert sorted(cleaned_collected.unwrap["a"].to_list()) == [1, 2, 2, 3]

            # get_unique requires collection — expect a warning to that effect
            with pytest.warns(UserWarning, match="get_unique.*requires collection"):
                unique_vals = backend.get_unique("a")
            assert sorted(unique_vals) == [1, 2, 3]

            Path(f.name).unlink()

    def test_streaming_json_lines(self):
        """Test streaming with JSON lines."""
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            df.write_ndjson(f.name)
            backend = PolarsBackend.from_json(f.name, lines=True, streaming=True)

            # Use manual iteration instead of list() to avoid __len__ call
            rows = []
            for row in backend:
                rows.append(row)

            assert len(rows) == 5
            assert rows[0] == {"a": 1}

            Path(f.name).unlink()

    def test_eager_mode_still_works(self):
        """Test that eager mode still works correctly after adding streaming."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend = PolarsBackend(df, streaming=False)

        assert not backend.is_streaming
        assert len(backend) == 3
        assert backend[0] == {"a": 1, "b": "x"}

        rows = list(backend)
        assert len(rows) == 3

    def test_collect_method(self):
        """Test that collect() materializes LazyFrame and switches to eager mode."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            backend = PolarsBackend.from_csv(f.name, streaming=True)

            assert backend.is_streaming

            # Call collect() to materialize
            collected_backend = backend.collect()

            # Should now be in eager mode
            assert not collected_backend.is_streaming
            assert len(collected_backend) == 3
            assert collected_backend[0] == {"a": 1}

            Path(f.name).unlink()

    def test_subsample_warning(self):
        """Test that subsample_by_column warns when in streaming mode."""
        df = pl.DataFrame({"a": [1, 1, 2, 2, 3, 3], "b": ["x", "y", "z", "w", "v", "u"]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            backend = PolarsBackend.from_csv(f.name, streaming=True)

            assert backend.is_streaming

            # Should warn about collection
            with pytest.warns(UserWarning, match="subsample_by_column.*requires collection"):
                result = backend.subsample_by_column("a", {1: 0.5, 2: 1.0})

            # Result should be in eager mode
            assert not result.is_streaming

            Path(f.name).unlink()

    def test_streaming_iter_does_not_fully_collect(self):
        """Iteration over a LazyFrame should not require collecting everything
        first — the backend should wrap `LazyFrame.collect_batches()`.
        """
        df = pl.DataFrame({"a": list(range(10)), "b": ["x"] * 10})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            backend = PolarsBackend.from_csv(f.name, streaming=True)

            # Underlying must still be a LazyFrame after constructing the iterator
            # (calling iter itself should not coerce the backend to eager).
            it = iter(backend)
            assert isinstance(backend.unwrap, pl.LazyFrame)

            first = next(it)
            assert first == {"a": 0, "b": "x"}

            rest = list(it)
            assert len(rest) == 9
            assert rest[-1] == {"a": 9, "b": "x"}

            Path(f.name).unlink()

    def test_streaming_iter_batches(self):
        """iter_batches should stream chunks from a LazyFrame, never
        materializing the whole thing at once.
        """
        df = pl.DataFrame({"a": list(range(50))})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            backend = PolarsBackend.from_csv(f.name, streaming=True)

            batches = list(backend.iter_batches(batch_size=10))

            # Every yielded batch should be eager (DataFrame), not a LazyFrame.
            for b in batches:
                assert not b.is_streaming
                assert isinstance(b.unwrap, pl.DataFrame)

            # All rows accounted for.
            total_rows = sum(len(b) for b in batches)
            assert total_rows == 50

            # Values recover the original series (order not guaranteed across
            # batches in streaming engines, so we sort).
            collected = sorted(v for b in batches for v in b.unwrap["a"].to_list())
            assert collected == list(range(50))

            Path(f.name).unlink()

    def test_eager_iter_batches(self):
        """iter_batches on an eager DataFrame still works and yields fixed-size
        slices.
        """
        df = pl.DataFrame({"a": list(range(25))})
        backend = PolarsBackend(df, streaming=False)

        batches = list(backend.iter_batches(batch_size=10))
        sizes = [len(b) for b in batches]
        assert sizes == [10, 10, 5]

    def test_column_exists_on_lazyframe(self):
        """column_exists must work on a streaming backend without raising
        or triggering full collection.
        """
        df = pl.DataFrame({"foo": [1, 2, 3], "bar": ["a", "b", "c"]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            backend = PolarsBackend.from_csv(f.name, streaming=True)

            assert backend.column_exists("foo")
            assert backend.column_exists("bar")
            assert not backend.column_exists("baz")

            # Still lazy afterwards.
            assert isinstance(backend.unwrap, pl.LazyFrame)

            Path(f.name).unlink()

    def test_get_unique_streaming_warns(self):
        """get_unique forces collection on a LazyFrame — warn the caller."""
        df = pl.DataFrame({"a": [1, 1, 2, 2, 3]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            backend = PolarsBackend.from_csv(f.name, streaming=True)

            with pytest.warns(UserWarning, match="get_unique.*requires collection"):
                uniques = backend.get_unique("a")

            assert uniques == [1, 2, 3]
            Path(f.name).unlink()

    def test_histogram_streaming_warns(self):
        """histogram forces collection on a LazyFrame — warn the caller."""
        df = pl.DataFrame({"a": [1, 1, 2, 2, 2, 3]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            backend = PolarsBackend.from_csv(f.name, streaming=True)

            with pytest.warns(UserWarning, match="histogram.*requires collection"):
                hist = backend.histogram("a")

            assert hist == {1: 2, 2: 3, 3: 1}
            Path(f.name).unlink()

    def test_from_parquet_kwargs_filtered_against_parquet_signature(self):
        """from_parquet must filter kwargs against `scan_parquet` /
        `read_parquet` signatures, not `scan_csv`.

        Regression: a copy-paste bug previously filtered kwargs against the
        `scan_csv` signature, silently stripping legitimate parquet-only
        arguments such as `row_index_name`.
        """
        df = pl.DataFrame({"a": [1, 2, 3]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet", delete=False) as f:
            df.write_parquet(f.name)

            # row_index_name is valid for scan_parquet / read_parquet but NOT
            # for scan_csv. With the bug it would be silently dropped; with the
            # fix it actually gets applied.
            backend_stream = PolarsBackend.from_parquet(
                f.name, streaming=True, row_index_name="idx"
            )
            assert "idx" in backend_stream.columns

            backend_eager = PolarsBackend.from_parquet(
                f.name, streaming=False, row_index_name="idx"
            )
            assert "idx" in backend_eager.columns

            Path(f.name).unlink()
