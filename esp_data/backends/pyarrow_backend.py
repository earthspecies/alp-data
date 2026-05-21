"""Pyarrow implementation of the DataFrameBackend protocol"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Iterator, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pa_fs
import pyarrow.parquet as pq
from pyarrow import csv as pa_csv
from pyarrow import json as pa_json

from .protocol import DataBackend


def _open_gcs_input_stream(path_str: str) -> tuple[pa_fs.GcsFileSystem, str]:
    """Return GcsFileSystem and bucket_and_key for a gs:// path.

    Parameters
    ----------
    path_str : str
        A path starting with ``gs://``

    Returns
    -------
    tuple[pa_fs.GcsFileSystem, str]
        GcsFileSystem instance and the path with the ``gs://`` prefix stripped
    """
    return pa_fs.GcsFileSystem(), path_str[len("gs://") :]


def _open_s3_input_stream(path_str: str) -> tuple[pa_fs.S3FileSystem, str]:
    """Return S3FileSystem and bucket_and_key for a s3:// path.

    Parameters
    ----------
    path_str : str
        A path starting with ``s3://``

    Returns
    -------
    tuple[pa_fs.S3FileSystem, str]
        S3FileSystem instance and the path with the ``s3://`` prefix stripped
    """
    return pa_fs.S3FileSystem(), path_str[len("s3://")]


class PyarrowBackend(DataBackend):
    """Pyarrow implementation of the DataBackend protocol.

    This backend wraps a pyarrow Table or RecordBatchReader and provides a unified
    interface for DataBackend operations that can work across different backend
    implementations.

    Supports both eager (Table) and streaming (RecordBatchReader) modes.

    Parameters
    ----------
    df : pa.Table | pa.RecordBatchReader
        The pyarrow Table or RecordBatchReader to wrap
    streaming : bool
        Whether the backend is in streaming mode
    streaming_chunk_size: int
        Number of rows per batch when iterating in streaming mode (default: 1000)
        1000 is a good number because its high enough to reduce I/O and any higher
        doesn't help because the main latency source in Dataset __getitem__ calls
        are in loading audio anyway.

    Examples
    --------
    >>> import pyarrow as pa
    >>> from esp_data.backends import PyarrowBackend
    >>> table = pa.table({
    ...     "species": ["cat", "dog", "fish", "cat", "dog", None],
    ...     "count": [5, 3, 8, 2, 7, 1]
    ... })
    >>> backend = PyarrowBackend(table)
    >>> row = backend[0]
    >>> filtered = backend.filter_isin("species", ["cat", "dog"])
    >>> assert filtered.unwrap["species"].to_pylist() == ["cat", "dog", "cat", "dog"]
    >>> print(backend.columns)
    ['species', 'count']
    >>> for row in backend:
    ...     print(row)
    ...     break
    {'species': 'cat', 'count': 5}
    >>> collected = backend.collect()
    >>> assert isinstance(collected.unwrap, pa.Table)
    """

    def __init__(
        self,
        df: pa.Table | pa.RecordBatchReader,
        *,
        streaming: bool = False,
        streaming_chunk_size: int = 1000,
    ) -> None:
        """Initialize the backend with a pyarrow Table or RecordBatchReader.

        Parameters
        ----------
        df : pa.Table | pa.RecordBatchReader
            The Table or RecordBatchReader to wrap
        streaming : bool, optional
            Whether to use streaming mode, by default False
        streaming_chunk_size : int, optional
            Number of rows per batch when iterating in streaming mode, by default 1000
        """
        self._df = df
        self._streaming = streaming
        self._streaming_chunk_size = streaming_chunk_size

        # Auto-detect streaming mode if RecordBatchReader is provided
        if isinstance(df, pa.RecordBatchReader) and not streaming:
            self._streaming = True

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        streaming: bool = False,
        streaming_chunk_size: int = 1000,
        **kwargs: Any,
    ) -> PyarrowBackend:
        """Read a CSV file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the CSV file (supports local and cloud paths via cloudpathlib)
        streaming : bool, optional
            If True, use streaming mode with RecordBatchReader, by default False
        streaming_chunk_size : int, optional
            Number of rows per batch in streaming mode, by default 1000
        **kwargs : Any
            Additional pyarrow-specific arguments

        Returns
        -------
        PyarrowBackend
            Backend instance wrapping the loaded Table or RecordBatchReader
        """
        path_str = str(path)
        if streaming:
            if path_str.startswith("gs://"):
                gcs, bucket_and_key = _open_gcs_input_stream(path_str)
                # Keep file handle open — reader holds a reference to it
                f = gcs.open_input_stream(bucket_and_key)
                reader = pa_csv.open_csv(f, **kwargs)
            elif path_str.startswith("gs://"):
                s3, bucket_and_key = _open_s3_input_stream(path_str)
                f = s3.open_input_stream(bucket_and_key)
                reader = pa_csv.open_csv(f, **kwargs)
            else:
                reader = pa_csv.open_csv(path_str, **kwargs)
            return cls(reader, streaming=True, streaming_chunk_size=streaming_chunk_size)
        else:
            if path_str.startswith("gs://"):
                gcs, bucket_and_key = _open_gcs_input_stream(path_str)
                with gcs.open_input_stream(bucket_and_key) as f:
                    df = pa_csv.read_csv(f, **kwargs)
            elif path_str.startswith("gs://"):
                s3, bucket_and_key = _open_s3_input_stream(path_str)
                f = s3.open_input_stream(bucket_and_key)
                reader = pa_csv.read_csv(f, **kwargs)
            else:
                df = pa_csv.read_csv(path_str, **kwargs)
            return cls(df, streaming=False)

    @classmethod
    def from_json(
        cls,
        path: str,
        *,
        lines: bool = False,
        streaming: bool = False,
        **kwargs: Any,
    ) -> PyarrowBackend:
        """Read a JSON file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the JSON file (supports local and cloud paths)
        lines : bool, optional
            Ignored for PyArrow backend (pyarrow.json always reads line-delimited),
            by default False
        streaming : bool, optional
            If True, use streaming mode — not supported for JSON, raises
            `NotImplementedError`, by default False
        **kwargs : Any
            Additional pyarrow-specific arguments passed to `pyarrow.json.read_json`

        Returns
        -------
        PyarrowBackend
            Backend instance wrapping the loaded Table

        Raises
        ------
        NotImplementedError
            If streaming=True, since pyarrow has no streaming JSON reader.
        """
        if streaming:
            raise NotImplementedError(
                "Streaming mode is not supported for JSON files with the PyArrow backend. "
                "Load eagerly or use the Polars backend with lines=True for streaming JSON."
            )

        path_str = str(path)
        if path_str.startswith("gs://"):
            gcs, bucket_and_key = _open_gcs_input_stream(path_str)
            with gcs.open_input_stream(bucket_and_key) as f:
                df = pa_json.read_json(f, **kwargs)
        else:
            df = pa_json.read_json(path_str, **kwargs)
        return cls(df, streaming=False)

    @classmethod
    def from_parquet(
        cls,
        path: str,
        *,
        streaming: bool = False,
        streaming_chunk_size: int = 1000,
        **kwargs: Any,
    ) -> PyarrowBackend:
        """Read a Parquet file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the Parquet file (supports local and cloud paths)
        streaming : bool, optional
            If True, use streaming mode with RecordBatchReader, by default False
        streaming_chunk_size : int, optional
            Number of rows per batch in streaming mode, by default 1000
        **kwargs : Any
            Additional pyarrow-specific arguments passed to `pyarrow.parquet.read_table`

        Returns
        -------
        PyarrowBackend
            Backend instance wrapping the loaded Table or RecordBatchReader
        """
        path_str = str(path)
        if streaming:
            if path_str.startswith("gs://"):
                gcs, bucket_and_key = _open_gcs_input_stream(path_str)
                parquet_file = pq.ParquetFile(bucket_and_key, filesystem=gcs)
            else:
                parquet_file = pq.ParquetFile(path_str)
            schema = parquet_file.schema_arrow
            reader = pa.RecordBatchReader.from_batches(
                schema,
                parquet_file.iter_batches(batch_size=streaming_chunk_size),
            )
            return cls(reader, streaming=True, streaming_chunk_size=streaming_chunk_size)
        else:
            if path_str.startswith("gs://"):
                gcs, bucket_and_key = _open_gcs_input_stream(path_str)
                df = pq.read_table(bucket_and_key, filesystem=gcs, **kwargs)
            else:
                df = pq.read_table(path_str, **kwargs)
            return cls(df, streaming=False)

    @property
    def is_streaming(self) -> bool:
        """Check if backend is in streaming mode.

        Returns
        -------
        bool
            True if in streaming mode, False otherwise
        """
        return self._streaming

    def _ensure_collected(self) -> pa.Table:
        """Ensure the Table is collected (not a RecordBatchReader).

        Returns
        -------
        pa.Table
            The underlying Table (reads and consumes stream if streaming)
        """
        if isinstance(self._df, pa.RecordBatchReader):
            return self._df.read_all()
        return self._df

    def collect(self) -> PyarrowBackend:
        """Materialize the RecordBatchReader and return an eager backend.

        Returns
        -------
        PyarrowBackend
            New backend in eager mode with materialized Table

        Notes
        -----
        If the backend is already in eager mode, returns a new backend wrapping
        the same Table. Calling this on a streaming backend consumes the reader.
        """
        return PyarrowBackend(self._ensure_collected(), streaming=False)

    def _ensure_not_streaming(self, operation: str) -> None:
        """Raise error if in streaming mode for operations that require eager evaluation.

        Parameters
        ----------
        operation : str
            Name of the operation being attempted

        Raises
        ------
        RuntimeError
            If backend is in streaming mode
        """
        if self._streaming:
            raise RuntimeError(
                f"Cannot perform '{operation}' in streaming mode. "
                f"RecordBatchReader requires explicit collection. "
                f"Consider using .collect() or iterate over the data."
            )

    def __getitem__(self, key: int | list[int] | slice) -> dict[str, Any] | PyarrowBackend:
        """Get row(s) from the Table using Pythonic indexing.

        Parameters
        ----------
        key : int | list[int] | slice
            - int: Get single row as dict
            - list[int]: Get multiple rows as new backend
            - slice: Get row range as new backend

        Returns
        -------
        dict[str, Any] | PyarrowBackend
            - dict if key is int (single row)
            - PyarrowBackend if key is list or slice (multiple rows)

        Raises
        ------
        IndexError
            If index is out of bounds
        TypeError
            If key type is not supported
        """
        self._ensure_not_streaming("__getitem__")
        df = self._df

        if isinstance(key, int):
            # Return single row as dict
            if key < 0:
                key += len(df)
            if key < 0 or key >= len(df):
                raise IndexError(f"Index {key} out of bounds for Table of length {len(df)}")
            row = df.take([key]).to_pydict()
            # Convert values from list to scalar
            return {k: v[0] for k, v in row.items()}
        elif isinstance(key, list):
            return PyarrowBackend(df.take(key), streaming=False)
        elif isinstance(key, slice):
            if key.step is not None:
                raise NotImplementedError("Step slicing is not supported. Use a list of indices.")
            offset = key.start if key.start is not None else 0
            length = (key.stop - offset) if key.stop is not None else None
            return PyarrowBackend(df.slice(offset=offset, length=length))
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")

    def __len__(self) -> int:
        """Get the number of rows in the Table.

        Returns
        -------
        int
            Number of rows
        """
        self._ensure_not_streaming("__len__")
        return len(self._df)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over Table rows as dictionaries.

        In streaming mode (RecordBatchReader), reads one batch at a time so the
        full result never needs to live in memory at once.
        In eager mode (Table), yields rows directly.

        Yields
        ------
        dict[str, Any]
            Dictionary for each row mapping column names to values
        """
        if isinstance(self._df, pa.RecordBatchReader):
            for batch in self._df:
                for row in batch.to_pylist():
                    yield row
        else:
            for batch in self._df.to_batches(
                max_chunksize=self._streaming_chunk_size if self._streaming else None
            ):
                for row in batch.to_pylist():
                    yield row

    def iter_batches(self, batch_size: int = 1000) -> Iterator[PyarrowBackend]:
        """Iterate over Table in batches.

        Parameters
        ----------
        batch_size : int, optional
            Number of rows per batch, by default 1000

        Yields
        ------
        PyarrowBackend
            Backend instances wrapping batches of rows.
            Yielded backends are always in eager mode.

        Notes
        -----
        In streaming mode, accumulates batches from the RecordBatchReader and
        yields chunks of exactly `batch_size` rows (last chunk may be smaller).
        Only one chunk worth of data is held in memory at a time.
        """
        if isinstance(self._df, pa.RecordBatchReader):
            pending: pa.Table | None = None
            for batch in self._df:
                chunk = pa.Table.from_batches([batch])
                pending = chunk if pending is None else pa.concat_tables([pending, chunk])
                while len(pending) >= batch_size:
                    yield PyarrowBackend(pending.slice(0, batch_size), streaming=False)
                    pending = pending.slice(batch_size)
            if pending is not None and len(pending) > 0:
                yield PyarrowBackend(pending, streaming=False)
        else:
            df = self._df
            for start_idx in range(0, len(df), batch_size):
                yield PyarrowBackend(df.slice(start_idx, batch_size), streaming=False)

    def filter_isin(
        self, column: str, values: list[Any], *, negate: bool = False
    ) -> PyarrowBackend:
        """Filter Table rows where column values are in (or not in) a list.

        Parameters
        ----------
        column : str
            Column name to filter on
        values : list[Any]
            List of values to match
        negate : bool, optional
            If True, keep rows NOT in values list, by default False

        Returns
        -------
        PyarrowBackend
            New backend with filtered Table
        """
        self._ensure_not_streaming("filter_isin")
        expr = pc.field(column).isin(values)
        if negate:
            expr = ~expr
        filtered_tb = self._df.filter(expr)
        return PyarrowBackend(filtered_tb, streaming=False)

    def drop_duplicates(
        self,
        subset: list[str] | None = None,
        *,
        keep: Literal["first", "last"] = "first",
    ) -> PyarrowBackend:
        """Remove duplicate rows from the Table.

        Parameters
        ----------
        subset : list[str] | None, optional
            Column names to consider for identifying duplicates.
            If None, use all columns, by default None
        keep : Literal["first", "last"], optional
            Which duplicate to keep, by default "first"

        Returns
        -------
        PyarrowBackend
            New backend with duplicates removed
        """
        self._ensure_not_streaming("drop_duplicates")
        df = self._df
        cols = subset if subset is not None else df.column_names

        idx_col = "_row_idx_"
        df_with_idx = df.append_column(idx_col, pa.array(range(len(df))))
        agg_fn = "min" if keep == "first" else "max"
        idx_table = df_with_idx.group_by(cols).aggregate([(idx_col, agg_fn)])
        indices = sorted(idx_table.column(f"{idx_col}_{agg_fn}").to_pylist())
        return PyarrowBackend(df.take(indices), streaming=False)

    def dropna(
        self,
        subset: list[str] | None = None,
    ) -> PyarrowBackend:
        """Remove rows with missing values.

        Parameters
        ----------
        subset : list[str] | None, optional
            Column names to consider for null detection.
            If None, check all columns, by default None

        Returns
        -------
        PyarrowBackend
            New backend with null rows removed
        """
        self._ensure_not_streaming("dropna")
        if subset is not None:
            mask = pc.is_valid(self._df.column(subset[0]))
            for col in subset[1:]:
                mask = pc.and_(mask, pc.is_valid(self._df.column(col)))
            return PyarrowBackend(self._df.filter(mask), streaming=False)
        return PyarrowBackend(self._df.drop_null(), streaming=False)

    def get_unique(self, column: str) -> list[Any]:
        """Get sorted unique values from a column.

        Parameters
        ----------
        column : str
            Column name

        Returns
        -------
        list[Any]
            Sorted list of unique values (nulls excluded)

        Notes
        -----
        In streaming mode, materializes the full stream to compute uniques.
        A UserWarning is emitted because this consumes the underlying reader.
        """
        if self._streaming:
            warnings.warn(
                "get_unique() requires collection of RecordBatchReader to compute uniques. "
                "The backend itself is not modified, but the underlying stream will be consumed.",
                UserWarning,
                stacklevel=2,
            )
        df = self._ensure_collected()
        unique_values = df.column(column).drop_null().unique().to_pylist()
        return sorted(unique_values)

    def histogram(self, column: str) -> dict[Any, int]:
        """Get value counts (histogram) for a column.

        Parameters
        ----------
        column : str
            Column name

        Returns
        -------
        dict[Any, int]
            Dictionary mapping unique values to their counts (nulls excluded)

        Notes
        -----
        In streaming mode, materializes the full stream to compute counts.
        A UserWarning is emitted because this consumes the underlying reader.
        """
        if self._streaming:
            warnings.warn(
                "histogram() requires collection of RecordBatchReader to compute counts. "
                "The backend itself is not modified, but the underlying stream will be consumed.",
                UserWarning,
                stacklevel=2,
            )
        df = self._ensure_collected()
        counts_table = (
            df.select([column]).drop_null().group_by(column).aggregate([([], "count_all")])
        )
        counts_dict = counts_table.to_pydict()
        return dict(zip(counts_dict[column], counts_dict["count_all"], strict=True))

    def map_column(
        self,
        column: str,
        mapping: dict[Any, Any],
        output_column: str,
        *,
        default: Any | None = None,  # noqa ANN401
    ) -> PyarrowBackend:
        """Create a new column by mapping values from an existing column.

        Parameters
        ----------
        column : str
            Source column name
        mapping : dict[Any, Any]
            Dictionary mapping source values to output values
        output_column : str
            Name of the new column to create
        default : Any, optional
            Value to use for unmapped keys, by default None

        Returns
        -------
        PyarrowBackend
            New backend with mapped column added
        """
        self._ensure_not_streaming("map_column")
        source = self._df.column(column).to_pylist()
        mapped = [mapping.get(v, default) for v in source]
        new_col = pa.array(mapped)
        new_df = self._df.append_column(output_column, new_col)
        return PyarrowBackend(new_df)

    def rename_columns(
        self,
        mapping: dict[str, str],
    ) -> PyarrowBackend:
        """Rename Table columns.

        Parameters
        ----------
        mapping : dict[str, str]
            Dictionary mapping old column names to new names

        Returns
        -------
        PyarrowBackend
            New backend with renamed columns
        """
        self._ensure_not_streaming("rename_columns")
        new_names = [mapping.get(name, name) for name in self._df.column_names]
        new_df = self._df.rename_columns(new_names)
        return PyarrowBackend(new_df)

    def add_column(
        self,
        column: str,
        values: Any,  # noqa ANN401
    ) -> PyarrowBackend:
        """Add a new column to the Table.

        Parameters
        ----------
        column : str
            Name of the new column
        values : Any
            Values for the new column (scalar or array-like)

        Returns
        -------
        PyarrowBackend
            New backend with new column added
        """
        self._ensure_not_streaming("add_column")
        if isinstance(values, (pa.Array, pa.ChunkedArray)):
            new_col = values
        elif isinstance(values, list):
            new_col = pa.array(values)
        else:
            # scalar — broadcast to length of table
            new_col = pa.array([values] * len(self._df))
        new_df = self._df.append_column(column, new_col)
        return PyarrowBackend(new_df)

    def select_columns(
        self,
        columns: list[str],
    ) -> PyarrowBackend:
        """Select a subset of columns from the Table.

        Parameters
        ----------
        columns : list[str]
            List of column names to keep

        Returns
        -------
        PyarrowBackend
            New backend with only specified columns
        """
        self._ensure_not_streaming("select_columns")
        return PyarrowBackend(self._df.select(columns))

    @classmethod
    def concat(
        cls,
        backends: list[PyarrowBackend],
        *,
        ignore_index: bool = True,
        sort: bool = False,
    ) -> PyarrowBackend:
        """Concatenate multiple backend instances vertically (row-wise).

        Parameters
        ----------
        backends : list[PyarrowBackend]
            List of backend instances to concatenate
        ignore_index : bool, optional
            Unused — present for API compatibility, by default True
        sort : bool, optional
            If True, sort columns alphabetically, by default False

        Returns
        -------
        PyarrowBackend
            New backend with concatenated data

        Raises
        ------
        RuntimeError
            If any input backend is in streaming mode.
        """
        for b in backends:
            if b._streaming:
                raise RuntimeError(
                    "Cannot concat streaming backends. Call .collect() on each backend first."
                )
        dfs = [backend._df for backend in backends]

        concatenated_df = pa.concat_tables(dfs)

        if sort:
            sorted_cols = sorted(concatenated_df.column_names)
            concatenated_df = concatenated_df.select(sorted_cols)

        return cls(concatenated_df)

    @property
    def columns(self) -> list[str]:
        """Get the list of column names.

        Returns
        -------
        list[str]
            List of column names
        """
        if self._streaming:
            return self._df.schema.names
        return self._df.column_names

    def column_exists(self, column: str) -> bool:
        """Check if a column exists in the DataFrame.

        Parameters
        ----------
        column : str
            Column name to look for

        Returns
        -------
        bool
            True if column exists, False otherwise
        """
        if self._streaming:
            return column in self._df.schema.names
        return column in self._df.column_names

    @property
    def unwrap(self) -> pa.Table | pa.RecordBatchReader:
        """Get the underlying Table or RecordBatchReader object.

        Returns
        -------
        pa.Table | pa.RecordBatchReader
            The underlying pyarrow Table or RecordBatchReader
        """
        return self._df

    def _sample_by_column_helper(
        self,
        df: pa.Table,
        column: str,
        values_dict: dict[str, Any],
        *,
        sample_fn: Callable[[pa.Table, Any], pa.Table],
        other_sample_fn: Callable[[pa.Table, Any], pa.Table],
        dict_name: str,
    ) -> PyarrowBackend:
        """Helper function for sampling by column values.

        Parameters
        ----------
        df : pa.Table
            Already-collected table to sample from
        column : str
            Column name to group by
        values_dict : dict[str, Any]
            Dictionary mapping column values to sampling parameters
        sample_fn : Callable[[pa.Table, Any], pa.Table]
            Function to sample a group given (group_df, value_from_dict).
            The function should handle seed internally via closure.
        other_sample_fn : Callable[[pa.Table, Any], pa.Table]
            Function to sample the "other" group given (other_df, other_value).
            The function should handle seed internally via closure.
        dict_name : str
            Name of the dictionary for error messages (e.g., "ratios", "target_counts")

        Returns
        -------
        PyarrowBackend
            New backend with sampled rows
        """
        groups = []

        unique_values_set = set(df.column(column).drop_null().unique().to_pylist())
        explicit_values = set(values_dict.keys()) - {"other"}

        for val, param in values_dict.items():
            if val == "other":
                continue

            mask = pc.equal(df.column(column), val)
            group_df = df.filter(mask)

            if len(group_df) == 0:
                if val not in unique_values_set:
                    warnings.warn(
                        f"Key {val!r} in {dict_name} not found in column '{column}'. "
                        "This may indicate a typo or type mismatch "
                        "(e.g., string key for int column). Skipping this key.",
                        UserWarning,
                        stacklevel=3,
                    )
                continue

            sampled = sample_fn(group_df, param)
            if len(sampled) > 0:
                groups.append(sampled)

        if "other" in values_dict:
            other_mask = pc.invert(pc.is_in(df.column(column), pa.array(list(explicit_values))))
            other_df = df.filter(other_mask)

            other_param = values_dict["other"]
            if len(other_df) == 0:
                sampled_other = other_df.slice(0, 0)
            else:
                sampled_other = other_sample_fn(other_df, other_param)

            if len(sampled_other) > 0:
                groups.append(sampled_other)

        if groups:
            result_df = pa.concat_tables(groups)
        else:
            result_df = df.slice(0, 0)

        return PyarrowBackend(result_df, streaming=False)

    def subsample_by_column(
        self,
        column: str,
        ratios: dict[str, float],
        *,
        seed: int = 42,
    ) -> PyarrowBackend:
        """Subsample rows by column values with specified ratios.

        For each unique value in the column, sample the specified ratio of rows.
        Special key "other" can be used to subsample all values not explicitly listed.

        If the backend is in streaming mode, a UserWarning will be issued and the
        RecordBatchReader will be consumed since sampling requires materialization.

        Note: The "other" key pools all unlisted values together and samples from
        the pooled group, rather than applying the ratio per unlisted category.

        Parameters
        ----------
        column : str
            Column name to group by
        ratios : dict[str, float]
            Dictionary mapping column values to sampling ratios (0.0 to 1.0).
            Special key "other" applies to all unlisted values (pooled together).
        seed : int, optional
            Random seed for reproducibility, by default 42

        Returns
        -------
        PyarrowBackend
            New backend with subsampled rows

        Raises
        ------
        KeyError
            If the specified column does not exist in the DataFrame
        ValueError
            If any ratio is negative or greater than 1.0
        """
        if self._streaming:
            warnings.warn(
                "subsample_by_column() requires collection of RecordBatchReader for sampling. "
                "The returned backend will be in eager mode (streaming=False).",
                UserWarning,
                stacklevel=2,
            )

        df = self._ensure_collected()

        if column not in df.column_names:
            raise KeyError(f"Column '{column}' not found in DataFrame columns.")

        for val, ratio in ratios.items():
            if ratio < 0.0:
                raise ValueError(
                    f"Ratio for value {val!r} is negative: {ratio}. Ratios must be >= 0.0"
                )
            if ratio > 1.0:
                raise ValueError(
                    f"Ratio for value {val!r} is greater than 1.0: {ratio}. "
                    "For ratios > 1.0, use upsample_by_column() instead."
                )

        rng = np.random.default_rng(seed=seed)

        def sample_by_ratio(group_df: pa.Table, ratio: float) -> pa.Table:
            if ratio >= 1.0:
                return group_df
            n = max(0, int(len(group_df) * ratio))
            if n == 0:
                return group_df.slice(0, 0)
            indices = rng.choice(len(group_df), size=n, replace=False).tolist()
            return group_df.take(indices)

        return self._sample_by_column_helper(
            df,
            column=column,
            values_dict=ratios,
            sample_fn=sample_by_ratio,
            other_sample_fn=sample_by_ratio,
            dict_name="ratios",
        )

    def upsample_by_column(
        self,
        column: str,
        target_counts: dict[str, int],
        *,
        seed: int = 42,
    ) -> PyarrowBackend:
        """Upsample rows by column values to target counts with replacement.

        For each unique value in the column, sample rows with replacement to reach
        the target count. If a category already has more rows than the target, it will
        be downsampled (without replacement) to the target count.

        If the backend is in streaming mode, a UserWarning will be issued and the
        RecordBatchReader will be consumed since sampling requires materialization.

        Note: The "other" key pools all unlisted values together and samples from
        the pooled group to reach the target count, rather than applying the target
        per unlisted category.

        Parameters
        ----------
        column : str
            Column name to group by
        target_counts : dict[str, int]
            Dictionary mapping column values to target sample counts.
            Special key "other" applies to all unlisted values (pooled together).
        seed : int, optional
            Random seed for reproducibility, by default 42

        Returns
        -------
        PyarrowBackend
            New backend with upsampled/downsampled rows

        Raises
        ------
        KeyError
            If the specified column does not exist in the DataFrame
        ValueError
            If any target count is negative
        TypeError
            If any target count is not an integer
        """
        if self._streaming:
            warnings.warn(
                "upsample_by_column() requires collection of RecordBatchReader for sampling. "
                "The returned backend will be in eager mode (streaming=False).",
                UserWarning,
                stacklevel=2,
            )

        df = self._ensure_collected()

        if column not in df.column_names:
            raise KeyError(f"Column '{column}' not found in DataFrame columns.")

        for val, target_count in target_counts.items():
            if not isinstance(target_count, int):
                raise TypeError(
                    f"Target count for value {val!r} must be an integer, "
                    f"got {type(target_count).__name__}"
                )
            if target_count < 0:
                raise ValueError(
                    f"Target count for value {val!r} is negative: {target_count}. "
                    "Target counts must be >= 0"
                )

        rng = np.random.default_rng(seed=seed)

        def sample_by_target_count(group_df: pa.Table, target_count: int) -> pa.Table:
            if target_count == 0:
                return group_df.slice(0, 0)
            replace = target_count > len(group_df)
            indices = rng.choice(len(group_df), size=target_count, replace=replace).tolist()
            return group_df.take(indices)

        return self._sample_by_column_helper(
            df,
            column=column,
            values_dict=target_counts,
            sample_fn=sample_by_target_count,
            other_sample_fn=sample_by_target_count,
            dict_name="target_counts",
        )

    def sample_rows(
        self,
        n: int,
        *,
        seed: int = 42,
        replace: bool = False,
    ) -> PyarrowBackend:
        """Randomly sample n rows from the DataFrame.

        Parameters
        ----------
        n : int
            Number of rows to sample
        seed : int, optional
            Random seed for reproducibility, by default 42
        replace : bool, optional
            Whether to sample with replacement, by default False

        Returns
        -------
        PyarrowBackend
            New backend with sampled rows
        """
        self._ensure_not_streaming("sample_rows")

        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(len(self._df), size=n, replace=replace).tolist()
        return PyarrowBackend(self._df.take(indices), streaming=False)

    def copy(self) -> PyarrowBackend:
        """Create a new backend wrapping the same underlying Table.

        Returns
        -------
        PyarrowBackend
            New backend instance wrapping the same `pa.Table`. This is a
            zero-copy operation: `pa.Table` is immutable so no data is
            duplicated. Matches the `DataBackend.copy` contract because no
            mutation is possible through the returned backend.
        """
        self._ensure_not_streaming("copy")
        return PyarrowBackend(self._df, streaming=False)

    def apply_fn(
        self,
        fn: Callable,  # noqa ANN401
        **fn_kwargs: Any,
    ) -> PyarrowBackend:
        """Apply a custom function to the underlying Table.

        Parameters
        ----------
        fn : Callable
            Function to apply. Should accept a `pa.Table` as its first argument
            and return a `pa.Table`.
        **fn_kwargs : Any
            Additional keyword arguments to pass to the function

        Returns
        -------
        PyarrowBackend
            New backend wrapping the result of the function application
        """
        self._ensure_not_streaming("apply_fn")
        result = fn(self._df, **fn_kwargs)
        return PyarrowBackend(result, streaming=False)

    def multilabel_from_features(
        self,
        input_features: list[str],
        output_feature: str,
        label_map: dict[Any, int] | None = None,
        allow_missing_labels: bool = False,
    ) -> tuple[PyarrowBackend, dict[Any, int]]:
        """Create a multi-label column by combining multiple input feature columns.
        Each row in the output column will contain a sorted list of integer IDs
        corresponding to the labels found in the specified input feature columns.

        Parameters
        ----------
        input_features : list[str]
            List of column names to use as sources for labels. Each column can
            contain single values or lists of values.
        output_feature : str
            Name of the output column to store the generated label lists.
        label_map : dict[Any, int] | None, optional
            Mapping of unique label values to integer IDs. If None, a mapping
            will be generated from the unique values in the input features.
        allow_missing_labels : bool, optional
            If True, rows with no labels will be included in the output.
            If False, rows with no labels will be dropped. Default is False.

        Returns
        -------
        tuple[PyarrowBackend, dict]
            A tuple containing:
            - New PyarrowBackend instance with the added multi-label column
            - The label_map used for mapping labels to IDs

        Raises
        ------
        ValueError
            If any input feature does not exist or is not of type List.
        """
        self._ensure_not_streaming("multilabel_from_features")
        df = self._df

        for f in input_features:
            if f not in df.column_names:
                raise ValueError(f"Input feature '{f}' does not exist in DataFrame.")
            col_type = df.schema.field(f).type
            if not pa.types.is_list(col_type):
                # Wrap scalar column values as single-element lists
                col = df.column(f)
                new_col = pa.array(
                    [[v] if v is not None else [] for v in col.to_pylist()],
                    type=pa.list_(col_type),
                )
                idx = df.column_names.index(f)
                df = df.set_column(idx, f, new_col)

        if label_map is None:
            uniques: set = set()
            for f in input_features:
                for val in df.column(f).to_pylist():
                    if val is not None:
                        uniques.update(v for v in val if v is not None)
            label_map = {lbl: idx for idx, lbl in enumerate(sorted(uniques))}

        n_rows = len(df)
        label_lists = []
        for i in range(n_rows):
            labels: set[int] = set()
            for f in input_features:
                val = df.column(f)[i].as_py()
                if val is None:
                    continue
                for v in val:
                    if v is not None and v in label_map:
                        labels.add(label_map[v])
            label_lists.append(sorted(labels))

        new_col = pa.array(label_lists, type=pa.list_(pa.int64()))
        new_df = df.append_column(output_feature, new_col)

        if not allow_missing_labels:
            mask = pc.greater(pc.list_value_length(new_df.column(output_feature)), 0)
            new_df = new_df.filter(mask)

        return PyarrowBackend(new_df, streaming=False), label_map

    def __repr__(self) -> str:
        """Return string representation of the backend.

        Returns
        -------
        str
            String representation showing backend type and Table shape
        """
        if self._streaming:
            return f"PyarrowBackend(streaming=True, chunk_size={self._streaming_chunk_size})"
        return f"PyarrowBackend(shape={self._df.shape})"
