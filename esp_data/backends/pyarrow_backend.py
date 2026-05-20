"""Pyarrow implementation of the DataFrameBackend protocol"""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Any, Callable, Iterator, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pa_fs
import pyarrow.parquet as pq
from pyarrow import csv as pa_csv
from pyarrow import json as pa_json

from .protocol import DataBackend

logger = logging.getLogger("esp_data")


class PyarrowBackend(DataBackend):
    """Pyarrow implementation of the DataBackend protocol.

    This backend wraps a pyarrow Table and provides a unified interface
    for DataBackend operations that can work across different backend implementations.

    Supports only Table mode currently.

    Parameters
    ----------
    df : pa.Table
        The pyarrow Table to wrap
    streaming : bool
        Whether the backend is in streaming mode (Not implemented yet)
    streaming_chunk_size: int
        Number of rows per batch when iterating in streaming mode (default: 1000)
        1000 is a good number because its high enough to reduce I/O and any higher
        doesn't help because the main latency source in Dataset __getitem__ calls
        are in loading audio anyway.
    """

    def __init__(
        self,
        df: pa.Table,
        *,
        streaming: bool = False,
        streaming_chunk_size: int = 1000,
    ) -> None:
        """Initialize the backend with a pyarrow Table.

        Parameters
        ----------
        df : pa.Table
            The Table to wrap
        streaming:
            Whether to use streaming mode, by default False
        streaming_chunk_size : int, optional
            Number of rows per batch when iterating in streaming mode, by default 1000
        """
        self._df = df
        self._streaming = streaming
        self._streaming_chunk_size = streaming_chunk_size

    @classmethod
    def from_csv(cls, path: str, *, streaming: bool = False, **kwargs: Any) -> "PyarrowBackend":
        """Read a CSV file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the CSV file (supports local and cloud paths via cloudpathlib)
        streaming : bool, optional
            If True, use streaming mode, by default False
        **kwargs : Any
            Additional pyarrow-specific arguments

        Returns
        -------
        PyarrowBackend
            Backend instance wrapping the loaded Table
        """
        # Filter out kwargs for any non-pyarrow argument
        valid_params = set(inspect.signature(pa_csv.read_csv).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        path_str = str(path)
        if path_str.startswith("gs://"):
            gcs = pa_fs.GcsFileSystem()
            bucket_and_key = path_str[len("gs://") :]
            with gcs.open_input_stream(bucket_and_key) as f:
                df = pa_csv.read_csv(f, **filtered_kwargs)
        else:
            df = pa_csv.read_csv(path_str, **filtered_kwargs)
        return cls(df, streaming=False)

    @classmethod
    def from_json(cls, path: str, *, streaming: bool = False, **kwargs: Any) -> "PyarrowBackend":
        """Read a JSON file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the JSON file (supports local and cloud paths)
        streaming : bool, optional
            If True, use streaming mode, by default False
        **kwargs : Any
            Additional pyarrow-specific arguments passed to `pyarrow.json.read_json`

        Returns
        -------
        PyarrowBackend
            Backend instance wrapping the loaded Table
        """
        valid_params = set(inspect.signature(pa_json.read_json).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        path_str = str(path)
        if path_str.startswith("gs://"):
            gcs = pa_fs.GcsFileSystem()
            bucket_and_key = path_str[len("gs://") :]
            with gcs.open_input_stream(bucket_and_key) as f:
                df = pa_json.read_json(f, **filtered_kwargs)
        else:
            df = pa_json.read_json(path_str, **filtered_kwargs)
        return cls(df, streaming=False)

    @classmethod
    def from_parquet(cls, path: str, *, streaming: bool = False, **kwargs: Any) -> "PyarrowBackend":
        """Read a Parquet file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the Parquet file (supports local and cloud paths)
        streaming : bool, optional
            If True, use streaming mode, by default False
        **kwargs : Any
            Additional pyarrow-specific arguments passed to `pyarrow.parquet.read_table`

        Returns
        -------
        PyarrowBackend
            Backend instance wrapping the loaded Table
        """
        valid_params = set(inspect.signature(pq.read_table).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        path_str = str(path)
        if path_str.startswith("gs://"):
            gcs = pa_fs.GcsFileSystem()
            bucket_and_key = path_str[len("gs://") :]
            df = pq.read_table(bucket_and_key, filesystem=gcs, **filtered_kwargs)
        else:
            df = pq.read_table(path_str, **filtered_kwargs)
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

    def __getitem__(self, key: int | list[int] | slice) -> dict[str, Any] | "PyarrowBackend":
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
        RuntimeError
            If backend is in streaming mode
        """
        if self._streaming:
            raise RuntimeError("Cannot use __getitem__ in streaming mode. Use iteration instead.")

        if isinstance(key, int):
            # Return single row as dict
            if key >= len(self._df):
                raise IndexError(f"Index {key} out of bounds for Table of length {len(self._df)}")
            row = self._df.take([key]).to_pydict()
            # Convert values from list to any
            for key, value in row.items():
                row[key] = value[0]
            return row
        elif isinstance(key, list):
            return PyarrowBackend(self._df.take(key), streaming=False)
        elif isinstance(key, slice):
            if key.start is not None:
                offset = key.start
            else:
                offset = 0
            if key.stop is not None:
                length = key.stop - offset
            else:
                length = None
            return PyarrowBackend(self._df.slice(offset=offset, length=length))
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
        df = self._ensure_collected()
        return len(df)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch in self._df.to_batches(
            max_chunksize=self._streaming_chunk_size if self._streaming else None
        ):
            for row in batch.to_pylist():
                yield row

    def filter_isin(
        self, column: str, values: list[Any], *, negate: bool = False
    ) -> "PyarrowBackend":
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
        expr = pc.field(column).isin(values)
        if negate:
            expr = ~expr
        filtered_tb = self._df.filter(expr)
        return PyarrowBackend(filtered_tb, streaming=self._streaming)

    def drop_duplicates(
        self,
        subset: list[str] | None = None,
        *,
        keep: Literal["first", "last"] = "first",
    ) -> "PyarrowBackend":
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
        df = self._ensure_collected()
        cols = subset if subset is not None else df.column_names

        # Build row indices grouped by key columns, keep first or last
        keys: dict[tuple, int] = {}
        for i in range(len(df)):
            key = tuple(df.column(c)[i].as_py() for c in cols)
            if key not in keys or keep == "last":
                keys[key] = i

        indices = sorted(keys.values())
        return PyarrowBackend(df.take(indices), streaming=False)

    def dropna(
        self,
        subset: list[str] | None = None,
    ) -> "PyarrowBackend":
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
        if subset is not None:
            mask = pc.is_valid(self._df.column(subset[0]))
            for col in subset[1:]:
                mask = pc.and_(mask, pc.is_valid(self._df.column(col)))
            return PyarrowBackend(self._df.filter(mask), streaming=self._streaming)
        return PyarrowBackend(self._df.drop_null(), streaming=self._streaming)

    def get_unique(self, column: str) -> list[Any]:
        """Get sorted unique values from a column

        Parameters
        ----------
        column : str
            Column name

        Returns
        -------
        list[Any]
            Sorted list of unique values (nulls excluded)
        """
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
        """
        df = self._ensure_collected()
        # Drop nulls and group by column to get counts
        counts_table = (
            df.select([column]).drop_null().group_by(column).aggregate([([], "count_all")])
        )
        # Convert to dictionary
        counts_dict = counts_table.to_pydict()
        return dict(zip(counts_dict[column], counts_dict["count_all"], strict=True))

    def map_column(
        self,
        column: str,
        mapping: dict[Any, Any],
        output_column: str,
        *,
        default: Any | None = None,  # noqa ANN401
    ) -> "PyarrowBackend":
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
        df = self._ensure_collected()
        source = df.column(column).to_pylist()
        mapped = [mapping.get(v, default) for v in source]
        new_col = pa.array(mapped)
        new_df = df.append_column(output_column, new_col)
        return PyarrowBackend(new_df)

    def rename_columns(
        self,
        mapping: dict[str, str],
    ) -> "PyarrowBackend":
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
        new_df = self._df.rename_columns(mapping)
        return PyarrowBackend(new_df)

    def add_column(
        self,
        column: str,
        values: Any,  # noqa ANN401
    ) -> "PyarrowBackend":
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
        new_df = self._df.append_column(column, [values])
        return PyarrowBackend(new_df)

    def select_columns(
        self,
        columns: list[str],
    ) -> "PyarrowBackend":
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
        return PyarrowBackend(self._df.select(columns))

    @classmethod
    def concat(
        cls,
        backends: list["PyarrowBackend"],
        *,
        ignore_index: bool = True,
        sort: bool = False,
    ) -> "PyarrowBackend":
        """Concatenate multiple backend instances vertically (row-wise).

        Parameters
        ----------
        backends : list[PyarrowBackend]
            List of backend instances to concatenate
        sort : bool, optional
            If True, sort columns alphabetically, by default False

        Returns
        -------
        PyarrowBackend
            New backend with concatenated data
        """
        dfs = [backend._df for backend in backends]

        concatenated_df = pa.concat_tables(dfs)

        if sort:
            # Sort columns alphabetically
            sorted_cols = sorted(concatenated_df.column_names)
            concatenated_df = concatenated_df.select(sorted_cols)

        return cls(concatenated_df)

    def _ensure_collected(self) -> pa.Table:
        """Ensure the Table is collected (not lazy).

        Returns
        -------
        pa.Table
            The underlying Table
        """
        return self._df

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
                f"LazyFrame operations require explicit collection. "
                f"Consider using .collect() or iterate over the data."
            )

    @property
    def columns(self) -> list[str]:
        """Get the list of column names.

        Returns
        -------
        list[str]
            List of column names
        """
        if self._streaming:
            return self._df.collect_schema().names()
        else:
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
        return column in self._df.column_names

    @property
    def unwrap(self) -> pa.Table:
        """Get the underlying Table object.

        Returns
        -------
        pa.Table
            The underlying pyarrow Table
        """
        return self._df

    def _sample_by_column_helper(
        self,
        column: str,
        values_dict: dict[str, Any],
        *,
        sample_fn: Callable[[pa.Table, Any], pa.Table],
        other_sample_fn: Callable[[pa.Table, Any], pa.Table],
        dict_name: str,
    ) -> "PyarrowBackend":
        """Helper function for sampling by column values.

        Parameters
        ----------
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
        df = self._ensure_collected()
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
            other_mask = ~pc.is_in(df.column(column), pa.array(list(explicit_values)))
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
    ) -> "PyarrowBackend":
        """Subsample rows by column values with specified ratios.

        For each unique value in the column, sample the specified ratio of rows.
        Special key "other" can be used to subsample all values not explicitly listed.

        If the backend is in streaming mode, a UserWarning will be issued and the
        LazyFrame will be collected since sampling requires materialization.

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
        import numpy as np

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
    ) -> "PyarrowBackend":
        """Upsample rows by column values to target counts with replacement.

        For each unique value in the column, sample rows with replacement to reach
        the target count. If a category already has more rows than the target, it will
        be downsampled (without replacement) to the target count.

        If the backend is in streaming mode, a UserWarning will be issued and the
        LazyFrame will be collected since sampling requires materialization.

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
        import numpy as np

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
    ) -> "PyarrowBackend":
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
        import numpy as np

        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(len(self._df), size=n, replace=replace).tolist()
        return PyarrowBackend(self._df.take(indices), streaming=self._streaming)

    def copy(self) -> "PyarrowBackend":
        """Create a copy of the backend with a copied Table.

        Returns
        -------
        PolarsBackend
            New backend instance with copied Table
        """
        # Preserve streaming mode
        return PyarrowBackend(
            self._df,
            streaming=self._streaming,
        )

    def apply_fn(
        self,
        fn: Callable,  # noqa ANN401
        **fn_kwargs: Any,
    ) -> "PyarrowBackend":
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
        result = fn(self._df, **fn_kwargs)
        return PyarrowBackend(result, streaming=self._streaming)

    def multilabel_from_features(
        self,
        input_features: list[str],
        output_feature: str,
        label_map: dict[Any, int] | None = None,
        allow_missing_labels: bool = True,
    ) -> tuple["PyarrowBackend", dict[Any, int]]:
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
        label_map : dict[str, Any] | None, optional
            Mapping of unique label values to integer IDs. If None, a mapping
            will be generated from the unique values in the input features.
        allow_missing_labels : bool, optional
            If True, rows with no labels will be included in the output.
            If False, rows with no labels will be dropped. Default is True.

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
        df = self._ensure_collected()

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

        return PyarrowBackend(new_df, streaming=self._streaming), label_map

    def __repr__(self) -> str:
        """Return string representation of the backend.

        Returns
        -------
        str
            String representation showing backend type and Table shape
        """
        if self._streaming:
            return f"PyarrowBackend(streaming=True, chunk_size={self._chunk_size})"
        return f"PyarrowBackend(shape={self._df.shape})"
