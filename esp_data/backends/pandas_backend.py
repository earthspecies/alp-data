"""Pandas implementation of the DataFrameBackend protocol."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import pandas as pd

from esp_data.io import AnyPathT, anypath

from .protocol import DataBackend


class PandasBackend(DataBackend):
    """Pandas implementation of the DataFrameBackend protocol.

    This backend wraps a pandas DataFrame and provides a unified interface
    for DataFrame operations that can work across different backend implementations.

    Supports both eager (in-memory) and streaming (chunked) modes.

    Parameters
    ----------
    df : pd.DataFrame | pd.io.parsers.TextFileReader
        The pandas DataFrame to wrap, or TextFileReader for streaming
    streaming : bool
        Whether the backend is in streaming mode
    streaming_chunk_size : int
        Number of rows per chunk in streaming mode

    Examples
    --------
    >>> import pandas as pd
    >>> from esp_data.backends import PandasBackend
    >>> df = pd.DataFrame({"col1": range(100), "col2": list("abcdefghij") * 10})
    >>> backend = PandasBackend(df, streaming=False)
    >>> backend[0]  # Get first row as dict
    {'col1': 0, 'col2': 'a'}
    >>> backend[[0, 5, 10]]  # Get rows 0, 5, 10 as new backend
    PandasBackend(shape=(3, 2))
    >>> filtered = backend.filter_isin("col2", ["a", "b"])
    >>> len(filtered)  # Number of rows with col2 in ['a', 'b']
    20
    >>> backend.columns  # List of column names
    ['col1', 'col2']
    >>> backend.column_exists("col1")  # Check if column exists
    True
    >>> sub = backend.subsample_by_column("col2", {"a": 0.5, "b": 0.5, "other": 0.1})
    >>> counts = sub.unwrap["col2"].count()  # Subsampled counts
    >>> assert counts <= 20
    """

    def __init__(
        self,
        df: pd.DataFrame | pd.io.parsers.TextFileReader,
        *,
        streaming: bool = False,
        streaming_chunk_size: int = 1000,
    ) -> None:
        """Initialize the backend with a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame | pd.io.parsers.TextFileReader
            The DataFrame to wrap, or TextFileReader for streaming mode
        streaming : bool, optional
            Whether to use streaming mode, by default False
        streaming_chunk_size : int, optional
            Number of rows per chunk in streaming mode, by default 1000
        """
        self._df = df
        self._streaming = streaming
        if isinstance(self._df, pd.DataFrame) and streaming:
            self._streaming = False  # Override if df is already loaded
        self._chunk_size = streaming_chunk_size

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        streaming: bool = False,
        streaming_chunk_size: int = 1000,
        **kwargs: Any,
    ) -> "PandasBackend":
        """Read a CSV file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the CSV file (supports local and cloud paths via cloudpathlib)
        streaming : bool, optional
            If True, use streaming mode with chunked reading, by default False
        streaming_chunk_size : int, optional
            Number of rows per chunk in streaming mode, by default 1000
        **kwargs : Any
            Additional pandas-specific arguments

        Returns
        -------
        PandasBackend
            Backend instance wrapping the loaded DataFrame
        """
        # Filter out kwargs for any non-pandas argument
        valid_params = set(inspect.signature(pd.read_csv).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        if streaming:
            # Use chunksize parameter for streaming mode
            reader = pd.read_csv(
                path,
                chunksize=streaming_chunk_size,
                **filtered_kwargs,
            )
            return cls(reader, streaming=True, streaming_chunk_size=streaming_chunk_size)
        else:
            df = pd.read_csv(path, **filtered_kwargs)
            return cls(df, streaming=False)

    @classmethod
    def from_json(
        cls,
        path: str,
        *,
        lines: bool = False,
        streaming: bool = False,
        streaming_chunk_size: int = 1000,
        **kwargs: Any,
    ) -> "PandasBackend":
        """Read a JSON file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the JSON file
        lines : bool, optional
            If True, read file as JSON lines (one JSON object per line),
            by default False
        streaming : bool, optional
            If True, use streaming mode with chunked reading, by default False
        streaming_chunk_size : int, optional
            Number of rows per chunk in streaming mode, by default 1000
        **kwargs : Any
            Additional pandas-specific arguments

        Returns
        -------
        PandasBackend
            Backend instance wrapping the loaded DataFrame
        """
        # Filter out kwargs for any non-pandas argument
        valid_params = set(inspect.signature(pd.read_json).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        if streaming and lines:
            # Use chunksize for JSON lines streaming
            reader = pd.read_json(
                path,
                lines=lines,
                chunksize=streaming_chunk_size,
                **filtered_kwargs,
            )
            return cls(reader, streaming=True, streaming_chunk_size=streaming_chunk_size)
        else:
            df = pd.read_json(path, lines=lines, **filtered_kwargs)
            return cls(df, streaming=False)

    @classmethod
    def from_parquet(cls, path: str, *, streaming: bool = False, **kwargs: Any) -> "PandasBackend":
        """Read a Parquet file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the Parquet file
        streaming : bool, optional
            If True, use streaming mode (not supported for parquet in pandas),
            by default False
        **kwargs : Any
            Additional pandas-specific arguments

        Returns
        -------
        PandasBackend
            Backend instance wrapping the loaded DataFrame

        Note
        ----
        Pandas does not natively support streaming parquet files.
        Consider using polars backend for large parquet files.
        """
        # Filter out kwargs for any non-pandas argument
        valid_params = set(inspect.signature(pd.read_parquet).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        if streaming:
            raise NotImplementedError(
                "Streaming mode is not supported for parquet files with pandas backend."
                "Consider using PolarsBackend for large parquet files."
            )
        df = pd.read_parquet(path, **filtered_kwargs)
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

    def __getitem__(self, key: int | list[int] | slice) -> dict[str, Any] | "PandasBackend":
        """Get row(s) from the DataFrame using Pythonic indexing.

        Parameters
        ----------
        key : int | list[int] | slice
            - int: Get single row as dict
            - list[int]: Get multiple rows as new backend
            - slice: Get row range as new backend

        Returns
        -------
        dict[str, Any] | PandasBackend
            - dict if key is int (single row)
            - PandasBackend if key is list or slice (multiple rows)

        Raises
        ------
        IndexError
            If index is out of bounds
        TypeError
            If key type is not supported
        RuntimeError
            If backend is in streaming mode

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"col1": range(100), "col2": list("abcdefghij") * 10})
        >>> from esp_data.backends import PandasBackend
        >>> backend = PandasBackend(df)
        >>> backend[0]  # Get first row as dict
        {'col1': 0, 'col2': 'a'}
        >>> backend[[0, 5, 10]]  # Get rows 0, 5, 10 as new backend
        PandasBackend(shape=(3, 2))
        >>> backend[5:]  # Get rows from index 5 to end
        PandasBackend(shape=(95, 2))
        >>> backend[:10]  # Get first 10 rows
        PandasBackend(shape=(10, 2))
        """
        if self._streaming:
            raise RuntimeError(
                "Cannot use __getitem__ in streaming mode. "
                "Use iteration instead: for row in backend: ..."
            )

        if isinstance(key, int):
            # Return single row as dict
            if key >= len(self._df):
                raise IndexError(
                    f"Index {key} out of bounds for DataFrame of length {len(self._df)}"
                )
            return self._df.iloc[key].to_dict()
        elif isinstance(key, (list, slice)):
            # Return multiple rows as new backend (not streaming)
            # .iloc handles both list and slice
            return PandasBackend(self._df.iloc[key], streaming=False)
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")

    def __len__(self) -> int:
        """Get the number of rows in the DataFrame.

        Returns
        -------
        int
            Number of rows

        Raises
        ------
        RuntimeError
            If backend is in streaming mode (length unknown until consumed)
        """
        if self._streaming:
            raise RuntimeError(
                "Cannot get length in streaming mode. "
                "Length is unknown until the stream is consumed."
            )
        return len(self._df)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over DataFrame rows as dictionaries.

        In streaming mode, yields rows from chunks as they are read.
        In eager mode, yields rows from the loaded DataFrame.

        Yields
        ------
        dict[str, Any]
            Dictionary for each row mapping column names to values
        """
        if self._streaming:
            # Streaming mode: iterate through chunks from reader
            for chunk in self._df:
                for _, row in chunk.iterrows():
                    yield row.to_dict()
        else:
            # Eager mode: iterate through loaded DataFrame
            for idx in range(len(self._df)):
                yield self._df.iloc[idx].to_dict()

    def _ensure_not_streaming(self, operation: str) -> None:
        """Raise error if in streaming mode for operations that require
        eager evaluation.

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
                f"Operations that modify the DataFrame require eager evaluation. "
                f"Consider loading data eagerly or use iteration for processing."
            )

    def filter_isin(
        self,
        column: str,
        values: list[Any],
        *,
        negate: bool = False,
    ) -> "PandasBackend":
        """Filter DataFrame rows where column values are in (or not in) a list.

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
        PandasBackend
            New backend with filtered DataFrame
        """
        self._ensure_not_streaming("filter_isin")
        mask = self._df[column].isin(values)
        if negate:
            mask = ~mask
        filtered_df = self._df[mask].reset_index(drop=True)
        return PandasBackend(filtered_df, streaming=False)

    def drop_duplicates(
        self,
        subset: list[str] | None = None,
        *,
        keep: Literal["first", "last"] = "first",
    ) -> "PandasBackend":
        """Remove duplicate rows from the DataFrame.

        Parameters
        ----------
        subset : list[str] | None, optional
            Column names to consider for identifying duplicates.
            If None, use all columns, by default None
        keep : Literal["first", "last"], optional
            Which duplicate to keep, by default "first"

        Returns
        -------
        PandasBackend
            New backend with duplicates removed
        """
        self._ensure_not_streaming("drop_duplicates")
        deduped_df = self._df.drop_duplicates(subset=subset, keep=keep, ignore_index=True)
        return PandasBackend(deduped_df, streaming=False)

    def dropna(
        self,
        subset: list[str] | None = None,
    ) -> "PandasBackend":
        """Remove rows with missing values.

        Parameters
        ----------
        subset : list[str] | None, optional
            Column names to consider for null detection.
            If None, check all columns, by default None

        Returns
        -------
        PandasBackend
            New backend with null rows removed
        """
        self._ensure_not_streaming("dropna")
        cleaned_df = self._df.dropna(subset=subset, ignore_index=True)
        return PandasBackend(cleaned_df, streaming=False)

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
        """
        self._ensure_not_streaming("get_unique")
        return sorted(self._df[column].dropna().unique().tolist())

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
        self._ensure_not_streaming("histogram")
        return self._df[column].value_counts().to_dict()

    def map_column(
        self,
        column: str,
        mapping: dict[Any, Any],
        output_column: str,
    ) -> "PandasBackend":
        """Create a new column by mapping values from an existing column.

        Parameters
        ----------
        column : str
            Source column name
        mapping : dict[Any, Any]
            Dictionary mapping source values to output values
        output_column : str
            Name of the new column to create

        Returns
        -------
        PandasBackend
            New backend with mapped column added
        """
        self._ensure_not_streaming("map_column")
        # Use assign to create new column (immutable operation)
        new_df = self._df.assign(**{output_column: self._df[column].map(mapping)})
        return PandasBackend(new_df, streaming=False)

    def rename_columns(
        self,
        mapping: dict[str, str],
    ) -> "PandasBackend":
        """Rename DataFrame columns.

        Parameters
        ----------
        mapping : dict[str, str]
            Dictionary mapping old column names to new names

        Returns
        -------
        PandasBackend
            New backend with renamed columns
        """
        self._ensure_not_streaming("rename_columns")
        renamed_df = self._df.rename(columns=mapping)
        return PandasBackend(renamed_df, streaming=False)

    def add_column(
        self,
        column: str,
        values: pd.Series | list[Any] | Any,  # noqa ANN401
    ) -> "PandasBackend":
        """Add a new column to the DataFrame.

        Parameters
        ----------
        column : str
            Name of the new column
        values : Any
            Values for the new column (scalar or array-like)

        Returns
        -------
        PandasBackend
            New backend with new column added
        """
        self._ensure_not_streaming("add_column")
        new_df = self._df.assign(**{column: values})
        return PandasBackend(new_df, streaming=False)

    def select_columns(
        self,
        columns: list[str],
    ) -> "PandasBackend":
        """Select a subset of columns from the DataFrame.

        Parameters
        ----------
        columns : list[str]
            List of column names to keep

        Returns
        -------
        PandasBackend
            New backend with only specified columns
        """
        self._ensure_not_streaming("select_columns")
        selected_df = self._df[columns]
        return PandasBackend(selected_df, streaming=False)

    @classmethod
    def concat(
        cls,
        backends: list["PandasBackend"],
        *,
        ignore_index: bool = True,
        sort: bool = False,
    ) -> "PandasBackend":
        """Concatenate multiple backend instances vertically (row-wise).

        Parameters
        ----------
        backends : list[PandasBackend]
            List of backend instances to concatenate
        ignore_index : bool, optional
            If True, reset index in result, by default True
        sort : bool, optional
            If True, sort columns alphabetically, by default False

        Returns
        -------
        PandasBackend
            New backend with concatenated data
        """
        dfs = [backend._df for backend in backends]
        concatenated_df = pd.concat(dfs, ignore_index=ignore_index, sort=sort)
        return cls(concatenated_df)

    @property
    def columns(self) -> list[str]:
        """Get the list of column names.

        Returns
        -------
        list[str]
            List of column names
        """
        return list(self._df.columns)

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
        return column in self._df.columns

    @property
    def unwrap(self) -> pd.DataFrame:
        """Get the underlying DataFrame object.

        Returns
        -------
        pd.DataFrame
            The underlying pandas DataFrame
        """
        return self._df

    def _sample_by_column_helper(
        self,
        column: str,
        values_dict: dict[str, Any],
        *,
        sample_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
        other_sample_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
        dict_name: str,
    ) -> "PandasBackend":
        """Helper function for sampling by column values.

        Parameters
        ----------
        column : str
            Column name to group by
        values_dict : dict[str, Any]
            Dictionary mapping column values to sampling parameters
        sample_fn : Callable[[pd.DataFrame, Any], pd.DataFrame]
            Function to sample a group given (group_df, value_from_dict).
            The function should handle seed internally via closure.
        other_sample_fn : Callable[[pd.DataFrame, Any], pd.DataFrame]
            Function to sample the "other" group given (other_df, other_value).
            The function should handle seed internally via closure.
        dict_name : str
            Name of the dictionary for error messages (e.g., "ratios", "target_counts")

        Returns
        -------
        PandasBackend
            New backend with sampled rows
        """
        groups = []

        # Use groupby for better performance and type handling
        grouped = self._df.groupby(column, group_keys=False, dropna=False)

        # Handle explicitly listed values
        explicit_values = set(values_dict.keys()) - {"other"}
        for val, param in values_dict.items():
            if val == "other":
                continue

            # Get group for this value
            try:
                group_df = grouped.get_group(val)
            except KeyError:
                # Category not present in data - warn user as this might indicate
                # a typo or type mismatch
                warnings.warn(
                    f"Key {val!r} in {dict_name} not found in column '{column}'. "
                    "This may indicate a typo or type mismatch "
                    "(e.g., string key for int column). Skipping this key.",
                    UserWarning,
                    stacklevel=3,
                )
                continue

            if len(group_df) == 0:
                continue

            # Sample using the provided function
            chosen_df = sample_fn(group_df, param)

            if len(chosen_df) > 0:
                groups.append(chosen_df)

        # Handle "other" category (pooled unlisted values)
        if "other" in values_dict:
            # Get all values not explicitly in values_dict (excluding "other" itself)
            # This includes NaN values if present
            mask_other = ~self._df[column].isin(explicit_values)
            other_df = self._df[mask_other]

            other_param = values_dict["other"]
            if len(other_df) == 0:
                chosen_other_df = other_df.iloc[0:0]
            else:
                chosen_other_df = other_sample_fn(other_df, other_param)

            if len(chosen_other_df) > 0:
                groups.append(chosen_other_df)

        # Concatenate all groups
        if groups:
            result_df = pd.concat(groups, ignore_index=True)
        else:
            result_df = self._df.iloc[0:0]  # Empty dataframe with same schema

        return PandasBackend(result_df, streaming=False)

    def subsample_by_column(
        self,
        column: str,
        ratios: dict[str, float],
        *,
        seed: int = 42,
    ) -> "PandasBackend":
        """Subsample rows by column values with specified ratios.

        For each unique value in the column, sample the specified ratio of rows.
        Special key "other" can be used to subsample all values not explicitly listed.

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
        PandasBackend
            New backend with subsampled rows

        Raises
        ------
        KeyError
            If the specified column does not exist in the DataFrame
        ValueError
            If any ratio is negative or greater than 1.0
        """
        self._ensure_not_streaming("subsample_by_column")

        # Validate column exists
        if column not in self._df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame columns.")

        # Validate ratios (including "other")
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

        import numpy as np

        rng = np.random.default_rng(seed=seed)

        def sample_by_ratio(group_df: pd.DataFrame, ratio: float) -> pd.DataFrame:
            """Sample group by ratio.

            Parameters
            ----------
            group_df : pd.DataFrame
                Group DataFrame to sample from
            ratio : float
                Sampling ratio (0.0 to 1.0)

            Returns
            -------
            pd.DataFrame
                Sampled DataFrame
            """
            if ratio >= 1.0:
                return group_df
            n = max(0, int(len(group_df) * ratio))
            if n == 0:
                return group_df.iloc[0:0]
            # Use NumPy RNG for consistency with upsampling
            indices = group_df.index.tolist()
            chosen_indices = rng.choice(indices, size=n, replace=False).tolist()
            return group_df.loc[chosen_indices]

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
    ) -> "PandasBackend":
        """Upsample rows by column values to target counts with replacement.

        For each unique value in the column, sample rows with replacement to reach
        the target count. If a category already has more rows than the target, it will
        be downsampled (without replacement) to the target count.

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
        PandasBackend
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
        self._ensure_not_streaming("upsample_by_column")

        # Validate column exists
        if column not in self._df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame columns.")

        # Validate target counts (including "other")
        for val, target_count in target_counts.items():
            if target_count < 0:
                raise ValueError(
                    f"Target count for value {val!r} is negative: {target_count}. "
                    "Target counts must be >= 0"
                )
            if not isinstance(target_count, int):
                raise TypeError(
                    f"Target count for value {val!r} must be an integer, "
                    f"got {type(target_count).__name__}"
                )

        import numpy as np

        rng = np.random.default_rng(seed=seed)

        def sample_by_target_count(group_df: pd.DataFrame, target_count: int) -> pd.DataFrame:
            """Sample group to target count (with upsampling support).

            Parameters
            ----------
            group_df : pd.DataFrame
                Group DataFrame to sample from
            target_count : int
                Target number of samples

            Returns
            -------
            pd.DataFrame
                Sampled DataFrame
            """
            if target_count == 0:
                return group_df.iloc[0:0]
            indices = group_df.index.tolist()
            if target_count <= len(group_df):
                # Downsample: sample without replacement (explicit replace=False)
                chosen_indices = rng.choice(indices, size=target_count, replace=False).tolist()
            else:
                # Upsample: sample with replacement
                chosen_indices = rng.choice(indices, size=target_count, replace=True).tolist()
            return group_df.loc[chosen_indices]

        def sample_other_by_target_count(other_df: pd.DataFrame, target_count: int) -> pd.DataFrame:
            """Sample 'other' group to target count.

            Parameters
            ----------
            other_df : pd.DataFrame
                Other group DataFrame to sample from
            target_count : int
                Target number of samples (already validated)

            Returns
            -------
            pd.DataFrame
                Sampled DataFrame
            """
            # Validation is done in the outer loop, so we can just call the sampling function
            return sample_by_target_count(other_df, target_count)

        return self._sample_by_column_helper(
            column=column,
            values_dict=target_counts,
            sample_fn=sample_by_target_count,
            other_sample_fn=sample_other_by_target_count,
            dict_name="target_counts",
        )

    def sample_rows(
        self,
        n: int,
        *,
        seed: int = 42,
        replace: bool = False,
    ) -> "PandasBackend":
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
        PandasBackend
            New backend with sampled rows
        """
        self._ensure_not_streaming("sample_rows")
        sampled_df = self._df.sample(n=n, random_state=seed, replace=replace, ignore_index=True)
        return PandasBackend(sampled_df, streaming=False)

    def copy(self) -> "PandasBackend":
        """Create a copy of the backend with a copied DataFrame.

        Returns
        -------
        PandasBackend
            New backend instance with copied DataFrame
        """
        self._ensure_not_streaming("copy")
        return PandasBackend(self._df.copy(), streaming=False)

    def _write_path(self, path: str) -> Path | AnyPathT:
        """Resolve a destination path string to a local or cloud path object.

        Parameters
        ----------
        path : str
            Destination path (local or ``gs://``/``s3://``).

        Returns
        -------
        Path | AnyPathT
            Resolved path object.
        """
        return anypath(path)

    def to_csv(self, path: str) -> None:
        """Write the DataFrame to a CSV file.

        Parameters
        ----------
        path : str
            Destination path (local or cloud, e.g. ``gs://bucket/file.csv``).
        """
        self._ensure_not_streaming("to_csv")
        dest = self._write_path(path)
        if isinstance(dest, Path):
            self._df.to_csv(str(dest), index=False)
        else:
            with dest.open("w") as f:
                self._df.to_csv(f, index=False)

    def to_jsonl(self, path: str) -> None:
        """Write the DataFrame to a JSON-Lines (ndjson) file.

        Parameters
        ----------
        path : str
            Destination path (local or cloud, e.g. ``gs://bucket/file.jsonl``).
        """
        self._ensure_not_streaming("to_jsonl")
        dest = self._write_path(path)
        if isinstance(dest, Path):
            self._df.to_json(str(dest), orient="records", lines=True)
        else:
            with dest.open("w") as f:
                self._df.to_json(f, orient="records", lines=True)

    def apply_fn(
        self,
        fn: Callable,
        fn_kwargs: dict[str, Any],
        apply_kwargs: dict[str, Any],
    ) -> "PandasBackend":
        """Apply a function to the DataFrame.

        Parameters
        ----------
        fn : Callable
            Function to apply to the DataFrame. Should accept a DataFrame
            as the first argument and return a modified DataFrame.
        apply_kwargs : dict
            Additional keyword arguments to pass to pandas.DataFrame.apply()
            For e.g. engine="numba"
        fn_kwargs : Any
            Additional keyword arguments to pass to the function

        Returns
        -------
        PandasBackend
            New backend with modified DataFrame

        Raises
        ------
        ValueError
            If the function does not return a pandas DataFrame
        """
        self._ensure_not_streaming("apply_fn")
        from functools import partial

        fn = partial(fn, **fn_kwargs)
        new_df = self._df.apply(fn, **apply_kwargs)
        if not isinstance(new_df, pd.DataFrame):
            raise ValueError("Function must return a pandas DataFrame.")
        return PandasBackend(new_df, streaming=False)

    def multilabel_from_features(
        self,
        input_features: list[str],
        output_feature: str,
        label_map: dict[str, Any] | None = None,
        allow_missing_labels: bool = False,
    ) -> tuple["PandasBackend", dict]:
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
            If False, rows with no labels will be dropped. Default is False.

        Returns
        -------
        tuple[PandasBackend, dict]
            A tuple containing:
            - New PandasBackend instance with the added multi-label column
            - The label_map used for mapping labels to IDs
        """
        self._ensure_not_streaming("multilabel_from_features")

        # Generate label map if not provided
        if label_map is None:
            uniques = set()
            for f in input_features:
                # explode turns empty lists into NaNs hence the dropna()
                uniques |= set(self._df[f].explode().dropna().unique())
            label_map = {lbl: idx for idx, lbl in enumerate(sorted(uniques))}

        def _row_to_ids(row: pd.Series) -> list | None:
            row_labels = []
            for f in input_features:
                if isinstance(row[f], list):
                    v = row[f]
                elif pd.isna(row[f]):
                    continue
                else:
                    v = [row[f]]
                row_labels.extend(map(lambda x: label_map[x], v))
            if not allow_missing_labels and len(row_labels) == 0:
                return None
            return sorted(row_labels)

        self._df[output_feature] = self._df[input_features].apply(_row_to_ids, axis="columns")
        df_clean = self._df.dropna(subset=output_feature, ignore_index=True)

        return PandasBackend(df_clean, streaming=False), label_map

    def __repr__(self) -> str:
        """Return string representation of the backend.

        Returns
        -------
        str
            String representation showing backend type and DataFrame shape
        """
        if self._streaming:
            return f"PandasBackend(streaming=True, chunk_size={self._chunk_size})"
        return f"PandasBackend(shape={self._df.shape})"
