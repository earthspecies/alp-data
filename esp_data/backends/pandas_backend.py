"""Pandas implementation of the DataFrameBackend protocol."""

from __future__ import annotations

from typing import Any, Iterator, Literal

import pandas as pd


class PandasBackend:
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
    >>> backend = PandasBackend.read_csv("data.csv")
    >>> row = backend[0]
    >>> filtered = backend.filter_isin("species", ["cat", "dog"])
    >>> # Streaming mode
    >>> backend = PandasBackend.read_csv("large.csv", streaming=True)
    >>> for row in backend:
    ...     process(row)
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
        self._chunk_size = streaming_chunk_size

    @classmethod
    def read_csv(
        cls,
        path: str,
        *,
        keep_default_na: bool = True,
        na_values: list[str] | None = None,
        streaming: bool = False,
        streaming_chunk_size: int = 1000,
        **kwargs: Any,  # noqa ANN401
    ) -> "PandasBackend":
        """Read a CSV file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the CSV file (supports local and cloud paths via cloudpathlib)
        keep_default_na : bool, optional
            Whether to include default NA values, by default True
        na_values : list[str] | None, optional
            Additional strings to recognize as NA/NaN, by default None
        streaming : bool, optional
            If True, use streaming mode with chunked reading, by default False
        streaming_chunk_size : int, optional
            Number of rows per chunk in streaming mode, by default 1000
        **kwargs : d
            Additional pandas-specific arguments

        Returns
        -------
        PandasBackend
            Backend instance wrapping the loaded DataFrame
        """
        if streaming:
            # Use chunksize parameter for streaming mode
            reader = pd.read_csv(
                path,
                keep_default_na=keep_default_na,
                na_values=na_values,
                chunksize=streaming_chunk_size,
                **kwargs,
            )
            return cls(reader, streaming=True, streaming_chunk_size=streaming_chunk_size)
        else:
            df = pd.read_csv(path, keep_default_na=keep_default_na, na_values=na_values, **kwargs)
            return cls(df, streaming=False)

    @classmethod
    def read_json(
        cls,
        path: str,
        *,
        lines: bool = False,
        orient: str = "records",
        streaming: bool = False,
        streaming_chunk_size: int = 1000,
        **kwargs: Any,  # noqa ANN401
    ) -> "PandasBackend":
        """Read a JSON file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the JSON file
        lines : bool, optional
            If True, read file as JSON lines (one JSON object per line), by default False
        orient : str, optional
            Expected JSON format, by default "records"
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
        if streaming and lines:
            # Use chunksize for JSON lines streaming
            reader = pd.read_json(
                path,
                lines=lines,
                orient=orient,
                chunksize=streaming_chunk_size,
                **kwargs,
            )
            return cls(reader, streaming=True, streaming_chunk_size=streaming_chunk_size)
        else:
            df = pd.read_json(path, lines=lines, orient=orient, **kwargs)
            return cls(df, streaming=False)

    @classmethod
    def read_parquet(
        cls,
        path: str,
        *,
        streaming: bool = False,
        **kwargs: Any,  # noqa ANN401
    ) -> "PandasBackend":
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
        if streaming:
            raise NotImplementedError(
                "Streaming mode is not supported for parquet files with pandas backend."
                "Consider using PolarsBackend for large parquet files."
            )
        df = pd.read_parquet(path, **kwargs)
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
                for idx in range(len(chunk)):
                    yield chunk.iloc[idx].to_dict()
        else:
            # Eager mode: iterate through loaded DataFrame
            for idx in range(len(self._df)):
                yield self._df.iloc[idx].to_dict()

    def iter_batches(self, batch_size: int = 1000) -> Iterator["PandasBackend"]:
        """Iterate over DataFrame in batches.

        Parameters
        ----------
        batch_size : int, optional
            Number of rows per batch, by default 1000
            Ignored in streaming mode (uses streaming_chunk_size from initialization)

        Yields
        ------
        PandasBackend
            Backend instances wrapping batches of up to batch_size rows
        """
        if self._streaming:
            # Streaming mode: yield chunks directly from reader
            for chunk in self._df:
                yield PandasBackend(chunk, streaming=False)
        else:
            # Eager mode: batch the loaded DataFrame
            for start_idx in range(0, len(self._df), batch_size):
                end_idx = min(start_idx + batch_size, len(self._df))
                yield PandasBackend(self._df.iloc[start_idx:end_idx], streaming=False)

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
        """Get unique values from a column.

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
        default : Any, optional
            Value to use for unmapped keys, by default -1

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

    # ==================== Concatenation ====================

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

    # ==================== Metadata Operations ====================

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

        Parameters
        ----------
        column : str
            Column name to group by
        ratios : dict[str, float]
            Dictionary mapping column values to sampling ratios (0.0 to 1.0).
            Special key "other" applies to all unlisted values.
        seed : int, optional
            Random seed for reproducibility, by default 42

        Returns
        -------
        PandasBackend
            New backend with subsampled rows
        """
        self._ensure_not_streaming("subsample_by_column")

        import numpy as np

        rng = np.random.default_rng(seed=seed)
        groups = []

        # Handle explicitly listed values
        for val, ratio in ratios.items():
            if val == "other":
                continue

            # Get indices for this value
            mask = self._df[column] == val
            indices = self._df.index[mask].tolist()

            # Sample if ratio < 1.0 and we have rows
            if ratio >= 1.0 or len(indices) == 0:
                chosen = indices
            else:
                n = int(len(indices) * ratio)
                chosen = rng.choice(indices, size=n, replace=False).tolist()

            if chosen:
                groups.append(self._df.loc[chosen])

        # Handle "other" category
        if "other" in ratios:
            # Get all values not explicitly in ratios (excluding "other" itself)
            explicit_values = set(ratios.keys()) - {"other"}
            mask_other = ~self._df[column].isin(explicit_values)
            indices_other = self._df.index[mask_other].tolist()

            ratio = ratios["other"]
            if ratio >= 1.0 or len(indices_other) == 0:
                chosen_other = indices_other
            else:
                n = int(len(indices_other) * ratio)
                chosen_other = rng.choice(indices_other, size=n, replace=False).tolist()

            if chosen_other:
                groups.append(self._df.loc[chosen_other])

        # Concatenate all groups
        if groups:
            result_df = pd.concat(groups, ignore_index=True)
        else:
            result_df = self._df.iloc[0:0]  # Empty dataframe with same schema

        return PandasBackend(result_df, streaming=False)

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
