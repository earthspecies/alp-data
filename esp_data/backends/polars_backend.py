"""Polars implementation of the DataFrameBackend protocol."""

from __future__ import annotations

from typing import Any, Iterator, Literal

import polars as pl


class PolarsBackend:
    """Polars implementation of the DataFrameBackend protocol.

    This backend wraps a polars DataFrame or LazyFrame and provides a unified interface
    for DataFrame operations that can work across different backend implementations.

    Supports both eager (DataFrame) and streaming (LazyFrame) modes.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The polars DataFrame or LazyFrame to wrap
    streaming : bool
        Whether the backend is in streaming mode (LazyFrame)

    Examples
    --------
    >>> backend = PolarsBackend.read_csv("data.csv")
    >>> row = backend[0]
    >>> filtered = backend.filter_isin("species", ["cat", "dog"])
    >>> # Streaming mode with LazyFrame
    >>> backend = PolarsBackend.read_csv("large.csv", streaming=True)
    >>> for row in backend:
    ...     process(row)
    """

    def __init__(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        *,
        streaming: bool = False,
    ) -> None:
        """Initialize the backend with a polars DataFrame or LazyFrame.

        Parameters
        ----------
        df : pl.DataFrame | pl.LazyFrame
            The DataFrame or LazyFrame to wrap
        streaming : bool, optional
            Whether to use streaming mode (LazyFrame), by default False
        """
        self._df = df
        self._streaming = streaming

        # Auto-detect streaming mode if LazyFrame is provided
        if isinstance(df, pl.LazyFrame) and not streaming:
            self._streaming = True

    @classmethod
    def read_csv(
        cls,
        path: str,
        *,
        keep_default_na: bool = True,
        na_values: list[str] | None = None,
        streaming: bool = False,
        **kwargs: Any,  # noqa ANN401
    ) -> "PolarsBackend":
        """Read a CSV file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the CSV file (supports local and cloud paths via cloudpathlib)
        keep_default_na : bool, optional
            Whether to include default NA values (pandas compatibility), by default True
        na_values : list[str] | None, optional
            Additional strings to recognize as NA/NaN (pandas compatibility), by default None
        streaming : bool, optional
            If True, use streaming mode with LazyFrame, by default False
        **kwargs : Any
            Additional polars-specific arguments

        Returns
        -------
        PolarsBackend
            Backend instance wrapping the loaded DataFrame or LazyFrame

        Notes
        -----
        For pandas compatibility, keep_default_na and na_values are translated to
        polars' null_values parameter. In polars, null_values is a string or list of strings.
        """
        # Translate pandas parameters to polars parameters
        polars_kwargs = kwargs.copy()

        # Handle na_values conversion from pandas to polars
        if na_values is not None:
            # In polars, null_values parameter handles this
            polars_kwargs["null_values"] = na_values
        elif not keep_default_na:
            # If keep_default_na is False, don't use default NA values
            # In polars, we can pass an empty list to null_values
            polars_kwargs["null_values"] = []

        if streaming:
            # Use scan_csv for lazy/streaming mode
            df = pl.scan_csv(path, **polars_kwargs)
            return cls(df, streaming=True)
        else:
            df = pl.read_csv(path, **polars_kwargs)
            return cls(df, streaming=False)

    @classmethod
    def read_json(
        cls,
        path: str,
        *,
        lines: bool = False,
        streaming: bool = False,
        **kwargs: Any,  # noqa ANN401
    ) -> "PolarsBackend":
        """Read a JSON file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the JSON file
        lines : bool, optional
            If True, read file as JSON lines (one JSON object per line), by default False
        streaming : bool, optional
            If True, use streaming mode with LazyFrame, by default False
        **kwargs : Any
            Additional polars-specific arguments

        Returns
        -------
        PolarsBackend
            Backend instance wrapping the loaded DataFrame
        """
        if streaming:
            if lines:
                df = pl.scan_ndjson(path, **kwargs)
            else:
                # Regular JSON doesn't have a scan equivalent in polars
                raise NotImplementedError(
                    "Streaming mode only supported for JSON lines (ndjson) format. "
                    "Set lines=True to use streaming."
                )
            return cls(df, streaming=True)
        else:
            if lines:
                df = pl.read_ndjson(path, **kwargs)
            else:
                df = pl.read_json(path, **kwargs)
            return cls(df, streaming=False)

    @classmethod
    def read_parquet(
        cls,
        path: str,
        *,
        streaming: bool = False,
        **kwargs: Any,  # noqa ANN401
    ) -> "PolarsBackend":
        """Read a Parquet file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the Parquet file
        streaming : bool, optional
            If True, use streaming mode with LazyFrame, by default False
        **kwargs : Any
            Additional polars-specific arguments

        Returns
        -------
        PolarsBackend
            Backend instance wrapping the loaded DataFrame or LazyFrame
        """
        if streaming:
            # Use scan_parquet for lazy/streaming mode
            df = pl.scan_parquet(path, **kwargs)
            return cls(df, streaming=True)
        else:
            df = pl.read_parquet(path, **kwargs)
            return cls(df, streaming=False)

    @property
    def is_streaming(self) -> bool:
        """Check if backend is in streaming mode.

        Returns
        -------
        bool
            True if in streaming mode (LazyFrame), False otherwise
        """
        return self._streaming

    def _ensure_collected(self) -> pl.DataFrame:
        """Ensure the DataFrame is collected (not lazy).

        Returns
        -------
        pl.DataFrame
            Collected DataFrame
        """
        if isinstance(self._df, pl.LazyFrame):
            return self._df.collect()
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

    # ==================== Row Access Operations ====================

    def __getitem__(self, key: int | list[int] | slice) -> dict[str, Any] | "PolarsBackend":
        """Get row(s) from the DataFrame using Pythonic indexing.

        Parameters
        ----------
        key : int | list[int] | slice
            - int: Get single row as dict
            - list[int]: Get multiple rows as new backend
            - slice: Get row range as new backend

        Returns
        -------
        dict[str, Any] | PolarsBackend
            - dict if key is int (single row)
            - PolarsBackend if key is list or slice (multiple rows)

        Raises
        ------
        IndexError
            If index is out of bounds
        TypeError
            If key type is not supported
        """
        self._ensure_not_streaming("__getitem__")
        df = self._ensure_collected()

        if isinstance(key, int):
            # Return single row as dict
            if key >= len(df):
                raise IndexError(f"Index {key} out of bounds for DataFrame of length {len(df)}")
            return df.row(key, named=True)
        elif isinstance(key, (list, slice)):
            # Return multiple rows as new backend (not streaming)
            return PolarsBackend(df[key], streaming=False)
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")

    def __len__(self) -> int:
        """Get the number of rows in the DataFrame.

        Returns
        -------
        int
            Number of rows
        """
        self._ensure_not_streaming("__len__")
        df = self._ensure_collected()
        return len(df)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over DataFrame rows as dictionaries.

        In streaming mode (LazyFrame), collects and yields rows.
        In eager mode (DataFrame), yields rows directly.

        Yields
        ------
        dict[str, Any]
            Dictionary for each row mapping column names to values
        """
        df = self._ensure_collected()
        for row in df.iter_rows(named=True):
            yield row

    def iter_batches(self, batch_size: int = 1000) -> Iterator["PolarsBackend"]:
        """Iterate over DataFrame in batches.

        Parameters
        ----------
        batch_size : int, optional
            Number of rows per batch, by default 1000

        Yields
        ------
        PolarsBackend
            Backend instances wrapping batches of up to batch_size rows

        Note
        ----
        In streaming mode, polars LazyFrame will be collected in one go.
        For truly streaming batch processing, iterate and manually batch.
        """
        df = self._ensure_collected()
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            yield PolarsBackend(df[start_idx:end_idx], streaming=False)

    def filter_isin(
        self,
        column: str,
        values: list[Any],
        *,
        negate: bool = False,
    ) -> "PolarsBackend":
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
        PolarsBackend
            New backend with filtered DataFrame
        """
        expr = pl.col(column).is_in(values)
        if negate:
            expr = ~expr
        filtered_df = self._df.filter(expr)
        # Preserve streaming mode - LazyFrame operations return LazyFrame
        return PolarsBackend(
            filtered_df,
            streaming=self._streaming,
        )

    # ==================== Deduplication ====================

    def drop_duplicates(
        self,
        subset: list[str] | None = None,
        *,
        keep: Literal["first", "last"] = "first",
    ) -> "PolarsBackend":
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
        PolarsBackend
            New backend with duplicates removed
        """
        deduped_df = self._df.unique(subset=subset, keep=keep)
        # Preserve streaming mode
        return PolarsBackend(deduped_df, streaming=self._streaming)

    # ==================== Null Handling ====================

    def dropna(
        self,
        subset: list[str] | None = None,
    ) -> "PolarsBackend":
        """Remove rows with missing values.

        Parameters
        ----------
        subset : list[str] | None, optional
            Column names to consider for null detection.
            If None, check all columns, by default None

        Returns
        -------
        PolarsBackend
            New backend with null rows removed
        """
        if subset:
            cleaned_df = self._df.drop_nulls(subset=subset)
        else:
            cleaned_df = self._df.drop_nulls()
        # Preserve streaming mode
        return PolarsBackend(
            cleaned_df,
            streaming=self._streaming,
        )

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
        df = self._ensure_collected()
        # Drop nulls and get unique values
        unique_values = df[column].drop_nulls().unique().to_list()
        return sorted(unique_values)

    def map_column(
        self,
        column: str,
        mapping: dict[Any, Any],
        output_column: str,
        *,
        default: Any | None = None,  # noqa ANN401
    ) -> "PolarsBackend":
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
        PolarsBackend
            New backend with mapped column added
        """
        # Use replace_strict for mapping (polars >= 0.19)
        # For values not in mapping, they will be replaced with default
        mapping_expr = (
            pl.col(column)
            .replace_strict(mapping, default=default, return_dtype=type(default))
            .alias(output_column)
        )
        new_df = self._df.with_columns(mapping_expr)
        # Preserve streaming mode
        return PolarsBackend(new_df, streaming=self._streaming)

    def rename_columns(
        self,
        mapping: dict[str, str],
    ) -> "PolarsBackend":
        """Rename DataFrame columns.

        Parameters
        ----------
        mapping : dict[str, str]
            Dictionary mapping old column names to new names

        Returns
        -------
        PolarsBackend
            New backend with renamed columns
        """
        renamed_df = self._df.rename(mapping)
        # Preserve streaming mode
        return PolarsBackend(renamed_df, streaming=self._streaming)

    def add_column(
        self,
        column: str,
        values: Any,  # noqa ANN401
    ) -> "PolarsBackend":
        """Add a new column to the DataFrame.

        Parameters
        ----------
        column : str
            Name of the new column
        values : Any
            Values for the new column (scalar or array-like)

        Returns
        -------
        PolarsBackend
            New backend with new column added
        """
        # Use lit for scalar values, otherwise assume it's a Series
        if not isinstance(values, pl.Series):
            new_df = self._df.with_columns(pl.lit(values).alias(column))
        else:
            new_df = self._df.with_columns(values.alias(column))
        # Preserve streaming mode
        return PolarsBackend(new_df, streaming=self._streaming)

    def select_columns(
        self,
        columns: list[str],
    ) -> "PolarsBackend":
        """Select a subset of columns from the DataFrame.

        Parameters
        ----------
        columns : list[str]
            List of column names to keep

        Returns
        -------
        PolarsBackend
            New backend with only specified columns
        """
        selected_df = self._df.select(columns)
        # Preserve streaming mode
        return PolarsBackend(selected_df, streaming=self._streaming)

    @classmethod
    def concat(
        cls,
        backends: list["PolarsBackend"],
        *,
        ignore_index: bool = True,
        sort: bool = False,
    ) -> "PolarsBackend":
        """Concatenate multiple backend instances vertically (row-wise).

        Parameters
        ----------
        backends : list[PolarsBackend]
            List of backend instances to concatenate
        ignore_index : bool, optional
            If True, reset index in result (Polars doesn't have row indices), by default True
        sort : bool, optional
            If True, sort columns alphabetically, by default False

        Returns
        -------
        PolarsBackend
            New backend with concatenated data
        """
        dfs = [backend._df for backend in backends]

        # Ensure all are collected if mixing lazy and eager
        collected_dfs = []
        for df in dfs:
            if isinstance(df, pl.LazyFrame):
                collected_dfs.append(df.collect())
            else:
                collected_dfs.append(df)

        concatenated_df = pl.concat(collected_dfs, how="vertical")

        if sort:
            # Sort columns alphabetically
            sorted_cols = sorted(concatenated_df.columns)
            concatenated_df = concatenated_df.select(sorted_cols)

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
        return self._df.columns

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
    def unwrap(self) -> pl.DataFrame | pl.LazyFrame:
        """Get the underlying DataFrame object.

        Returns
        -------
        pl.DataFrame | pl.LazyFrame
            The underlying polars DataFrame or LazyFrame
        """
        return self._df

    # ==================== Sampling Operations ====================

    def subsample_by_column(
        self,
        column: str,
        ratios: dict[str, float],
        *,
        seed: int = 42,
    ) -> "PolarsBackend":
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
        PolarsBackend
            New backend with subsampled rows
        """
        groups = []

        # Handle explicitly listed values
        for val, ratio in ratios.items():
            if val == "other":
                continue

            # Filter for this value
            mask = pl.col(column) == val
            group_df = self._df.filter(mask)

            # Sample if ratio < 1.0
            if ratio >= 1.0:
                sampled = group_df
            else:
                # Calculate number of rows to sample
                # For LazyFrame, we need to collect to get length
                if isinstance(group_df, pl.LazyFrame):
                    group_collected = group_df.collect()
                    n = int(len(group_collected) * ratio)
                    if n > 0:
                        sampled = group_collected.sample(n=n, seed=seed)
                    else:
                        sampled = group_collected.head(0)
                else:
                    n = int(len(group_df) * ratio)
                    if n > 0:
                        sampled = group_df.sample(n=n, seed=seed)
                    else:
                        sampled = group_df.head(0)

            groups.append(sampled)

        # Handle "other" category
        if "other" in ratios:
            # Get all values not explicitly in ratios (excluding "other" itself)
            explicit_values = list(set(ratios.keys()) - {"other"})
            mask_other = ~pl.col(column).is_in(explicit_values)
            other_df = self._df.filter(mask_other)

            ratio = ratios["other"]
            if ratio >= 1.0:
                sampled_other = other_df
            else:
                # For LazyFrame, we need to collect to get length
                if isinstance(other_df, pl.LazyFrame):
                    other_collected = other_df.collect()
                    n = int(len(other_collected) * ratio)
                    if n > 0:
                        sampled_other = other_collected.sample(n=n, seed=seed)
                    else:
                        sampled_other = other_collected.head(0)
                else:
                    n = int(len(other_df) * ratio)
                    if n > 0:
                        sampled_other = other_df.sample(n=n, seed=seed)
                    else:
                        sampled_other = other_df.head(0)

            groups.append(sampled_other)

        # Concatenate all groups
        if groups:
            # Ensure all groups are DataFrame (not LazyFrame) for concat
            collected_groups = []
            for g in groups:
                if isinstance(g, pl.LazyFrame):
                    collected_groups.append(g.collect())
                else:
                    collected_groups.append(g)
            result_df = pl.concat(collected_groups)
        else:
            # Return empty dataframe with same schema
            if isinstance(self._df, pl.LazyFrame):
                result_df = self._df.head(0).collect()
            else:
                result_df = self._df.head(0)

        # Note: streaming mode is lost because we had to collect for sampling
        return PolarsBackend(result_df, streaming=False)

    def sample_rows(
        self,
        n: int,
        *,
        seed: int = 42,
        replace: bool = False,
    ) -> "PolarsBackend":
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
        PolarsBackend
            New backend with sampled rows
        """
        sampled_df = self._df.sample(n=n, seed=seed, with_replacement=replace)
        # Preserve streaming mode
        return PolarsBackend(sampled_df, streaming=self._streaming)

    # ==================== Utility Methods ====================

    def copy(self) -> "PolarsBackend":
        """Create a copy of the backend with a copied DataFrame.

        Returns
        -------
        PolarsBackend
            New backend instance with copied DataFrame
        """
        # Preserve streaming mode
        return PolarsBackend(
            self._df.clone(),
            streaming=self._streaming,
        )

    def __repr__(self) -> str:
        """Return string representation of the backend.

        Returns
        -------
        str
            String representation showing backend type and DataFrame shape
        """
        df = self._ensure_collected()
        return f"PolarsBackend(shape={df.shape})"
