"""Polars implementation of the DataFrameBackend protocol."""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Any, Callable, Iterator, Literal

import polars as pl

from .protocol import DataBackend

logger = logging.getLogger("esp_data")


class PolarsBackend(DataBackend):
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
    >>> import polars as pl
    >>> from esp_data.backends import PolarsBackend
    >>> df = pl.DataFrame({
    ...     "species": ["cat", "dog", "fish", "cat", "dog", None],
    ...     "count": [5, 3, 8, 2, 7, 1]
    ... })
    >>> backend = PolarsBackend(df)
    >>> row = backend[0]
    >>> filtered = backend.filter_isin("species", ["cat", "dog"])
    >>> assert filtered.unwrap["species"].to_list() == ["cat", "dog", "cat", "dog"]
    >>> # Streaming mode with LazyFrame
    >>> df = pl.LazyFrame({
    ...     "species": ["cat", "dog", "fish", "cat", "dog", None],
    ...     "count": [5, 3, 8, 2, 7, 1]
    ... })
    >>> backend = PolarsBackend(df, streaming=True)
    >>> assert isinstance(backend.unwrap, pl.LazyFrame)
    >>> print(backend.columns)
    ['species', 'count']
    >>> for row in backend:
    ...     print(row)
    ...     break
    {'species': 'cat', 'count': 5}
    >>> collected = backend.collect()
    >>> assert isinstance(collected.unwrap, pl.DataFrame)
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
    def from_csv(
        cls, path: str, *, streaming: bool = False, **kwargs: dict[str, Any]
    ) -> "PolarsBackend":
        """Read a CSV file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the CSV file (supports local and cloud paths via cloudpathlib)
        streaming : bool, optional
            If True, use streaming mode with LazyFrame, by default False
        **kwargs : Any
            Additional polars-specific arguments

        Returns
        -------
        PolarsBackend
            Backend instance wrapping the loaded DataFrame or LazyFrame
        """
        # Filter out kwargs for any non-polars argument
        if streaming:
            valid_params = set(inspect.signature(pl.scan_csv).parameters.keys())
        else:
            valid_params = set(inspect.signature(pl.read_csv).parameters.keys())

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        if streaming:
            # Use scan_csv for lazy/streaming mode
            df = pl.scan_csv(path, **filtered_kwargs)
            return cls(df, streaming=True)
        else:
            df = pl.read_csv(path, **filtered_kwargs)
            return cls(df, streaming=False)

    @classmethod
    def from_json(
        cls,
        path: str,
        *,
        lines: bool = False,
        streaming: bool = False,
        **kwargs: dict[str, Any],
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
        # Filter out kwargs for any non-polars argument
        valid_params = set()
        if streaming and lines:
            valid_params = set(inspect.signature(pl.scan_ndjson).parameters.keys())
        elif not streaming and lines:
            valid_params = set(inspect.signature(pl.read_ndjson).parameters.keys())
        elif not streaming and not lines:
            valid_params = set(inspect.signature(pl.read_json).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        if streaming:
            if lines:
                df = pl.scan_ndjson(path, **filtered_kwargs)
            else:
                # Regular JSON doesn't have a scan equivalent in polars
                raise NotImplementedError(
                    "Streaming mode only supported for JSON lines (ndjson) format. "
                    "Set lines=True to use streaming."
                )
            return cls(df, streaming=True)
        else:
            if lines:
                df = pl.read_ndjson(path, **filtered_kwargs)
            else:
                df = pl.read_json(path, **filtered_kwargs)
            return cls(df, streaming=False)

    @classmethod
    def from_parquet(
        cls,
        path: str,
        *,
        streaming: bool = False,
        **kwargs: dict[str, Any],
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
        # Filter out kwargs for any non-polars argument
        valid_params = set(inspect.signature(pl.scan_csv).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        if streaming:
            # Use scan_parquet for lazy/streaming mode
            df = pl.scan_parquet(path, **filtered_kwargs)
            return cls(df, streaming=True)
        else:
            df = pl.read_parquet(path, **filtered_kwargs)
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

    def collect(self) -> "PolarsBackend":
        """Materialize the LazyFrame and return an eager backend.

        This method collects the LazyFrame into a DataFrame and returns a new
        backend with streaming mode disabled.

        Returns
        -------
        PolarsBackend
            New backend in eager mode with materialized DataFrame

        Notes
        -----
        If the backend is already in eager mode, returns a copy of the backend.
        """
        collected_df = self._ensure_collected()
        return PolarsBackend(collected_df, streaming=False)

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

        Notes
        -----
        In streaming mode (LazyFrame), this operation preserves the lazy computation.
        Call `.collect()` to materialize the filtered result into an eager backend.
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

        Notes
        -----
        In streaming mode (LazyFrame), this operation preserves the lazy computation.
        Call `.collect()` to materialize the deduplicated result into an eager backend.
        """
        deduped_df = self._df.unique(subset=subset, keep=keep)
        # Preserve streaming mode
        return PolarsBackend(deduped_df, streaming=self._streaming)

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

        Notes
        -----
        In streaming mode (LazyFrame), this operation preserves the lazy computation.
        Call `.collect()` to materialize the cleaned result into an eager backend.
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
        # For values not in mapping, they will be replaced with default
        _return_dtype = type(default) if default is not None else type(list(mapping.values())[0])
        mapping_expr = (
            pl.col(column)
            .replace_strict(mapping, default=default, return_dtype=_return_dtype)
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
        # if values is a scalar, create a series with the same length as df
        if not isinstance(values, (list, pl.Series)):
            df = self._ensure_collected()
            values = [values] * len(df)
            new_df = self._df.with_columns(pl.Series(column, values))
        else:
            new_df = self._df.with_columns(pl.Series(column, values))
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

        # Use diagonal concat to handle potentially mismatched columns (fills with null)
        # This is needed because the concat may be called with backends that have
        # different columns in "soft" merge mode
        # TODO: This doesn't work when there are dataype mismatches between the dfs
        # being concatenated
        concatenated_df = pl.concat(collected_dfs, how="diagonal_relaxed")

        if sort:
            # Sort columns alphabetically
            sorted_cols = sorted(concatenated_df.columns)
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
        return self._df.collect_schema().names()

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

        Warnings
        --------
        UserWarning
            If backend is in streaming mode, warns that LazyFrame will be collected
            since sampling requires materialization.
        """
        if self._streaming:
            warnings.warn(
                "subsample_by_column() requires collection of LazyFrame for sampling. "
                "The returned backend will be in eager mode (streaming=False).",
                UserWarning,
                stacklevel=2,
            )

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

    def apply_fn(
        self,
        fn: Callable,  # noqa ANN401
        fn_kwargs: dict[str, Any],
        apply_kwargs: dict[str, Any],
    ) -> "PolarsBackend":
        """Apply a custom function to rows and create a new column.

        Parameters
        ----------
        fn : Any
            Function to apply to each row. Should accept a dict of column values.
        fn_kwargs : dict
            Additional keyword arguments to pass to the function
        apply_kwargs : dict
            Additional keyword arguments to pass to polars.DataFrame.map_rows()

        Returns
        -------
        PolarsBackend
            New backend with the new column added

        Notes
        -----
        This method collects the DataFrame if in streaming mode, as polars does not
        support arbitrary row-wise functions in LazyFrame. The returned backend will
        be in eager mode (streaming=False).
        """
        logger.warning(
            "It's highly discouraged to use apply_fn with Polars"
            "Instead consider creating an Expression and using the"
            " backend.unwrap.with_columns() method"
        )
        self._ensure_not_streaming("apply_fn")

        # use partial to bind fn_kwargs to fn
        from functools import partial

        fn_partial = partial(fn, **fn_kwargs)
        df = self._ensure_collected()
        # Apply function row-wise and create new column
        df = self._df.map_rows(fn_partial, **apply_kwargs)
        return PolarsBackend(df, streaming=False)

    def multilabel_from_features(
        self,
        input_features: list[str],
        output_feature: str,
        label_map: dict[Any, int] | None = None,
        allow_missing_labels: bool = True,
    ) -> tuple["PolarsBackend", dict[Any, int]]:
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
        tuple[PolarsBackend, dict]
            A tuple containing:
            - New PolarsBackend instance with the added multi-label column
            - The label_map used for mapping labels to IDs

        Raises
        ------
        ValueError
            If any input feature does not exist or is not of type List.
        """
        df = self._ensure_collected()

        # Check that every input feature is a pl.List type
        for f in input_features:
            if f not in df.columns:
                raise ValueError(f"Input feature '{f}' does not exist in DataFrame.")
            if not df[f].dtype.base_type() == pl.List:
                # try to convert to list
                # first, get the base type of the column
                btype = df[f].dtype.base_type()
                df = df.with_columns(pl.col(f).cast(pl.List(btype)))

        if label_map is None:
            uniques = set()
            for f in input_features:
                # explode turns empty lists into NaNs hence the dropna()
                uniques |= set(df.select(pl.col(f).explode()).drop_nulls().to_series().to_list())
            label_map = {lbl: idx for idx, lbl in enumerate(sorted(uniques))}

        # Build mapping expressions - each should be an Expr that returns a list
        mapping_exprs = []
        for f in input_features:
            # Map each list element to its ID, handling empty lists
            mapping_expr = (
                pl.when(pl.col(f).list.len() > 0)
                .then(
                    pl.col(f)
                    .list.eval(pl.element().replace_strict(label_map, default=None))
                    .list.drop_nulls()
                )
                .otherwise(pl.lit([], dtype=pl.List(pl.Int64)))
            )
            mapping_exprs.append(mapping_expr)

        # Combine all mapped lists into a single list per row, then sort and deduplicate
        combined_expr = (
            pl.concat_list(mapping_exprs)
            .list.drop_nulls()
            .list.unique()
            .list.sort()
            .alias(output_feature)
        )

        new_df = df.with_columns(combined_expr)

        # Filter out rows with empty labels if required
        if not allow_missing_labels:
            new_df = new_df.filter(pl.col(output_feature).list.len() > 0)

        return PolarsBackend(new_df, streaming=self._streaming), label_map

    def __repr__(self) -> str:
        """Return string representation of the backend.

        Returns
        -------
        str
            String representation showing backend type and DataFrame shape
        """
        df = self._ensure_collected()
        return f"PolarsBackend(shape={df.shape})"
