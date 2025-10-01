"""Protocol definition for DataFrame backend operations.

This module defines the interface that all backend implementations must follow.
It enables support for multiple dataframe libraries (pandas, polars, etc.) through
a common abstraction layer.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal, Protocol, TypeVar, overload

# Type alias for backend return type (allows chaining)
DataFrameBackendT = TypeVar("DataFrameBackend")


class DataFrameBackend(Protocol):
    """Protocol defining the interface all DataFrame backends must implement.

    This protocol uses the Adapter pattern where the backend wraps a DataFrame
    and provides a unified interface. The wrapped DataFrame is stored as an
    instance attribute, making the API more Pythonic and cleaner.

    Examples
    --------
    >>> backend = PandasBackend.read_csv("data.csv")
    >>> row = backend[0]  # Get first row as dict
    >>> filtered = backend.filter_isin("species", ["cat", "dog"])
    >>> for row in filtered:
    ...     print(row)
    """

    @classmethod
    def read_csv(
        cls,
        path: str,
        *,
        keep_default_na: bool = True,
        na_values: list[str] | None = None,
        streaming: bool = False,
        **kwargs: Any,  # noqa ANN401
    ) -> DataFrameBackendT:
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
            If True, use streaming mode (lazy evaluation). In streaming mode,
            __getitem__ is disabled and data is processed via iteration.
            By default False.
        **kwargs : Any
            Additional backend-specific arguments

        Returns
        -------
        DataFrameBackend
            Backend instance wrapping the loaded DataFrame

        Examples
        --------
        >>> backend = PandasBackend.read_csv("data.csv")
        >>> backend = PolarsBackend.read_csv("gs://bucket/data.csv", streaming=True)
        """
        ...

    @classmethod
    def read_json(
        cls,
        path: str,
        *,
        lines: bool = False,
        orient: str = "records",
        streaming: bool = False,
        **kwargs: Any,  # noqa ANN401
    ) -> DataFrameBackendT:
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
            If True, use streaming mode (lazy evaluation), by default False
        **kwargs : Any
            Additional backend-specific arguments

        Returns
        -------
        DataFrameBackend
            Backend instance wrapping the loaded DataFrame

        Examples
        --------
        >>> backend = PandasBackend.read_json("data.jsonl", lines=True)
        """
        ...

    @classmethod
    def read_parquet(
        cls,
        path: str,
        *,
        streaming: bool = False,
        **kwargs: Any,  # noqa ANN401
    ) -> DataFrameBackendT:
        """Read a Parquet file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the Parquet file
        streaming : bool, optional
            If True, use streaming mode (lazy evaluation), by default False
        **kwargs : Any
            Additional backend-specific arguments

        Returns
        -------
        DataFrameBackend
            Backend instance wrapping the loaded DataFrame

        Examples
        --------
        >>> backend = PandasBackend.read_parquet("data.parquet")
        """
        ...

    def __init__(self, df: Any, *, streaming: bool = False) -> None:  # noqa ANN401
        """Wrap an existing DataFrame.

        Parameters
        ----------
        df : Any
            The DataFrame to wrap (e.g., pd.DataFrame, pl.DataFrame).
            For streaming mode with pandas, this can be a TextFileReader.
        streaming : bool, optional
            If True, use streaming mode where __getitem__ is disabled
            and iteration processes data in chunks, by default False

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3]})
        >>> backend = PandasBackend(df)
        >>> # Streaming mode
        >>> backend = PandasBackend.read_csv("large.csv", streaming=True)
        """
        ...

    @property
    def is_streaming(self) -> bool:
        """Check if backend is in streaming mode.

        Returns
        -------
        bool
            True if in streaming mode, False otherwise

        Examples
        --------
        >>> backend = PandasBackend.read_csv("data.csv", streaming=True)
        >>> backend.is_streaming
        True
        """
        ...

    @overload
    def __getitem__(self, key: int) -> dict[str, Any]:
        """Get a single row as a dictionary."""
        ...

    @overload
    def __getitem__(self, key: list[int]) -> DataFrameBackendT:
        """Get multiple rows by list of indices."""
        ...

    @overload
    def __getitem__(self, key: slice) -> DataFrameBackendT:
        """Get rows by slice."""
        ...

    def __getitem__(self, key: int | list[int] | slice) -> dict[str, Any] | DataFrameBackendT:
        """Get row(s) from the DataFrame using Pythonic indexing.

        Parameters
        ----------
        key : int | list[int] | slice
            - int: Get single row as dict
            - list[int]: Get multiple rows as new backend
            - slice: Get row range as new backend

        Returns
        -------
        dict[str, Any] | DataFrameBackend
            - dict if key is int (single row)
            - DataFrameBackend if key is list or slice (multiple rows)

        Raises
        ------
        IndexError
            If index is out of bounds
        TypeError
            If key type is not supported
        RuntimeError
            If backend is in streaming mode (use iteration instead)

        Examples
        --------
        >>> backend[0]  # Get first row as dict
        {'col1': 1, 'col2': 'a'}
        >>> backend[[0, 5, 10]]  # Get rows 0, 5, 10 as new backend
        >>> backend[5:]  # Get rows from index 5 to end
        >>> backend[:10]  # Get first 10 rows

        Note
        ----
        In streaming mode, __getitem__ is disabled. Use iteration instead:
        >>> for row in backend:
        ...     process(row)
        """
        ...

    def __len__(self) -> int:
        """Get the number of rows in the DataFrame.

        Returns
        -------
        int
            Number of rows

        Examples
        --------
        >>> len(backend)
        100
        """
        ...

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over DataFrame rows as dictionaries.

        Yields
        ------
        dict[str, Any]
            Dictionary for each row mapping column names to values

        Examples
        --------
        >>> for row in backend:
        ...     print(row['column_name'])
        """
        ...

    def iter_batches(self, batch_size: int = 1000) -> Iterator[DataFrameBackendT]:
        """Iterate over DataFrame in batches.

        This provides a streaming interface for batch processing,
        useful for large datasets.

        Parameters
        ----------
        batch_size : int, optional
            Number of rows per batch, by default 1000

        Yields
        ------
        DataFrameBackend
            Backend instances wrapping batches of up to batch_size rows

        Examples
        --------
        >>> for batch in backend.iter_batches(batch_size=100):
        ...     process_batch(batch)
        """
        ...

    def filter_isin(
        self,
        column: str,
        values: list[Any],
        *,
        negate: bool = False,
    ) -> DataFrameBackendT:
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
        DataFrameBackend
            New backend with filtered DataFrame

        Examples
        --------
        >>> filtered = backend.filter_isin("species", ["cat", "dog"])
        >>> excluded = backend.filter_isin("species", ["cat"], negate=True)
        """
        ...

    def drop_duplicates(
        self,
        subset: list[str] | None = None,
        *,
        keep: Literal["first", "last"] = "first",
    ) -> DataFrameBackendT:
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
        DataFrameBackend
            New backend with duplicates removed

        Examples
        --------
        >>> deduped = backend.drop_duplicates(subset=["id"])
        """
        ...

    def dropna(
        self,
        subset: list[str] | None = None,
    ) -> DataFrameBackendT:
        """Remove rows with missing values.

        Parameters
        ----------
        subset : list[str] | None, optional
            Column names to consider for null detection.
            If None, check all columns, by default None

        Returns
        -------
        DataFrameBackend
            New backend with null rows removed

        Examples
        --------
        >>> cleaned = backend.dropna(subset=["species"])
        """
        ...

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

        Examples
        --------
        >>> backend.get_unique("species")
        ['cat', 'dog', 'bird']
        """
        ...

    def map_column(
        self,
        column: str,
        mapping: dict[Any, Any],
        output_column: str,
        *,
        default: Any | None = None,  # noqa ANN401
    ) -> DataFrameBackendT:
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
        DataFrameBackend
            New backend with mapped column added

        Examples
        --------
        >>> label_map = {"cat": 0, "dog": 1, "bird": 2}
        >>> labeled = backend.map_column("species", label_map, "label")
        """
        ...

    def rename_columns(
        self,
        mapping: dict[str, str],
    ) -> DataFrameBackendT:
        """Rename DataFrame columns.

        Parameters
        ----------
        mapping : dict[str, str]
            Dictionary mapping old column names to new names

        Returns
        -------
        DataFrameBackend
            New backend with renamed columns

        Examples
        --------
        >>> renamed = backend.rename_columns({"old_name": "new_name"})
        """
        ...

    def add_column(
        self,
        column: str,
        values: Any,  # noqa ANN401
    ) -> DataFrameBackendT:
        """Add a new column to the DataFrame.

        Parameters
        ----------
        column : str
            Name of the new column
        values : Any
            Values for the new column (scalar or array-like)

        Returns
        -------
        DataFrameBackend
            New backend with new column added

        Examples
        --------
        >>> with_col = backend.add_column("new_col", 0)
        >>> with_col = backend.add_column("ids", range(len(backend)))
        """
        ...

    def select_columns(
        self,
        columns: list[str],
    ) -> DataFrameBackendT:
        """Select a subset of columns from the DataFrame.

        Parameters
        ----------
        columns : list[str]
            List of column names to keep

        Returns
        -------
        DataFrameBackend
            New backend with only specified columns

        Examples
        --------
        >>> subset = backend.select_columns(["species", "count"])
        """
        ...

    @classmethod
    def concat(
        cls,
        backends: list[DataFrameBackendT],
        *,
        ignore_index: bool = True,
        sort: bool = False,
    ) -> DataFrameBackendT:
        """Concatenate multiple backend instances vertically (row-wise).

        Parameters
        ----------
        backends : list[DataFrameBackend]
            List of backend instances to concatenate
        ignore_index : bool, optional
            If True, reset index in result, by default True
        sort : bool, optional
            If True, sort columns alphabetically, by default False

        Returns
        -------
        DataFrameBackend
            New backend with concatenated data

        Examples
        --------
        >>> combined = PandasBackend.concat([backend1, backend2, backend3])
        """
        ...

    @property
    def columns(self) -> list[str]:
        """Get the list of column names.

        Returns
        -------
        list[str]
            List of column names

        Examples
        --------
        >>> backend.columns
        ['species', 'count', 'location']
        """
        ...

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

        Examples
        --------
        >>> backend.column_exists("species")
        True
        """
        ...

    @property
    def unwrap(self) -> Any:  # noqa: ANN401
        """Get the underlying DataFrame object.

        This is useful when you need to access backend-specific functionality
        or pass the DataFrame to functions that expect the native type.

        Returns
        -------
        Any
            The underlying DataFrame (e.g., pd.DataFrame, pl.DataFrame)

        Examples
        --------
        >>> df = backend.unwrap  # Get the raw pandas/polars DataFrame
        >>> isinstance(df, pd.DataFrame)
        True
        """
        ...

    def subsample_by_column(
        self,
        column: str,
        ratios: dict[str, float],
        *,
        seed: int = 42,
    ) -> DataFrameBackendT:
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
        DataFrameBackend
            New backend with subsampled rows

        Examples
        --------
        >>> ratios = {"cat": 0.5, "dog": 0.3, "other": 0.1}
        >>> subsampled = backend.subsample_by_column("species", ratios, seed=42)
        """
        ...

    def sample_rows(
        self,
        n: int,
        *,
        seed: int = 42,
        replace: bool = False,
    ) -> DataFrameBackendT:
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
        DataFrameBackend
            New backend with sampled rows

        Examples
        --------
        >>> sampled = backend.sample_rows(100, seed=42)
        """
        ...

    def copy(self) -> DataFrameBackendT:
        """Create a copy of the backend with a copied DataFrame.

        Returns
        -------
        DataFrameBackend
            New backend instance with copied DataFrame

        Examples
        --------
        >>> backend_copy = backend.copy()
        """
        ...
