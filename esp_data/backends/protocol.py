"""Protocol definition for DataFrame backend operations.

This module defines the interface that all backend implementations must follow.
It enables support for multiple dataframe libraries (pandas, polars, etc.) through
a common abstraction layer.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Literal, Protocol, Type, TypeVar, overload

from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend

# Type alias for backend return type (allows chaining)
DataBackendT = TypeVar("DataBackend")


class DataBackend(Protocol):
    """Protocol defining the interface all DataFrame backends must implement.

    This protocol uses the Adapter pattern where the backend wraps a DataFrame
    and provides a unified interface. The wrapped DataFrame is stored as an
    instance attribute, making the API more Pythonic and cleaner.
    """

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        streaming: bool = False,
        **kwargs: Any,  # noqa ANN401
    ) -> DataBackendT:
        """Read a CSV file and return a wrapped data backend.

        Parameters
        ----------
        path : str
            Path to the CSV file (supports local and cloud paths via cloudpathlib)
        streaming : bool, optional
            If True, use streaming mode (lazy evaluation). In streaming mode,
            __getitem__ is disabled and data is processed via iteration.
            By default False.
        **kwargs : Any
            Additional backend-specific arguments

        Returns
        -------
        DataBackend
            Backend instance wrapping the loaded data
        """
        ...

    @classmethod
    def from_json(
        cls,
        path: str,
        *,
        lines: bool = False,
        streaming: bool = False,
        **kwargs: Any,  # noqa ANN401
    ) -> DataBackendT:
        """Read a JSON file and return a wrapped DataFrame backend.

        Parameters
        ----------
        path : str
            Path to the JSON file
        lines : bool, optional
            If True, read file as JSON lines (one JSON object per line), by default False
        streaming : bool, optional
            If True, use streaming mode (lazy evaluation), by default False
        **kwargs : Any
            Additional backend-specific arguments

        Returns
        -------
        DataBackend
            Backend instance wrapping the loaded DataFrame
        """
        ...

    @classmethod
    def from_parquet(
        cls,
        path: str,
        *,
        streaming: bool = False,
        **kwargs: Any,  # noqa ANN401
    ) -> DataBackendT:
        """Read a Parquet file and return a wrapped data backend.

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
        DataBackend
            Backend instance wrapping the loaded data object
        """
        ...

    def __init__(self, df: Any, *, streaming: bool = False) -> None:  # noqa ANN401
        """Wrap an existing data object.

        Parameters
        ----------
        df : Any
            The data to wrap (e.g., pd.DataFrame, pl.DataFrame).

        streaming : bool, optional
            If True, use streaming mode where __getitem__ is disabled
            and iteration processes data in chunks, by default False
        """
        ...

    @property
    def is_streaming(self) -> bool:
        """Check if backend is in streaming mode.

        Returns
        -------
        bool
            True if in streaming mode, False otherwise
        """
        ...

    @overload
    def __getitem__(self, key: int) -> dict[str, Any]:
        """Get a single row as a dictionary."""
        ...

    @overload
    def __getitem__(self, key: list[int]) -> DataBackendT:
        """Get multiple rows by list of indices."""
        ...

    @overload
    def __getitem__(self, key: slice) -> DataBackendT:
        """Get rows by slice."""
        ...

    def __getitem__(self, key: int | list[int] | slice) -> dict[str, Any] | DataBackendT:
        """Get row(s) from the DataFrame using Pythonic indexing.

        Parameters
        ----------
        key : int | list[int] | slice
            - int: Get single row as dict
            - list[int]: Get multiple rows as new backend
            - slice: Get row range as new backend

        Returns
        -------
        dict[str, Any] | DataBackend
            - dict if key is int (single row)
            - DataBackend if key is list or slice (multiple rows)

        Raises
        ------
        IndexError
            If index is out of bounds
        TypeError
            If key type is not supported
        RuntimeError
            If backend is in streaming mode (use iteration instead)

        Note
        ----
        In streaming mode, __getitem__ is disabled. Use iteration instead:
        for row in backend:
            process(row)
        """
        ...

    def __len__(self) -> int:
        """Get the number of rows in the DataFrame.

        Returns
        -------
        int
            Number of rows
        """
        ...

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over DataFrame rows as dictionaries.

        Yields
        ------
        dict[str, Any]
            Dictionary for each row mapping column names to values
        """
        ...

    def filter_isin(
        self,
        column: str,
        values: list[Any],
        *,
        negate: bool = False,
    ) -> DataBackendT:
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
        DataBackend
            New backend with filtered DataFrame
        """
        ...

    def drop_duplicates(
        self,
        subset: list[str] | None = None,
        *,
        keep: Literal["first", "last"] = "first",
    ) -> DataBackendT:
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
        DataBackend
            New backend with duplicates removed
        """
        ...

    def dropna(
        self,
        subset: list[str] | None = None,
    ) -> DataBackendT:
        """Remove rows with missing values.

        Parameters
        ----------
        subset : list[str] | None, optional
            Column names to consider for null detection.
            If None, check all columns, by default None

        Returns
        -------
        DataBackend
            New backend with null rows removed
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
        """
        ...

    def map_column(
        self,
        column: str,
        mapping: dict[Any, Any],
        output_column: str,
        *,
        default: Any | None = None,  # noqa ANN401
    ) -> DataBackendT:
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
        DataBackend
            New backend with mapped column added
        """
        ...

    def rename_columns(
        self,
        mapping: dict[str, str],
    ) -> DataBackendT:
        """Rename DataFrame columns.

        Parameters
        ----------
        mapping : dict[str, str]
            Dictionary mapping old column names to new names

        Returns
        -------
        DataBackend
            New backend with renamed columns
        """
        ...

    def add_column(
        self,
        column: str,
        values: Any,  # noqa ANN401
    ) -> DataBackendT:
        """Add a new column to the DataFrame.

        Parameters
        ----------
        column : str
            Name of the new column
        values : Any
            Values for the new column (scalar or array-like)

        Returns
        -------
        DataBackend
            New backend with new column added
        """
        ...

    def select_columns(
        self,
        columns: list[str],
    ) -> DataBackendT:
        """Select a subset of columns from the DataFrame.

        Parameters
        ----------
        columns : list[str]
            List of column names to keep

        Returns
        -------
        DataBackend
            New backend with only specified columns
        """
        ...

    @classmethod
    def concat(
        cls,
        backends: list[DataBackendT],
        *,
        ignore_index: bool = True,
        sort: bool = False,
    ) -> DataBackendT:
        """Concatenate multiple backend instances vertically (row-wise).

        Parameters
        ----------
        backends : list[DataBackend]
            List of backend instances to concatenate
        ignore_index : bool, optional
            If True, reset index in result, by default True
        sort : bool, optional
            If True, sort columns alphabetically, by default False

        Returns
        -------
        DataBackend
            New backend with concatenated data
        """
        ...

    @property
    def columns(self) -> list[str]:
        """Get the list of column names.

        Returns
        -------
        list[str]
            List of column names
        """
        ...

    def column_exists(self, column: str) -> bool:
        """Check if a column exists in the data.

        Parameters
        ----------
        column : str
            Column name to look for
        """
        ...

    @property
    def unwrap(self) -> Any:  # noqa: ANN401
        """Get the underlying data object.

        This is useful when you need to access backend-specific functionality
        or pass the data to functions that expect the native type.

        Returns
        -------
        Any
            The underlying data (e.g., pd.DataFrame, pl.DataFrame)
        """
        ...

    def subsample_by_column(
        self,
        column: str,
        ratios: dict[str, float],
        *,
        seed: int = 42,
    ) -> DataBackendT:
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
        DataBackend
            New backend with subsampled rows
        """
        ...

    def sample_rows(
        self,
        n: int,
        *,
        seed: int = 42,
        replace: bool = False,
    ) -> DataBackendT:
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
        DataBackend
            New backend with sampled rows
        """
        ...

    def copy(self) -> DataBackendT:
        """Create a copy of the backend with a copied DataFrame.

        Returns
        -------
        DataBackend
            New backend instance with copied DataFrame
        """
        ...

    def apply_fn(self, fn: Callable, **fn_kwargs: dict) -> DataBackendT:
        """Apply a custom function to the underlying DataFrame.

        Parameters
        ----------
        fn : Callable
            Function to apply. It should accept the underlying data type
            (e.g., pd.DataFrame, pl.DataFrame) as the first argument.
        **fn_kwargs : Any
            Keyword arguments to pass to the function

        Returns
        -------
        DataBackend
            New backend wrapping the result of the function application
        """
        ...


# Type alias for supported backend names
BackendType = Literal["pandas", "polars"]

# Registry mapping backend names to their implementation classes
_BACKEND_REGISTRY: dict[str, Type[DataBackendT]] = {
    "pandas": PandasBackend,
    "polars": PolarsBackend,
}


def get_backend(backend: BackendType) -> Type[DataBackend]:
    """Get the backend class for the specified backend type.

    Parameters
    ----------
    backend : BackendType
        Name of the backend ("pandas" or "polars")

    Returns
    -------
    Type[DataBackend]
        The backend class (not an instance)

    Raises
    ------
    ValueError
        If the backend name is not recognized

    Examples
    --------
    >>> backend_cls = get_backend("pandas")
    >>> assert backend_cls is PandasBackend
    >>> backend_cls = get_backend("polars")
    >>> assert backend_cls is PolarsBackend
    """
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends: {list(_BACKEND_REGISTRY.keys())}"
        )
    return _BACKEND_REGISTRY[backend]


def list_backends() -> list[str]:
    """List all registered backend names.

    Returns
    -------
    list[str]
        List of registered backend names

    Examples
    --------
    >>> list_backends()
    ['pandas', 'polars']
    """
    return list(_BACKEND_REGISTRY.keys())
