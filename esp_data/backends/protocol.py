"""Protocol definition for data backend operations.

This module defines the interface that all backend implementations must follow.
It enables support for multiple data handling libraries (pandas, polars, etc.) through
a common abstraction layer.

Two protocols are defined:
- StreamingBackend: For streaming-only data formats (e.g., WebDataset/tar files)
  that only support iteration, not random access.
- DataBackend: For in-memory data formats (e.g., pandas, polars DataFrames)
  that support both random access and iteration.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Literal, Protocol, overload


class StreamingBackend(Protocol):
    """Protocol defining the interface for streaming-only data backends.

    This protocol is designed for data formats that only support sequential
    iteration, such as WebDataset (tar files). These backends cannot support:
    - Random access (__getitem__)
    - Known length (__len__)
    - Operations requiring full data scan (drop_duplicates, get_unique, histogram)
    - Sampling operations (subsample_by_column, upsample_by_column, sample_rows)

    Implementations should apply filtering and transformations lazily during iteration.
    """

    @property
    def is_streaming(self) -> bool:
        """Check if backend is in streaming mode.

        For StreamingBackend implementations, this should always return True.

        Returns
        -------
        bool
            Always True for streaming backends
        """
        ...

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples as dictionaries.

        This is the primary access method for streaming backends.

        Yields
        ------
        dict[str, Any]
            Dictionary for each sample mapping field names to values
        """
        ...

    @property
    def columns(self) -> list[str]:
        """Get the list of column/field names.

        Note: This may require peeking at the first sample, which could
        have side effects depending on the implementation.

        Returns
        -------
        list[str]
            List of column/field names
        """
        ...

    def column_exists(self, column: str) -> bool:
        """Check if a column/field exists in the data.

        Parameters
        ----------
        column : str
            Column name to look for

        Returns
        -------
        bool
            True if column exists, False otherwise
        """
        ...

    @property
    def unwrap(self) -> Any:  # noqa ANN401
        """Get the underlying data object.

        This is useful when you need to access backend-specific functionality
        or pass the data to functions that expect the native type.

        Returns
        -------
        Any
            The underlying data (e.g., wds.WebDataset)
        """
        ...

    def filter_isin(
        self,
        column: str,
        values: list[Any],
        *,
        negate: bool = False,
    ) -> "StreamingBackend":
        """Filter samples where column values are in (or not in) a list.

        This filter is applied lazily during iteration.

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
        StreamingBackend
            Self with filter configured (applied during iteration)
        """
        ...

    def dropna(
        self,
        subset: list[str] | None = None,
    ) -> "StreamingBackend":
        """Remove samples with missing values.

        This filter is applied lazily during iteration.

        Parameters
        ----------
        subset : list[str] | None, optional
            Column names to consider for null detection.
            If None, check all columns, by default None

        Returns
        -------
        StreamingBackend
            Self with dropna configured (applied during iteration)
        """
        ...

    def map_column(
        self,
        column: str,
        mapping: dict[Any, Any],
        output_column: str,
        *,
        default: Any | None = None,  # noqa ANN401
    ) -> "StreamingBackend":
        """Create a new column by mapping values from an existing column.

        This transformation is applied lazily during iteration.

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
        StreamingBackend
            Self with mapping configured (applied during iteration)
        """
        ...

    def apply_fn(
        self,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> "StreamingBackend":
        """Apply a custom function to each sample during iteration.

        Parameters
        ----------
        fn : Callable[[dict[str, Any]], dict[str, Any]]
            Function to apply. Should accept a sample dict and return
            a transformed sample dict.

        Returns
        -------
        StreamingBackend
            Self with function configured (applied during iteration)
        """
        ...

    def save_to(
        self,
        path: str | Any,  # noqa ANN401
        format: str = "webdataset",
        **kwargs: Any,
    ) -> int:
        """Write samples to disk or cloud storage.

        Parameters
        ----------
        path : str | Any
            Destination directory (local or cloud).
        format : str, optional
            Output format, by default ``"webdataset"``.
        **kwargs : Any
            Additional backend-specific arguments.

        Returns
        -------
        int
            Number of samples written.
        """
        ...


class DataBackend(Protocol):
    """Protocol defining the interface all data backends must implement.

    This protocol uses the Adapter pattern where the backend wraps a data
    and provides a unified interface. The wrapped data is stored as an
    instance attribute, making the API more Pythonic and cleaner.
    """

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        streaming: bool = False,
        **kwargs: Any,
    ) -> "DataBackend":
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
        **kwargs: Any,
    ) -> "DataBackend":
        """Read a JSON file and return a wrapped data backend.

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
            Backend instance wrapping the loaded data
        """
        ...

    @classmethod
    def from_parquet(cls, path: str, *, streaming: bool = False, **kwargs: Any) -> "DataBackend":
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

    @classmethod
    def from_path(cls, path: str, *, streaming: bool = False, **kwargs: Any) -> "DataBackend":
        """Load a tabular file, dispatching on extension.

        Parameters
        ----------
        path : str
            Path to a ``.parquet``, ``.csv``, ``.json``, ``.jsonl``,
            or ``.ndjson`` file.
        streaming : bool, optional
            Whether to use streaming mode, by default False.
        **kwargs : Any
            Additional backend-specific arguments.

        Returns
        -------
        DataBackend
            Backend instance wrapping the loaded data.

        Raises
        ------
        ValueError
            If the file extension is not supported.
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
    def __getitem__(self, key: list[int]) -> "DataBackend":
        """Get multiple rows by list of indices."""
        ...

    @overload
    def __getitem__(self, key: slice) -> "DataBackend":
        """Get rows by slice."""
        ...

    def __getitem__(self, key):
        """Get row(s) from the dataset using Pythonic indexing.

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
        """Get the number of rows in the dataset.

        Returns
        -------
        int
            Number of rows
        """
        ...

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over rows as dictionaries.

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
    ) -> "DataBackend":
        """Filter rows where column values are in (or not in) a list.

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
            New backend with filtered data
        """
        ...

    def drop_duplicates(
        self,
        subset: list[str] | None = None,
        *,
        keep: Literal["first", "last"] = "first",
    ) -> "DataBackend":
        """Remove duplicate rows from the data.

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
    ) -> "DataBackend":
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
        ...

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
        ...

    def map_column(
        self,
        column: str,
        mapping: dict[Any, Any],
        output_column: str,
        *,
        default: Any | None = None,  # noqa ANN401
    ) -> "DataBackend":
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
    ) -> "DataBackend":
        """Rename data columns.

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
    ) -> "DataBackend":
        """Add a new column to the data.

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
    ) -> "DataBackend":
        """Select a subset of columns from the data.

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
        backends: list["DataBackend"],
        *,
        ignore_index: bool = True,
        sort: bool = False,
    ) -> "DataBackend":
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
    def unwrap(self) -> Any:  # noqa ANN401
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
    ) -> "DataBackend":
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
        DataBackend
            New backend with subsampled rows

        Raises
        ------
        KeyError
            If the specified column does not exist in the DataFrame
        ValueError
            If any ratio is negative or greater than 1.0
        """
        ...

    def upsample_by_column(
        self,
        column: str,
        target_counts: dict[str, int],
        *,
        seed: int = 42,
    ) -> "DataBackend":
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
        DataBackend
            New backend with upsampled/downsampled rows

        Raises
        ------
        KeyError
            If the specified column does not exist in the DataFrame
        ValueError
            If any target count is negative
        """
        ...

    def sample_rows(
        self,
        n: int,
        *,
        seed: int = 42,
        replace: bool = False,
    ) -> "DataBackend":
        """Randomly sample n rows from the data.

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

    def copy(self) -> "DataBackend":
        """Create a copy of the backend with a copied data.

        Returns
        -------
        DataBackend
            New backend instance with copied data
        """
        ...

    def apply_fn(self, fn: Callable, **fn_kwargs: dict) -> "DataBackend":
        """Apply a custom function to the underlying data.

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

    def multilabel_from_features(
        self,
        input_features: list[str],
        output_feature: str,
        label_map: dict[str, Any] | None = None,
        allow_missing_labels: bool = False,
    ) -> tuple["DataBackend", dict]:
        """Create a multilabel column from multiple feature columns.

        Parameters
        ----------
        input_features : list[str]
            List of input feature column names to combine
        output_feature : str
            Name of the output multilabel column
        label_map : dict[str, Any] | None, optional
            Optional mapping from input feature values to output labels,
            by default None
        allow_missing_labels : bool, optional
            If True, ignore missing labels in input features,
            by default False

        Returns
        -------
        tuple[DataBackend, dict]
            New backend with multilabel column and metadata dictionary
        """
        ...

    def save_to(self, path: str, format: str = "webdataset", **kwargs: Any) -> int:
        """Save data to a file.

        Parameters
        ----------
        path : str
            Destination path (supports local and cloud paths)
        format : str, optional
            Output format. Supported: ``"webdataset"``. By default ``"webdataset"``.
        **kwargs : Any
            Additional backend-specific arguments passed to the underlying writer

        Returns
        -------
        int
            Number of samples written.

        Raises
        ------
        ValueError
            If `format` is not supported
        """
        ...
