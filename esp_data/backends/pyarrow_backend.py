"""Pyarrow implementation of the DataFrameBackend protocol"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Literal

import pyarrow as pa
import pyarrow.compute as pc

from .protocol import DataBackend

logger = logging.getLogger("esp_data")


class PyarrowBackend(DataBackend):
    """Pyarrow implementation of the DataFrameBackend protocol.

    TO BE IMPLEMENTED
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
    def from_csv(cls) -> None:
        pass

    @classmethod
    def from_json(cls) -> None:
        pass

    @classmethod
    def from_parquet(cls) -> None:
        pass

    @property
    def is_streaming(self) -> None:
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
            offset = key.start
            length = key.stop - key.start
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

    @property
    def unwrap(self) -> pa.Table:
        """Get the underlying Table object.

        Returns
        -------
        pa.Table
            The underlying pyarrow Table
        """
        return self._df

    def _ensure_collected(self) -> pa.Table:
        """Ensure the Table is collected (not lazy).

        Returns
        -------
        pl.DataFrame
            Collected DataFrame
        """
        if isinstance(self._df, str):
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
