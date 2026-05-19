"""Pyarrow implementation of the DataFrameBackend protocol"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import pyarrow as pa

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
