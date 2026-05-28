"""Backend system for unified data operations across pandas, polars, and other libraries.

This module provides a protocol-based backend system that allows alp-data to work
with multiple data libraries through a common interface.

Two protocols are defined:
- StreamingDataBackend: For streaming-only data formats (e.g., WebDataset/tar files)
  that only support iteration, not random access.
- DataBackend: For in-memory data formats (e.g., pandas, polars DataFrames)
  that support both random access and iteration.
"""

from .backends import BackendType, get_backend
from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend
from .protocol import DataBackend, StreamingDataBackend

__all__ = [
    # Protocols
    "DataBackend",
    "StreamingDataBackend",
    "PandasBackend",
    "PolarsBackend",
    # Registry
    "BackendType",
    "get_backend",
]
