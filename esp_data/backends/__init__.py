"""Backend system for unified data operations across pandas, polars, and other libraries.

This module provides a protocol-based backend system that allows esp-data to work
with multiple data libraries through a common interface.
"""

from .backends import BackendType, get_backend
from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend
from .protocol import DataBackend
from .pyarrow_backend import PyarrowBackend

__all__ = [
    "DataBackend",
    "PandasBackend",
    "PolarsBackend",
    "PyarrowBackend",
    "BackendType",
    "get_backend",
]
