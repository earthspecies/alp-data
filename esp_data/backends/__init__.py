"""Backend system for unified DataFrame operations across pandas, polars, and other libraries.

This module provides a protocol-based backend system that allows esp-data to work
with multiple DataFrame libraries through a common interface.
"""

from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend
from .protocol import (
    BackendType,
    get_backend,
    list_backends,
    register_backend,
)

__all__ = [
    "PandasBackend",
    "PolarsBackend",
    "BackendType",
    "get_backend",
    "register_backend",
    "list_backends",
]
