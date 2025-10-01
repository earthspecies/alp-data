"""Backend system for unified DataFrame operations across pandas, polars, and other libraries.

This module provides a protocol-based backend system that allows esp-data to work
with multiple DataFrame libraries through a common interface.

Examples
--------
>>> from esp_data.backends import get_backend, BackendType
>>>
>>> # Using pandas backend
>>> backend = get_backend("pandas")
>>> df_backend = backend.read_csv("data.csv")
>>> row = df_backend[0]
>>>
>>> # Using polars backend
>>> backend = get_backend("polars")
>>> df_backend = backend.read_csv("data.csv")
>>> filtered = df_backend.filter_isin("species", ["cat", "dog"])
"""

from typing import Literal, Type

from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend
from .protocol import DataFrameBackend

# Type alias for supported backend names
BackendType = Literal["pandas", "polars"]

# Registry mapping backend names to their implementation classes
_BACKEND_REGISTRY: dict[str, Type[DataFrameBackend]] = {
    "pandas": PandasBackend,
    "polars": PolarsBackend,
}


def get_backend(backend: BackendType) -> Type[DataFrameBackend]:
    """Get the backend class for the specified backend type.

    Parameters
    ----------
    backend : BackendType
        Name of the backend ("pandas" or "polars")

    Returns
    -------
    Type[DataFrameBackend]
        The backend class (not an instance)

    Raises
    ------
    ValueError
        If the backend name is not recognized

    Examples
    --------
    >>> backend_cls = get_backend("pandas")
    >>> df_backend = backend_cls.read_csv("data.csv")
    >>>
    >>> backend_cls = get_backend("polars")
    >>> df_backend = backend_cls.read_csv("data.csv")
    """
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends: {list(_BACKEND_REGISTRY.keys())}"
        )
    return _BACKEND_REGISTRY[backend]


def register_backend(name: str, backend_class: Type[DataFrameBackend]) -> None:
    """Register a custom backend implementation.

    This allows users to add support for additional DataFrame libraries
    beyond pandas and polars.

    Parameters
    ----------
    name : str
        Name to register the backend under
    backend_class : Type[DataFrameBackend]
        The backend class implementing the DataFrameBackend protocol

    Raises
    ------
    ValueError
        If a backend with this name is already registered

    Examples
    --------
    >>> class DuckDBBackend:
    ...     # Implement DataFrameBackend protocol
    ...     pass
    >>>
    >>> register_backend("duckdb", DuckDBBackend)
    >>> backend_cls = get_backend("duckdb")
    """
    if name in _BACKEND_REGISTRY:
        raise ValueError(
            f"Backend '{name}' is already registered. "
            f"Use a different name or unregister the existing backend first."
        )
    _BACKEND_REGISTRY[name] = backend_class


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


__all__ = [
    "DataFrameBackend",
    "PandasBackend",
    "PolarsBackend",
    "BackendType",
    "get_backend",
    "register_backend",
    "list_backends",
]
