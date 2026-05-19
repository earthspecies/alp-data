"""Serves as a registry of sorts and type definitions for supported backends.

When more backends are added, they should be registered here.
"""

from typing import Literal, Type

from esp_data.backends.pandas_backend import PandasBackend
from esp_data.backends.polars_backend import PolarsBackend
from esp_data.backends.protocol import DataBackend
from esp_data.backends.pyarrow_backend import PyarrowBackend

BackendType = Literal["pandas", "polars", "pyarrow"]


# We need to add new backends here when they are implemented
_BACKEND_REGISTRY: dict[str, Type[DataBackend]] = {
    "pandas": PandasBackend,
    "polars": PolarsBackend,
    "pyarrow": PyarrowBackend,
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
