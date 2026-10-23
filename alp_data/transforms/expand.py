"""Use an arbitrary function to map each row of a table to some number of output rows.

`ExpandByFunction` expands rows based on any "row method" which takes a row dict as input and
returns the child rows to emit.

This is the template transform for e.g. making a dataset of videos into a dataset of shorter clips.

Examples
--------
>>> from alp_data.transforms import ExpandByFunction, register_row_method
>>> @register_row_method()
... def duplicate(row, n=2):
...     return [{} for _ in range(n)]
>>> transform = ExpandByFunction(method="duplicate", kwargs={"n": 3})

The same transform in a dataset config:

>>> from alp_data import DatasetConfig
>>> cfg = DatasetConfig(
...     dataset_name="mammalps",
...     split="benchmark-1-val",
...     transformations=[
...         {"type": "expand_by_function", "method": "duplicate", "kwargs": {"n": 3}},
...     ],
... )
"""

import importlib
import logging
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from alp_data.backends.protocol import DataBackend

from . import register_transform

logger = logging.getLogger("alp_data")

RowMethod = Callable[..., list[dict[str, Any]]]

_ROW_METHODS: dict[str, RowMethod] = {}


def register_row_method(name: str | None = None) -> Callable[[RowMethod], RowMethod]:
    """Register a row method so that configs can refer to it by name.

    Parameters
    ----------
    name : str | None, optional
        Name to register under. Defaults to the function's `__name__`.

    Returns
    -------
    Callable[[RowMethod], RowMethod]
        A decorator that registers the function and returns it unchanged.

    Notes
    -----
    Applying the decorator raises `ValueError` if a different function is already
    registered under the same name.
    """

    def decorator(fn: RowMethod) -> RowMethod:
        key = name or fn.__name__
        if key in _ROW_METHODS and _ROW_METHODS[key] is not fn:
            raise ValueError(f"Row method {key!r} is already registered.")
        _ROW_METHODS[key] = fn
        return fn

    return decorator


def resolve_row_method(spec: str | RowMethod) -> RowMethod:
    """Turn a row method specification into a callable.

    Parameters
    ----------
    spec : str | RowMethod
        A callable (returned as is), the name of a registered row method, or an
        import path of the form `"package.module:function"`.

    Returns
    -------
    RowMethod
        The resolved function.

    Raises
    ------
    KeyError
        If `spec` is a plain name that is not registered.
    TypeError
        If `spec` is neither a string nor a callable, or resolves to a non-callable.

    Notes
    -----
    For an import path, propagates `ImportError` if the module cannot be imported and
    `AttributeError` if it has no such attribute.
    """
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"Row method must be a string or a callable, got {type(spec).__name__}")
    if ":" in spec:
        module_name, attr = spec.split(":", 1)
        fn = getattr(importlib.import_module(module_name), attr)
    else:
        if spec not in _ROW_METHODS:
            raise KeyError(f"Unknown row method {spec!r}. Registered: {sorted(_ROW_METHODS)}")
        fn = _ROW_METHODS[spec]
    if not callable(fn):
        raise TypeError(f"Row method {spec!r} resolved to a non-callable {type(fn).__name__}")
    return fn


def _empty(backend: DataBackend) -> DataBackend:
    # An empty list is ambiguous for polars (it selects zero columns); a slice is not.
    return backend[0:0]


class ExpandByFunctionConfig(BaseModel):
    type: Literal["expand_by_function"]
    method: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ExpandByFunction:
    """Expand each row into zero, one or many rows with a row method.

    The row method is called as `method(row, **kwargs)` with the row as a dict and
    returns a list of dicts, one per child row. Each dict holds only the fields to set
    or override on the child; all other fields are copied from the parent. Returning
    `[]` drops the row, and `[{}]` keeps it unchanged.

    Works with any eager backend (pandas, polars) through the `DataBackend` protocol.

    Parameters
    ----------
    method : str | RowMethod
        The row method, or a specification accepted by `resolve_row_method`.
    kwargs : dict[str, Any] | None, optional
        Extra keyword arguments passed to `method` on every call.

    Notes
    -----
    Construction propagates the exceptions of `resolve_row_method` if `method`
    cannot be resolved.

    Examples
    --------
    >>> import pandas as pd
    >>> from alp_data.backends import PandasBackend
    >>> def halves(row):
    ...     mid = (row["start_frame"] + row["end_frame"]) // 2
    ...     return [{"end_frame": mid}, {"start_frame": mid}]
    >>> backend = PandasBackend(pd.DataFrame({"start_frame": [0], "end_frame": [10]}))
    >>> expanded, meta = ExpandByFunction(method=halves)(backend)
    >>> len(expanded), meta["n_out"]
    (2, 2)
    """

    def __init__(self, *, method: str | RowMethod, kwargs: dict[str, Any] | None = None) -> None:
        self.method = resolve_row_method(method)
        self.kwargs = kwargs or {}

    @classmethod
    def from_config(cls, cfg: ExpandByFunctionConfig) -> "ExpandByFunction":
        return cls(method=cfg.method, kwargs=cfg.kwargs)

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Expand every row of `backend`.

        Parameters
        ----------
        backend : DataBackend
            The backend wrapping the table to expand.

        Returns
        -------
        tuple[DataBackend, dict]
            The expanded backend (same type as input) and metadata with the number of
            input rows (`n_in`), output rows (`n_out`) and dropped input rows
            (`n_dropped`).

        Raises
        ------
        TypeError
            If the row method returns something other than a list.

        Notes
        -----
        Propagates any exception raised by the row method.
        """
        name = getattr(self.method, "__name__", repr(self.method))
        row_indices: list[int] = []
        parents: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        n_in = 0
        n_dropped = 0
        for i, row in enumerate(backend):
            n_in += 1
            children = self.method(row, **self.kwargs)
            if not isinstance(children, list):
                raise TypeError(
                    f"Row method {name!r} must return a list of dicts, "
                    f"got {type(children).__name__}"
                )
            if not children:
                n_dropped += 1
            for child in children:
                row_indices.append(i)
                parents.append(row)
                updates.append(child)

        if not row_indices:
            expanded = _empty(backend)
        else:
            expanded = backend[row_indices]
            columns = list(dict.fromkeys(key for update in updates for key in update))
            for column in columns:
                values = [
                    update[column] if column in update else parent.get(column)
                    for parent, update in zip(parents, updates, strict=True)
                ]
                expanded = expanded.add_column(column, values)

        metadata = {"n_in": n_in, "n_out": len(row_indices), "n_dropped": n_dropped}
        logger.info("ExpandByFunction(%s): %s", name, metadata)
        return expanded, metadata


register_transform(ExpandByFunctionConfig, ExpandByFunction)
