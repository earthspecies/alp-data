"""Use an arbitrary function to map each row of a table to some number of output rows.

`ExpandByFunction` expands rows based on any "row method" which takes a row dict as input and
returns the child rows to emit.

This is the template transform for e.g. making a dataset of videos into a dataset of shorter clips.

Row methods are referred to in configs by the name they were registered under with
`register_row_method`. Only registered functions can be selected from a config; no module is
imported because of what a config says. In Python, a callable can also be passed directly.

The registry is global to the Python process. A row method registered in any module (your
script, or another module it imports) can be used by name once the code calling
`register_row_method` has run, i.e. once its module is imported. Registrations do not carry
over to other processes: a new process only knows alp-data's built-in row methods until it
imports the modules registering the others. Names must be unique within a process.

Examples
--------
>>> from alp_data.transforms import ExpandByFunction, register_row_method
>>> @register_row_method("duplicate_rows")
... def duplicate(row, n=2):
...     return [{} for _ in range(n)]
>>> transform = ExpandByFunction(method=duplicate, kwargs={"n": 3})

The same transform in a dataset config, given the function itself or its registered name:

>>> from alp_data import DatasetConfig
>>> cfg = DatasetConfig(
...     dataset_name="mammalps",
...     split="benchmark-1-val",
...     transformations=[
...         {"type": "expand_by_function", "method": duplicate, "kwargs": {"n": 3}},
...         {"type": "expand_by_function", "method": "duplicate_rows", "kwargs": {"n": 3}},
...         {
...             "type": "expand_by_function",
...             "method": "split_clip_by_frames",
...             "kwargs": {"clip_length": 16},
...         },
...     ],
... )
"""

import logging
from collections.abc import Callable
from typing import Any, Literal, overload

from pydantic import BaseModel, Field, field_validator

from alp_data.backends.protocol import DataBackend

from . import register_transform

logger = logging.getLogger("alp_data")

RowMethod = Callable[..., list[dict[str, Any]]]

_ROW_METHOD_REGISTRY: dict[str, RowMethod] = {}


def _register_row_method(name: str, fn: RowMethod, overwrite: bool = False) -> RowMethod:
    if not callable(fn):
        raise TypeError(f"Row method must be callable, got {type(fn).__name__}")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(
            f"Row method name must be a valid Python identifier, got {name!r}; "
            "pass an explicit name to register_row_method"
        )
    if name in _ROW_METHOD_REGISTRY and not overwrite:
        raise ValueError(
            f"Row method {name!r} is already registered. Please use a unique name, or "
            "pass `overwrite=True` to replace it."
        )
    _ROW_METHOD_REGISTRY[name] = fn
    return fn


@overload
def register_row_method(name: RowMethod) -> RowMethod: ...


@overload
def register_row_method(
    name: str | None = None, *, overwrite: bool = False
) -> Callable[[RowMethod], RowMethod]: ...


def register_row_method(
    name: str | RowMethod | None = None, *, overwrite: bool = False
) -> RowMethod | Callable[[RowMethod], RowMethod]:
    """Register a row method so configs can refer to it by name.

    Use it as a bare decorator to register a function under its `__name__`, or with a
    name to register it under that name instead:

    >>> @register_row_method
    ... def keep_row(row):
    ...     return [{}]
    >>> @register_row_method("drop_all_rows")
    ... def drop(row):
    ...     return []

    When the decorator runs (in an import or in the top-level script),
    the decorated function is added to a registry global to the Python process.

    Parameters
    ----------
    name : str | RowMethod | None, optional
        The name to register under, or the row method itself when used as a bare
        decorator. If None, the decorated function's `__name__` is used.
    overwrite : bool, optional
        If True, replace any row method already registered under the name instead of
        raising, by default False. This includes alp-data's built-in row methods.

    Returns
    -------
    RowMethod | Callable[[RowMethod], RowMethod]
        The row method itself when used as a bare decorator, otherwise a decorator that
        registers and returns the row method.

    Raises
    ------
    ValueError
        If the name is already registered and `overwrite` is False, or is not a valid
        Python identifier (e.g. a lambda's `"<lambda>"`).
    TypeError
        If the registered object is not callable, or has no `__name__` and no name is given.
    """  # noqa: DOC502 - raised by `_register_row_method` and `_default_name`
    if callable(name):
        return _register_row_method(_default_name(name), name)

    def decorator(fn: RowMethod) -> RowMethod:
        return _register_row_method(
            name if name is not None else _default_name(fn), fn, overwrite=overwrite
        )

    return decorator


def _default_name(fn: RowMethod) -> str:
    fn_name = getattr(fn, "__name__", None)
    if fn_name is None:
        raise TypeError(
            f"Row method {fn!r} has no __name__; pass an explicit name to register_row_method"
        )
    return fn_name


def _get_row_method(name: str) -> RowMethod:
    """Look up a registered row method by name.

    Parameters
    ----------
    name : str
        The name the row method was registered under.

    Returns
    -------
    RowMethod
        The registered row method.

    Raises
    ------
    ValueError
        If no row method is registered under `name`.
    """
    try:
        return _ROW_METHOD_REGISTRY[name]
    except KeyError:
        hint = ""
        if "." in name or ":" in name:
            hint = (
                " Import paths are not supported. Decorate the function with "
                "`register_row_method` and refer to it by its registered name."
            )
        raise ValueError(
            f"No row method registered under {name!r}. Registered row methods: "
            f"{sorted(_ROW_METHOD_REGISTRY)}. Make sure the module defining it is "
            f"imported.{hint}"
        ) from None


def _empty(backend: DataBackend) -> DataBackend:
    # An empty list is ambiguous for polars (it selects zero columns); a slice is not.
    return backend[0:0]


class ExpandByFunctionConfig(BaseModel):
    type: Literal["expand_by_function"]
    method: str | RowMethod
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("method")
    @classmethod
    def _check_registered(cls, method: str | RowMethod) -> str | RowMethod:
        """Check that a row method given by name is registered.

        Parameters
        ----------
        method : str | RowMethod
            The row method or its registered name.

        Returns
        -------
        str | RowMethod
            `method`, unchanged.

        Raises
        ------
        ValueError
            If `method` is a name with no registered row method.
        """  # noqa: DOC502 - raised by `_get_row_method`
        if isinstance(method, str):
            _get_row_method(method)
        return method


class ExpandByFunction:
    """Expand each row into zero, one or many rows with a row method.

    The row method is called as `method(row, **kwargs)` with the row as a dict and
    returns a list of dicts, one per child row. Each dict holds only the fields to set
    or override on the child; all other fields are copied from the parent. Returning
    `[]` drops the row.

    All rows must produce children with the same keys, across the dataset. Those keys
    correspond to the columns that get changed in the children - all other keys stay
    with their values in the original dataset.

    Works with any eager backend (pandas, polars) through the `DataBackend` protocol.

    Parameters
    ----------
    method : str | RowMethod
        The row method, or the name it was registered under with `register_row_method`.
    kwargs : dict[str, Any] | None, optional
        Extra keyword arguments passed to `method` on every call.

    Raises
    ------
    TypeError
        If `method` is neither a string nor a callable.
    ValueError
        If `method` is a name with no registered row method.

    Examples
    --------
    >>> import pandas as pd
    >>> from alp_data.backends import PandasBackend
    >>> def halves(row):
    ...     start, end = row["start_frame"], row["end_frame"]
    ...     mid = (start + end) // 2
    ...     return [
    ...         {"start_frame": start, "end_frame": mid},
    ...         {"start_frame": mid, "end_frame": end},
    ...     ]
    >>> backend = PandasBackend(pd.DataFrame({"start_frame": [0], "end_frame": [10]}))
    >>> expanded, meta = ExpandByFunction(method=halves)(backend)
    >>> len(expanded), meta["n_out"]
    (2, 2)
    """

    def __init__(self, *, method: str | RowMethod, kwargs: dict[str, Any] | None = None) -> None:
        if isinstance(method, str):
            self.method = _get_row_method(method)
        elif callable(method):
            self.method = method
        else:
            raise TypeError(
                f"Row method must be a registered name or a callable, got {type(method).__name__}"
            )
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
            If the row method returns something other than a list of dicts.
        ValueError
            If two children set different keys.
        ValueError
            If backend has streaming=True.

        Notes
        -----
        Propagates any exception raised by the row method.
        """
        if backend.is_streaming:
            raise ValueError("ExpandByFunction requires an eager (non-streaming) backend")
        name = getattr(self.method, "__name__", repr(self.method))
        # Index of each child's parent row, one entry per child
        parent_indices: list[int] = []
        # Only the columns the children override (not the whole table): column name ->
        # the value each child sets for it, in child order. The column names are read
        # and fixed from the first child's columns.
        child_values: dict[str, list[Any]] | None = None
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
                if not isinstance(child, dict):
                    raise TypeError(
                        f"Row method {name!r} must return a list of dicts, "
                        f"got a list containing {type(child).__name__}"
                    )
                if child_values is None:
                    # Column names haven't been set yet: set them from the first child
                    child_values = {column_name: [] for column_name in child}
                elif child.keys() != child_values.keys():
                    raise ValueError(
                        f"Row method {name!r} returned a child with keys {list(child)} "
                        f"for row {i}, but earlier children had keys {list(child_values)}; "
                        "all children must set the same keys"
                    )
                for column_name, column_values in child_values.items():
                    column_values.append(child[column_name])
                parent_indices.append(i)

        if not parent_indices:
            expanded = _empty(backend)
        else:
            # Repeat each parent row once per child; this carries over every column
            expanded = backend[parent_indices]
            # Then replace just the overridden columns with the children's values
            for column_name, column_values in child_values.items():
                expanded = expanded.add_column(column_name, column_values)

        metadata = {"n_in": n_in, "n_out": len(parent_indices), "n_dropped": n_dropped}
        logger.info("ExpandByFunction(%s): %s", name, metadata)
        return expanded, metadata


register_transform(ExpandByFunctionConfig, ExpandByFunction)
