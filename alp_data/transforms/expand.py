"""Use an arbitrary function to map each row of a table to some number of output rows.

`ExpandByFunction` expands rows based on any "row method" which takes a row dict as input and
returns the child rows to emit.

This is the template transform for e.g. making a dataset of videos into a dataset of shorter clips.

Examples
--------
>>> from alp_data.transforms import ExpandByFunction
>>> def duplicate(row, n=2):
...     return [{} for _ in range(n)]
>>> transform = ExpandByFunction(method=duplicate, kwargs={"n": 3})

The same transform in a dataset config, given the function itself or its import path
(`"package.module.function"` or `"package.module:function"`):

>>> from alp_data import DatasetConfig
>>> cfg = DatasetConfig(
...     dataset_name="mammalps",
...     split="benchmark-1-val",
...     transformations=[
...         {"type": "expand_by_function", "method": duplicate, "kwargs": {"n": 3}},
...     ],
... )
>>> cfg = DatasetConfig(
...     dataset_name="mammalps",
...     split="benchmark-1-val",
...     transformations=[
...         {
...             "type": "expand_by_function",
...             "method": "alp_data.transforms.video_sample.split_clip_by_frames",
...             "kwargs": {"clip_length": 16},
...         },
...     ],
... )
"""

import functools
import importlib
import logging
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from alp_data.backends.protocol import DataBackend

from . import register_transform

logger = logging.getLogger("alp_data")

RowMethod = Callable[..., list[dict[str, Any]]]


def resolve_row_method(spec: str | RowMethod) -> RowMethod:
    """Turn a row method specification into a callable.

    Parameters
    ----------
    spec : str | RowMethod
        A callable (returned as is) or an import path, either
        `"package.module.function"` or `"package.module:function"`.

    Returns
    -------
    RowMethod
        The resolved function.

    Raises
    ------
    TypeError
        If `spec` is neither a string nor a callable, or resolves to a non-callable.
    ValueError
        If `spec` is a string that is not an import path.

    Notes
    -----
    Propagates `ImportError` if the module cannot be imported and `AttributeError` if
    it has no such attribute.
    """
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"Row method must be a string or a callable, got {type(spec).__name__}")
    if ":" in spec:
        module_name, attr_path = spec.split(":", 1)
    else:
        module_name, _, attr_path = spec.rpartition(".")
    if not module_name or not attr_path:
        raise ValueError(
            f"Row method {spec!r} is not an import path; expected "
            "'package.module.function' or 'package.module:function'"
        )
    module = importlib.import_module(module_name)
    fn = functools.reduce(getattr, attr_path.split("."), module)
    if not callable(fn):
        raise TypeError(f"Row method {spec!r} resolved to a non-callable {type(fn).__name__}")
    return fn


def _empty(backend: DataBackend) -> DataBackend:
    # An empty list is ambiguous for polars (it selects zero columns); a slice is not.
    return backend[0:0]


class ExpandByFunctionConfig(BaseModel):
    type: Literal["expand_by_function"]
    method: str | RowMethod
    kwargs: dict[str, Any] = Field(default_factory=dict)


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
        The row method, or its import path (`"package.module.function"` or
        `"package.module:function"`); see `resolve_row_method`.
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
            If the row method returns something other than a list of dicts.
        ValueError
            If two children set different keys.

        Notes
        -----
        Propagates any exception raised by the row method.
        """
        name = getattr(self.method, "__name__", repr(self.method))
        row_indices: list[int] = []
        # Column values of the children, keyed by the first child's keys
        columns: dict[str, list[Any]] | None = None
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
                if columns is None:  # Set columns for the first time
                    columns = {key: [] for key in child}
                elif child.keys() != columns.keys():
                    raise ValueError(
                        f"Row method {name!r} returned a child with keys {list(child)} "
                        f"for row {i}, but earlier children had keys {list(columns)}; "
                        "all children must set the same keys"
                    )
                for key, values in columns.items():
                    values.append(child[key])
                row_indices.append(i)

        if not row_indices:
            expanded = _empty(backend)
        else:
            expanded = backend[row_indices]
            for column, values in columns.items():
                expanded = expanded.add_column(column, values)

        metadata = {"n_in": n_in, "n_out": len(row_indices), "n_dropped": n_dropped}
        logger.info("ExpandByFunction(%s): %s", name, metadata)
        return expanded, metadata


register_transform(ExpandByFunctionConfig, ExpandByFunction)
