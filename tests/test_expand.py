from typing import Any

import pandas as pd
import polars as pl
import pytest

from alp_data.backends import PandasBackend, PolarsBackend
from alp_data.backends.protocol import DataBackend
from alp_data.transforms import (
    ExpandByFunction,
    ExpandByFunctionConfig,
    register_row_method,
    transform_from_config,
)
from alp_data.transforms.expand import resolve_row_method
from alp_data.transforms.video_sample import clip_to_num_frames


def _make_backend(backend_type: str, data: dict[str, list]) -> DataBackend:
    if backend_type == "pandas":
        return PandasBackend(pd.DataFrame(data))
    return PolarsBackend(pl.DataFrame(data))


def _rows(backend: DataBackend) -> list[dict[str, Any]]:
    return list(backend)


DATA = {"name": ["a", "b", "c"], "n": [0, 1, 3]}


@register_row_method("test_expand_repeat_n")
def repeat_n(row: dict[str, Any], label: str = "copy") -> list[dict[str, Any]]:
    """Emit `row["n"]` children, numbered with `idx`.

    Returns
    -------
    list[dict[str, Any]]
        One update per child.
    """
    return [{"idx": i, "label": label} for i in range(row["n"])]


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_expand_drop_keep_and_multiply(backend_type: str) -> None:
    """Test if rows can be dropped, kept unchanged, or expanded into many rows."""
    backend = _make_backend(backend_type, DATA)
    out, meta = ExpandByFunction(method=repeat_n)(backend)

    assert meta == {"n_in": 3, "n_out": 4, "n_dropped": 1}
    rows = _rows(out)
    assert [(r["name"], r["idx"]) for r in rows] == [("b", 0), ("c", 0), ("c", 1), ("c", 2)]
    # Fields not in the update are inherited from the parent
    assert [r["n"] for r in rows] == [1, 3, 3, 3]
    assert all(r["label"] == "copy" for r in rows)


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_expand_keep_unchanged(backend_type: str) -> None:
    backend = _make_backend(backend_type, DATA)
    out, meta = ExpandByFunction(method=lambda row: [{}])(backend)
    assert meta == {"n_in": 3, "n_out": 3, "n_dropped": 0}
    assert _rows(out) == _rows(backend)


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_expand_override_existing_column(backend_type: str) -> None:
    backend = _make_backend(backend_type, DATA)
    out, _ = ExpandByFunction(method=lambda row: [{"n": row["n"] * 10}])(backend)
    assert [r["n"] for r in _rows(out)] == [0, 10, 30]


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_expand_drop_all(backend_type: str) -> None:
    """Test if dropping every row gives an empty table with the original columns."""
    backend = _make_backend(backend_type, DATA)
    out, meta = ExpandByFunction(method=lambda row: [])(backend)
    assert meta == {"n_in": 3, "n_out": 0, "n_dropped": 3}
    assert len(out) == 0
    assert out.columns == backend.columns


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_expand_kwargs(backend_type: str) -> None:
    backend = _make_backend(backend_type, DATA)
    out, _ = ExpandByFunction(method=repeat_n, kwargs={"label": "x"})(backend)
    assert all(r["label"] == "x" for r in _rows(out))


@pytest.mark.parametrize("bad_return", [None, {}, ({},)])
def test_expand_non_list_return_raises(bad_return: Any) -> None:  # noqa: ANN401
    backend = _make_backend("pandas", DATA)
    with pytest.raises(TypeError, match="must return a list"):
        ExpandByFunction(method=lambda row: bad_return)(backend)


def test_resolve_row_method() -> None:
    """Test if row methods resolve from callables, registered names and import paths."""
    assert resolve_row_method(repeat_n) is repeat_n
    assert resolve_row_method("test_expand_repeat_n") is repeat_n
    assert resolve_row_method("clip_to_num_frames") is clip_to_num_frames
    path = "alp_data.transforms.video_sample:clip_to_num_frames"
    assert resolve_row_method(path) is clip_to_num_frames


def test_resolve_row_method_errors() -> None:
    with pytest.raises(KeyError, match="Unknown row method"):
        resolve_row_method("not_registered")
    with pytest.raises(TypeError, match="must be a string or a callable"):
        resolve_row_method(42)
    with pytest.raises(TypeError, match="non-callable"):
        resolve_row_method("alp_data.transforms.expand:_ROW_METHODS")
    with pytest.raises(AttributeError):
        resolve_row_method("alp_data.transforms.expand:does_not_exist")


def test_register_row_method_conflict() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_row_method("test_expand_repeat_n")(lambda row: [])

    # Re-registering the same function is allowed
    assert register_row_method("test_expand_repeat_n")(repeat_n) is repeat_n


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_expand_transform_methods_equivalence(backend_type: str) -> None:
    """
    Test that transform_from_config, manual, and from_config produce the same result.
    """
    config = ExpandByFunctionConfig(
        type="expand_by_function", method="test_expand_repeat_n", kwargs={"label": "x"}
    )
    transforms = [
        ExpandByFunction(method="test_expand_repeat_n", kwargs={"label": "x"}),
        ExpandByFunction.from_config(config),
        transform_from_config(config),
    ]
    results = [_rows(t(_make_backend(backend_type, DATA))[0]) for t in transforms]
    assert results[0] == results[1] == results[2]
    assert len(results[0]) == 4
