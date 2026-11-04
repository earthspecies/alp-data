from typing import Any

import pandas as pd
import polars as pl
import pytest

from alp_data import DatasetConfig
from alp_data.backends import PandasBackend, PolarsBackend
from alp_data.backends.protocol import DataBackend
from alp_data.transforms import ExpandByFunction, ExpandByFunctionConfig, transform_from_config
from alp_data.transforms.expand import resolve_row_method
from alp_data.transforms.video_sample import split_clip_by_frames


def _make_backend(backend_type: str, data: dict[str, list]) -> DataBackend:
    if backend_type == "pandas":
        return PandasBackend(pd.DataFrame(data))
    return PolarsBackend(pl.DataFrame(data))


def _rows(backend: DataBackend) -> list[dict[str, Any]]:
    return list(backend)


DATA = {"name": ["a", "b", "c"], "n": [0, 1, 3]}


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


@pytest.mark.parametrize("bad_child", [None, 1, "x", ["n", 1]])
def test_expand_non_dict_child_raises(bad_child: Any) -> None:  # noqa: ANN401
    backend = _make_backend("pandas", DATA)
    with pytest.raises(TypeError, match="list containing"):
        ExpandByFunction(method=lambda row: [bad_child])(backend)


@pytest.mark.parametrize(
    "method",
    [
        # different keys within one row
        pytest.param(lambda row: [{"a": 1}, {"b": 2}], id="within-row"),
        # a subset of the keys within one row
        pytest.param(lambda row: [{"a": 1, "b": 2}, {"a": 1}], id="subset"),
        # unchanged `{}` for some rows, updates for others
        pytest.param(lambda row: [{}] if row["n"] == 0 else [{"n": 1}], id="across-rows"),
    ],
)
@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_expand_mismatched_keys_raise(backend_type: str, method: Any) -> None:  # noqa: ANN401
    backend = _make_backend(backend_type, DATA)
    with pytest.raises(ValueError, match="all children must set the same keys"):
        ExpandByFunction(method=method)(backend)


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_expand_key_order_does_not_matter(backend_type: str) -> None:
    """Test if children may list the same keys in a different order."""
    backend = _make_backend(backend_type, DATA)
    method = lambda row: [{"x": 1, "y": 2}, {"y": 3, "x": 4}]  # noqa: E731
    out, meta = ExpandByFunction(method=method)(backend)
    assert meta["n_out"] == 6
    assert [(r["x"], r["y"]) for r in _rows(out)] == [(1, 2), (4, 3)] * 3


REPEAT_N_PATHS = [f"{__name__}.repeat_n", f"{__name__}:repeat_n"]


def test_resolve_row_method() -> None:
    """Test if row methods resolve from callables and both import path forms."""
    assert resolve_row_method(repeat_n) is repeat_n
    for path in REPEAT_N_PATHS:
        assert resolve_row_method(path) is repeat_n
    for path in [
        "alp_data.transforms.video_sample.split_clip_by_frames",
        "alp_data.transforms.video_sample:split_clip_by_frames",
    ]:
        assert resolve_row_method(path) is split_clip_by_frames


def test_resolve_row_method_nested_attribute() -> None:
    """Test if the colon form resolves a dotted attribute path."""
    path = "alp_data.transforms.expand:ExpandByFunction.from_config"
    assert resolve_row_method(path) == ExpandByFunction.from_config


@pytest.mark.parametrize("spec", ["split_clip_by_frames", "module:", ":function", "."])
def test_resolve_row_method_not_an_import_path(spec: str) -> None:
    with pytest.raises(ValueError, match="not an import path"):
        resolve_row_method(spec)


def test_resolve_row_method_errors() -> None:
    with pytest.raises(TypeError, match="must be a string or a callable"):
        resolve_row_method(42)
    with pytest.raises(TypeError, match="non-callable"):
        resolve_row_method("alp_data.transforms.expand.logger")
    with pytest.raises(AttributeError):
        resolve_row_method("alp_data.transforms.expand:does_not_exist")
    with pytest.raises(ImportError):
        resolve_row_method("alp_data.no_such_module.function")


def test_config_accepts_callable_and_import_path() -> None:
    """Test if a dataset config accepts a row method as a callable or an import path."""
    cfg = DatasetConfig(
        dataset_name="mammalps",
        transformations=[
            {"type": "expand_by_function", "method": repeat_n},
            {"type": "expand_by_function", "method": REPEAT_N_PATHS[0]},
        ],
    )
    assert [t.method for t in cfg.transformations] == [repeat_n, REPEAT_N_PATHS[0]]
    assert all(ExpandByFunction.from_config(t).method is repeat_n for t in cfg.transformations)


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_expand_transform_methods_equivalence(backend_type: str) -> None:
    """
    Test that transform_from_config, manual, and from_config produce the same result.
    """
    config = ExpandByFunctionConfig(
        type="expand_by_function", method=REPEAT_N_PATHS[1], kwargs={"label": "x"}
    )
    transforms = [
        ExpandByFunction(method=repeat_n, kwargs={"label": "x"}),
        ExpandByFunction.from_config(config),
        transform_from_config(config),
    ]
    results = [_rows(t(_make_backend(backend_type, DATA))[0]) for t in transforms]
    assert results[0] == results[1] == results[2]
    assert len(results[0]) == 4
