import functools
import importlib
import sys
import textwrap
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest
from pydantic import ValidationError

from alp_data import DatasetConfig
from alp_data.backends import PandasBackend, PolarsBackend
from alp_data.backends.protocol import DataBackend
from alp_data.transforms import (
    ExpandByFunction,
    ExpandByFunctionConfig,
    expand,
    register_row_method,
    transform_from_config,
)
from alp_data.transforms.video_sample import split_clip_by_duration, split_clip_by_frames


def _make_backend(backend_type: str, data: dict[str, list]) -> DataBackend:
    if backend_type == "pandas":
        return PandasBackend(pd.DataFrame(data))
    return PolarsBackend(pl.DataFrame(data))


def _rows(backend: DataBackend) -> list[dict[str, Any]]:
    return list(backend)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test a copy of the row method registry, so registrations don't leak."""
    monkeypatch.setattr(expand, "_ROW_METHOD_REGISTRY", dict(expand._ROW_METHOD_REGISTRY))


@pytest.fixture
def write_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], str]]:
    """Return a factory that writes a module, standing in for another user script.

    The factory takes the module source and returns a unique module name, importable
    but not yet imported. The modules are removed from `sys.modules` on teardown.

    Yields
    ------
    Callable[[str], str]
        The module factory.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    names: list[str] = []

    def factory(source: str) -> str:
        name = f"user_rows_{uuid.uuid4().hex[:8]}"
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(source))
        names.append(name)
        return name

    yield factory

    for name in names:
        sys.modules.pop(name, None)


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


def test_builtin_row_methods_registered() -> None:
    assert expand._get_row_method("split_clip_by_frames") is split_clip_by_frames
    assert expand._get_row_method("split_clip_by_duration") is split_clip_by_duration


def test_register_bare_decorator_uses_function_name() -> None:
    assert register_row_method(repeat_n) is repeat_n
    assert expand._get_row_method("repeat_n") is repeat_n


@pytest.mark.parametrize(
    "call",
    [lambda: register_row_method("custom"), lambda: register_row_method(name="custom")],
    ids=["positional", "keyword"],
)
def test_register_with_explicit_name(call: Callable[[], Any]) -> None:
    assert call()(repeat_n) is repeat_n
    assert expand._get_row_method("custom") is repeat_n
    assert "repeat_n" not in expand._ROW_METHOD_REGISTRY


def test_register_empty_call_uses_function_name() -> None:
    register_row_method()(repeat_n)
    assert expand._get_row_method("repeat_n") is repeat_n


def test_register_any_function_in_script() -> None:
    """Test if a function defined in the calling script can be registered and used by name."""

    def tag(row: dict[str, Any], value: str = "x") -> list[dict[str, Any]]:
        return [{"tag": value}]

    register_row_method(tag)
    cfg = DatasetConfig(
        dataset_name="mammalps",
        transformations=[{"type": "expand_by_function", "method": "tag", "kwargs": {"value": "y"}}],
    )
    out, _ = transform_from_config(cfg.transformations[0])(_make_backend("pandas", DATA))
    assert {r["tag"] for r in _rows(out)} == {"y"}


def test_register_duplicate_in_same_script_raises() -> None:
    register_row_method(repeat_n)
    with pytest.raises(ValueError, match="'repeat_n' is already registered"):
        register_row_method("repeat_n")(lambda row: [])
    assert expand._get_row_method("repeat_n") is repeat_n


@pytest.mark.parametrize(
    "decorator",
    [register_row_method(overwrite=True), register_row_method("repeat_n", overwrite=True)],
    ids=["function-name", "explicit-name"],
)
def test_register_overwrite_replaces(decorator: Callable[[Any], Any]) -> None:
    """Test if `overwrite=True` replaces an existing registration, as when re-running a cell."""
    register_row_method(repeat_n)

    def repeat_n_v2(row: dict[str, Any]) -> list[dict[str, Any]]:
        return [{}]

    repeat_n_v2.__name__ = "repeat_n"
    assert decorator(repeat_n_v2) is repeat_n_v2
    assert expand._get_row_method("repeat_n") is repeat_n_v2


def test_register_overwrite_new_name() -> None:
    """Test if `overwrite=True` also registers a name that isn't taken yet."""
    register_row_method("fresh", overwrite=True)(repeat_n)
    assert expand._get_row_method("fresh") is repeat_n


def test_register_duplicate_of_builtin_raises() -> None:
    """Test if a user function can't take over a name alp-data registered in another module."""

    def split_clip_by_frames(row: dict[str, Any]) -> list[dict[str, Any]]:
        return [{}]

    with pytest.raises(ValueError, match="'split_clip_by_frames' is already registered"):
        register_row_method(split_clip_by_frames)
    assert expand._get_row_method("split_clip_by_frames") is not split_clip_by_frames


def test_register_duplicate_across_modules_raises(write_module: Callable[[str], str]) -> None:
    """Test if two modules registering the same name clash on the second import."""
    source = """
        from alp_data.transforms import register_row_method


        @register_row_method("shared_name")
        def fn(row):
            return [{}]
    """
    first = importlib.import_module(write_module(source))
    with pytest.raises(ValueError, match="'shared_name' is already registered"):
        importlib.import_module(write_module(source))
    assert expand._get_row_method("shared_name") is first.fn


def test_registered_in_other_module_after_import(write_module: Callable[[str], str]) -> None:
    """Test if a row method registered in another module is usable once it is imported."""
    name = write_module("""
        from alp_data.transforms import register_row_method


        @register_row_method
        def from_other_script(row):
            return [{"source": "other"}]
    """)
    with pytest.raises(ValidationError, match="No row method registered"):
        ExpandByFunctionConfig(type="expand_by_function", method="from_other_script")
    importlib.import_module(name)
    config = ExpandByFunctionConfig(type="expand_by_function", method="from_other_script")
    out, _ = transform_from_config(config)(_make_backend("pandas", DATA))
    assert {r["source"] for r in _rows(out)} == {"other"}


def test_config_does_not_import_modules(write_module: Callable[[str], str]) -> None:
    """Test if naming a module in a config neither imports it nor runs its code."""
    name = write_module("raise RuntimeError('imported!')\n\n\ndef fn(row):\n    return [{}]\n")
    for method in [f"{name}.fn", f"{name}:fn"]:
        with pytest.raises(ValidationError, match="Import paths are not supported"):
            ExpandByFunctionConfig(type="expand_by_function", method=method)
    assert name not in sys.modules


@pytest.mark.parametrize("name", ["", "has space", "pkg.module.fn", "pkg:fn", "1st"])
def test_register_invalid_name_raises(name: str) -> None:
    with pytest.raises(ValueError, match="valid Python identifier"):
        register_row_method(name)(repeat_n)


def test_register_lambda_needs_explicit_name() -> None:
    with pytest.raises(ValueError, match="valid Python identifier"):
        register_row_method(lambda row: [])
    fn = register_row_method("named_lambda")(lambda row: [])
    assert expand._get_row_method("named_lambda") is fn


def test_register_nameless_callable_needs_explicit_name() -> None:
    partial = functools.partial(repeat_n, label="p")
    with pytest.raises(TypeError, match="no __name__"):
        register_row_method(partial)
    register_row_method("partial_repeat")(partial)
    assert expand._get_row_method("partial_repeat") is partial


def test_register_non_callable_raises() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        register_row_method("not_callable")(42)


def test_unregistered_name_raises() -> None:
    with pytest.raises(ValueError, match="No row method registered under 'nope'"):
        ExpandByFunction(method="nope")


@pytest.mark.parametrize("path", [f"{__name__}.repeat_n", f"{__name__}:repeat_n"])
def test_import_path_rejected_with_hint(path: str) -> None:
    """Test if import paths are not resolved and the error points to the registry."""
    with pytest.raises(ValueError, match="Import paths are not supported"):
        ExpandByFunction(method=path)


def test_non_string_non_callable_method_raises() -> None:
    with pytest.raises(TypeError, match="registered name or a callable"):
        ExpandByFunction(method=42)


def test_config_accepts_callable_and_registered_name() -> None:
    """Test if a dataset config accepts a row method as a callable or a registered name."""
    register_row_method(repeat_n)
    cfg = DatasetConfig(
        dataset_name="mammalps",
        transformations=[
            {"type": "expand_by_function", "method": repeat_n},
            {"type": "expand_by_function", "method": "repeat_n"},
        ],
    )
    assert [t.method for t in cfg.transformations] == [repeat_n, "repeat_n"]
    assert all(ExpandByFunction.from_config(t).method is repeat_n for t in cfg.transformations)


@pytest.mark.parametrize("method", ["not_registered", f"{__name__}.repeat_n"])
def test_config_rejects_unregistered_name(method: str) -> None:
    """Test if an unknown name fails when the config is validated, not when it is built."""
    with pytest.raises(ValidationError, match="No row method registered"):
        ExpandByFunctionConfig(type="expand_by_function", method=method)
    with pytest.raises(ValidationError, match="No row method registered"):
        DatasetConfig(
            dataset_name="mammalps",
            transformations=[{"type": "expand_by_function", "method": method}],
        )


def test_config_round_trips_registered_name() -> None:
    """Test if a config with a registered name serializes and reloads."""
    register_row_method(repeat_n)
    config = ExpandByFunctionConfig(type="expand_by_function", method="repeat_n")
    dumped = config.model_dump()
    assert dumped["method"] == "repeat_n"
    assert ExpandByFunctionConfig(**dumped) == config


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_expand_transform_methods_equivalence(backend_type: str) -> None:
    """
    Test that transform_from_config, manual, and from_config produce the same result.
    """
    register_row_method(repeat_n)
    config = ExpandByFunctionConfig(
        type="expand_by_function", method="repeat_n", kwargs={"label": "x"}
    )
    transforms = [
        ExpandByFunction(method=repeat_n, kwargs={"label": "x"}),
        ExpandByFunction.from_config(config),
        transform_from_config(config),
    ]
    results = [_rows(t(_make_backend(backend_type, DATA))[0]) for t in transforms]
    assert results[0] == results[1] == results[2]
    assert len(results[0]) == 4
