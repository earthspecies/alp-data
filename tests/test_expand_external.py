"""Namespace stress tests: row methods defined outside of alp-data.

Each test writes a throwaway package into `tmp_path` (outside the repository), puts it
on `sys.path` for the duration of the test, and refers to its functions by callable or
import path. Package names are unique per test, so nothing is cached between tests.
"""

import functools
import sys
import textwrap
import types
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from alp_data import DatasetConfig
from alp_data.backends import PandasBackend
from alp_data.backends.protocol import DataBackend
from alp_data.transforms import ExpandByFunction, transform_from_config
from alp_data.transforms import video_sample as alp_video_sample
from alp_data.transforms.expand import resolve_row_method

DATA = {"name": ["a", "b"], "n": [1, 2]}

# A module that defines a function with the same name as an alp-data row method,
# plus helpers covering other kinds of callables.
ROWS_SOURCE = """
TAG = "{tag}"


def split_clip_by_frames(row, **kwargs):
    return [{{"source": TAG}}]


def repeat(row, label="copy"):
    return [{{"idx": i, "label": label, "source": TAG}} for i in range(row["n"])]


class Expander:
    def __init__(self, k=2):
        self.k = k

    def __call__(self, row):
        return [{{"idx": i, "source": TAG}} for i in range(self.k)]

    @staticmethod
    def keep(row):
        return [{{"source": TAG}}]


NOT_CALLABLE = 42
"""


def _backend() -> DataBackend:
    return PandasBackend(pd.DataFrame(DATA))


def _rows(backend: DataBackend) -> list[dict[str, Any]]:
    return list(backend)


@pytest.fixture
def make_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., str]]:
    """Return a factory that writes a throwaway package outside alp-data.

    The factory takes a tag (stored in the package's `rows` module as `TAG`) and
    optional extra `{relative path: source}` files, and returns the package name.
    The packages are removed from `sys.modules` on teardown.

    Yields
    ------
    Callable[..., str]
        The package factory.
    """
    root = tmp_path / "external"
    root.mkdir()
    monkeypatch.syspath_prepend(str(root))
    names: list[str] = []

    def factory(tag: str = "external", extra_files: dict[str, str] | None = None) -> str:
        name = f"ext_rows_{uuid.uuid4().hex[:8]}"
        pkg = root / name
        files = {
            "__init__.py": "",
            "rows.py": ROWS_SOURCE.format(tag=tag),
            **(extra_files or {}),
        }
        for rel, source in files.items():
            path = pkg / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(source))
        names.append(name)
        return name

    yield factory

    for module in [m for m in sys.modules if any(m.split(".")[0] == n for n in names)]:
        del sys.modules[module]


def test_package_lives_outside_alp_data(make_package: Callable[..., str]) -> None:
    """Test if the throwaway package is importable but not part of alp-data."""
    pkg = make_package()
    fn = resolve_row_method(f"{pkg}.rows.repeat")
    module_file = Path(sys.modules[fn.__module__].__file__).resolve()
    alp_data_dir = Path(alp_video_sample.__file__).resolve().parents[1]
    assert alp_data_dir not in module_file.parents


@pytest.mark.parametrize("form", ["{pkg}.rows.repeat", "{pkg}.rows:repeat"])
def test_external_import_path(make_package: Callable[..., str], form: str) -> None:
    pkg = make_package()
    out, meta = ExpandByFunction(method=form.format(pkg=pkg), kwargs={"label": "x"})(_backend())
    assert meta == {"n_in": 2, "n_out": 3, "n_dropped": 0}
    assert [(r["name"], r["idx"], r["label"]) for r in _rows(out)] == [
        ("a", 0, "x"),
        ("b", 0, "x"),
        ("b", 1, "x"),
    ]


def test_external_module_imported_lazily(make_package: Callable[..., str]) -> None:
    """Test if a module that was never imported is imported on resolution."""
    pkg = make_package()
    assert f"{pkg}.rows" not in sys.modules
    ExpandByFunction(method=f"{pkg}.rows.repeat")
    assert f"{pkg}.rows" in sys.modules


def test_external_nested_subpackage(make_package: Callable[..., str]) -> None:
    pkg = make_package(
        extra_files={
            "deep/__init__.py": "",
            "deep/er/__init__.py": "",
            "deep/er/rows.py": "def halve(row):\n    return [{'half': row['n'] / 2}]\n",
        }
    )
    for path in [f"{pkg}.deep.er.rows.halve", f"{pkg}.deep.er.rows:halve"]:
        out, _ = ExpandByFunction(method=path)(_backend())
        assert [r["half"] for r in _rows(out)] == [0.5, 1.0]


def test_external_name_shadowing_alp_data(make_package: Callable[..., str]) -> None:
    """Test if same-named functions in different packages don't collide."""
    pkg_a, pkg_b = make_package(tag="a"), make_package(tag="b")
    fn_a = resolve_row_method(f"{pkg_a}.rows.split_clip_by_frames")
    fn_b = resolve_row_method(f"{pkg_b}.rows:split_clip_by_frames")
    builtin = resolve_row_method("alp_data.transforms.video_sample.split_clip_by_frames")

    assert len({fn_a, fn_b, builtin}) == 3
    assert builtin is alp_video_sample.split_clip_by_frames
    for fn, tag in [(fn_a, "a"), (fn_b, "b")]:
        out, _ = ExpandByFunction(method=fn)(_backend())
        assert {r["source"] for r in _rows(out)} == {tag}


def test_external_attribute_paths(make_package: Callable[..., str]) -> None:
    """Test if the colon form reaches attributes nested inside a module."""
    pkg = make_package()
    out, _ = ExpandByFunction(method=f"{pkg}.rows:Expander.keep")(_backend())
    assert len(out) == 2


@pytest.mark.parametrize(
    "make_callable",
    [
        pytest.param(lambda rows: rows.Expander(k=3), id="callable-instance"),
        pytest.param(lambda rows: functools.partial(rows.repeat, label="p"), id="partial"),
        pytest.param(lambda rows: lambda row: rows.repeat(row, label="p"), id="lambda"),
        pytest.param(lambda rows: rows.Expander.keep, id="staticmethod"),
    ],
)
def test_external_callable_kinds(
    make_package: Callable[..., str], make_callable: Callable[[types.ModuleType], Any]
) -> None:
    """Test if non-function callables from outside alp-data work when passed directly."""
    pkg = make_package()
    rows = __import__(f"{pkg}.rows", fromlist=["rows"])
    fn = make_callable(rows)
    assert resolve_row_method(fn) is fn
    out, meta = ExpandByFunction(method=fn)(_backend())
    assert meta["n_out"] > 0
    assert {r["source"] for r in _rows(out)} == {"external"}


def test_closure_redefinition_does_not_conflict() -> None:
    """Test if redefining a function (as when re-running a notebook cell) is fine."""
    outs = []
    for k in [1, 2]:

        def expand(row: dict[str, Any], k: int = k) -> list[dict[str, Any]]:
            return [{} for _ in range(k)]

        outs.append(len(ExpandByFunction(method=expand)(_backend())[0]))
    assert outs == [2, 4]


def test_in_memory_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test if a module created at runtime, with no file, resolves by import path."""
    name = f"ext_mem_{uuid.uuid4().hex[:8]}"
    module = types.ModuleType(name)
    exec("def tag(row):\n    return [{'source': 'memory'}]\n", module.__dict__)  # noqa: S102
    monkeypatch.setitem(sys.modules, name, module)
    out, _ = ExpandByFunction(method=f"{name}.tag")(_backend())
    assert {r["source"] for r in _rows(out)} == {"memory"}


@pytest.mark.parametrize("as_path", [False, True], ids=["callable", "import-path"])
def test_external_through_dataset_config(make_package: Callable[..., str], as_path: bool) -> None:
    """Test if a dataset config builds a working transform from an external function."""
    pkg = make_package()
    path = f"{pkg}.rows:repeat"
    method = path if as_path else resolve_row_method(path)
    cfg = DatasetConfig(
        dataset_name="mammalps",
        transformations=[
            {"type": "expand_by_function", "method": method, "kwargs": {"label": "cfg"}}
        ],
    )
    out, meta = transform_from_config(cfg.transformations[0])(_backend())
    assert meta["n_out"] == 3
    assert {r["label"] for r in _rows(out)} == {"cfg"}


def test_external_errors(make_package: Callable[..., str]) -> None:
    pkg = make_package(extra_files={"broken.py": "raise RuntimeError('boom on import')\n"})
    with pytest.raises(AttributeError):
        ExpandByFunction(method=f"{pkg}.rows.does_not_exist")
    with pytest.raises(TypeError, match="non-callable"):
        ExpandByFunction(method=f"{pkg}.rows:NOT_CALLABLE")
    with pytest.raises(ModuleNotFoundError):
        ExpandByFunction(method=f"{pkg}.missing.fn")
    with pytest.raises(RuntimeError, match="boom on import"):
        ExpandByFunction(method=f"{pkg}.broken.fn")


def test_external_bad_return_type(make_package: Callable[..., str]) -> None:
    pkg = make_package(extra_files={"bad.py": "def as_dict(row):\n    return {}\n"})
    with pytest.raises(TypeError, match="must return a list"):
        ExpandByFunction(method=f"{pkg}.bad.as_dict")(_backend())
