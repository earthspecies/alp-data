import pandas as pd
import polars as pl
import pytest

from alp_data.backends import PandasBackend, PolarsBackend
from alp_data.transforms import SelectColumns, SelectColumnsConfig


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_select_subset(backend_type: str) -> None:
    """Selecting a subset of columns keeps only those columns."""
    if backend_type == "pandas":
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        backend = PolarsBackend(df)

    transform = SelectColumns(columns=["a", "c"])
    result, metadata = transform(backend)

    assert list(result.columns) == ["a", "c"]
    assert metadata == {}
    assert len(result) == 2


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_select_single_column(backend_type: str) -> None:
    """Selecting a single column works correctly."""
    if backend_type == "pandas":
        df = pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]})
        backend = PolarsBackend(df)

    transform = SelectColumns(columns=["x"])
    result, _ = transform(backend)

    assert list(result.columns) == ["x"]
    assert len(result) == 3


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_select_all_columns(backend_type: str) -> None:
    """Selecting all columns returns a backend with the same shape."""
    if backend_type == "pandas":
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        backend = PolarsBackend(df)

    transform = SelectColumns(columns=["a", "b", "c"])
    result, _ = transform(backend)

    assert list(result.columns) == ["a", "b", "c"]
    assert len(result) == 1


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_missing_column_raises_key_error(backend_type: str) -> None:
    """KeyError is raised when a requested column does not exist."""
    if backend_type == "pandas":
        df = pd.DataFrame({"a": [1], "b": [2]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"a": [1], "b": [2]})
        backend = PolarsBackend(df)

    transform = SelectColumns(columns=["a", "missing"])
    with pytest.raises(KeyError, match="missing"):
        transform(backend)


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_preserves_column_order(backend_type: str) -> None:
    """Output columns follow the order specified in the config, not the source."""
    if backend_type == "pandas":
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        backend = PolarsBackend(df)

    transform = SelectColumns(columns=["c", "a"])
    result, _ = transform(backend)

    assert list(result.columns) == ["c", "a"]


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_manual_vs_config(backend_type: str) -> None:
    """Manual instantiation and from_config produce the same result."""
    if backend_type == "pandas":
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        backend = PolarsBackend(df)

    manual = SelectColumns(columns=["a", "c"])
    manual_result, _ = manual(backend)

    config = SelectColumnsConfig(type="select_columns", columns=["a", "c"])
    config_result, _ = SelectColumns.from_config(config)(backend)

    if backend_type == "pandas":
        pd.testing.assert_frame_equal(manual_result.unwrap, config_result.unwrap)
    else:
        m = manual_result.unwrap
        c = config_result.unwrap
        if isinstance(m, pl.LazyFrame):
            m = m.collect()
        if isinstance(c, pl.LazyFrame):
            c = c.collect()
        assert m.equals(c)


def test_config_validation_empty_columns() -> None:
    """Config rejects an empty columns list."""
    with pytest.raises(ValueError, match="columns"):
        SelectColumnsConfig(type="select_columns", columns=[])
