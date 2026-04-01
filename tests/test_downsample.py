import pandas as pd
import polars as pl
import pytest

from esp_data.backends import PandasBackend, PolarsBackend
from esp_data.transforms import Downsample, DownsampleConfig


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_downsample(backend_type: str) -> None:
    """Test that downsample keeps approximately the right fraction of rows."""
    if backend_type == "pandas":
        df = pd.DataFrame({"species": ["bee"] * 200, "value": range(200)})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"species": ["bee"] * 200, "value": range(200)})
        backend = PolarsBackend(df)

    config = DownsampleConfig(type="downsample", fraction=0.5, seed=42)
    transform = Downsample.from_config(config)
    result, meta = transform(backend)

    assert meta == {}
    assert abs(len(result) - 100) <= 1


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_downsample_full_fraction(backend_type: str) -> None:
    """Test that fraction=1.0 keeps all rows."""
    if backend_type == "pandas":
        df = pd.DataFrame({"species": ["bee", "ant", "fly"], "value": [1, 2, 3]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"species": ["bee", "ant", "fly"], "value": [1, 2, 3]})
        backend = PolarsBackend(df)

    transform = Downsample(fraction=1.0, seed=42)
    result, _ = transform(backend)

    assert len(result) == 3


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_downsample_small_fraction(backend_type: str) -> None:
    """Test that a very small fraction still keeps at least 1 row."""
    if backend_type == "pandas":
        df = pd.DataFrame({"species": ["bee"] * 10, "value": range(10)})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"species": ["bee"] * 10, "value": range(10)})
        backend = PolarsBackend(df)

    transform = Downsample(fraction=0.01, seed=42)
    result, _ = transform(backend)

    assert len(result) >= 1


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_downsample_manual_vs_config(backend_type: str) -> None:
    """Test that manual instantiation and from_config produce the same result."""
    if backend_type == "pandas":
        df = pd.DataFrame({"species": ["bee"] * 100, "value": range(100)})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"species": ["bee"] * 100, "value": range(100)})
        backend = PolarsBackend(df)

    manual_transform = Downsample(fraction=0.5, seed=42)
    manual_result, _ = manual_transform(backend)

    config = DownsampleConfig(type="downsample", fraction=0.5, seed=42)
    config_transform = Downsample.from_config(config)
    config_result, _ = config_transform(backend)

    assert len(manual_result) == len(config_result)


def test_downsample_config_validation() -> None:
    """Test that invalid fraction values are rejected."""
    with pytest.raises(Exception):
        DownsampleConfig(type="downsample", fraction=0.0)

    with pytest.raises(Exception):
        DownsampleConfig(type="downsample", fraction=1.5)

    with pytest.raises(Exception):
        DownsampleConfig(type="downsample", fraction=-0.1)
