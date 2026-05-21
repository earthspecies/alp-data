import pandas as pd
import polars as pl
import pytest

from esp_data.backends import PandasBackend, PolarsBackend
from esp_data.transforms import Upsample, UpsampleConfig


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_upsample(backend_type: str) -> None:
    """Test that upsample produces factor * original rows."""
    if backend_type == "pandas":
        df = pd.DataFrame({"species": ["bee", "ant", "fly"], "value": [1, 2, 3]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"species": ["bee", "ant", "fly"], "value": [1, 2, 3]})
        backend = PolarsBackend(df)

    config = UpsampleConfig(type="upsample", factor=3)
    transform = Upsample.from_config(config)
    result, meta = transform(backend)

    assert meta == {}
    assert len(result) == 9


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_upsample_preserves_rows(backend_type: str) -> None:
    """Test that all original rows appear in the upsampled result."""
    if backend_type == "pandas":
        df = pd.DataFrame({"species": ["bee", "ant"], "value": [10, 20]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"species": ["bee", "ant"], "value": [10, 20]})
        backend = PolarsBackend(df)

    transform = Upsample(factor=2)
    result, _ = transform(backend)

    if backend_type == "pandas":
        result_df = result.unwrap
        assert set(result_df["species"].tolist()) == {"bee", "ant"}
        assert sorted(result_df["value"].tolist()) == [10, 10, 20, 20]
    else:
        result_df = result.unwrap
        if isinstance(result_df, pl.LazyFrame):
            result_df = result_df.collect()
        assert set(result_df["species"].to_list()) == {"bee", "ant"}
        assert sorted(result_df["value"].to_list()) == [10, 10, 20, 20]


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_upsample_manual_vs_config(backend_type: str) -> None:
    """Test that manual instantiation and from_config produce the same result."""
    if backend_type == "pandas":
        df = pd.DataFrame({"species": ["bee", "ant"], "value": [1, 2]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"species": ["bee", "ant"], "value": [1, 2]})
        backend = PolarsBackend(df)

    manual_transform = Upsample(factor=4)
    manual_result, _ = manual_transform(backend)

    config = UpsampleConfig(type="upsample", factor=4)
    config_transform = Upsample.from_config(config)
    config_result, _ = config_transform(backend)

    assert len(manual_result) == len(config_result) == 8


def test_upsample_config_validation() -> None:
    """Test that invalid factor values are rejected."""
    with pytest.raises(Exception):
        UpsampleConfig(type="upsample", factor=1)

    with pytest.raises(Exception):
        UpsampleConfig(type="upsample", factor=0)

    with pytest.raises(Exception):
        UpsampleConfig(type="upsample", factor=-1)
