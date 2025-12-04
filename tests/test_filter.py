from typing import Callable, Literal

import pandas as pd
import polars as pl
import pytest

from esp_data.backends import PandasBackend, PolarsBackend
from esp_data.transforms import Filter, FilterConfig, transform_from_config

# TODO (milad) add tests for returned metadata


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_filter(backend_type: str) -> None:
    """Test filtering with both pandas and polars backends."""
    # Create test data
    if backend_type == "pandas":
        df = pd.DataFrame(
            {
                "source": ["xeno-canto", "iNaturalist", "Watkins", "other"],
                "class": ["birds", "mammals", "amphibians", "reptiles"],
                "value": [1, 2, 3, 4],
            }
        )
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(
            {
                "source": ["xeno-canto", "iNaturalist", "Watkins", "other"],
                "class": ["birds", "mammals", "amphibians", "reptiles"],
                "value": [1, 2, 3, 4],
            }
        )
        backend = PolarsBackend(df)

    # Test include operation
    config = FilterConfig(
        type="filter",
        property="source",
        values=["xeno-canto", "iNaturalist"],
        mode="include",
    )
    filter_transform = Filter.from_config(config)
    filtered_backend, _ = filter_transform(backend)

    assert len(filtered_backend) == 2
    if backend_type == "pandas":
        filtered_df = filtered_backend.unwrap
        assert set(filtered_df["source"]) == {"xeno-canto", "iNaturalist"}
    else:
        filtered_df = filtered_backend.unwrap.collect() if isinstance(filtered_backend.unwrap, pl.LazyFrame) else filtered_backend.unwrap
        assert set(filtered_df["source"].to_list()) == {"xeno-canto", "iNaturalist"}

    # Test exclude operation
    config = FilterConfig(
        type="filter",
        property="source",
        values=["xeno-canto", "iNaturalist"],
        mode="exclude",
    )
    filter_transform = Filter.from_config(config)
    filtered_backend, _ = filter_transform(backend)

    assert len(filtered_backend) == 2
    if backend_type == "pandas":
        filtered_df = filtered_backend.unwrap
        assert set(filtered_df["source"]) == {"Watkins", "other"}
    else:
        filtered_df = filtered_backend.unwrap.collect() if isinstance(filtered_backend.unwrap, pl.LazyFrame) else filtered_backend.unwrap
        assert set(filtered_df["source"].to_list()) == {"Watkins", "other"}


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_filter_transform_methods_equivalence(backend_type: str) -> None:
    """
    Test that transform_from_config, manual, and from_config produce the same result for
    Filter (include/exclude) with both backends.
    """

    def _get_sorted_result(backend, transform: Callable):
        result_backend, _ = transform(backend)
        if backend_type == "pandas":
            result = result_backend.unwrap
            return result.sort_values(by=["source", "value"]).reset_index(drop=True)
        else:
            result = result_backend.unwrap
            if isinstance(result, pl.LazyFrame):
                result = result.collect()
            return result.sort(by=["source", "value"])

    def _all_filter_methods(
        backend,
        property: str,
        values: list[str],
        mode: Literal["include", "exclude"],
    ) -> list:
        manual = Filter(property=property, values=values, mode=mode)
        config = FilterConfig(
            type="filter", property=property, values=values, mode=mode
        )
        from_config_transform = Filter.from_config(config)
        from_registry_transform = transform_from_config(config)
        return [
            _get_sorted_result(backend, manual),
            _get_sorted_result(backend, from_config_transform),
            _get_sorted_result(backend, from_registry_transform),
        ]

    if backend_type == "pandas":
        df = pd.DataFrame(
            {
                "source": ["xeno-canto", "iNaturalist", "Watkins", "other"],
                "class": ["birds", "mammals", "amphibians", "reptiles"],
                "value": [1, 2, 3, 4],
            }
        )
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(
            {
                "source": ["xeno-canto", "iNaturalist", "Watkins", "other"],
                "class": ["birds", "mammals", "amphibians", "reptiles"],
                "value": [1, 2, 3, 4],
            }
        )
        backend = PolarsBackend(df)

    values = ["xeno-canto", "iNaturalist"]
    for mode in ["include", "exclude"]:
        results = _all_filter_methods(backend, property="source", values=values, mode=mode)
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if backend_type == "pandas":
                    pd.testing.assert_frame_equal(results[i], results[j])
                else:
                    assert results[i].equals(results[j])
