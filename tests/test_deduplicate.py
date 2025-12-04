import pandas as pd
import polars as pl
import pytest

from esp_data.backends import PandasBackend, PolarsBackend
from esp_data.transforms import Deduplicate


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_deduplicate(backend_type: str) -> None:
    """Test deduplication with both pandas and polars backends."""
    if backend_type == "pandas":
        df = pd.DataFrame(
            {"species": ["bee", "bee", "butterfly", "bee"], "count": [10, 10, 5, 10]}
        )
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(
            {"species": ["bee", "bee", "butterfly", "bee"], "count": [10, 10, 5, 10]}
        )
        backend = PolarsBackend(df)

    transform = Deduplicate(subset=["species"], keep_first=True)
    deduplicated_backend, _ = transform(backend)

    if backend_type == "pandas":
        expected_df = pd.DataFrame({"species": ["bee", "butterfly"], "count": [10, 5]})
        result = deduplicated_backend.unwrap.reset_index(drop=True)
        pd.testing.assert_frame_equal(result, expected_df)
    else:
        expected_df = pl.DataFrame({"species": ["bee", "butterfly"], "count": [10, 5]})
        result = deduplicated_backend.unwrap
        assert sorted(result["species"].to_list()) == sorted(
            expected_df["species"].to_list()
        )
        assert sorted(result["count"].to_list()) == sorted(
            expected_df["count"].to_list()
        )
