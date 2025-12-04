import pandas as pd
import polars as pl
import pytest

from esp_data.backends import PandasBackend, PolarsBackend
from esp_data.transforms import Subsample, SubsampleConfig

# TODO (milad) add tests for returned metadata


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_subsample(backend_type: str) -> None:
    """Test subsampling with both pandas and polars backends."""
    # Create test data with known class distribution
    if backend_type == "pandas":
        df = pd.DataFrame(
            {
                "class": ["birds"] * 100 + ["mammals"] * 100 + ["amphibians"] * 100,
                "value": range(300),
            }
        )
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(
            {
                "class": ["birds"] * 100 + ["mammals"] * 100 + ["amphibians"] * 100,
                "value": range(300),
            }
        )
        backend = PolarsBackend(df)

    # Test subsampling with different ratios
    config = SubsampleConfig(
        type="subsample",
        property="class",
        ratios={"birds": 0.5, "mammals": 0.3, "amphibians": 0.7},
    )
    subsample_transform = Subsample.from_config(config)
    subsampled_backend, _ = subsample_transform(backend)

    # Check that the ratios are approximately correct
    if backend_type == "pandas":
        subsampled_df = subsampled_backend.unwrap
        class_counts = subsampled_df["class"].value_counts()
        assert abs(class_counts["birds"] / 100 - 0.5) < 0.1
        assert abs(class_counts["mammals"] / 100 - 0.3) < 0.1
        assert abs(class_counts["amphibians"] / 100 - 0.7) < 0.1
    else:
        subsampled_df = subsampled_backend.unwrap
        if isinstance(subsampled_df, pl.LazyFrame):
            subsampled_df = subsampled_df.collect()
        class_counts = subsampled_df.group_by("class").count()
        birds_count = class_counts.filter(pl.col("class") == "birds")["count"][0]
        mammals_count = class_counts.filter(pl.col("class") == "mammals")["count"][0]
        amphibians_count = class_counts.filter(pl.col("class") == "amphibians")["count"][0]
        assert abs(birds_count / 100 - 0.5) < 0.1
        assert abs(mammals_count / 100 - 0.3) < 0.1
        assert abs(amphibians_count / 100 - 0.7) < 0.1


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_subsample_with_other(backend_type: str) -> None:
    """Test subsampling with 'other' category."""
    # Create test data
    if backend_type == "pandas":
        df = pd.DataFrame(
            {
                "class": ["birds"] * 100 + ["mammals"] * 100 + ["amphibians"] * 100,
                "value": range(300),
            }
        )
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(
            {
                "class": ["birds"] * 100 + ["mammals"] * 100 + ["amphibians"] * 100,
                "value": range(300),
            }
        )
        backend = PolarsBackend(df)

    # Test with 'other' class
    config = SubsampleConfig(
        type="subsample",
        property="class",
        ratios={"birds": 0.5, "other": 0.2},
    )
    subsample_transform = Subsample.from_config(config)
    subsampled_backend, _ = subsample_transform(backend)

    # Check that 'other' class (mammals + amphibians) is subsampled correctly
    if backend_type == "pandas":
        subsampled_df = subsampled_backend.unwrap
        other_count = len(
            subsampled_df[subsampled_df["class"].isin(["mammals", "amphibians"])]
        )
        assert abs(other_count / 200 - 0.2) < 0.1
    else:
        subsampled_df = subsampled_backend.unwrap
        if isinstance(subsampled_df, pl.LazyFrame):
            subsampled_df = subsampled_df.collect()
        other_count = len(
            subsampled_df.filter(pl.col("class").is_in(["mammals", "amphibians"]))
        )
        assert abs(other_count / 200 - 0.2) < 0.1


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_subsample_manual_vs_config(backend_type: str) -> None:
    """Test that manual instantiation and from_config produce the same result."""
    if backend_type == "pandas":
        df = pd.DataFrame(
            {
                "class": ["birds"] * 100 + ["mammals"] * 100 + ["amphibians"] * 100,
                "value": range(300),
            }
        )
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(
            {
                "class": ["birds"] * 100 + ["mammals"] * 100 + ["amphibians"] * 100,
                "value": range(300),
            }
        )
        backend = PolarsBackend(df)

    ratios = {"birds": 0.5, "mammals": 0.3, "amphibians": 0.7}

    # Manual instantiation
    manual_transform = Subsample(property="class", ratios=ratios, seed=42)
    manual_result_backend, _ = manual_transform(backend)

    # Using config and from_config
    config = SubsampleConfig(type="subsample", property="class", ratios=ratios, seed=42)
    config_transform = Subsample.from_config(config)
    config_result_backend, _ = config_transform(backend)

    # Get underlying dataframes
    if backend_type == "pandas":
        manual_result = manual_result_backend.unwrap
        config_result = config_result_backend.unwrap

        # Sort and reset index for comparison
        manual_sorted = manual_result.sort_values(by=["class", "value"]).reset_index(
            drop=True
        )
        config_sorted = config_result.sort_values(by=["class", "value"]).reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(manual_sorted, config_sorted)
    else:
        manual_result = manual_result_backend.unwrap
        config_result = config_result_backend.unwrap

        if isinstance(manual_result, pl.LazyFrame):
            manual_result = manual_result.collect()
        if isinstance(config_result, pl.LazyFrame):
            config_result = config_result.collect()

        # Sort for comparison
        manual_sorted = manual_result.sort(by=["class", "value"])
        config_sorted = config_result.sort(by=["class", "value"])

        assert manual_sorted.equals(config_sorted)
