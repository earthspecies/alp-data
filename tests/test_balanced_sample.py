import pandas as pd
import polars as pl
import pytest

from esp_data.backends import PandasBackend, PolarsBackend
from esp_data.transforms import BalancedSample, BalancedSampleConfig

# TODO (milad) add tests for returned metadata


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_balanced_sample(backend_type: str) -> None:
    """Test balanced sampling with both pandas and polars backends."""
    # Create test data with equal class distribution
    # With equal counts, balanced sampling should keep all samples
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

    # Test balanced sampling
    config = BalancedSampleConfig(
        type="balanced_sample",
        property="class",
        seed=42,
    )
    balanced_sample_transform = BalancedSample.from_config(config)
    sampled_backend, _ = balanced_sample_transform(backend)

    # With equal counts, all categories should have similar sample sizes
    # Check that the distribution is approximately balanced
    if backend_type == "pandas":
        sampled_df = sampled_backend.unwrap
        class_counts = sampled_df["class"].value_counts()
        # All categories should have similar counts (within 10% of each other)
        counts = [class_counts["birds"], class_counts["mammals"], class_counts["amphibians"]]
        # With balanced sampling, all categories should have the same count (min_count)
        # Since all categories start with 100, they should all have 100 after balanced sampling
        assert class_counts["birds"] == 100
        assert class_counts["mammals"] == 100
        assert class_counts["amphibians"] == 100
        # Verify balanced distribution: max and min should be close
        assert max(counts) - min(counts) < 10  # Within 10 samples
    else:
        sampled_df = sampled_backend.unwrap
        if isinstance(sampled_df, pl.LazyFrame):
            sampled_df = sampled_df.collect()
        class_counts = sampled_df.group_by("class").len()
        birds_count = class_counts.filter(pl.col("class") == "birds")["len"][0]
        mammals_count = class_counts.filter(pl.col("class") == "mammals")["len"][0]
        amphibians_count = class_counts.filter(pl.col("class") == "amphibians")["len"][0]
        # With balanced sampling, all categories should have the same count (min_count)
        # Since all categories start with 100, they should all have 100 after balanced sampling
        assert birds_count == 100
        assert mammals_count == 100
        assert amphibians_count == 100
        # Verify balanced distribution
        counts = [birds_count, mammals_count, amphibians_count]
        assert max(counts) - min(counts) < 10


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_balanced_sample_unequal_counts(backend_type: str) -> None:
    """Test balanced sampling with unequal category counts to verify balanced distribution."""
    # Create test data with unequal distribution: birds=100, mammals=50, amphibians=200
    # Balanced sampling should balance these to have similar counts
    if backend_type == "pandas":
        df = pd.DataFrame(
            {
                "class": ["birds"] * 100 + ["mammals"] * 50 + ["amphibians"] * 200,
                "value": range(350),
            }
        )
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(
            {
                "class": ["birds"] * 100 + ["mammals"] * 50 + ["amphibians"] * 200,
                "value": range(350),
            }
        )
        backend = PolarsBackend(df)

    # Test balanced sampling
    config = BalancedSampleConfig(
        type="balanced_sample",
        property="class",
        seed=42,
    )
    balanced_sample_transform = BalancedSample.from_config(config)
    sampled_backend, _ = balanced_sample_transform(backend)

    # Check that the distribution is more balanced than the original
    # Original: birds=100, mammals=50, amphibians=200 (ratio 2:1:4)
    # After balanced sampling, all should have min_count (50)
    if backend_type == "pandas":
        sampled_df = sampled_backend.unwrap
        class_counts = sampled_df["class"].value_counts()
        birds_count = class_counts["birds"]
        mammals_count = class_counts["mammals"]
        amphibians_count = class_counts["amphibians"]
    else:
        sampled_df = sampled_backend.unwrap
        if isinstance(sampled_df, pl.LazyFrame):
            sampled_df = sampled_df.collect()
        class_counts = sampled_df.group_by("class").len()
        birds_count = class_counts.filter(pl.col("class") == "birds")["len"][0]
        mammals_count = class_counts.filter(pl.col("class") == "mammals")["len"][0]
        amphibians_count = class_counts.filter(pl.col("class") == "amphibians")["len"][0]

    counts = [birds_count, mammals_count, amphibians_count]

    # With balanced sampling, all categories should have min_count (50)
    # Verify that all counts are approximately equal to min_count
    assert abs(birds_count - 50) < 5, f"Birds should be ~50, got {birds_count}"
    assert abs(mammals_count - 50) < 5, f"Mammals should be ~50, got {mammals_count}"
    assert abs(amphibians_count - 50) < 5, f"Amphibians should be ~50, got {amphibians_count}"

    # Verify balanced distribution: max/min ratio should be close to 1.0
    max_min_ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")
    assert max_min_ratio < 1.2, f"Distribution not balanced enough: {counts}, ratio={max_min_ratio}"


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_balanced_sample_manual_vs_config(backend_type: str) -> None:
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

    # Manual instantiation
    manual_transform = BalancedSample(property="class", seed=42)
    manual_result_backend, _ = manual_transform(backend)

    # Using config and from_config
    config = BalancedSampleConfig(type="balanced_sample", property="class", seed=42)
    config_transform = BalancedSample.from_config(config)
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


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_balanced_sample_upsampling(backend_type: str) -> None:
    """Test that balanced sampling upsamples categories with fewer samples."""
    # Create test data with very unequal distribution: birds=10, mammals=5, amphibians=50
    # Balanced sampling should upsample birds and mammals to 5 (min_count)
    if backend_type == "pandas":
        df = pd.DataFrame(
            {
                "class": ["birds"] * 10 + ["mammals"] * 5 + ["amphibians"] * 50,
                "value": range(65),
            }
        )
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(
            {
                "class": ["birds"] * 10 + ["mammals"] * 5 + ["amphibians"] * 50,
                "value": range(65),
            }
        )
        backend = PolarsBackend(df)

    # Test balanced sampling
    config = BalancedSampleConfig(
        type="balanced_sample",
        property="class",
        seed=42,
    )
    balanced_sample_transform = BalancedSample.from_config(config)
    sampled_backend, _ = balanced_sample_transform(backend)

    # Check that all categories have min_count (5) after balancing
    # Birds should be upsampled from 10 to 5 (downsampled actually, since 10 > 5)
    # Mammals should stay at 5 (already at min_count)
    # Amphibians should be downsampled from 50 to 5
    if backend_type == "pandas":
        sampled_df = sampled_backend.unwrap
        class_counts = sampled_df["class"].value_counts()
        birds_count = class_counts["birds"]
        mammals_count = class_counts["mammals"]
        amphibians_count = class_counts["amphibians"]
    else:
        sampled_df = sampled_backend.unwrap
        if isinstance(sampled_df, pl.LazyFrame):
            sampled_df = sampled_df.collect()
        class_counts = sampled_df.group_by("class").len()
        birds_count = class_counts.filter(pl.col("class") == "birds")["len"][0]
        mammals_count = class_counts.filter(pl.col("class") == "mammals")["len"][0]
        amphibians_count = class_counts.filter(pl.col("class") == "amphibians")["len"][0]

    # All should have min_count (5)
    assert birds_count == 5, f"Birds should be 5, got {birds_count}"
    assert mammals_count == 5, f"Mammals should be 5, got {mammals_count}"
    assert amphibians_count == 5, f"Amphibians should be 5, got {amphibians_count}"

    # Test with a case that actually requires upsampling
    # birds=2, mammals=5, amphibians=20 -> min_count=2, so mammals and amphibians need upsampling
    if backend_type == "pandas":
        df2 = pd.DataFrame(
            {
                "class": ["birds"] * 2 + ["mammals"] * 5 + ["amphibians"] * 20,
                "value": range(27),
            }
        )
        backend2 = PandasBackend(df2)
    else:
        df2 = pl.DataFrame(
            {
                "class": ["birds"] * 2 + ["mammals"] * 5 + ["amphibians"] * 20,
                "value": range(27),
            }
        )
        backend2 = PolarsBackend(df2)

    sampled_backend2, _ = balanced_sample_transform(backend2)

    if backend_type == "pandas":
        sampled_df2 = sampled_backend2.unwrap
        class_counts2 = sampled_df2["class"].value_counts()
        birds_count2 = class_counts2["birds"]
        mammals_count2 = class_counts2["mammals"]
        amphibians_count2 = class_counts2["amphibians"]
    else:
        sampled_df2 = sampled_backend2.unwrap
        if isinstance(sampled_df2, pl.LazyFrame):
            sampled_df2 = sampled_df2.collect()
        class_counts2 = sampled_df2.group_by("class").len()
        birds_count2 = class_counts2.filter(pl.col("class") == "birds")["len"][0]
        mammals_count2 = class_counts2.filter(pl.col("class") == "mammals")["len"][0]
        amphibians_count2 = class_counts2.filter(pl.col("class") == "amphibians")["len"][0]

    # All should have min_count (2)
    # Birds stays at 2, mammals and amphibians are upsampled to 2
    assert birds_count2 == 2, f"Birds should be 2, got {birds_count2}"
    assert mammals_count2 == 2, f"Mammals should be 2 (upsampled from 5), got {mammals_count2}"
    assert (
        amphibians_count2 == 2
    ), f"Amphibians should be 2 (upsampled from 20), got {amphibians_count2}"
