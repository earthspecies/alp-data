import pandas as pd
import polars as pl
import pytest

from esp_data.backends import PandasBackend, PolarsBackend
from esp_data.transforms import BalancedSample, BalancedSampleConfig


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
        strategy="min",
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
def test_balanced_sample_strategy_max(backend_type: str) -> None:
    """Test balanced sampling with max strategy."""
    # Create test data: birds=50, mammals=100, amphibians=200
    # Max strategy should upsample all to 200
    if backend_type == "pandas":
        df = pd.DataFrame(
            {
                "class": ["birds"] * 50 + ["mammals"] * 100 + ["amphibians"] * 200,
                "value": range(350),
            }
        )
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(
            {
                "class": ["birds"] * 50 + ["mammals"] * 100 + ["amphibians"] * 200,
                "value": range(350),
            }
        )
        backend = PolarsBackend(df)

    config = BalancedSampleConfig(
        type="balanced_sample",
        property="class",
        strategy="max",
        seed=42,
    )
    balanced_sample_transform = BalancedSample.from_config(config)
    sampled_backend, _ = balanced_sample_transform(backend)

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

    # All should have max_count (200)
    assert birds_count == 200, f"Birds should be 200, got {birds_count}"
    assert mammals_count == 200, f"Mammals should be 200, got {mammals_count}"
    assert amphibians_count == 200, f"Amphibians should be 200, got {amphibians_count}"


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_balanced_sample_strategy_median(backend_type: str) -> None:
    """Test balanced sampling with median strategy."""
    # Create test data: birds=50, mammals=100, amphibians=200
    # Median of [50, 100, 200] = 100
    if backend_type == "pandas":
        df = pd.DataFrame(
            {
                "class": ["birds"] * 50 + ["mammals"] * 100 + ["amphibians"] * 200,
                "value": range(350),
            }
        )
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(
            {
                "class": ["birds"] * 50 + ["mammals"] * 100 + ["amphibians"] * 200,
                "value": range(350),
            }
        )
        backend = PolarsBackend(df)

    config = BalancedSampleConfig(
        type="balanced_sample",
        property="class",
        strategy="median",
        seed=42,
    )
    balanced_sample_transform = BalancedSample.from_config(config)
    sampled_backend, _ = balanced_sample_transform(backend)

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

    # All should have median_count (100)
    assert birds_count == 100, f"Birds should be 100, got {birds_count}"
    assert mammals_count == 100, f"Mammals should be 100, got {mammals_count}"
    assert amphibians_count == 100, f"Amphibians should be 100, got {amphibians_count}"


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_balanced_sample_strategy_mean(backend_type: str) -> None:
    """Test balanced sampling with mean strategy."""
    # Create test data: birds=60, mammals=90, amphibians=150
    # Mean of [60, 90, 150] = 100
    if backend_type == "pandas":
        df = pd.DataFrame(
            {
                "class": ["birds"] * 60 + ["mammals"] * 90 + ["amphibians"] * 150,
                "value": range(300),
            }
        )
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(
            {
                "class": ["birds"] * 60 + ["mammals"] * 90 + ["amphibians"] * 150,
                "value": range(300),
            }
        )
        backend = PolarsBackend(df)

    config = BalancedSampleConfig(
        type="balanced_sample",
        property="class",
        strategy="mean",
        seed=42,
    )
    balanced_sample_transform = BalancedSample.from_config(config)
    sampled_backend, _ = balanced_sample_transform(backend)

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

    # All should have mean_count (100)
    assert birds_count == 100, f"Birds should be 100, got {birds_count}"
    assert mammals_count == 100, f"Mammals should be 100, got {mammals_count}"
    assert amphibians_count == 100, f"Amphibians should be 100, got {amphibians_count}"


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_balanced_sample_strategy_median_with_range(backend_type: str) -> None:
    """Test balanced sampling with median_with_range strategy."""
    # Create test data: birds=50, mammals=100, amphibians=200
    # Median = 100, with range_fraction=0.2, bounds are [80, 120]
    # birds (50) -> 80 (below lower bound, upsample)
    # mammals (100) -> 100 (within range, keep)
    # amphibians (200) -> 120 (above upper bound, downsample)
    if backend_type == "pandas":
        df = pd.DataFrame(
            {
                "class": ["birds"] * 50 + ["mammals"] * 100 + ["amphibians"] * 200,
                "value": range(350),
            }
        )
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(
            {
                "class": ["birds"] * 50 + ["mammals"] * 100 + ["amphibians"] * 200,
                "value": range(350),
            }
        )
        backend = PolarsBackend(df)

    config = BalancedSampleConfig(
        type="balanced_sample",
        property="class",
        strategy="median_with_range",
        range_fraction=0.2,
        seed=42,
    )
    balanced_sample_transform = BalancedSample.from_config(config)
    sampled_backend, _ = balanced_sample_transform(backend)

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

    # birds should be clamped to lower_bound (80)
    assert birds_count == 80, f"Birds should be 80, got {birds_count}"
    # mammals should stay at 100 (within range)
    assert mammals_count == 100, f"Mammals should be 100, got {mammals_count}"
    # amphibians should be clamped to upper_bound (120)
    assert amphibians_count == 120, f"Amphibians should be 120, got {amphibians_count}"


def test_balanced_sample_range_fraction_validation() -> None:
    """Test that range_fraction is validated for median_with_range strategy."""
    # Valid range_fraction
    config = BalancedSampleConfig(
        type="balanced_sample",
        property="class",
        strategy="median_with_range",
        range_fraction=0.5,
    )
    assert config.range_fraction == 0.5

    # Invalid range_fraction (>= 1)
    with pytest.raises(ValueError, match="range_fraction must be between 0 and 1"):
        BalancedSampleConfig(
            type="balanced_sample",
            property="class",
            strategy="median_with_range",
            range_fraction=1.0,
        )

    # Invalid range_fraction (<= 0)
    with pytest.raises(ValueError, match="range_fraction must be between 0 and 1"):
        BalancedSampleConfig(
            type="balanced_sample",
            property="class",
            strategy="median_with_range",
            range_fraction=0.0,
        )

    # range_fraction not validated for other strategies
    config2 = BalancedSampleConfig(
        type="balanced_sample",
        property="class",
        strategy="min",
        range_fraction=2.0,  # This should be allowed for non-median_with_range
    )
    assert config2.range_fraction == 2.0
