import pandas as pd
import polars as pl
import pytest

from esp_data.backends import PandasBackend, PolarsBackend
from esp_data.transforms import UniformSample, UniformSampleConfig

# TODO (milad) add tests for returned metadata


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_uniform_sample(backend_type: str) -> None:
    """Test uniform sampling with both pandas and polars backends."""
    # Create test data with equal class distribution
    # With equal counts, inverse probabilities should result in equal ratios
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

    # Test uniform sampling
    config = UniformSampleConfig(
        type="uniform_sample",
        property="class",
        ratio=0.5,
        seed=42,
    )
    uniform_sample_transform = UniformSample.from_config(config)
    sampled_backend, _ = uniform_sample_transform(backend)

    # With equal counts, all categories should have similar sample sizes
    # Check that the distribution is approximately uniform
    if backend_type == "pandas":
        sampled_df = sampled_backend.unwrap
        class_counts = sampled_df["class"].value_counts()
        # All categories should have similar counts (within 10% of each other)
        counts = [class_counts["birds"], class_counts["mammals"], class_counts["amphibians"]]
        assert abs(class_counts["birds"] / 100 - 0.5) < 0.1
        assert abs(class_counts["mammals"] / 100 - 0.5) < 0.1
        assert abs(class_counts["amphibians"] / 100 - 0.5) < 0.1
        # Verify uniform distribution: max and min should be close
        assert max(counts) - min(counts) < 10  # Within 10 samples
    else:
        sampled_df = sampled_backend.unwrap
        if isinstance(sampled_df, pl.LazyFrame):
            sampled_df = sampled_df.collect()
        class_counts = sampled_df.group_by("class").len()
        birds_count = class_counts.filter(pl.col("class") == "birds")["len"][0]
        mammals_count = class_counts.filter(pl.col("class") == "mammals")["len"][0]
        amphibians_count = class_counts.filter(pl.col("class") == "amphibians")["len"][0]
        assert abs(birds_count / 100 - 0.5) < 0.1
        assert abs(mammals_count / 100 - 0.5) < 0.1
        assert abs(amphibians_count / 100 - 0.5) < 0.1
        # Verify uniform distribution
        counts = [birds_count, mammals_count, amphibians_count]
        assert max(counts) - min(counts) < 10


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_uniform_sample_unequal_counts(backend_type: str) -> None:
    """Test uniform sampling with unequal category counts to verify uniform distribution."""
    # Create test data with unequal distribution: birds=100, mammals=50, amphibians=200
    # Uniform sampling should balance these to have similar counts
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

    # Test uniform sampling with ratio=0.5
    config = UniformSampleConfig(
        type="uniform_sample",
        property="class",
        ratio=0.5,
        seed=42,
    )
    uniform_sample_transform = UniformSample.from_config(config)
    sampled_backend, _ = uniform_sample_transform(backend)

    # Check that the distribution is more uniform than the original
    # Original: birds=100, mammals=50, amphibians=200 (ratio 2:1:4)
    # After uniform sampling, counts should be more balanced
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

    # Verify that mammals (originally smallest) is not the smallest anymore
    # or at least the difference is reduced
    # With inverse probabilities, mammals should be sampled at higher rate
    # The distribution should be more balanced than 2:1:4
    # Check that max/min ratio is less than original (which was 200/50 = 4.0)
    max_min_ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")
    assert max_min_ratio < 3.0, f"Distribution not uniform enough: {counts}, ratio={max_min_ratio}"

    # Verify mammals count is closer to others (it was 50, should be sampled more)
    # With ratio=0.5 and min_count=50, target is 25 per category
    # But inverse probabilities will adjust this
    assert mammals_count >= 20, "Mammals should be sampled (inverse probability should help)"


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_uniform_sample_manual_vs_config(backend_type: str) -> None:
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

    ratio = 0.5

    # Manual instantiation
    manual_transform = UniformSample(property="class", ratio=ratio, seed=42)
    manual_result_backend, _ = manual_transform(backend)

    # Using config and from_config
    config = UniformSampleConfig(type="uniform_sample", property="class", ratio=ratio, seed=42)
    config_transform = UniformSample.from_config(config)
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
