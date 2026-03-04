import pandas as pd
import polars as pl
import pytest

from esp_data.backends import PandasBackend, PolarsBackend
from esp_data.transforms import LongTailUpsample, LongTailUpsampleConfig


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_above_threshold_untouched(backend_type: str) -> None:
    """Categories at or above sufficient_threshold keep their original count."""
    if backend_type == "pandas":
        df = pd.DataFrame({"class": ["common"] * 500, "value": range(500)})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"class": ["common"] * 500, "value": range(500)})
        backend = PolarsBackend(df)

    config = LongTailUpsampleConfig(
        type="long_tail_upsample",
        property="class",
        sufficient_threshold=300,
        max_repeats=5,
        seed=42,
    )
    transform = LongTailUpsample.from_config(config)
    result, _ = transform(backend)

    counts = result.histogram("class")
    assert counts["common"] == 500


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_below_threshold_upsampled_to_threshold(backend_type: str) -> None:
    """Categories below threshold are upsampled to it when max_repeats allows."""
    # 100 examples * max_repeats=5 = 500 >= threshold=300, so target = 300
    if backend_type == "pandas":
        df = pd.DataFrame({"class": ["moderate"] * 100, "value": range(100)})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"class": ["moderate"] * 100, "value": range(100)})
        backend = PolarsBackend(df)

    config = LongTailUpsampleConfig(
        type="long_tail_upsample",
        property="class",
        sufficient_threshold=300,
        max_repeats=5,
        seed=42,
    )
    transform = LongTailUpsample.from_config(config)
    result, _ = transform(backend)

    counts = result.histogram("class")
    assert counts["moderate"] == 300


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_rare_capped_by_max_repeats(backend_type: str) -> None:
    """Very rare categories are capped by max_repeats, not reaching threshold."""
    # 3 examples * max_repeats=5 = 15 < threshold=300, so target = 15
    if backend_type == "pandas":
        df = pd.DataFrame({"class": ["rare"] * 3, "value": range(3)})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"class": ["rare"] * 3, "value": range(3)})
        backend = PolarsBackend(df)

    config = LongTailUpsampleConfig(
        type="long_tail_upsample",
        property="class",
        sufficient_threshold=300,
        max_repeats=5,
        seed=42,
    )
    transform = LongTailUpsample.from_config(config)
    result, _ = transform(backend)

    counts = result.histogram("class")
    assert counts["rare"] == 15


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_long_tail_distribution(backend_type: str) -> None:
    """End-to-end test with a realistic long-tail distribution."""
    # common=500, moderate=100, uncommon=40, rare=3
    # threshold=300, max_repeats=5
    # Expected: common=500, moderate=300, uncommon=200, rare=15
    classes = ["common"] * 500 + ["moderate"] * 100 + ["uncommon"] * 40 + ["rare"] * 3
    values = list(range(len(classes)))

    if backend_type == "pandas":
        df = pd.DataFrame({"class": classes, "value": values})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"class": classes, "value": values})
        backend = PolarsBackend(df)

    config = LongTailUpsampleConfig(
        type="long_tail_upsample",
        property="class",
        sufficient_threshold=300,
        max_repeats=5,
        seed=42,
    )
    transform = LongTailUpsample.from_config(config)
    result, meta = transform(backend)

    counts = result.histogram("class")
    assert counts["common"] == 500  # untouched
    assert counts["moderate"] == 300  # min(300, 100*5=500) = 300
    assert counts["uncommon"] == 200  # min(300, 40*5=200) = 200
    assert counts["rare"] == 15  # min(300, 3*5=15) = 15

    assert meta["histogram_before"] == {"common": 500, "moderate": 100, "uncommon": 40, "rare": 3}
    assert meta["histogram_after"] == counts


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_max_repeats_one_leaves_data_unchanged(backend_type: str) -> None:
    """With max_repeats=1, no upsampling occurs regardless of threshold."""
    classes = ["common"] * 500 + ["rare"] * 3
    values = list(range(len(classes)))

    if backend_type == "pandas":
        df = pd.DataFrame({"class": classes, "value": values})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"class": classes, "value": values})
        backend = PolarsBackend(df)

    config = LongTailUpsampleConfig(
        type="long_tail_upsample",
        property="class",
        sufficient_threshold=300,
        max_repeats=1,
        seed=42,
    )
    transform = LongTailUpsample.from_config(config)
    result, meta = transform(backend)

    counts = result.histogram("class")
    assert counts["common"] == 500
    assert counts["rare"] == 3
    assert meta["histogram_before"] == meta["histogram_after"]


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_missing_property_raises_key_error(backend_type: str) -> None:
    """KeyError is raised when the property column does not exist."""
    if backend_type == "pandas":
        df = pd.DataFrame({"species": ["a", "b"], "value": [1, 2]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"species": ["a", "b"], "value": [1, 2]})
        backend = PolarsBackend(df)

    transform = LongTailUpsample(
        property="class",
        sufficient_threshold=10,
        max_repeats=3,
    )
    with pytest.raises(KeyError, match="class"):
        transform(backend)


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_manual_vs_config(backend_type: str) -> None:
    """Manual instantiation and from_config produce the same per-class counts."""
    classes = ["a"] * 50 + ["b"] * 10 + ["c"] * 2
    values = list(range(len(classes)))

    if backend_type == "pandas":
        df = pd.DataFrame({"class": classes, "value": values})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"class": classes, "value": values})
        backend = PolarsBackend(df)

    manual = LongTailUpsample(
        property="class",
        sufficient_threshold=30,
        max_repeats=4,
        seed=42,
    )
    manual_result, manual_meta = manual(backend)

    config = LongTailUpsampleConfig(
        type="long_tail_upsample",
        property="class",
        sufficient_threshold=30,
        max_repeats=4,
        seed=42,
    )
    config_result, config_meta = LongTailUpsample.from_config(config)(backend)

    manual_counts = manual_result.histogram("class")
    config_counts = config_result.histogram("class")
    assert manual_counts == config_counts
    assert len(manual_result) == len(config_result)
    assert manual_meta == config_meta


def test_config_validation_sufficient_threshold() -> None:
    """sufficient_threshold must be >= 1."""
    with pytest.raises(ValueError, match="sufficient_threshold"):
        LongTailUpsampleConfig(
            type="long_tail_upsample",
            property="class",
            sufficient_threshold=0,
            max_repeats=3,
        )


def test_config_validation_max_repeats() -> None:
    """max_repeats must be >= 1."""
    with pytest.raises(ValueError, match="max_repeats"):
        LongTailUpsampleConfig(
            type="long_tail_upsample",
            property="class",
            sufficient_threshold=100,
            max_repeats=0,
        )
