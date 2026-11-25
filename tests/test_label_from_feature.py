import pandas as pd
import polars as pl
import pytest

from alp_data.backends import PandasBackend, PolarsBackend
from alp_data.transforms import LabelFromFeature


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
@pytest.mark.parametrize(
    "feature, data, expected_labels, expected_map",
    [
        # Single column, all strings
        (
            "col1",
            {"col1": ["banana", "apple", "banana", "orange"]},
            [1, 0, 1, 2],
            {"apple": 0, "banana": 1, "orange": 2},
        ),
        # Single column, with NaN
        (
            "col1",
            {"col1": ["banana", "apple", None, "orange"]},
            [1, 0, 2],
            {"apple": 0, "banana": 1, "orange": 2},
        ),
    ],
)
def test_label_from_feature(
    backend_type: str,
    feature: str,
    data: dict,
    expected_labels: list[int],
    expected_map: dict[str, int],
) -> None:
    """Test label from feature with both pandas and polars backends."""
    if backend_type == "pandas":
        df = pd.DataFrame(data)
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame(data)
        backend = PolarsBackend(df)

    t = LabelFromFeature(feature=feature)
    result_backend, meta = t(backend)

    if backend_type == "pandas":
        result_df = result_backend.unwrap
        assert result_df["label"].tolist() == expected_labels
    else:
        result_df = result_backend.unwrap
        if isinstance(result_df, pl.LazyFrame):
            result_df = result_df.collect()
        assert result_df["label"].to_list() == expected_labels

    assert meta["label_map"] == expected_map
    assert meta["num_classes"] == len(expected_map)


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_label_from_feature_with_label_map(backend_type: str) -> None:
    """Test label from feature with provided label map."""
    if backend_type == "pandas":
        df = pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"col1": ["banana", "apple", "banana", "orange"]})
        backend = PolarsBackend(df)

    label_map = {"apple": 0, "banana": 1, "orange": 2, "grape": 3}
    t = LabelFromFeature(feature="col1", label_map=label_map)
    result_backend, meta = t(backend)

    if backend_type == "pandas":
        result_df = result_backend.unwrap
        assert result_df["label"].tolist() == [1, 0, 1, 2]
    else:
        result_df = result_backend.unwrap
        if isinstance(result_df, pl.LazyFrame):
            result_df = result_df.collect()
        assert result_df["label"].to_list() == [1, 0, 1, 2]

    assert meta["label_map"] == label_map
    assert meta["num_classes"] == len(label_map)


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_label_from_feature_with_noncontiguous_indices(backend_type: str) -> None:
    """Test label from feature with non-contiguous indices."""
    if backend_type == "pandas":
        df = pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"col1": ["banana", "apple", "banana", "orange"]})
        backend = PolarsBackend(df)

    label_map = {"apple": 100, "banana": 101, "orange": 102}
    t = LabelFromFeature(feature="col1", label_map=label_map)
    result_backend, meta = t(backend)

    if backend_type == "pandas":
        result_df = result_backend.unwrap
        assert result_df["label"].tolist() == [101, 100, 101, 102]
    else:
        result_df = result_backend.unwrap
        if isinstance(result_df, pl.LazyFrame):
            result_df = result_df.collect()
        assert result_df["label"].to_list() == [101, 100, 101, 102]

    assert meta["label_map"] == label_map
    assert meta["num_classes"] == len(label_map)


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_label_from_feature_label_map_remains_none(backend_type: str) -> None:
    """Test that label_map attribute remains None after transform."""
    if backend_type == "pandas":
        df = pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"]})
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({"col1": ["banana", "apple", "banana", "orange"]})
        backend = PolarsBackend(df)

    t = LabelFromFeature(feature="col1")
    assert t.label_map is None
    result_backend, meta = t(backend)
    assert t.label_map is None
    assert sorted(meta["label_map"].keys()) == ["apple", "banana", "orange"]
    assert meta["num_classes"] == 3

    if backend_type == "pandas":
        result_df = result_backend.unwrap
        assert result_df["label"].tolist() == [1, 0, 1, 2]
    else:
        result_df = result_backend.unwrap
        if isinstance(result_df, pl.LazyFrame):
            result_df = result_df.collect()
        assert result_df["label"].to_list() == [1, 0, 1, 2]


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_label_from_feature_with_label_map_two_cols(backend_type: str) -> None:
    """Test label from feature with two columns."""
    if backend_type == "pandas":
        df = pd.DataFrame({
            "col1": ["banana", "apple", "banana", "orange"],
            "col2": ["dog", "cat", "dog", "mouse"]
        })
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({
            "col1": ["banana", "apple", "banana", "orange"],
            "col2": ["dog", "cat", "dog", "mouse"]
        })
        backend = PolarsBackend(df)

    label_map = {"apple": 0, "banana": 1, "orange": 2, "dog": 3, "cat": 4, "mouse": 5}
    t = LabelFromFeature(feature="col1", label_map=label_map)
    result_backend, meta = t(backend)

    if backend_type == "pandas":
        result_df = result_backend.unwrap
        assert result_df["label"].tolist() == [1, 0, 1, 2]
    else:
        result_df = result_backend.unwrap
        if isinstance(result_df, pl.LazyFrame):
            result_df = result_df.collect()
        assert result_df["label"].to_list() == [1, 0, 1, 2]

    assert meta["label_map"] == label_map
    assert meta["num_classes"] == len(label_map)


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_label_from_feature_with_label_map_two_cols_identicalnames(backend_type: str) -> None:
    """Test label from feature with identical column names."""
    if backend_type == "pandas":
        df = pd.DataFrame({
            "label": ["banana", "apple", "banana", "orange"],
            "col2": ["dog", "cat", "dog", "mouse"]
        })
        backend = PandasBackend(df)
    else:
        df = pl.DataFrame({
            "label": ["banana", "apple", "banana", "orange"],
            "col2": ["dog", "cat", "dog", "mouse"]
        })
        backend = PolarsBackend(df)

    label_map = {"apple": 0, "banana": 1, "orange": 2, "dog": 3, "cat": 4, "mouse": 5}
    t = LabelFromFeature(feature="label", label_map=label_map, output_feature="label", override=True)
    result_backend, meta = t(backend)

    if backend_type == "pandas":
        result_df = result_backend.unwrap
        assert result_df["label"].tolist() == [1, 0, 1, 2]
    else:
        result_df = result_backend.unwrap
        if isinstance(result_df, pl.LazyFrame):
            result_df = result_df.collect()
        assert result_df["label"].to_list() == [1, 0, 1, 2]

    assert meta["label_map"] == label_map
    assert meta["num_classes"] == len(label_map)
