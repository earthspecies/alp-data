import pandas as pd
import pytest

from esp_data.transforms import MultiLabelFromFeatures


@pytest.mark.parametrize(
    "features, df, expected_labels, expected_map",
    [
        # Single column, all strings
        (
            ["col1"],
            pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"]}),
            [[1], [0], [1], [2]],
            {"apple": 0, "banana": 1, "orange": 2},
        ),
        # Single column, all lists
        (
            ["col1"],
            pd.DataFrame(
                {
                    "col1": [
                        ["banana", "apple"],
                        ["apple"],
                        ["banana", "orange"],
                        ["orange"],
                    ]
                }
            ),
            [[0, 1], [0], [1, 2], [2]],
            {"apple": 0, "banana": 1, "orange": 2},
        ),
        # Single column, mix of strings and lists
        (
            ["col1"],
            pd.DataFrame(
                {"col1": ["banana", ["apple"], ["banana", "orange"], "orange"]}
            ),
            [[1], [0], [1, 2], [2]],
            {"apple": 0, "banana": 1, "orange": 2},
        ),
        # Multiple columns, mix of strings and lists
        (
            ["col1", "col2"],
            pd.DataFrame(
                {
                    "col1": ["banana", ["apple"], ["banana", "orange"], "orange"],
                    "col2": [["grape"], "kiwi", ["grape", "melon"], "melon"],
                }
            ),
            [[1, 2], [0, 3], [1, 2, 4, 5], [4, 5]],
            {"apple": 0, "banana": 1, "grape": 2, "kiwi": 3, "melon": 4, "orange": 5},
        ),
        # Multiple columns, some rows have NaN in just one column
        (
            ["col1", "col2"],
            pd.DataFrame(
                {
                    "col1": ["banana", ["apple"], float("nan"), "orange"],
                    "col2": [["grape"], float("nan"), ["grape", "melon"], "melon"],
                }
            ),
            [[1, 2], [0], [2, 3], [3, 4]],
            {"apple": 0, "banana": 1, "grape": 2, "melon": 3, "orange": 4},
        ),
        # Multiple columns, some rows have NaN in both columns
        (
            ["col1", "col2"],
            pd.DataFrame(
                {
                    "col1": [
                        float("nan"),
                        ["apple"],
                        float("nan"),
                        "orange",
                        float("nan"),
                    ],
                    "col2": [
                        float("nan"),
                        float("nan"),
                        ["grape", "melon"],
                        float("nan"),
                        [],
                    ],
                }
            ),
            [[0], [1, 2], [3]],
            {"apple": 0, "grape": 1, "melon": 2, "orange": 3},
        ),
    ],
)
def test_multilabel_from_features(
    features: list[str],
    df: pd.DataFrame,
    expected_labels: list[list[str]],
    expected_map: dict[str, int],
) -> None:
    t = MultiLabelFromFeatures(features=features, allow_missing_labels=False)
    df_out, meta = t(df.copy())
    assert df_out["label"].tolist() == expected_labels
    assert meta["label_map"] == expected_map
    assert meta["num_classes"] == len(expected_map)


def test_multilabel_from_features_with_extra_labels() -> None:
    df = pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"]})
    label_map = {"apple": 0, "banana": 1, "orange": 2, "grape": 3, "melon": 4}
    t = MultiLabelFromFeatures(features=["col1"], label_map=label_map, allow_missing_labels=False)
    df_out, meta = t(df.copy())
    # Only present labels should be used in output
    assert df_out["label"].tolist() == [[1], [0], [1], [2]]
    assert meta["label_map"] == label_map
    assert meta["num_classes"] == len(label_map)


def test_multilabel_from_features_with_noncontiguous_indices() -> None:
    df = pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"]})
    label_map = {"apple": 100, "banana": 101, "orange": 102}
    t = MultiLabelFromFeatures(features=["col1"], label_map=label_map, allow_missing_labels=False)
    df_out, meta = t(df.copy())
    assert df_out["label"].tolist() == [[101], [100], [101], [102]]
    assert meta["label_map"] == label_map
    assert meta["num_classes"] == len(label_map)


def test_multilabel_from_features_label_map_remains_none() -> None:
    df = pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"]})
    t = MultiLabelFromFeatures(features=["col1"], allow_missing_labels=False)
    assert t.label_map is None
    df_out, meta = t(df.copy())
    # The transform should still not set t.label_map
    assert t.label_map is None
    # The output should be correct
    assert sorted(meta["label_map"].keys()) == ["apple", "banana", "orange"]
    assert meta["num_classes"] == 3
    assert df_out["label"].tolist() == [[1], [0], [1], [2]]


def test_multilabel_from_features_allow_missing_labels() -> None:
    """Ensure that the *allow_missing_labels* flag correctly retains or drops rows with no labels."""
    df = pd.DataFrame({"col1": ["banana", [], "orange"]})

    # When allow_missing_labels=True (default) we expect to **keep** rows with no labels.
    t_keep = MultiLabelFromFeatures(features=["col1"], allow_missing_labels=True)
    df_keep, meta_keep = t_keep(df.copy())

    # The label map should contain only the present, non-empty labels
    assert meta_keep["label_map"] == {"banana": 0, "orange": 1}
    assert meta_keep["num_classes"] == 2
    # All original rows should remain, with empty rows represented by an empty list
    assert df_keep["label"].tolist() == [[0], [], [1]]

    # When allow_missing_labels=False we expect rows with no labels to be **dropped**.
    t_drop = MultiLabelFromFeatures(features=["col1"], allow_missing_labels=False)
    df_drop, meta_drop = t_drop(df.copy())

    # Metadata should be identical to the previous run
    assert meta_drop == meta_keep
    # The resulting DataFrame should only contain the rows with labels ("banana" and "orange")
    assert df_drop["label"].tolist() == [[0], [1]]
    assert len(df_drop) == 2


def test_multilabel_from_features_outputfeature_already_present() -> None:
    """Ensure that an AssertionError is raised if the output feature already exists and override is False."""
    df = pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"],
                       "label": ["dog", "cat", "dog", "mouse"]})
    t = MultiLabelFromFeatures(features=["label"], output_feature="label", override=False)
    with pytest.raises(AssertionError, match="Feature already exists in DataFrame"):
        t(df)

    t = MultiLabelFromFeatures(features=["label"], output_feature="label", override=True)
    df_out, meta = t(df)
    assert df_out["label"].tolist() == [[1], [0], [1], [2]]
