import logging
from typing import Any, Literal

from pydantic import BaseModel

from esp_data.backends import DataBackend

from . import register_transform

logger = logging.Logger("esp_data")


class MultiLabelFromFeaturesConfig(BaseModel):
    type: Literal["labels_from_features"]
    features: str | list[str]
    label_map: dict[Any, int] | None = None
    output_feature: str = "label"
    override: bool = False


class MultiLabelFromFeatures:
    """
    A transform that generates multi-label targets from one or more feature columns.

    This class goes through one or more specified columns and generates a mapping of
    unique values to integer IDs. It then uses this mapping to generate a new column
    where each row contains a list of integer label IDs corresponding to the unique
    values found in the specified feature columns. It is useful for preparing data for
    multi-label classification tasks, where each sample may be associated with multiple
    labels.

    Notes
    -----
    If element values are themselves lists, the transform will explode them first before
    constructing the mapping dictionary and converting the values.

    Parameters
    ----------
    features : list[str]
        The names of the columns in the DataFrame to use as sources for the labels. Each
        column can contain a single value or a list of values per row.
    label_map : dict[Any, int] | None, default=None
        A mapping of unique values to integer IDs. If not provided, the transform will
        generate a mapping based on the unique values in the specified feature columns.
    output_feature : str, default="label"
        The name of the output column to store the generated label lists.
    override : bool, default=False
        If False and the output_feature already exists in the dataset, an error is
        raised. If True, the output_feature will be overwritten.
    allow_missing_labels : bool, default=True
        If True, rows with no labels will be included in the output. If False, rows with
        no labels will be dropped.

    Methods
    -------
    from_config(cfg: MultiLabelFromFeaturesConfig) -> MultiLabelFromFeatures
        Instantiates the transform from a configuration object.
    __call__(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]
        Applies the transform to the DataFrame, returning the modified DataFrame and
        metadata about the label mapping.

    Examples
    -------
    >>> import pandas as pd
    >>> from esp_data.transforms import MultiLabelFromFeatures
    >>> config = MultiLabelFromFeaturesConfig(
    ...     type="labels_from_features",
    ...     features=["tags", "categories"],
    ...     label_map=None,
    ...     output_feature="labels",
    ...     override=False
    ... )
    >>> df = pd.DataFrame({
    ...     "tags": [["cat", "dog"], ["bird"], ["cat"]],
    ...     "categories": [["mammal"], ["avian"], []]
    ... })
    >>> from esp_data.backends import PandasBackend
    >>> backend = PandasBackend(df)
    >>> transform = MultiLabelFromFeatures.from_config(config)
    >>> transformed_df, metadata = transform(backend)
    >>> metadata["label_map"]
    {'avian': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'mammal': 4}
    """

    def __init__(
        self,
        *,
        features: list[str],
        label_map: dict[Any, int] | None = None,
        output_feature: str = "label",
        override: bool = False,
        allow_missing_labels: bool = True,
    ) -> None:
        self.features = features
        self.label_map = label_map
        self.override = override
        self.output_feature = output_feature
        self.allow_missing_labels = allow_missing_labels

    @classmethod
    def from_config(cls, cfg: MultiLabelFromFeaturesConfig) -> "MultiLabelFromFeatures":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        if self.output_feature in backend.columns and not self.override:
            raise AssertionError(
                "Feature already exists in DataFrame. Set `override=True` to replace it."
            )

        backend, label_map = backend.multilabel_from_features(
            input_features=self.features,
            label_map=self.label_map,
            output_feature=self.output_feature,
            allow_missing_labels=self.allow_missing_labels,
        )

        metadata = {
            "label_feature": self.features,
            "label_map": label_map,
            "num_classes": len(label_map),
        }

        return backend, metadata


register_transform(MultiLabelFromFeaturesConfig, MultiLabelFromFeatures)
