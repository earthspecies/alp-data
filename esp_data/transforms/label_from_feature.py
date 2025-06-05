import logging
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from . import register_transform

logger = logging.Logger("esp_data")


class LabelFromFeatureConfig(BaseModel):
    type: Literal["label_from_feature"]
    feature: str
    label_map: dict[str, int] | None = None
    output_feature: str = "label"
    override: bool = False


class LabelFromFeature:
    """Transform to create a label feature from an existing feature in a DataFrame.

    This transform maps the values of a specified feature to integer labels.

    Arguments
    ---------
    feature: str
        The name of the feature in the DataFrame from which to create labels.
    label_map: dict[str, int] | None
        A mapping of feature values to integer labels. If None, the labels will be
        created from the unique values in the feature.
    output_feature: str
        The name of the new feature to store the labels. Defaults to "label".
    override: bool
        If True, will override the output feature if it already exists in the DataFrame.
        If False, will raise an AssertionError if the output feature already exists.

    Example
    -------
    >>> df = pd.DataFrame({"species": ["cat", "dog", "bird", "cat"]})
    >>> transform = LabelFromFeature(feature="species", output_feature="label")
    >>> transformed_df, metadata = transform(df)
    >>> print(transformed_df)
        species  label
    0      cat      0
    1      dog      1
    2     bird      2
    """

    def __init__(
        self,
        *,
        feature: str,
        label_map: dict[str, int] | None = None,
        output_feature: str = "label",
        override: bool = False,
    ) -> None:
        self.feature = feature
        self.label_map = label_map
        self.override = override
        self.output_feature = output_feature

    @classmethod
    def from_config(cls, cfg: LabelFromFeatureConfig) -> "LabelFromFeature":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Apply the transformation to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to transform.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            A tuple containing the transformed DataFrame and metadata about the labels.

        Raises
        -------
        AssertionError
            If the output feature already exists in the DataFrame and override is False.
        """
        if self.output_feature in df and not self.override:
            raise AssertionError(
                "Feature already exists in DataFrame. "
                "Set `override=True` to replace it."
            )

        df_clean = df.dropna(subset=[self.feature])
        if len(df_clean) != len(df):
            logger.warning(
                f"Dropped {len(df) - len(df_clean)} rows with {self.feature}=NaN"
            )

        if self.label_map is None:
            uniques = sorted(df_clean[self.feature].unique())
            label_map = {lbl: idx for idx, lbl in enumerate(uniques)}
        else:
            label_map = self.label_map

        df_clean.loc[:, [self.output_feature]] = df_clean[self.feature].map(label_map)

        metadata = {
            "label_feature": self.feature,
            "label_map": label_map,
            "num_classes": len(label_map),
        }

        return df_clean, metadata


register_transform(LabelFromFeatureConfig, LabelFromFeature)
