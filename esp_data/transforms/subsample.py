import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from . import register_transform

logger = logging.Logger("esp_data")


class SubsampleConfig(BaseModel):
    type: Literal["subsample"]
    property: str
    ratios: dict[str, float]

    # TODO (milad) we support "other" in ratios?

    @field_validator("ratios")
    @classmethod
    def is_in_range(cls, ratios: dict[str, float]) -> dict[str, float]:
        if not all(0 <= r <= 1 for r in ratios.values()):
            raise ValueError("All ratios must be in [0, 1]")
        return ratios


class Subsample:
    """Subsample data based on property ratios.

    This transform subsamples a DataFrame based on the specified ratios for each value
    of a given property. It allows for controlling the representation of different
    categories in the dataset by specifying how much of each category to keep.
    The property is a column in the DataFrame, and the ratios are specified as a
    dictionary where keys are property values and values are the ratios of samples to
    keep for each property value. The "other" category can be used to specify a ratio
    for all other values not explicitly listed in the ratios dictionary.

    Arguments
    ---------
    property: str
        The name of the property (column) to subsample by.

    ratios: dict[str, float]
        A dictionary where keys are the values of the property and values are the
        ratios of samples to keep for each value. The ratios should be in the range
        [0, 1]. If "other" is included as a key, it will subsample all other values
        not explicitly listed in the ratios dictionary.

    Example
    -------
    >>> from esp_data.transforms import Subsample, SubsampleConfig
    >>> config = SubsampleConfig(
    ...     type="subsample",
    ...     property="species",
    ...     ratios={
    ...         "bee": 0.5,
    ...         "butterfly": 0.3,
    ...         "other": 0.1
    ...     }
    >>> )
    >>> subsample_transform = Subsample.from_config(config)
    >>> df = pd.DataFrame({
    ...     "species": ["bee", "bee", "butterfly", "ant", "butterfly", "spider"],
    ...     "count": [10, 5, 8, 2, 3, 1]
    ... })
    >>> subsampled_df, _ = subsample_transform(df)
    >>> print(subsampled_df)
    """

    def __init__(self, property: str, ratios: dict[str, float]) -> None:
        self.property = property
        self.ratios = ratios

    @classmethod
    def from_config(cls, cfg: SubsampleConfig) -> "Subsample":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Apply the subsample transformation.

        Arguments
        ---------
            data: The data to subsample (DataFrame or dict).

        Returns
        -------
            tuple[pd.DataFrame, dict]: A tuple containing:
                The subsampled data (same type as input).

        Raises
        ------
            TypeError: If the data type is not supported.
            KeyError: If the specified property is not found in the DataFrame columns.
        """
        if self.property not in data.columns:
            raise KeyError(
                f"Property '{self.property}' not found in the DataFrame columns."
            )

        if isinstance(data, pd.DataFrame):
            return self._subsample_dataframe(data), {}
        # if isinstance(data, dict):
        #     return self._subsample_dict(data), None
        raise TypeError(f"Unsupported data type: {type(data)}")

    def _choose_keys(self, keys: list[Any], ratio: float) -> list[Any]:
        """Return a subsample of *keys* of size `ceil(len(keys)*ratio)`.

        Args:
            keys: List of keys to subsample from.
            ratio: Ratio of keys to select.

        Returns:
            List[Any]: The selected keys.
        """
        if ratio >= 1.0 or len(keys) == 0:
            return keys
        n = int(len(keys) * ratio)
        rng = np.random.default_rng(seed=42)
        return rng.choice(keys, size=n, replace=False).tolist()

    def _subsample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subsample a pandas DataFrame.

        Args:
            df: The DataFrame to subsample.

        Returns:
            pd.DataFrame: The subsampled DataFrame.
        """
        groups = []

        for val, ratio in self.ratios.items():
            if val == "other":
                continue
            idx = df.index[df[self.property] == val].tolist()
            chosen = self._choose_keys(idx, ratio)
            groups.append(df.loc[chosen])

        if "other" in self.ratios:
            mask_other = ~df[self.property].isin(self.ratios.keys() - {"other"})
            idx_other = df.index[mask_other].tolist()
            chosen_other = self._choose_keys(idx_other, self.ratios["other"])
            groups.append(df.loc[chosen_other])

        return pd.concat(groups, ignore_index=True)


register_transform(SubsampleConfig, Subsample)
