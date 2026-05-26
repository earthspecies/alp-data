import logging
from typing import Literal

from pydantic import BaseModel, field_validator

from esp_data.backends.protocol import DataBackend

from . import register_transform

logger = logging.Logger("esp_data")


class SubsampleConfig(BaseModel):
    type: Literal["subsample"]
    property: str
    ratios: dict[str, float]
    seed: int = 42

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

    Works with any backend (pandas, polars) through the DataBackend protocol.

    Parameters
    ----------
    property: str
        The name of the property (column) to subsample by.

    ratios: dict[str, float]
        A dictionary where keys are the values of the property and values are the
        ratios of samples to keep for each value. The ratios should be in the range
        [0, 1]. If "other" is included as a key, it will subsample all other values
        not explicitly listed in the ratios dictionary.

    Examples
    -------
    >>> from esp_data.transforms import Subsample, SubsampleConfig
    >>> from esp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> config = SubsampleConfig(
    ...     type="subsample",
    ...     property="species",
    ...     ratios={
    ...         "bee": 0.5,
    ...         "butterfly": 0.3,
    ...         "other": 0.1
    ...     })
    >>> subsample_transform = Subsample.from_config(config)
    >>> df = pd.DataFrame({
    ...     "species": ["bee", "bee", "butterfly", "ant", "butterfly", "spider"],
    ...     "count": [10, 5, 8, 2, 3, 1]
    ... })
    >>> backend = PandasBackend(df)
    >>> subsampled_backend, _ = subsample_transform(backend)
    """

    def __init__(self, property: str, ratios: dict[str, float], seed: int = 42) -> None:
        self.property = property
        self.ratios = ratios
        self.seed = seed

    @classmethod
    def from_config(cls, cfg: SubsampleConfig) -> "Subsample":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """
        Apply the subsample transformation.

        Parameters
        ----------
        backend: DataBackend
            The backend wrapping the dataframe to subsample

        Returns
        -------
        tuple[DataBackend, dict]: A tuple containing:
            The subsampled backend (same type as input).
            The metadata dictionary (empty placeholder for future use).

        Raises
        ------
        KeyError
            If the specified property is not found in the DataFrame columns.
        """
        if self.property not in backend.columns:
            raise KeyError(f"Property '{self.property}' not found in the DataFrame columns.")

        # Use backend's subsample_by_column method
        subsampled_backend = backend.subsample_by_column(
            column=self.property,
            ratios=self.ratios,
            seed=self.seed,
        )

        return subsampled_backend, {}


register_transform(SubsampleConfig, Subsample)
