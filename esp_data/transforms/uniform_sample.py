import logging
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from . import register_transform

logger = logging.Logger("esp_data")


class UniformSampleConfig(BaseModel):
    type: Literal["uniform_sample"]
    property: str
    ratio: float

    @field_validator("ratio")
    @classmethod
    def is_in_range(cls, ratio: float) -> float:
        if not 0 <= ratio <= 1:
            raise ValueError("Ratio must be in [0, 1]")
        return ratio


# TODO (Gagan) I'm a bit confused by what UniformSample is ... it sounds like a subset
# of Subsample just that the ratios "fixed" internally to make the dataset uniformly
# distributed across a certain property. But here a scalar "ratio" is being provided
# which would apply to all the unique groups in a property, and hence will not
# distribute uniformly.


class UniformSample:
    """Uniformly sample data based on a property.

    This transform uniformly samples a DataFrame based on a specified property and a
    ratio. It is SUPPOSED 🚨 to ensure that the resulting DataFrame has a
    uniform distribution of the specified property across the samples.

    Parameters
    ----------
    property: str
        The name of the property (column) to sample by.
    ratio: float
        The ratio of samples to keep for each unique value of the property. This should
        be a float in the range [0, 1], where 1 means all samples are kept and 0 means
        no samples are kept.
    """

    def __init__(self, property: str, ratio: float) -> None:
        self.property = property
        self.ratio = ratio

    @classmethod
    def from_config(cls, cfg: UniformSampleConfig) -> "UniformSample":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if isinstance(data, pd.DataFrame):
            return self._uniform_sample_dataframe(data), {}
        # if isinstance(data, dict):
        #     return self._uniform_sample_dict(data)
        raise TypeError(f"Unsupported data type: {type(data)}")

    def _uniform_sample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Uniformly sample a pandas DataFrame.

        Returns:
        --------
        tuple[pd.DataFrame, dict]:
            A tuple containing the sampled DataFrame and an empty dictionary for
            metadata. The dictionary can be used to store any additional information
            about the sampling process, if needed.
        """

        # Group by the property and sample uniformly
        groups = []
        for _, group in df.groupby(self.property):
            n_samples = max(1, int(len(group) * self.ratio))
            # TODO is this the right way to set up the random seed? Do we want to fix it
            # here?
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(len(group), size=n_samples, replace=False)
            groups.append(group.iloc[sampled_indices])

        return pd.concat(groups, ignore_index=True), {}


register_transform(UniformSampleConfig, UniformSample)
