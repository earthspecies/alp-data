import logging
from typing import Literal

from pydantic import BaseModel, field_validator

from esp_data.backends.protocol import DataBackend

from . import register_transform

logger = logging.Logger("esp_data")


class UniformSampleConfig(BaseModel):
    type: Literal["uniform_sample"]
    property: str
    ratio: float
    seed: int = 42

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
    seed: int
        Random seed for reproducibility. Defaults to 42.

    Examples
    -------
    >>> from esp_data.transforms import UniformSample, UniformSampleConfig
    >>> from esp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> config = UniformSampleConfig(
    ...     type="uniform_sample",
    ...     property="species",
    ...     ratio=0.5,
    ...     seed=42
    ... )
    >>> uniform_sample_transform = UniformSample.from_config(config)
    >>> df = pd.DataFrame({
    ...     "species": ["bee", "bee", "butterfly", "ant", "butterfly", "spider"],
    ...     "count": [10, 5, 8, 2, 3, 1]
    ... })
    >>> backend = PandasBackend(df)
    >>> sampled_backend, _ = uniform_sample_transform(backend)
    """

    def __init__(self, property: str, ratio: float, seed: int = 42) -> None:
        self.property = property
        self.ratio = ratio
        self.seed = seed

    @classmethod
    def from_config(cls, cfg: UniformSampleConfig) -> "UniformSample":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Apply the uniform sample transformation.

        This transform creates a uniform distribution by computing inverse probabilities.
        For each category, it calculates the ratio needed to achieve equal representation
        across all categories. The `ratio` parameter controls the target sample size
        relative to the minimum category count.

        Parameters
        ----------
        backend: DataBackend
            The backend wrapping the dataframe to sample.

        Returns
        -------
        tuple[DataBackend, dict]: A tuple containing:
            The sampled backend (same type as input).
            The metadata dictionary (empty placeholder for future use).

        Raises
        ------
        KeyError
            If the specified property is not found in the DataFrame columns.
        """
        if self.property not in backend.columns:
            raise KeyError(f"Property '{self.property}' not found in the DataFrame columns.")

        # Get all unique values for the property
        unique_values = backend.get_unique(self.property)

        # Get counts for each category to compute inverse probabilities
        # We need to use unwrap temporarily to get category counts
        df = backend.unwrap
        if hasattr(df, "value_counts"):  # pandas
            import pandas as pd

            if isinstance(df, pd.DataFrame):
                category_counts = df[self.property].value_counts().to_dict()
            else:
                # Handle streaming case - collect if needed
                category_counts = {}
                for val in unique_values:
                    mask = df[self.property] == val
                    category_counts[val] = int(mask.sum())
        else:  # polars
            import polars as pl

            if isinstance(df, pl.LazyFrame):
                df = df.collect()
            category_counts = df.group_by(self.property).len().to_dict(as_series=False)
            category_counts = dict(
                zip(category_counts[self.property], category_counts["len"], strict=True)
            )

        if not category_counts:
            # Empty dataset
            return backend, {}

        total_count = sum(category_counts.values())
        num_categories = len(unique_values)

        # Compute inverse probabilities as Gagan suggested
        # Original probability of category i: p_i = count_i / total_count
        # Target uniform probability: p_target = 1 / num_categories
        # Inverse probability ratio: ratio_i = p_target / p_i
        #   = total_count / (num_categories * count_i)
        # This gives us the relative sampling rate needed for each category
        inverse_prob_ratios = {}
        for value in unique_values:
            count = category_counts[value]
            if count == 0:
                inverse_prob_ratios[value] = 0.0
            else:
                inverse_prob_ratios[value] = total_count / (num_categories * count)

        # Normalize ratios so max is 1.0 (since we can't oversample)
        # Then scale by self.ratio to control overall sample size
        max_ratio = max(inverse_prob_ratios.values()) if inverse_prob_ratios else 1.0
        if max_ratio > 0:
            ratios = {
                value: min(1.0, (inv_ratio / max_ratio) * self.ratio)
                for value, inv_ratio in inverse_prob_ratios.items()
            }
        else:
            ratios = {value: 0.0 for value in unique_values}

        # Use backend's subsample_by_column method
        sampled_backend = backend.subsample_by_column(
            column=self.property,
            ratios=ratios,
            seed=self.seed,
        )

        return sampled_backend, {}


register_transform(UniformSampleConfig, UniformSample)
