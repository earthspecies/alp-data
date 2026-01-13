import logging
from typing import Literal

from pydantic import BaseModel

from esp_data.backends.protocol import DataBackend

from . import register_transform

logger = logging.Logger("esp_data")


class BalancedSampleConfig(BaseModel):
    type: Literal["balanced_sample"]
    property: str
    seed: int = 42


class BalancedSample:
    """Balance data by sampling to equalize category counts.

    This transform balances a DataFrame based on a specified property to
    ensure that the resulting DataFrame has a balanced distribution of the specified
    property across the samples. All categories are sampled to have the same count,
    based on the minimum category count in the dataset.

    Parameters
    ----------
    property: str
        The name of the property (column) to sample by.
    seed: int
        Random seed for reproducibility. Defaults to 42.

    Examples
    -------
    >>> from esp_data.transforms import BalancedSample, BalancedSampleConfig
    >>> from esp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> config = BalancedSampleConfig(
    ...     type="balanced_sample",
    ...     property="species",
    ...     seed=42
    ... )
    >>> balanced_sample_transform = BalancedSample.from_config(config)
    >>> df = pd.DataFrame({
    ...     "species": ["bee", "bee", "butterfly", "ant", "butterfly", "spider"],
    ...     "count": [10, 5, 8, 2, 3, 1]
    ... })
    >>> backend = PandasBackend(df)
    >>> sampled_backend, _ = balanced_sample_transform(backend)
    """

    def __init__(self, property: str, seed: int = 42) -> None:
        self.property = property
        self.seed = seed

    @classmethod
    def from_config(cls, cfg: BalancedSampleConfig) -> "BalancedSample":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Apply the balanced sample transformation.

        This transform creates a balanced distribution by sampling all categories
        to have the same count. For each category, it calculates the ratio needed
        to achieve equal representation across all categories. All categories are
        sampled to have the same count, based on the minimum category count in
        the dataset.

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
        category_counts = backend.histogram(self.property)

        if not category_counts:
            # Empty dataset
            return backend, {}

        # Find minimum count across all categories
        min_count = min(category_counts.values())

        # Compute target counts for each category to achieve balanced distribution
        # All categories should have the same count (min_count)
        # For categories with more samples than min_count, we undersample
        # For categories with fewer samples, we upsample with replacement
        target_counts = {}
        for value in unique_values:
            count = category_counts[value]
            if count == 0:
                target_counts[value] = 0
            else:
                # Target count is min_count for all categories
                target_counts[value] = min_count

        # Use backend's upsample_by_column method
        # This handles both upsampling (with replacement) and downsampling (without replacement)
        sampled_backend = backend.upsample_by_column(
            column=self.property,
            target_counts=target_counts,
            seed=self.seed,
        )

        return sampled_backend, {}


register_transform(BalancedSampleConfig, BalancedSample)
