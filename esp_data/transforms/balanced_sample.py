import logging
import statistics
from typing import Literal

from pydantic import BaseModel, model_validator

from esp_data.backends.protocol import DataBackend

from . import register_transform

logger = logging.Logger("esp_data")


class BalancedSampleConfig(BaseModel):
    type: Literal["balanced_sample"]
    property: str
    strategy: Literal["min", "max", "median", "mean", "median_with_range"] = "median"
    range_fraction: float = 0.2
    seed: int = 42

    @model_validator(mode="after")
    def validate_range_fraction(self) -> "BalancedSampleConfig":
        if self.strategy == "median_with_range":
            if not 0 < self.range_fraction < 1:
                raise ValueError("range_fraction must be between 0 and 1 (exclusive)")
        return self


class BalancedSample:
    """Balance data by sampling to equalize category counts.

    This transform balances a DataFrame based on a specified property to
    ensure that the resulting DataFrame has a balanced distribution of the specified
    property across the samples.

    Parameters
    ----------
    property: str
        The name of the property (column) to sample by.
    strategy: str
        The balancing strategy to use. Options are:
        - "min": Sample all categories to the minimum count (downsamples larger)
        - "max": Sample all categories to the maximum count (upsamples smaller)
        - "median": Sample all categories to the median count (default)
        - "mean": Sample all categories to the mean count
        - "median_with_range": Clamp each category to a range around the median
    range_fraction: float
        Only used with "median_with_range" strategy. The fraction of the median
        to use as the range. E.g., 0.2 means targets are clamped to
        [median * 0.8, median * 1.2]. Defaults to 0.2.
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
    ...     strategy="median",
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

    def __init__(
        self,
        property: str,
        strategy: Literal["min", "max", "median", "mean", "median_with_range"] = "median",
        range_fraction: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.property = property
        self.strategy = strategy
        self.range_fraction = range_fraction
        self.seed = seed

    @classmethod
    def from_config(cls, cfg: BalancedSampleConfig) -> "BalancedSample":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Apply the balanced sample transformation.

        This transform creates a balanced distribution by sampling categories
        based on the selected strategy.

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

        # Get counts for each category to compute target counts
        category_counts = backend.histogram(self.property)

        if not category_counts:
            # Empty dataset
            return backend, {}

        counts = list(category_counts.values())

        # Compute target count based on strategy
        if self.strategy == "min":
            target_count = min(counts)
            target_counts = {v: target_count for v in unique_values if category_counts[v] > 0}
        elif self.strategy == "max":
            target_count = max(counts)
            target_counts = {v: target_count for v in unique_values if category_counts[v] > 0}
        elif self.strategy == "median":
            target_count = int(statistics.median(counts))
            target_counts = {v: target_count for v in unique_values if category_counts[v] > 0}
        elif self.strategy == "mean":
            target_count = round(statistics.mean(counts))
            target_counts = {v: target_count for v in unique_values if category_counts[v] > 0}
        elif self.strategy == "median_with_range":
            median_count = statistics.median(counts)
            lower_bound = int(median_count * (1 - self.range_fraction))
            upper_bound = int(median_count * (1 + self.range_fraction))
            # Clamp each category to [lower_bound, upper_bound]
            target_counts = {}
            for value in unique_values:
                count = category_counts[value]
                if count > 0:
                    target_counts[value] = max(lower_bound, min(upper_bound, count))

        # Handle categories with zero count
        for value in unique_values:
            if category_counts[value] == 0:
                target_counts[value] = 0

        # Use backend's upsample_by_column method
        # This handles both upsampling (with replacement) and downsampling (without replacement)
        sampled_backend = backend.upsample_by_column(
            column=self.property,
            target_counts=target_counts,
            seed=self.seed,
        )

        return sampled_backend, {}


register_transform(BalancedSampleConfig, BalancedSample)
