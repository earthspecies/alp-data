import logging
from typing import Literal

from pydantic import BaseModel, field_validator

from alp_data.backends.protocol import DataBackend

from . import register_transform

logger = logging.Logger("alp_data")


class LongTailUpsampleConfig(BaseModel):
    type: Literal["long_tail_upsample"]
    property: str
    sufficient_threshold: int
    max_repeats: int
    seed: int = 42

    @field_validator("sufficient_threshold")
    @classmethod
    def validate_sufficient_threshold(cls, v: int) -> int:
        if v < 1:
            raise ValueError("`sufficient_threshold` must be >= 1")
        return v

    @field_validator("max_repeats")
    @classmethod
    def validate_max_repeats(cls, v: int) -> int:
        if v < 1:
            raise ValueError("`max_repeats` must be >= 1")
        return v


class LongTailUpsample:
    """Upsample under-represented categories without excessive repetition.

    Designed for long-tail distributions (e.g. bioacoustic species counts) where
    a few categories dominate and many categories have very few examples. This
    transform lifts the tail towards a `sufficient_threshold` while capping how
    many times any single example can be repeated via `max_repeats`.

    For each category with count *c*:

    - If ``c >= sufficient_threshold``: the category is left untouched.
    - If ``c < sufficient_threshold``: the target becomes
      ``min(sufficient_threshold, c * max_repeats)``.

    This produces a gradual compression of the distribution: well-represented
    categories keep all their data, moderately-represented categories are boosted
    to the threshold, and very rare categories are boosted as much as possible
    without repeating any single example more than `max_repeats` times.

    Parameters
    ----------
    property : str
        The name of the property (column) to balance on.
    sufficient_threshold : int
        Categories with at least this many examples are left as-is. Categories
        below this count are upsampled towards it, subject to `max_repeats`.
    max_repeats : int
        Maximum number of times any individual example may appear in the output.
        Prevents over-fitting on very rare categories.
    seed : int
        Random seed for reproducibility. Defaults to 42.

    Examples
    -------
    >>> from alp_data.transforms import LongTailUpsample, LongTailUpsampleConfig
    >>> from alp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> config = LongTailUpsampleConfig(
    ...     type="long_tail_upsample",
    ...     property="species",
    ...     sufficient_threshold=6,
    ...     max_repeats=3,
    ...     seed=42,
    ... )
    >>> transform = LongTailUpsample.from_config(config)
    >>> df = pd.DataFrame({
    ...     "species": (
    ...         ["common"] * 10
    ...         + ["moderate"] * 4
    ...         + ["rare"] * 1
    ...     ),
    ... })
    >>> backend = PandasBackend(df)
    >>> result, _ = transform(backend)
    >>> # common (10): untouched — already >= 6
    >>> # moderate (4): upsampled to min(6, 4*3)=6
    >>> # rare (1): upsampled to min(6, 1*3)=3
    """

    def __init__(
        self,
        property: str,
        sufficient_threshold: int,
        max_repeats: int,
        seed: int = 42,
    ) -> None:
        self.property = property
        self.sufficient_threshold = sufficient_threshold
        self.max_repeats = max_repeats
        self.seed = seed

    @classmethod
    def from_config(cls, cfg: LongTailUpsampleConfig) -> "LongTailUpsample":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Apply the long-tail upsample transformation.

        Categories below `sufficient_threshold` are upsampled towards it,
        bounded by `max_repeats`. Categories at or above the threshold are
        left unchanged.

        Parameters
        ----------
        backend : DataBackend
            The backend wrapping the dataframe to transform.

        Returns
        -------
        tuple[DataBackend, dict]
            A tuple containing the transformed backend (same type as input)
            and a metadata dictionary with keys ``histogram_before`` and
            ``histogram_after``, each mapping category values to their counts.

        Raises
        ------
        KeyError
            If the specified property is not found in the DataFrame columns.
        """
        if self.property not in backend.columns:
            raise KeyError(f"Property '{self.property}' not found in the DataFrame columns.")

        category_counts = backend.histogram(self.property)

        if not category_counts:
            return backend, {"histogram_before": {}, "histogram_after": {}}

        target_counts: dict[str, int] = {}
        for value, count in category_counts.items():
            if count == 0:
                target_counts[value] = 0
            elif count >= self.sufficient_threshold:
                target_counts[value] = count
            else:
                target_counts[value] = min(
                    self.sufficient_threshold,
                    count * self.max_repeats,
                )

        sampled_backend = backend.upsample_by_column(
            column=self.property,
            target_counts=target_counts,
            seed=self.seed,
        )

        histogram_after = sampled_backend.histogram(self.property)

        return sampled_backend, {
            "histogram_before": category_counts,
            "histogram_after": histogram_after,
        }


register_transform(LongTailUpsampleConfig, LongTailUpsample)
