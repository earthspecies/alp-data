"""A dedicated transform to remove duplicate rows from a DataFrame."""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from esp_data.backends.protocol import DataFrameBackend

from . import register_transform

logger = logging.Logger("esp_data")


class DeduplicateConfig(BaseModel):
    type: Literal["deduplicate"]
    subset: list[str] | None = Field(
        default_factory=None,
        description="List of columns to consider for deduplication. "
        "If empty, all columns are considered.",
    )
    keep_first: bool = Field(
        default=True,
        description="If True, keeps the first occurrence of duplicates. "
        "If False, keeps the last occurrence.",
    )


class Deduplicate:
    """A transform to remove duplicate rows from a DataFrame.

    This transform removes duplicate rows based on specified columns or all columns if
    none are specified. It can keep either the first or last occurrence of duplicates.

    Works with any backend (pandas, polars) through the DataFrameBackend protocol.

    Parameters
    ----------
    subset: list[str]
        List of column names to consider for deduplication. If empty, all columns are
        considered.
    keep_first: bool
        If True, keeps the first occurrence of duplicates. If False, keeps the last
        occurrence.

    Examples
    --------
    >>> from esp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "species": ["bee", "bee", "butterfly", "bee"],
    ...     "count": [10, 10, 5, 10]
    ... })
    >>> backend = PandasBackend(df)
    >>> transform = Deduplicate(subset=["species"], keep_first=True)
    >>> deduplicated_backend, _ = transform(backend)
    """

    def __init__(self, *, subset: list[str] | None = None, keep_first: bool = True) -> None:
        self.subset = subset
        self.keep_first = keep_first

    @classmethod
    def from_config(cls, cfg: DeduplicateConfig) -> "Deduplicate":
        return cls(subset=cfg.subset, keep_first=cfg.keep_first)

    def __call__(self, backend: DataFrameBackend) -> tuple[DataFrameBackend, dict]:
        """Remove duplicate rows from the backend.

        Args:
            backend: The backend wrapping the dataframe to deduplicate

        Returns:
            The deduplicated backend (same type as input) and empty metadata dict.
        """
        deduplicated_backend = backend.drop_duplicates(
            subset=self.subset,
            keep="first" if self.keep_first else "last",
        )
        return deduplicated_backend, {}


register_transform(DeduplicateConfig, Deduplicate)
