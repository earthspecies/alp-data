"""A dedicated transform to remove duplicate rows from a DataFrame."""

import logging
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from . import register_transform

logger = logging.Logger("esp_data")


class DeduplicateConfig(BaseModel):
    type: Literal["deduplicate"]
    subset: Optional[list[str]] = Field(
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
    >>> df = pd.DataFrame({
    ...     "species": ["bee", "bee", "butterfly", "bee"],
    ...     "count": [10, 10, 5, 10]
    ... })
    >>> transform = Deduplicate(subset=["species"], keep_first=True)
    >>> deduplicated_df, _ = transform(df)
    >>> print(deduplicated_df.species.unique().tolist())
    ['bee', 'butterfly']
    """

    def __init__(self, *, subset: list[str] | None = None, keep_first: bool = True) -> None:
        self.subset = subset
        self.keep_first = keep_first

    @classmethod
    def from_config(cls, cfg: DeduplicateConfig) -> "Deduplicate":
        return cls(subset=cfg.subset, keep_first=cfg.keep_first)

    def __call__(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        return df.drop_duplicates(
            subset=self.subset, keep="first" if self.keep_first else "last"
        ), {}


register_transform(DeduplicateConfig, Deduplicate)
