import logging
from typing import Literal

from pydantic import BaseModel, field_validator

from esp_data.backends.protocol import DataBackend

from . import register_transform

logger = logging.Logger("esp_data")


class SelectColumnsConfig(BaseModel):
    type: Literal["select_columns"]
    columns: list[str]

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("`columns` must contain at least one column name")
        return v


class SelectColumns:
    """Select a subset of columns from the dataset.

    This transform keeps only the specified columns and drops all others.

    Works with any backend (pandas, polars) through the DataBackend protocol.

    Parameters
    ----------
    columns : list[str]
        List of column names to keep.

    Examples
    -------
    >>> from esp_data.transforms import SelectColumns, SelectColumnsConfig
    >>> from esp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> config = SelectColumnsConfig(
    ...     type="select_columns",
    ...     columns=["species", "audio"],
    ... )
    >>> transform = SelectColumns.from_config(config)
    >>> df = pd.DataFrame({
    ...     "species": ["bee", "ant"],
    ...     "audio": ["/a.wav", "/b.wav"],
    ...     "extra": [1, 2],
    ... })
    >>> backend = PandasBackend(df)
    >>> result, _ = transform(backend)
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    @classmethod
    def from_config(cls, cfg: SelectColumnsConfig) -> "SelectColumns":
        return cls(columns=cfg.columns)

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Select the specified columns from the backend.

        Parameters
        ----------
        backend : DataBackend
            The backend wrapping the dataframe to transform.

        Returns
        -------
        tuple[DataBackend, dict]
            A tuple containing the transformed backend with only the selected
            columns and an empty metadata dictionary.

        Raises
        ------
        KeyError
            If any of the specified columns are not found in the backend.
        """
        missing = [c for c in self.columns if c not in backend.columns]
        if missing:
            raise KeyError(
                f"Columns {missing} not found in the DataFrame. "
                f"Available columns: {backend.columns}"
            )

        return backend.select_columns(self.columns), {}


register_transform(SelectColumnsConfig, SelectColumns)
