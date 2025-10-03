import logging
from typing import Literal

from pydantic import BaseModel

from esp_data.backends.protocol import DataBackendT

from . import register_transform

logger = logging.Logger("esp_data")


class FilterConfig(BaseModel):
    type: Literal["filter"]
    mode: Literal["include", "exclude"] = "include"
    property: str
    values: list[str]


class Filter:
    """Filter data based on property values.

    This transform filters a DataFrame based on the values of a specified property.
    It can either include or exclude rows based on the specified values. The property
    is a column in the DataFrame, and the values are the values to filter by.

    Works with any backend (pandas, polars) through the DataBackendT protocol.

    Parameters
    ----------
    property: str
        The name of the property (column) to filter by.
    values: list[str]
        The values to include or exclude from the DataFrame.
    mode: Literal["include", "exclude"]
        The mode of filtering. If "include", only rows with the specified values
        in the property will be kept. If "exclude", rows with the specified values
        will be removed from the DataFrame.

    Examples
    -------
    >>> from esp_data.transforms import Filter
    >>> from esp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> filter_transform = Filter(property="species", values=["bee", "butterfly"],
    ...     mode="include")
    >>> df = pd.DataFrame({"species": ["bee", "ant", "butterfly", "spider"],
    ...     "count": [10, 5, 8, 2]})
    >>> backend = PandasBackend(df)
    >>> filtered_backend, _ = filter_transform(backend)
    """

    def __init__(
        self,
        *,
        property: str,
        values: list[str],
        mode: Literal["include", "exclude"] = "include",
    ) -> None:
        """
        Initialize the filter.
        """

        self.mode = mode
        self.property = property
        self.values = values

    @classmethod
    def from_config(cls, cfg: FilterConfig) -> "Filter":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, backend: DataBackendT) -> tuple[DataBackendT, dict]:
        """Filter the data based on property values.

        Args:
            backend: The backend wrapping the dataframe to filter

        Returns:
            The filtered backend (same type as input) and empty metadata dict.
        """
        # Use backend's filter_isin method
        negate = self.mode == "exclude"
        filtered_backend = backend.filter_isin(self.property, self.values, negate=negate)

        return filtered_backend, {}


register_transform(FilterConfig, Filter)
