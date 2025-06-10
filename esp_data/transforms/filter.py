import logging
from typing import Literal

import pandas as pd
from pydantic import BaseModel

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

    Returns
    -------
    None

    Examples
    --------
    >>> from esp_data.transforms import Filter
    >>> filter_transform = Filter(property="species", values=["bee", "butterfly"],
    ...     mode="include")
    >>> df = pd.DataFrame({"species": ["bee", "ant", "butterfly", "spider"],
    ...     "count": [10, 5, 8, 2]})
    >>> filtered_df, _ = filter_transform(df)
    >>> assert filtered_df["species"].tolist() == ["bee", "butterfly"]
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

    def __call__(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Filter the data based on property values.

        Parameters
        ----------
        data : pd.DataFrame
            The data dataframe to filter.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            A tuple containing:
            - The filtered data (same type as input)
            - An empty dictionary for metadata

        Raises
        ------
        TypeError
            If the data type is not supported.
        """
        if isinstance(data, pd.DataFrame):
            return self._filter_dataframe(data), {}
        # elif isinstance(data, dict):
        #     return self._filter_dict(data), {}
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to filter.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame.
        """
        if self.mode == "include":
            return df[df[self.property].isin(self.values)]
        else:
            return df[~df[self.property].isin(self.values)]

    # Right now we assume dataframe (though that will change soon)

    # def _filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
    #     """Filter a dictionary of data.

    #     Args:
    #         data: The dictionary to filter.

    #     Returns:
    #         Dict[str, Any]: The filtered dictionary.
    #     """
    #     if self.mode == "include":
    #         return {k: v for k, v in data.items() if v[self.property] in self.values}
    #     else:
    #         return {
    #             k: v for k, v in data.items() if v[self.property] not in self.values
    #         }


register_transform(FilterConfig, Filter)
