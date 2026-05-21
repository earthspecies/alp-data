"""Transform to duplicate all rows N times."""

from typing import Literal

from pydantic import BaseModel, Field

from esp_data.backends.protocol import DataBackend

from . import register_transform


class UpsampleConfig(BaseModel):
    type: Literal["upsample"]
    factor: int = Field(ge=2)


class Upsample:
    """Repeat every row in a dataset a fixed number of times.

    This transform concatenates the dataset with itself ``factor`` times,
    effectively duplicating every row.  Useful when a small dataset needs to
    be epoch-matched with larger ones during training.

    Works with any backend (pandas, polars) through the DataBackend protocol.

    Parameters
    ----------
    factor : int
        Number of copies of the full dataset to concatenate.  Must be >= 2.

    Examples
    -------
    >>> from esp_data.transforms import Upsample, UpsampleConfig
    >>> from esp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> config = UpsampleConfig(type="upsample", factor=3)
    >>> upsample_transform = Upsample.from_config(config)
    >>> df = pd.DataFrame({"species": ["bee", "ant"], "count": [3, 1]})
    >>> backend = PandasBackend(df)
    >>> upsampled_backend, _ = upsample_transform(backend)
    >>> len(upsampled_backend)
    6
    """

    def __init__(self, factor: int) -> None:
        self.factor = factor

    @classmethod
    def from_config(cls, cfg: UpsampleConfig) -> "Upsample":
        return cls(**cfg.model_dump(exclude=("type",)))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Repeat every row ``factor`` times.

        Parameters
        ----------
        backend : DataBackend
            The backend wrapping the data to upsample.

        Returns
        -------
        tuple[DataBackend, dict]
            The upsampled backend and empty metadata dict.
        """
        return type(backend).concat([backend] * self.factor), {}


register_transform(UpsampleConfig, Upsample)
