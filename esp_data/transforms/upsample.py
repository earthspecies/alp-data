"""Transform to duplicate all rows N times."""

from typing import Literal

from pydantic import BaseModel, Field

from esp_data.backends.protocol import DataBackend

from . import register_transform


class UpsampleConfig(BaseModel):
    """Configuration for the Upsample transform.

    Attributes
    ----------
    type : Literal["upsample"]
        Discriminator field for transform registry.
    factor : int
        Number of times to repeat every row. Must be >= 2.
    """

    type: Literal["upsample"]
    factor: int = Field(ge=2)


class Upsample:
    """Repeat every row in a dataset a fixed number of times.

    Parameters
    ----------
    factor : int
        Number of copies of the full dataset to concatenate.

    Examples
    --------
    >>> from esp_data.transforms import Upsample
    >>> from esp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> df = pd.DataFrame({"species": ["bee", "ant"], "count": [3, 1]})
    >>> backend = PandasBackend(df)
    >>> upsampled, _ = Upsample(factor=3)(backend)
    >>> len(upsampled)
    6
    """

    def __init__(self, factor: int) -> None:
        self.factor = factor

    @classmethod
    def from_config(cls, cfg: UpsampleConfig) -> "Upsample":
        """Create an Upsample transform from configuration.

        Parameters
        ----------
        cfg : UpsampleConfig
            The configuration object.

        Returns
        -------
        Upsample
            A new Upsample instance.
        """
        return cls(factor=cfg.factor)

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
        result = type(backend).concat([backend] * self.factor)
        return result, {}


register_transform(UpsampleConfig, Upsample)
