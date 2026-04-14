"""Transform to randomly keep a fraction of rows."""

from typing import Literal

from pydantic import BaseModel, Field

from esp_data.backends.protocol import DataBackend

from . import register_transform


class DownsampleConfig(BaseModel):
    """Configuration for the Downsample transform.

    Attributes
    ----------
    type : Literal["downsample"]
        Discriminator field for transform registry.
    fraction : float
        Fraction of rows to keep, in (0, 1].
    seed : int
        Random seed for reproducibility.
    """

    type: Literal["downsample"]
    fraction: float = Field(gt=0.0, le=1.0)
    seed: int = 42


class Downsample:
    """Randomly sample a fraction of rows from the dataset.

    Parameters
    ----------
    fraction : float
        Fraction of rows to retain, in (0, 1].
    seed : int
        Random seed for reproducibility.

    Examples
    --------
    >>> from esp_data.transforms import Downsample
    >>> from esp_data.backends import PolarsBackend
    >>> import polars as pl
    >>> df = pl.DataFrame({"species": ["bee", "ant", "fly"], "count": [3, 1, 2]})
    >>> backend = PolarsBackend(df)
    >>> downsampled, _ = Downsample(fraction=0.5, seed=42)(backend)
    >>> len(downsampled) <= 3
    True
    """

    def __init__(self, fraction: float, seed: int = 42) -> None:
        self.fraction = fraction
        self.seed = seed

    @classmethod
    def from_config(cls, cfg: DownsampleConfig) -> "Downsample":
        """Create a Downsample transform from configuration.

        Parameters
        ----------
        cfg : DownsampleConfig
            The configuration object.

        Returns
        -------
        Downsample
            A new Downsample instance.
        """
        return cls(fraction=cfg.fraction, seed=cfg.seed)

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Randomly keep ``fraction`` of the rows.

        Parameters
        ----------
        backend : DataBackend
            The backend wrapping the data to downsample.

        Returns
        -------
        tuple[DataBackend, dict]
            The downsampled backend and empty metadata dict.
        """
        n = round(len(backend) * self.fraction)
        n = max(n, 1)
        result = backend.sample_rows(n=n, seed=self.seed)
        return result, {}


register_transform(DownsampleConfig, Downsample)
