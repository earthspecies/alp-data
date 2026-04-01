"""Transform to randomly keep a fraction of rows."""

from typing import Literal

from pydantic import BaseModel, Field

from esp_data.backends.protocol import DataBackend

from . import register_transform


class DownsampleConfig(BaseModel):
    type: Literal["downsample"]
    fraction: float = Field(gt=0.0, le=1.0)
    seed: int = 42


class Downsample:
    """Randomly sample a fraction of rows from the dataset.

    This transform keeps a random subset of rows, specified as a fraction of
    the total.  Useful for quickly reducing dataset size for development or
    ablation experiments.

    Works with any backend (pandas, polars) through the DataBackend protocol.

    Parameters
    ----------
    fraction : float
        Fraction of rows to retain, in (0, 1].
    seed : int
        Random seed for reproducibility.

    Examples
    -------
    >>> from esp_data.transforms import Downsample, DownsampleConfig
    >>> from esp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> config = DownsampleConfig(type="downsample", fraction=0.5, seed=42)
    >>> downsample_transform = Downsample.from_config(config)
    >>> df = pd.DataFrame({"species": ["bee"] * 100, "count": range(100)})
    >>> backend = PandasBackend(df)
    >>> downsampled_backend, _ = downsample_transform(backend)
    """

    def __init__(self, fraction: float, seed: int = 42) -> None:
        self.fraction = fraction
        self.seed = seed

    @classmethod
    def from_config(cls, cfg: DownsampleConfig) -> "Downsample":
        return cls(**cfg.model_dump(exclude=("type",)))

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
        n = max(1, round(len(backend) * self.fraction))
        return backend.sample_rows(n=n, seed=self.seed), {}


register_transform(DownsampleConfig, Downsample)
