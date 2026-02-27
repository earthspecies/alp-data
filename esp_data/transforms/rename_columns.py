"""Transform to rename columns in a backend."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from esp_data.backends.protocol import DataBackend

from . import register_transform


class RenameColumnsConfig(BaseModel):
    type: Literal["rename_columns"]
    mapping: dict[str, str] = Field(
        description="Mapping of old column names to new column names.",
    )


class RenameColumns:
    """Rename columns in a backend via a mapping dict.

    Parameters
    ----------
    mapping : dict[str, str]
        Mapping of ``{old_name: new_name}``.

    Examples
    --------
    >>> from esp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> df = pd.DataFrame({"soundType": ["song", "call"], "species_common": ["Robin", "Wren"]})
    >>> backend = PandasBackend(df)
    >>> transform = RenameColumns(mapping={"soundType": "behavior"})
    >>> new_backend, _ = transform(backend)
    >>> "behavior" in new_backend.columns
    True
    """

    def __init__(self, *, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    @classmethod
    def from_config(cls, cfg: RenameColumnsConfig) -> "RenameColumns":
        return cls(mapping=cfg.mapping)

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict[str, Any]]:
        return backend.rename_columns(self.mapping), {}


register_transform(RenameColumnsConfig, RenameColumns)
