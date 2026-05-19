"""Pyarrow implementation of the DataFrameBackend protocol"""

import logging

from .protocol import DataBackend

logger = logging.getLogger("esp_data")


class PyarrowBackend(DataBackend):
    """Pyarrow implementation of the DataFrameBackend protocol.

    TO BE IMPLEMENTED
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def from_csv(cls) -> None:
        pass

    @classmethod
    def from_json(cls) -> None:
        pass

    @classmethod
    def from_parquet(cls) -> None:
        pass

    @property
    def is_streaming(self) -> None:
        pass
