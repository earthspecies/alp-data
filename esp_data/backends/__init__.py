"""Backend system for unified data operations across pandas, polars, and other libraries.

This module provides a protocol-based backend system that allows esp-data to work
with multiple data libraries through a common interface.

Two protocols are defined:
- StreamingBackend: For streaming-only data formats (e.g., WebDataset/tar files)
- DataBackend: For in-memory data formats (e.g., pandas, polars DataFrames)
"""

from .backends import BackendType, get_backend
from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend
from .protocol import DataBackend, StreamingBackend
from .webdataset_backend import (
    WebDatasetBackend,
    audio_decoder,
    audio_encoder,
    json_decoder,
    json_encoder,
    load_webdataset,
    make_file_opener_for_wds,
)

__all__ = [
    # Protocols
    "DataBackend",
    "StreamingBackend",
    # DataBackend implementations
    "PandasBackend",
    "PolarsBackend",
    # StreamingBackend implementations
    "WebDatasetBackend",
    # WebDataset utilities
    "audio_decoder",
    "audio_encoder",
    "json_decoder",
    "json_encoder",
    "load_webdataset",
    "make_file_opener_for_wds",
    # Registry
    "BackendType",
    "get_backend",
]
