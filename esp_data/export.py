"""Public export API for writing datasets to supported formats."""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from esp_data.backends.webdataset_utils import (
    audio_decoder,
    audio_encoder,
    json_decoder,
    json_encoder,
)
from esp_data.io import anypath

__all__ = [
    "export_to",
    "audio_encoder",
    "json_encoder",
    "audio_decoder",
    "json_decoder",
]

_SUPPORTED_FORMATS = ("webdataset",)


def export_to(
    iterable: Iterator[dict[str, Any]] | Iterable[dict[str, Any]],
    path: str,
    format: str = "webdataset",
    **kwargs: Any,
) -> tuple[int, str]:
    """Export an iterable of samples to a file.

    Parameters
    ----------
    iterable : Iterator[dict[str, Any]] | Iterable[dict[str, Any]]
        Iterable of sample dicts to write.
    path : str
        Destination directory (local or cloud).
    format : str, optional
        Output format. Supported: ``"webdataset"``. By default ``"webdataset"``.
    **kwargs : Any
        Additional arguments passed to the format writer.
        For ``"webdataset"``: accepts ``encoder_fn``, ``shard_pattern``,
        ``maxcount``, ``maxsize`` (see `write_to_webdataset`).

    Returns
    -------
    tuple[int, str]
        A tuple containing the number of samples written and the format used.

    Raises
    ------
    ValueError
        If `format` is not supported.
    """
    if format == "webdataset":
        from esp_data.backends.webdataset_utils import write_to_webdataset

        return write_to_webdataset(iterable, anypath(path), **kwargs), "webdataset"
    raise ValueError(f"Unsupported format: {format!r}. Supported formats: {_SUPPORTED_FORMATS}")
