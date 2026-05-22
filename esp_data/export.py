"""Public export API for writing datasets to supported formats."""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from esp_data.io import anypath

_SUPPORTED_FORMATS = ("webdataset",)


def export_dataset(
    iterable: Iterator[dict[str, Any]] | Iterable[dict[str, Any]],
    path: str,
    format: str = "webdataset",
    **kwargs: Any,
) -> dict[str, Any]:
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
    dict[str, Any] :
        Summary of export results

    Raises
    ------
    ValueError
        If `format` is not supported.
    """
    if format == "webdataset":
        from esp_data.backends.webdataset_utils import write_to_webdataset

        return write_to_webdataset(iterable, anypath(path), **kwargs), "webdataset"
    raise ValueError(f"Unsupported format: {format!r}. Supported formats: {_SUPPORTED_FORMATS}")
