"""Attach precomputed selection-table TSV strings to dataset rows.

Designed for incorporating an upstream SED model's pseudo-labels as a new
column on a dataset (e.g. BirdCODE detections on top of Xeno-Canto). The
shard format produced by ``run_large_scale_postprocessing`` is:

* one ``shard_*.npz`` file per shard,
* each shard contains a ``file_ids`` array plus ``table_<i>`` entries
  storing TSV strings, with the same ordering as ``file_ids``.

This transform builds the full ``{file_id: tsv_string}`` index in RAM at
construction time and attaches the TSV string per row by matching ``id_column``
against the shard ``file_ids``. Rows with no shard entry get an empty string.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from esp_data.backends.protocol import DataBackend
from esp_data.io import filesystem_from_path

from . import register_transform

logger = logging.getLogger("esp_data")


class AddPrecomputedSelectionTablesConfig(BaseModel):
    """Configuration for the AddPrecomputedSelectionTables transform.

    Attributes
    ----------
    type : Literal["add_precomputed_selection_tables"]
        Discriminator field for transform registry.
    path : str
        Directory holding ``shard_*.npz`` selection-table files
        (local path or cloud URI such as ``gs://...``).
    id_column : str or None
        Column in the backend whose values index into the shards.  If
        ``None``, the transform falls back to the backend's
        ``_originals_path_column`` attribute (set on every esp-data
        dataset), then to a column literally named ``"audio_path"``.
    output_column : str
        Name of the column added by this transform.
    """

    type: Literal["add_precomputed_selection_tables"]
    path: str
    id_column: str | None = None
    output_column: str = "selection_table"


def _is_cloud_path(path: str) -> bool:
    """Check whether ``path`` carries a ``scheme://`` prefix.

    Returns:
        ``True`` if the path looks like a remote/cloud URI.
    """
    return "://" in path


def _list_shards(path: str) -> list[str]:
    """List every ``shard_*.npz`` under ``path``.

    Parameters
    ----------
    path : str
        Local or cloud directory holding the shards.

    Returns
    -------
    list[str]
        Sorted shard paths.  Cloud paths retain their ``scheme://`` prefix.
    """
    if _is_cloud_path(path):
        fs = filesystem_from_path(path)
        proto, stripped = path.split("://", 1)
        stripped = stripped.rstrip("/")
        matches = fs.glob(f"{stripped}/shard_*.npz")
        return sorted(f"{proto}://{m}" for m in matches)
    from pathlib import Path

    return sorted(str(p) for p in Path(path).glob("shard_*.npz"))


def _load_selection_table_shard(path: str) -> dict[str, str]:
    """Load one shard file into a ``{file_id: tsv}`` mapping.

    Parameters
    ----------
    path : str
        Local or cloud path to a ``shard_*.npz`` file.

    Returns
    -------
    dict[str, str]
        Mapping from file identifier (string) to TSV selection-table string.
    """
    if _is_cloud_path(path):
        fs = filesystem_from_path(path)
        proto, stripped = path.split("://", 1)
        with fs.open(stripped, "rb") as f:
            buf = io.BytesIO(f.read())
    else:
        with open(path, "rb") as f:
            buf = io.BytesIO(f.read())

    data = np.load(buf, allow_pickle=False)
    file_ids = data["file_ids"].tolist()
    return {str(fid): str(data[f"table_{i}"][0]) for i, fid in enumerate(file_ids)}


def _infer_id_column(backend: DataBackend, id_column: str | None) -> str:
    """Resolve which column to use as the per-row shard lookup key.

    Returns
    -------
    str
        The column name to use.

    Raises
    ------
    ValueError
        If no suitable column can be found.
    """
    if id_column is not None:
        return id_column
    col = getattr(backend, "_originals_path_column", None)
    if col is not None:
        return col
    if "audio_path" in backend.columns:
        return "audio_path"
    raise ValueError(
        "Could not infer id_column for AddPrecomputedSelectionTables. "
        "Pass `id_column` explicitly or expose `_originals_path_column` on the backend."
    )


class AddPrecomputedSelectionTables:
    """Attach precomputed per-row selection-table TSV strings.

    All shards under ``path`` are loaded once at construction time into an
    in-memory ``{file_id: tsv}`` dict.  ``__call__`` then walks the backend,
    appends the TSV per row (empty string for rows with no shard entry),
    and returns a new backend with the added column.

    Parameters
    ----------
    path : str
        Directory holding ``shard_*.npz`` files.
    id_column : str or None
        Column to use as the shard lookup key.  If ``None``, inferred at
        call time from ``backend._originals_path_column``.
    output_column : str
        Name of the column to add (default ``"selection_table"``).
    """

    def __init__(
        self,
        *,
        path: str,
        id_column: str | None = None,
        output_column: str = "selection_table",
    ) -> None:
        self.path = path
        self.id_column = id_column
        self.output_column = output_column
        self._index: dict[str, str] = self._load_index()

    def _load_index(self) -> dict[str, str]:
        """Eagerly merge all ``shard_*.npz`` files at ``self.path``.

        Returns
        -------
        dict[str, str]
            Combined ``{file_id: tsv}`` mapping across every shard.

        Raises
        ------
        ValueError
            If no ``shard_*.npz`` files are found under ``self.path``.
        """
        shard_paths = _list_shards(self.path)
        if not shard_paths:
            raise ValueError(
                f"No shard_*.npz files found under '{self.path}'. "
                "Check the directory name (date suffix, hyphen vs underscore, etc.)."
            )
        index: dict[str, str] = {}
        for shard_path in shard_paths:
            index.update(_load_selection_table_shard(shard_path))
        logger.info(
            "AddPrecomputedSelectionTables: loaded %d entries from %d shards at %s",
            len(index),
            len(shard_paths),
            self.path,
        )
        return index

    @classmethod
    def from_config(
        cls,
        cfg: AddPrecomputedSelectionTablesConfig,
    ) -> "AddPrecomputedSelectionTables":
        return cls(**cfg.model_dump(exclude={"type"}))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict[str, Any]]:
        """Add the precomputed selection-table column to ``backend``.

        Parameters
        ----------
        backend : DataBackend
            Dataset rows including the id column.

        Returns
        -------
        tuple[DataBackend, dict[str, Any]]
            New backend with the ``output_column`` added, and metadata
            ``{"matched": int, "unmatched": int}``.

        Raises
        ------
        KeyError
            If the resolved id column is not present in the backend.
        """
        id_column = _infer_id_column(backend, self.id_column)
        if id_column not in backend.columns:
            raise KeyError(
                f"Column '{id_column}' not found in backend. "
                f"Available columns: {list(backend.columns)}"
            )

        matched = 0
        unmatched = 0
        values: list[str] = []
        for row in backend:
            tsv = self._index.get(str(row[id_column]), "")
            if tsv:
                matched += 1
            else:
                unmatched += 1
            values.append(tsv)

        return (
            backend.add_column(self.output_column, values),
            {"matched": matched, "unmatched": unmatched},
        )


register_transform(AddPrecomputedSelectionTablesConfig, AddPrecomputedSelectionTables)
