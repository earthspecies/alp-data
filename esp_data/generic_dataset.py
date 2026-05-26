"""Generic dataset backed by any supported backend."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Sequence

from esp_data.backends import BackendType
from esp_data.dataset import Dataset, DatasetInfo, GenericDatasetConfig, register_dataset
from esp_data.io import AnyPathT, anypath, filesystem_from_path, read_yaml

logger = logging.getLogger(__name__)

_FORMAT_DEFAULT_BACKEND: dict[str, BackendType] = {
    "webdataset": "webdataset",
    "parquet": "polars",
}

_FORMAT_COMPATIBLE_BACKENDS: dict[str, set[BackendType]] = {
    "webdataset": {"webdataset"},
    "parquet": {"pandas", "polars"},
}


@register_dataset
class GenericDataset(Dataset):
    """A dataset loaded directly from a path rather than a registered source.

    Wraps any supported backend (polars, pandas, webdataset) with a minimal
    Dataset interface. Intended as the return type of `Dataset.from_path` and
    as a config-driven loader via `GenericDatasetConfig`.

    Unlike named dataset classes (e.g. `AnuraSetStrong`), `GenericDataset`
    performs no custom per-sample processing — `__iter__` and `__getitem__`
    yield rows directly from the backend.

    If a ``config.yaml`` file exists alongside the data (in the same directory),
    its contents are used to populate `info`.
    """

    info = DatasetInfo(
        name="generic_dataset",
        owner="ESP Data Team",
        split_paths={"default": "virtual://generic"},
        version="0.1.0",
        description="A dataset loaded from a local or cloud path.",
        sources=["Unknown"],
        license="Unknown",
    )

    def __init__(
        self,
        path: str | AnyPathT,
        *,
        backend: BackendType | None = None,
        **kwargs: Any,
    ) -> None:
        """Load a dataset from a path that points to a directory.

        The export format is read from ``config.yaml`` (written by `Dataset.save_to`).
        If `backend` is not provided, a default backend is chosen for the format.
        If `backend` is provided, it must be compatible with the stored format.

        Parameters
        ----------
        path : str | AnyPathT
            Source path (local or cloud).
        backend : BackendType | None
            Backend to use for loading. If ``None``, the default backend for the
            stored format is used (``"pandas"`` for ``"parquet"``,
            ``"webdataset"`` for ``"webdataset"``).
        **kwargs : Any
            Additional arguments forwarded to the backend's ``from_path``
            (e.g. ``data_processor`` for the webdataset backend).

        Raises
        ------
        ValueError
            If no ``config.yaml`` is found, or if `backend` is incompatible
            with the stored format.
        """
        self.path = anypath(path)
        fs = filesystem_from_path(self.path)
        info_path = self.path / "config.yaml"
        if fs.exists(info_path):
            config_dict = read_yaml(info_path)
            self.info = DatasetInfo(**config_dict["info"])
        else:
            raise ValueError(
                f"No config.yaml found at {info_path}. Cannot infer format or streaming mode."
            )

        # Support both old configs (backend key) and new configs (format key).
        fmt = config_dict.get("format") or config_dict.get("backend")

        if backend is not None:
            compatible = _FORMAT_COMPATIBLE_BACKENDS.get(fmt, {backend})
            if backend not in compatible:
                raise ValueError(
                    f"Backend {backend!r} is not compatible with format {fmt!r}. "
                    f"Compatible backends: {compatible}"
                )
            resolved_backend: BackendType = backend
        else:
            resolved_backend = _FORMAT_DEFAULT_BACKEND.get(fmt, fmt)

        super().__init__(backend=resolved_backend, streaming=config_dict["streaming"])

        self._data = self._backend_class.from_path(self.path, streaming=self._streaming, **kwargs)

    @property
    def available_splits(self) -> Sequence[str]:
        return list(self.info.split_paths.keys())

    @property
    def columns(self) -> Sequence[str]:
        return self._data.columns

    def _load(self) -> None:
        pass

    def __len__(self) -> int:
        if self._streaming:
            raise NotImplementedError(
                "Length is not available for streaming backends. Iterate instead."
            )
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._streaming:
            raise RuntimeError(
                "Random access is not supported for streaming backends. Iterate instead."
            )
        return self._data[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from self._data

    def __str__(self) -> str:
        return f"GenericDataset(name={self.info.name}, version={self.info.version})"

    @classmethod
    def from_config(  # type: ignore[override]
        cls, dataset_config: GenericDatasetConfig
    ) -> tuple["GenericDataset", dict[str, Any]]:
        """Create a GenericDataset from a :class:`GenericDatasetConfig`.

        Parameters
        ----------
        dataset_config : GenericDatasetConfig
            Configuration specifying path, backend, and streaming mode.

        Returns
        -------
        tuple[GenericDataset, dict]
            Dataset instance and empty metadata dict.
        """
        return cls(dataset_config.path), {}
