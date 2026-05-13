"""Generic dataset backed by any supported backend."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Sequence

from esp_data.backends import BackendType, get_backend
from esp_data.dataset import Dataset, DatasetInfo, GenericDatasetConfig, register_dataset
from esp_data.io import AnyPathT, anypath, read_yaml, filesystem_from_path

logger = logging.getLogger(__name__)


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
        **kwargs: Any,
    ) -> None:
        """Load a dataset from a path.

        The backend and streaming mode are determined automatically from
        ``info.yaml`` (written by `Dataset.save_to`) or ``config.yaml`` when
        present. Falls back to ``"polars"`` / non-streaming if no config file
        is found.

        Parameters
        ----------
        path : str | AnyPathT
            Source path (local or cloud). For webdataset directories, must
            contain sharded tar files and an ``info.yaml``.
        **kwargs : Any
            Additional arguments forwarded to the backend's ``from_path``
            (e.g. ``data_processor`` for the webdataset backend).
        """
        self.path = anypath(path)
        fs = filesystem_from_path(self.path)
        info_path = self.path / "info.yaml"
        if fs.exists(info_path):
            info_dict = read_yaml(info_path)
            self.info = DatasetInfo(**info_dict)
            logger.info(f"Loaded dataset info : {self.info}")
            logger.info(f"Loaded dataset info from {info_path}")

        backend = self.info.backend
        streaming = self.info.streaming
        super().__init__(backend=backend, streaming=streaming)

        # Step 3: load data with the resolved backend.
        self._data = get_backend(backend).from_path(anypath(path), streaming=streaming, **kwargs)

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
