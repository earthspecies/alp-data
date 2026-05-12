"""Generic dataset backed by any supported backend."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Sequence

from esp_data.backends import BackendType, get_backend
from esp_data.dataset import Dataset, DatasetInfo, GenericDatasetConfig, register_dataset
from esp_data.io import AnyPathT, anypath, read_yaml

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
        backend: BackendType = "polars",
        streaming: bool = False,
        **kwargs: Any,
    ) -> None:
        """Load a dataset from a path.

        Parameters
        ----------
        path : str | AnyPathT
            Source path (local or cloud). For ``backend="webdataset"``, must be
            a directory containing sharded tar files.
        backend : BackendType, optional
            Backend to use for loading. By default ``"polars"``.
        streaming : bool, optional
            Whether to use streaming mode. By default ``False``.
        **kwargs : Any
            Additional arguments forwarded to the backend's ``from_path``.
        """
        super().__init__(backend=backend, streaming=streaming)

        self._data = get_backend(backend).from_path(anypath(path), streaming=streaming, **kwargs)

        resolved = anypath(path)
        config_dir = resolved if resolved.is_dir() else resolved.parent
        config_file = config_dir / "config.yaml"
        if config_file.exists():
            try:
                self.info = DatasetInfo.model_validate(read_yaml(str(config_file)))
            except Exception as e:
                logger.warning("Failed to load config.yaml from %s: %s", config_file, e)

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
        return cls(
            dataset_config.path,
            backend=dataset_config.backend,
            streaming=dataset_config.streaming,
        ), {}
