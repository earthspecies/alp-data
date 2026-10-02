"""Read a pack written by `alp_data.export.pack` as a map-style `Dataset`."""

from __future__ import annotations

import warnings
from typing import Any, Iterator, Sequence

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import model_validator

from alp_data.backends import BackendType
from alp_data.dataset import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    register_config,
    register_dataset,
)
from alp_data.export.columns import (
    BOOKKEEPING_COLS,
    OFFSET_COL,
    SHARD_COL,
    SIZE_COL,
)
from alp_data.export.serializers import decode_audio, decode_value
from alp_data.io import AnyPathT, anypath, filesystem_from_path
from alp_data.io.packed_media_store import PackedMediaStore

CONFIG_FILE = "config.yaml"
TABLE_FILE = "table.parquet"
MEDIA_DIR = "media"


def _as_list(value: Any) -> Any:  # noqa: ANN401
    """Turn the numpy arrays pyarrow makes out of list columns back into lists.

    Parameters
    ----------
    value : Any
        A cell of a pandas column that came from an arrow list column.

    Returns
    -------
    Any
        A list when `value` is a numpy array, otherwise `value` unchanged.
    """
    return value.tolist() if isinstance(value, np.ndarray) else value


@register_config
class PackedDatasetConfig(DatasetConfig):
    """Configuration for loading a pack.

    Everything that was frozen at pack time (split, version, sample rate,
    source transforms) is read from the pack's `config.yaml`. Of the inherited
    fields, `sample_rate`, `data_root`, and `streaming` therefore have no
    effect and setting them raises a warning; `split` may be given and is
    checked against the pack when the dataset is built.

    Attributes
    ----------
    dataset_name : str
        Always `"packed_dataset"`.
    path : str
        Directory holding `config.yaml`, `table.parquet`, and `media/`.
        Any `anypath` target.
    backend : BackendType
        Table backend, `"pandas"` or `"polars"`.
    transformations : list | None
        Transforms to apply on top of the packed table.
    output_take_and_give : dict[str, str] | None
        Output mapping applied on top of the one frozen into the pack.
    """

    dataset_name: str = "packed_dataset"
    path: str = ""

    @model_validator(mode="after")
    def _warn_about_fixed_fields(self) -> "PackedDatasetConfig":
        ignored = [
            name
            for name, unset in (
                ("sample_rate", self.sample_rate is None),
                ("data_root", self.data_root is None),
                ("streaming", not self.streaming),
            )
            if not unset
        ]
        if ignored:
            warnings.warn(
                f"PackedDatasetConfig ignores {', '.join(ignored)}: a pack fixes them at "
                "pack time and reports them through PackedDataset.info and .sample_rate.",
                UserWarning,
                stacklevel=2,
            )
        return self


@register_dataset
class PackedDataset(Dataset):
    """A dataset backed by a pack: a parquet table plus tar shards of audio.

    Reading sample `i` is one table lookup and one range read. The audio is
    decoded as it was stored: no mono conversion or resampling happens here,
    because the source dataset already did that before packing.

    Parameters
    ----------
    path : str | AnyPathT
        Pack directory.
    backend : BackendType, optional
        Table backend, by default `"polars"`.
    output_take_and_give : dict[str, str] | None, optional
        Output mapping applied on top of the frozen one.

    Raises
    ------
    FileNotFoundError
        If `path` holds no `config.yaml`.
    """

    info = DatasetInfo(
        name="packed_dataset",
        owner="alp_data",
        split_paths={"train": "virtual://packed"},
        version="0.0.0",
        description="A dataset frozen with alp_data.export.pack",
        sources="alp_data",
    )

    def __init__(
        self,
        path: str | AnyPathT,
        backend: BackendType = "polars",
        output_take_and_give: dict[str, str] | None = None,
    ) -> None:
        super().__init__(output_take_and_give=output_take_and_give, backend=backend)
        self.path = anypath(str(path))
        self._fs = filesystem_from_path(self.path)
        config_path = str(self.path / CONFIG_FILE)
        if not self._fs.exists(config_path):
            raise FileNotFoundError(f"No {CONFIG_FILE} at {self.path}; is this a pack?")
        with self._fs.open(config_path, "r") as f:
            self.pack_config: dict[str, Any] = yaml.safe_load(f)

        self.split = self.pack_config.get("split") or "train"
        self.sample_rate = self.pack_config.get("sample_rate")
        self.audio_key: str = self.pack_config["audio_key"]
        self.sample_rate_key: str | None = self.pack_config.get("sample_rate_key")
        self.opaque_columns: dict[str, str] = self.pack_config.get("opaque_columns") or {}
        self.info = DatasetInfo(
            name=self.pack_config.get("name") or "packed_dataset",
            owner="alp_data",
            split_paths={self.split: str(self.path)},
            version=self.pack_config.get("version") or "0.0.0",
            description=f"Pack of {self.pack_config.get('name')} at {self.path}",
            sources=str(self.pack_config.get("source", {}).get("dataset_name", "unknown")),
        )
        shards = self.pack_config["shards"]
        self._store = PackedMediaStore(
            self.path / MEDIA_DIR,
            [s["name"] for s in shards],
            [s["size"] for s in shards] if all("size" in s for s in shards) else None,
        )
        self._load()

    def _load(self) -> None:
        with self._fs.open(str(self.path / TABLE_FILE), "rb") as f:
            table = pq.read_table(f)
        if self._backend_class.__name__ == "PandasBackend":
            df = table.to_pandas()
            for name, field in zip(df.columns, table.schema, strict=True):
                if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
                    df[name] = df[name].map(_as_list)
            self._data = self._backend_class(df)
        else:
            self._data = self._backend_class(pl.from_arrow(table))

    @classmethod
    def from_config(cls, cfg: PackedDatasetConfig) -> tuple[PackedDataset, dict[str, Any]]:
        """Build a `PackedDataset` from its config.

        Parameters
        ----------
        cfg : PackedDatasetConfig
            The configuration.

        Returns
        -------
        tuple[PackedDataset, dict[str, Any]]
            The dataset and transform metadata, empty when no transforms ran.

        Raises
        ------
        ValueError
            If the config names a `split` and the pack holds a different one.
        """
        ds = cls(cfg.path, backend=cfg.backend, output_take_and_give=cfg.output_take_and_give)
        if "split" in cfg.model_fields_set and cfg.split != ds.split:
            raise ValueError(
                f"Config asks for split {cfg.split!r} but the pack at {cfg.path} holds "
                f"{ds.split!r}; a pack is one split, frozen at pack time."
            )
        meta = ds.apply_transformations(cfg.transformations) if cfg.transformations else {}
        return ds, meta

    @property
    def columns(self) -> Sequence[str]:
        return self._data.columns

    @property
    def available_splits(self) -> Sequence[str]:
        return [self.split]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        n = len(self._data)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {n}")
        return self._process(self._data[idx])

    def __str__(self) -> str:
        return (
            f"PackedDataset({self.info.name} v{self.info.version}, {len(self)} rows, {self.path})"
        )

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        data = self._store.read(int(row[SHARD_COL]), int(row[OFFSET_COL]), int(row[SIZE_COL]))
        audio, sample_rate = decode_audio(data)
        item = {
            key: decode_value(value, self.opaque_columns.get(key))
            for key, value in row.items()
            if key not in BOOKKEEPING_COLS
        }
        item[self.audio_key] = audio
        if self.sample_rate_key is not None:
            item[self.sample_rate_key] = int(sample_rate)
        if self.output_take_and_give:
            item = {give: item[take] for take, give in self.output_take_and_give.items()}
        return item

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_fs", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._fs = filesystem_from_path(self.path)
