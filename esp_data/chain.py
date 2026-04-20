import copy
import gc
import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from esp_data.backends.polars_backend import PolarsBackend
from esp_data.dataset import (
    ChainedDatasetConfig,
    Dataset,
    DatasetConfig,
    DatasetInfo,
    SaveFormat,
    dataset_from_config,
    register_dataset,
)
from esp_data.transforms import transform_from_config

logger = logging.getLogger(__name__)


def _base_dataset_key(cfg: DatasetConfig) -> str:
    """Return a hashable key identifying the base dataset (excluding transforms).

    Two configs that differ only in ``transformations`` will produce the
    same key, allowing the loaded data to be shared.

    Returns
    -------
    str
        A JSON string of all config fields except ``transformations``.
    """
    d = cfg.model_dump(exclude={"transformations"})
    return json.dumps(d, sort_keys=True, default=str)


def _transforms_prefix_key(base_key: str, transforms: list) -> str:
    """Return a hashable key for a base dataset + a prefix of applied transforms.

    Parameters
    ----------
    base_key : str
        Key from ``_base_dataset_key`` identifying the base dataset.
    transforms : list
        Ordered list of transform config objects (Pydantic models) applied so far.

    Returns
    -------
    str
        A deterministic JSON string encoding ``base_key`` and the transform configs.
    """
    tf_dumps = [cfg.model_dump() for cfg in transforms]
    return base_key + "|" + json.dumps(tf_dumps, sort_keys=True, default=str)


CHAIN_CACHE_DIR = Path("./chain_cache")


def _cleanup_dir(path: str) -> None:
    """Remove a directory, ignoring errors."""
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


class ChainException(Exception):
    """Exception raised when dataset chaining fails."""

    pass


@register_dataset
class ChainedDataset(Dataset):
    """Helper class to chain multiple datasets for iteration and indexing.

    This class allows iterating over multiple datasets as if they were a single
    dataset.  When built via :meth:`from_config`, the transformed data for each
    sub-dataset is written to an Arrow IPC file under ``./chain_cache/`` and
    then memory-mapped at runtime so that physical RAM usage is managed by the
    OS page cache rather than the Python heap.

    Parameters
    ----------
    datasets : list[Dataset]
        List of datasets to concatenate for iteration.

    Examples
    --------
    >>> from esp_data.datasets import InsectSet459, BirdSet
    >>> from esp_data.chain import ChainedDataset
    >>> dataset1 = InsectSet459(split="validation")
    >>> dataset2 = BirdSet(split="HSN-test")
    >>> concat_iter = ChainedDataset([dataset1, dataset2])
    >>> total_length = len(dataset1) + len(dataset2)
    >>> item = next(iter(concat_iter))
    >>> assert len(concat_iter) == total_length, \
        "Concatenated iterator length should match sum of source datasets lengths"
    """

    info = DatasetInfo(
        name="chained_dataset",
        owner="ESP Data Team",
        split_paths={"chained": "virtual://chained_dataset"},
        version="0.2.0",
        description="A dataset created by chaining multiple datasets for iteration.",
        sources=["Multiple datasets"],
        license="CC0-1.0",
    )

    def __init__(self, datasets: list[Dataset]) -> None:
        if not datasets:
            raise ChainException("At least one dataset must be provided")

        if not all(isinstance(ds, Dataset) for ds in datasets):
            raise ChainException("All objects must be Dataset instances")

        streaming_modes = {ds.streaming for ds in datasets}
        if len(streaming_modes) > 1:
            raise ChainException(
                "All datasets must have the same streaming mode "
                "to be concatenated into a ConcatenatedDataset."
            )
        _streaming = streaming_modes.pop()

        super().__init__(streaming=_streaming)

        self._source_datasets = datasets
        try:
            self._lengths = [len(ds) for ds in datasets]
            self._total_length = sum(self._lengths)
        except RuntimeError:
            self._lengths = []
            self._total_length = -1

        self._all_columns: list[str] = []
        col_set: set[str] = set()
        for ds in datasets:
            col_set.update(ds.columns)
        self._all_columns = sorted(col_set)

        self._data: PolarsBackend | None = None
        self._cache_dir: str | None = None

    @property
    def columns(self) -> list[str]:
        if self._data is not None:
            return [c for c in self._data.columns if c != "_chain_idx"]
        return self._all_columns

    @property
    def available_splits(self) -> list[str]:
        return ["chained"]

    def _load(self) -> None:
        pass

    def __del__(self) -> None:
        if getattr(self, "_cache_dir", None) is not None:
            _cleanup_dir(self._cache_dir)

    def __len__(self) -> int:
        if self._streaming:
            raise RuntimeError("Length is not supported in streaming mode")
        return self._total_length

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._data is not None:
            for row in self._data:
                chain_idx = int(row.pop("_chain_idx"))
                yield self._source_datasets[chain_idx]._process(row)
        else:
            for dataset in self._source_datasets:
                if getattr(dataset, "_data", None) is None:
                    continue
                for item in dataset:
                    yield item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get item by global index across chained datasets.

        Parameters
        ----------
        idx : int
            Global index across all chained datasets.

        Returns
        -------
        dict[str, Any]
            The item at the specified global index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        RuntimeError
            If indexing is attempted in streaming mode.
        """
        if self._streaming:
            raise RuntimeError("Indexing is not supported in streaming mode")

        if idx < 0:
            raise IndexError("Negative indexing is not supported")

        if idx >= self._total_length:
            raise IndexError(
                f"Index {idx} out of bounds for concatenated dataset of length {self._total_length}"
            )

        if self._data is not None:
            for attempt in range(11):
                _idx = idx if attempt == 0 else random.randint(0, len(self._data) - 1)
                row = dict(self._data[_idx])
                chain_idx = int(row.pop("_chain_idx"))
                try:
                    return self._source_datasets[chain_idx]._process(row)
                except (FileNotFoundError, OSError):
                    if attempt >= 10:
                        raise
                    logging.warning(
                        f"ChainedDataset: skipping missing audio at idx={_idx}, attempt {attempt + 1}/10"
                    )

        cumulative_length = 0
        for dataset, length in zip(self._source_datasets, self._lengths, strict=True):
            if idx < cumulative_length + length:
                return dataset[idx - cumulative_length]
            cumulative_length += length

    @classmethod
    def from_config(
        cls, chain_config: ChainedDatasetConfig
    ) -> tuple["ChainedDataset", dict[str, Any]]:
        """Create a ChainedDataset from a ChainedDatasetConfig object.

        When multiple entries share the same base dataset (same name, split,
        sample rate, etc.) only the first triggers a GCS/disk read.  Subsequent
        entries get a cheap clone of the cached backend and then apply their own
        transformations on top.

        Transform results are also cached at each step.  When two entries
        share the same base dataset *and* the same leading transforms
        (e.g. ``filter -> window_annotations -> annotation_features``) but
        diverge later (e.g. different ``chat`` template), only the
        divergent tail is recomputed.

        Cache entries are evicted eagerly: once all dataset entries that
        could benefit from a given cached prefix have been processed, that
        prefix is removed from the cache to free memory.

        After all entries are built their backends are written to Arrow IPC
        files under ``./chain_cache/`` and the in-memory DataFrames are freed.
        The final ``ChainedDataset`` reads from memory-mapped IPC so physical
        RAM is managed by the OS page cache.

        Parameters
        ----------
        chain_config : ChainedDatasetConfig
            Configuration object specifying the datasets to chain.

        Returns
        -------
        tuple[ChainedDataset, dict]
            A tuple containing the ChainedDataset instance
            and metadata about transformations applied.
        """
        base_cache: dict[str, Dataset] = {}
        transform_cache: dict[str, tuple[Any, dict[str, Any]]] = {}
        datasets: list[Dataset] = []
        lengths: list[int] = []
        metadata: dict[str, Any] = {}

        config_hash = hashlib.sha256(
            chain_config.model_dump_json(exclude_none=False).encode()
        ).hexdigest()[:16]
        CHAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{config_hash}_pid{os.getpid()}_",
                dir=CHAIN_CACHE_DIR,
            )
        )
        ipc_files: list[str] = []
        in_memory_backends: dict[int, Any] = {}

        # ------------------------------------------------------------------
        # Pre-scan: reference counts for transform prefix cache eviction
        # ------------------------------------------------------------------
        prefix_refcount: dict[str, int] = {}
        for cfg in chain_config.datasets:
            if not cfg.transformations:
                continue
            ck = _base_dataset_key(cfg)
            for i in range(1, len(cfg.transformations) + 1):
                pk = _transforms_prefix_key(ck, cfg.transformations[:i])
                prefix_refcount[pk] = prefix_refcount.get(pk, 0) + 1

        # ------------------------------------------------------------------
        # Pre-scan: reference counts for base dataset cache eviction
        # ------------------------------------------------------------------
        base_refcount: dict[str, int] = {}
        for cfg in chain_config.datasets:
            ck = _base_dataset_key(cfg)
            base_refcount[ck] = base_refcount.get(ck, 0) + 1

        # ------------------------------------------------------------------
        # Main construction loop
        # ------------------------------------------------------------------
        all_columns: set[str] = set()
        is_streaming = False

        num_entries = len(chain_config.datasets)
        for entry_idx, cfg in enumerate(chain_config.datasets):
            entry_label = f"[{entry_idx}/{num_entries}] {cfg.dataset_name}/{cfg.split}"
            cache_key = _base_dataset_key(cfg)

            if cache_key not in base_cache:
                no_tf_cfg = cfg.model_copy(update={"transformations": None})
                base_ds, _ = dataset_from_config(no_tf_cfg)
                base_cache[cache_key] = base_ds
                if not base_ds.streaming:
                    logger.info(
                        "ChainedDataset: loaded base %s (%d rows)",
                        entry_label,
                        len(base_ds),
                    )
                else:
                    is_streaming = True
                    logger.info(
                        "ChainedDataset: loaded base %s (streaming)",
                        entry_label,
                    )

            ds = copy.copy(base_cache[cache_key])

            meta: dict[str, Any] = {}
            if cfg.transformations:
                transforms = cfg.transformations

                best_prefix_len = 0
                for i in range(len(transforms), 0, -1):
                    prefix_key = _transforms_prefix_key(cache_key, transforms[:i])
                    if prefix_key in transform_cache:
                        best_prefix_len = i
                        break

                if best_prefix_len > 0:
                    prefix_key = _transforms_prefix_key(cache_key, transforms[:best_prefix_len])
                    cached_data, cached_meta = transform_cache[prefix_key]
                    ds._data = cached_data.copy()
                    meta = dict(cached_meta)
                    logger.info(
                        "ChainedDataset: %s — reusing cached result for %d/%d transforms",
                        entry_label,
                        best_prefix_len,
                        len(transforms),
                    )
                else:
                    ds._data = base_cache[cache_key]._data.copy()

                for i in range(best_prefix_len, len(transforms)):
                    tf_cfg = transforms[i]
                    tf = transform_from_config(tf_cfg)
                    ds._data, tf_meta = tf(ds._data)
                    meta[tf_cfg.type] = tf_meta

                    applied_key = _transforms_prefix_key(cache_key, transforms[: i + 1])
                    if applied_key not in transform_cache:
                        transform_cache[applied_key] = (ds._data, dict(meta))

                    if not ds._data.is_streaming and len(ds._data) == 0:
                        logger.warning(
                            "ChainedDataset: %s has 0 rows after transform '%s' "
                            "(%d/%d), skipping remaining transforms",
                            entry_label,
                            tf_cfg.type,
                            i + 1,
                            len(transforms),
                        )
                        break

                for i in range(1, len(transforms) + 1):
                    pk = _transforms_prefix_key(cache_key, transforms[:i])
                    prefix_refcount[pk] -= 1
                    if prefix_refcount[pk] <= 0 and pk in transform_cache:
                        del transform_cache[pk]
                        logger.debug("ChainedDataset: evicted cache for prefix %s", pk[:80])
            else:
                ds._data = base_cache[cache_key]._data.copy()

            # ----------------------------------------------------------
            # For non-streaming datasets: write to IPC and free memory
            # ----------------------------------------------------------
            if not is_streaming:
                entry_len = len(ds._data)
                if entry_len == 0:
                    logger.warning(
                        "ChainedDataset: %s has 0 rows after transforms, skipping",
                        entry_label,
                    )
                    ds._data = None
                    datasets.append(ds)
                    metadata[f"{cfg.dataset_name}_metadata"] = meta
                    base_refcount[cache_key] -= 1
                    if base_refcount[cache_key] <= 0 and cache_key in base_cache:
                        del base_cache[cache_key]
                    continue
                lengths.append(entry_len)
                all_columns.update(ds._data.columns)

                ipc_path = str(cache_dir / f"{entry_idx}.arrow")
                try:
                    ds._data._df.with_columns(
                        pl.lit(entry_idx).cast(pl.Int32).alias("_chain_idx")
                    ).write_ipc(ipc_path)
                    ipc_files.append(ipc_path)
                    logger.info(
                        "ChainedDataset: built entry %d/%d (%s/%s, %d rows) -> %s",
                        entry_idx,
                        num_entries,
                        cfg.dataset_name,
                        cfg.split,
                        entry_len,
                        ipc_path,
                    )
                    ds._data = None
                except Exception as exc:
                    logger.warning(
                        "ChainedDataset: IPC write failed for entry %d (%s/%s): %s. "
                        "Keeping in memory.",
                        entry_idx,
                        cfg.dataset_name,
                        cfg.split,
                        exc,
                    )
                    in_memory_backends[entry_idx] = ds._data
                    ds._data = None

            datasets.append(ds)
            metadata[f"{cfg.dataset_name}_metadata"] = meta

            # ----------------------------------------------------------
            # Eagerly free base dataset when all its consumers are done
            # ----------------------------------------------------------
            base_refcount[cache_key] -= 1
            if base_refcount[cache_key] <= 0 and cache_key in base_cache:
                logger.info(
                    "ChainedDataset: freed base dataset cache for %s/%s",
                    cfg.dataset_name,
                    cfg.split,
                )
                del base_cache[cache_key]

        del base_cache, transform_cache, prefix_refcount, base_refcount
        gc.collect()

        # ------------------------------------------------------------------
        # Streaming path: fall back to the old behaviour (no IPC)
        # ------------------------------------------------------------------
        if is_streaming:
            _cleanup_dir(str(cache_dir))
            chained = cls(datasets)
            chained._cache_dir = None
            return chained, metadata

        # ------------------------------------------------------------------
        # Consolidate: memory-map all IPC files into a single DataFrame
        # ------------------------------------------------------------------
        mmap_dfs: list[pl.DataFrame] = []
        for f in ipc_files:
            mmap_dfs.append(pl.read_ipc(f, memory_map=True))
        for entry_idx, backend in in_memory_backends.items():
            mmap_dfs.append(
                backend._df.with_columns(pl.lit(entry_idx).cast(pl.Int32).alias("_chain_idx"))
            )
        del in_memory_backends

        if mmap_dfs:
            consolidated_df = pl.concat(mmap_dfs, how="diagonal_relaxed", rechunk=False)
            del mmap_dfs
            consolidated = PolarsBackend(consolidated_df)
        else:
            consolidated = None

        total_rows = sum(lengths)
        logger.info(
            "ChainedDataset: consolidated %d sub-datasets (%d total rows, %d IPC files)",
            len(datasets),
            total_rows,
            len(ipc_files),
        )

        gc.collect()

        # ------------------------------------------------------------------
        # Assemble the ChainedDataset, bypassing __init__ len() calls
        # (sub-datasets no longer have _data)
        # ------------------------------------------------------------------
        chained = object.__new__(cls)
        chained._streaming = False
        chained._backend_class = None
        chained.output_take_and_give = None
        chained._source_datasets = datasets
        chained._lengths = lengths
        chained._total_length = total_rows
        chained._all_columns = sorted(all_columns)
        chained._data = consolidated
        chained._cache_dir = str(cache_dir)

        return chained, metadata

    def save_data(self, path: str, fmt: SaveFormat = "csv") -> None:
        """Save the consolidated data to a single file.

        Parameters
        ----------
        path : str
            Destination file path (local or cloud).
        fmt : SaveFormat
            Output format: ``"csv"`` or ``"jsonl"``.

        Raises
        ------
        ChainException
            If no data is available to save.
        """
        if self._data is None:
            raise ChainException("No data available to save.")

        cols = [c for c in self._data.columns if c != "_chain_idx"]
        saveable = PolarsBackend(self._data._df.select(cols))
        if fmt == "csv":
            saveable.to_csv(path)
        elif fmt == "jsonl":
            saveable.to_jsonl(path)

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version})\n"
            f"Description: {self.info.description}\n"
            f"Length: {self._total_length}\n"
            f"Columns: {', '.join(self.columns)}\n"
            f"Source datasets: {len(self._source_datasets)}"
        )
