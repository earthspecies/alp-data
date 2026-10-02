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


CHAIN_CACHE_DIR = Path(os.environ.get("ESP_DATA_CHAIN_CACHE_DIR", "./chain_cache"))

# Sidecar file written alongside the Arrow files of a *persistent* chain cache.
CHAIN_CACHE_SIDECAR = "chain_cache_manifest.json"


def _esp_data_code_version() -> str:
    """Best-effort esp_data code version, used to invalidate persistent caches
    when the library changes.

    Overridable via ``ESP_DATA_CHAIN_CACHE_CODE_VERSION`` (e.g. a git SHA the
    launcher injects). Falls back to the installed package version, or ``"dev"``
    for an editable checkout without version metadata.

    Returns
    -------
    str
        A short string identifying the code version.
    """
    override = os.environ.get("ESP_DATA_CHAIN_CACHE_CODE_VERSION")
    if override:
        return override
    try:
        from importlib.metadata import version

        return version("esp-data")
    except Exception:
        return "dev"


def chain_cache_signature(chain_config: ChainedDatasetConfig) -> str:
    """Deterministic signature keying a persistent chain cache directory.

    Combines the full chain config, the esp_data code version, and an optional
    manual data-version tag (``ESP_DATA_CHAIN_CACHE_DATA_VERSION``). The config
    hash cannot see *remote source-data content* (e.g. a re-uploaded manifest
    CSV whose path is unchanged), so the data-version tag is the escape hatch:
    bump it to force a rebuild after a source-data change.

    Parameters
    ----------
    chain_config : ChainedDatasetConfig
        The resolved chain configuration.

    Returns
    -------
    str
        A filesystem-safe signature of the form ``{config_hash}_{combined}``.
    """
    config_hash = hashlib.sha256(
        chain_config.model_dump_json(exclude_none=False).encode()
    ).hexdigest()[:16]
    code = _esp_data_code_version()
    data_ver = os.environ.get("ESP_DATA_CHAIN_CACHE_DATA_VERSION", "0")
    combined = hashlib.sha256(f"{config_hash}|{code}|{data_ver}".encode()).hexdigest()[:12]
    return f"{config_hash}_{combined}"


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
        # Whether this instance owns (and must clean up) its cache dir. A cache
        # loaded from a persistent, externally-managed location sets this False
        # so ranks exiting never delete a shared cache.
        self._owns_cache_dir: bool = True

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
        if getattr(self, "_owns_cache_dir", True) and getattr(self, "_cache_dir", None) is not None:
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

        # Per-sample fault tolerance: a single unreadable/corrupt example (e.g. a
        # missing GCS object or a decode error) must not crash a long distributed
        # run. On failure we log and retry with a different random index so the
        # returned batch size stays identical across ranks (keeping DDP
        # collectives in sync). We re-raise only after every attempt fails, which
        # signals a systemic problem rather than a sporadic bad file.
        max_retries = int(os.environ.get("ESP_DATA_GETITEM_MAX_RETRIES", "10"))
        current_idx = idx
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self._load_item(current_idx)
            except Exception as exc:  # noqa: BLE001 - tolerate sporadic data read failures
                last_exc = exc
                logger.warning(
                    "ChainedDataset.__getitem__ failed for index %d [%s] (attempt %d/%d): %s: %s",
                    current_idx,
                    self._dataset_label_for_index(current_idx),
                    attempt + 1,
                    max_retries + 1,
                    type(exc).__name__,
                    exc,
                )
                current_idx = random.randint(0, self._total_length - 1)

        raise RuntimeError(
            f"ChainedDataset.__getitem__ failed after {max_retries + 1} attempts; "
            f"last error: {last_exc}"
        ) from last_exc

    def _dataset_label_for_index(self, idx: int) -> str:
        """Best-effort source-dataset label for an index, used in error logs.

        Failures in ``__getitem__`` only report the raised exception, which is
        not enough to attribute a bad row to its source dataset in a large
        chained mixture. This resolves the index back to its source dataset
        name (falling back to the class name) and must never raise.
        """
        try:
            if self._data is not None:
                chain_idx = int(self._data[idx]["_chain_idx"])
                dataset = self._source_datasets[chain_idx]
            else:
                cumulative_length = 0
                dataset = None
                for candidate, length in zip(self._source_datasets, self._lengths, strict=True):
                    if idx < cumulative_length + length:
                        dataset = candidate
                        break
                    cumulative_length += length
                if dataset is None:
                    return "unknown"
            info = getattr(dataset, "info", None)
            return getattr(info, "name", None) or type(dataset).__name__
        except Exception:  # noqa: BLE001 - labeling must never break the data path
            return "unknown"

    def _load_item(self, idx: int) -> dict[str, Any]:
        """Load and process a single item by global index (without retry).

        Parameters
        ----------
        idx : int
            Global index across all chained datasets. Assumed to be in-bounds.

        Returns
        -------
        dict[str, Any]
            The processed item at the specified global index.
        """
        if self._data is not None:
            row = self._data[idx]
            chain_idx = int(row.pop("_chain_idx"))
            return self._source_datasets[chain_idx]._process(row)

        cumulative_length = 0
        for dataset, length in zip(self._source_datasets, self._lengths, strict=True):
            if idx < cumulative_length + length:
                return dataset[idx - cumulative_length]
            cumulative_length += length

    @classmethod
    def from_config(
        cls,
        chain_config: ChainedDatasetConfig,
        *,
        cache_dir: str | Path | None = None,
        owns_cache_dir: bool = True,
        resume: bool = False,
    ) -> tuple["ChainedDataset", dict[str, Any]]:
        """Create a ChainedDataset from a ChainedDatasetConfig object.

        When ``resume`` is True, any entry whose ``{entry_idx}.arrow`` already
        exists under ``cache_dir`` is reused as-is (no base reload / transform
        re-run), so a crashed persistent build can continue where it stopped
        instead of restarting. Safe only when ``cache_dir`` is keyed to this
        exact config (as in :meth:`build_persistent_cache`).

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

        if cache_dir is None:
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
        else:
            # Caller supplied an explicit cache dir (e.g. a persistent build):
            # write Arrow files directly there and let the caller own cleanup.
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
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

            # Resume: if a prior (crashed) build already wrote this entry's Arrow
            # file, reuse it instead of re-loading the base from GCS + re-running
            # transforms. cache_dir is signature-keyed so the file matches THIS
            # config. A partial/corrupt file (crash mid-write) is removed + rebuilt.
            if resume and not is_streaming:
                _ipc = Path(cache_dir) / f"{entry_idx}.arrow"
                if _ipc.exists():
                    _n = 0
                    try:
                        _df = pl.read_ipc(str(_ipc), memory_map=True)
                        _n = _df.height
                    except Exception:
                        _n = 0
                    if _n > 0:
                        lengths.append(_n)
                        all_columns.update(_df.columns)
                        ipc_files.append(str(_ipc))
                        logger.info(
                            "ChainedDataset: RESUME reuse entry %d/%d (%d rows) <- %s",
                            entry_idx,
                            num_entries,
                            _n,
                            _ipc,
                        )
                        continue
                    try:
                        _ipc.unlink()
                    except Exception:
                        pass

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
            if owns_cache_dir:
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
        chained._owns_cache_dir = owns_cache_dir

        return chained, metadata

    @classmethod
    def persistent_cache_dir(cls, chain_config: ChainedDatasetConfig, root: str | Path) -> Path:
        """Return the persistent cache directory for a config under ``root``.

        Parameters
        ----------
        chain_config : ChainedDatasetConfig
            The resolved chain configuration.
        root : str | Path
            The persistent cache root directory.

        Returns
        -------
        Path
            ``root / signature`` where signature is :func:`chain_cache_signature`.
        """
        return Path(root) / chain_cache_signature(chain_config)

    @classmethod
    def cache_is_valid(cls, chain_config: ChainedDatasetConfig, root: str | Path) -> bool:
        """Return whether a complete, matching persistent cache exists under ``root``.

        Checks that the sidecar exists, is marked complete, has a signature
        matching ``chain_config``, and that every referenced Arrow file is present.

        Parameters
        ----------
        chain_config : ChainedDatasetConfig
            The resolved chain configuration.
        root : str | Path
            The persistent cache root directory.

        Returns
        -------
        bool
            True if a valid, complete cache is present.
        """
        d = cls.persistent_cache_dir(chain_config, root)
        sidecar = d / CHAIN_CACHE_SIDECAR
        if not sidecar.exists():
            return False
        try:
            meta = json.loads(sidecar.read_text())
        except Exception:
            return False
        if not meta.get("complete"):
            return False
        if meta.get("signature") != chain_cache_signature(chain_config):
            return False
        return all((d / e["file"]).exists() for e in meta.get("entries", []))

    @classmethod
    def build_persistent_cache(
        cls,
        chain_config: ChainedDatasetConfig,
        root: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Build a chain and write it to a persistent, reusable cache directory.

        Intended to be run ONCE, single-process (e.g. a CPU pre-build job), so
        the expensive windowing/transform work is not repeated on every GPU
        launch nor duplicated across DDP ranks. The result is a set of Arrow IPC
        files plus a sidecar manifest under ``root / signature``.

        Parameters
        ----------
        chain_config : ChainedDatasetConfig
            The resolved chain configuration to materialize.
        root : str | Path
            The persistent cache root directory.
        overwrite : bool
            If True, rebuild even when a valid cache already exists.

        Returns
        -------
        Path
            The persistent cache directory that was written.

        Raises
        ------
        ChainException
            If the config is streaming (persistent caching is non-streaming only).
        """
        dest = cls.persistent_cache_dir(chain_config, root)
        if not overwrite and cls.cache_is_valid(chain_config, root):
            logger.info("ChainedDataset: persistent cache already valid at %s", dest)
            return dest
        # overwrite -> wipe and rebuild from scratch. Otherwise keep any partial
        # cache left by a crashed run so from_config(resume=True) skips the
        # entries already built (the signature dir guarantees the partial matches
        # this config) and continues from where it stopped.
        if overwrite and dest.exists():
            _cleanup_dir(str(dest))
        arrow_dir = dest / "arrow"
        arrow_dir.mkdir(parents=True, exist_ok=True)

        # Build directly into arrow_dir; owns_cache_dir=False so discarding the
        # returned dataset does not delete the freshly-written cache.
        chained, _ = cls.from_config(
            chain_config, cache_dir=arrow_dir, owns_cache_dir=False, resume=not overwrite
        )
        if chained._data is None:
            raise ChainException(
                "build_persistent_cache does not support streaming or empty chains"
            )

        entries = []
        for f in sorted(arrow_dir.glob("*.arrow"), key=lambda p: int(p.stem)):
            n = pl.read_ipc(str(f), memory_map=True).height
            entries.append({"entry_idx": int(f.stem), "file": f"arrow/{f.name}", "length": n})

        columns = [c for c in (chained._all_columns or []) if c != "_chain_idx"]
        sidecar = {
            "signature": chain_cache_signature(chain_config),
            "config_hash": hashlib.sha256(
                chain_config.model_dump_json(exclude_none=False).encode()
            ).hexdigest()[:16],
            "code_version": _esp_data_code_version(),
            "data_version": os.environ.get("ESP_DATA_CHAIN_CACHE_DATA_VERSION", "0"),
            "is_streaming": False,
            "columns": columns,
            "total_length": chained._total_length,
            "num_entries": len(chain_config.datasets),
            "entries": entries,
            "complete": True,
        }
        (dest / CHAIN_CACHE_SIDECAR).write_text(json.dumps(sidecar, indent=2))

        # Release the build-time mmap handles; the Arrow files persist on disk
        # (owns_cache_dir=False, so no cleanup on GC).
        del chained
        gc.collect()
        logger.info(
            "ChainedDataset: wrote persistent cache -> %s (%d entries, %d rows)",
            dest,
            len(entries),
            sidecar["total_length"],
        )
        return dest

    @classmethod
    def from_cache(
        cls, chain_config: ChainedDatasetConfig, root: str | Path
    ) -> tuple["ChainedDataset", dict[str, Any]]:
        """Load a prebuilt persistent chain cache instead of rebuilding.

        Memory-maps the cached Arrow files (read-only, shareable across DDP
        ranks) and reconstructs the per-entry source datasets *without*
        transformations so their ``_process`` (lazy audio decode) is available.
        The returned dataset never deletes the cache (``_owns_cache_dir`` False).

        Parameters
        ----------
        chain_config : ChainedDatasetConfig
            The resolved chain configuration (used to verify the signature and
            to reconstruct source datasets).
        root : str | Path
            The persistent cache root directory.

        Returns
        -------
        tuple[ChainedDataset, dict]
            The loaded dataset and an (empty) metadata dict.

        Raises
        ------
        ChainException
            If the cache is missing or its signature does not match the config.
        """
        dest = cls.persistent_cache_dir(chain_config, root)
        sidecar = dest / CHAIN_CACHE_SIDECAR
        if not sidecar.exists():
            raise ChainException(f"No persistent chain cache sidecar at {sidecar}")
        meta = json.loads(sidecar.read_text())
        if meta.get("signature") != chain_cache_signature(chain_config):
            raise ChainException(
                f"Chain cache at {dest} is stale for this config "
                f"(signature mismatch); rebuild it."
            )

        mmap_dfs = [
            pl.read_ipc(str(dest / e["file"]), memory_map=True)
            for e in meta["entries"]
        ]
        consolidated = PolarsBackend(
            pl.concat(mmap_dfs, how="diagonal_relaxed", rechunk=False)
        )

        # Reconstruct source datasets (no transforms), indexed by entry_idx to
        # match the cached ``_chain_idx`` column. ``_process`` needs the dataset
        # object but not its base table, so free ``_data`` to keep this cheap.
        datasets: list[Dataset] = []
        for cfg in chain_config.datasets:
            no_tf_cfg = cfg.model_copy(update={"transformations": None})
            ds, _ = dataset_from_config(no_tf_cfg)
            ds._data = None
            datasets.append(ds)

        chained = object.__new__(cls)
        chained._streaming = False
        chained._backend_class = None
        chained.output_take_and_give = None
        chained._source_datasets = datasets
        chained._lengths = [e["length"] for e in meta["entries"]]
        chained._total_length = int(meta["total_length"])
        chained._all_columns = sorted(meta.get("columns", []))
        chained._data = consolidated
        chained._cache_dir = None
        chained._owns_cache_dir = False

        logger.info(
            "ChainedDataset: loaded persistent cache from %s (%d rows, %d entries)",
            dest,
            chained._total_length,
            len(datasets),
        )
        return chained, {}

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
