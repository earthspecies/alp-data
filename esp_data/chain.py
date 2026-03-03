import copy
import json
import logging
from typing import Any, Iterator

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


class ChainException(Exception):
    """Exception raised when dataset chaining fails."""

    pass


@register_dataset
class ChainedDataset(Dataset):
    """Helper class to chain multiple datasets for iteration and indexing.

    This class allows iterating over multiple datasets as if they were a single dataset.

    Parameters
    ----------
    datasets : list[Dataset]
        List of datasets to concatenate for iteration

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
        version="0.1.0",
        description="A dataset created by chaining multiple datasets for iteration.",
        sources=["Multiple datasets"],
        license="CC0-1.0",
    )

    def __init__(self, datasets: list[Dataset]) -> None:
        if not datasets:
            raise ChainException("At least one dataset must be provided")

        if not all(isinstance(ds, Dataset) for ds in datasets):
            raise ChainException("All objects must be Dataset instances")

        # determine streaming mode based on source datasets
        # all datasets must have the same streaming mode
        streaming_modes = {ds.streaming for ds in datasets}
        if len(streaming_modes) > 1:
            raise ChainException(
                "All datasets must have the same streaming mode "
                "to be concatenated into a ConcatenatedDataset."
            )
        _streaming = streaming_modes.pop()

        # _backend_class doesn'gt matter here since we override all data access methods
        super().__init__(streaming=_streaming)

        self._source_datasets = datasets
        try:
            self._lengths = [len(ds) for ds in datasets]
            self._total_length = sum(self._lengths)
        except RuntimeError:
            self._lengths = []
            self._total_length = -1

        self._all_columns = set()
        for ds in datasets:
            self._all_columns.update(ds.columns)
        self._all_columns = sorted(list(self._all_columns))

    @property
    def columns(self) -> list[str]:
        return self._all_columns

    @property
    def available_splits(self) -> list[str]:
        return ["chained"]

    def _load(self) -> None:
        pass  # Data is already loaded

    def __len__(self) -> int:
        if self._streaming:
            raise RuntimeError("Length is not supported in streaming mode")
        return self._total_length

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for dataset in self._source_datasets:
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

        # Determine which dataset the index falls into
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
        (e.g. ``filter → window_annotations → annotation_features``) but
        diverge later (e.g. different ``chat`` template), only the
        divergent tail is recomputed.

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
        metadata: dict[str, Any] = {}

        for cfg in chain_config.datasets:
            cache_key = _base_dataset_key(cfg)

            if cache_key not in base_cache:
                no_tf_cfg = cfg.model_copy(update={"transformations": None})
                base_ds, _ = dataset_from_config(no_tf_cfg)
                base_cache[cache_key] = base_ds
                logger.info(
                    "ChainedDataset: loaded base %s/%s (%d rows)",
                    cfg.dataset_name,
                    cfg.split,
                    len(base_ds),
                )

            ds = copy.copy(base_cache[cache_key])

            meta: dict[str, Any] = {}
            if cfg.transformations:
                transforms = cfg.transformations

                # Find the longest already-cached prefix of transforms
                best_prefix_len = 0
                for i in range(len(transforms), 0, -1):
                    prefix_key = _transforms_prefix_key(cache_key, transforms[:i])
                    if prefix_key in transform_cache:
                        best_prefix_len = i
                        break

                if best_prefix_len > 0:
                    prefix_key = _transforms_prefix_key(
                        cache_key, transforms[:best_prefix_len]
                    )
                    cached_data, cached_meta = transform_cache[prefix_key]
                    ds._data = cached_data.copy()
                    meta = dict(cached_meta)
                    logger.info(
                        "ChainedDataset: %s — reusing cached result for %d/%d transforms",
                        cfg.dataset_name,
                        best_prefix_len,
                        len(transforms),
                    )
                else:
                    ds._data = base_cache[cache_key]._data.copy()

                # Apply only the remaining transforms
                for i in range(best_prefix_len, len(transforms)):
                    tf_cfg = transforms[i]
                    tf = transform_from_config(tf_cfg)
                    ds._data, tf_meta = tf(ds._data)
                    meta[tf_cfg.type] = tf_meta

                    applied_key = _transforms_prefix_key(
                        cache_key, transforms[: i + 1]
                    )
                    if applied_key not in transform_cache:
                        transform_cache[applied_key] = (ds._data, dict(meta))
            else:
                ds._data = base_cache[cache_key]._data.copy()

            datasets.append(ds)
            metadata[f"{cfg.dataset_name}_metadata"] = meta

        chained = cls(datasets)
        return chained, metadata

    def save_data(self, path: str, fmt: SaveFormat = "csv") -> None:
        """Concatenate all source datasets' backends and save to a single file.

        Parameters
        ----------
        path : str
            Destination file path (local or cloud).
        fmt : SaveFormat
            Output format: ``"csv"`` or ``"jsonl"``.

        Raises
        ------
        ChainException
            If none of the source datasets have loaded data.
        """
        backends = [
            ds._data for ds in self._source_datasets if getattr(ds, "_data", None) is not None
        ]
        if not backends:
            raise ChainException("No source datasets have loaded data.")

        backend_cls = type(backends[0])
        merged = backend_cls.concat(backends)
        if fmt == "csv":
            merged.to_csv(path)
        elif fmt == "jsonl":
            merged.to_jsonl(path)

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version})\n"
            f"Description: {self.info.description}\n"
            f"Length: {len(self)}\n"
            f"Columns: {', '.join(self.columns)}\n"
            f"Source datasets: {len(self._datasets)}"
        )
