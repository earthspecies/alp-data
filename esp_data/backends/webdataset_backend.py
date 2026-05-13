"""WebDataset implementation of the StreamingBackend protocol."""

from typing import Any, Callable, Iterator

import webdataset as wds

from esp_data.backends.webdataset_utils import audio_decoder
from esp_data.io import AnyPathT, anypath

from .protocol import StreamingBackend


def _load_webdataset(
    path: str | AnyPathT,
    file_pattern: str = "shard*tar",
    data_processor: Callable | None = audio_decoder,
    shuffle_size: int | None = None,
    batch_size: int | None = None,
    shard_shuffle: bool = False,
    shard_shuffle_size: int = 1000,
    split_by_worker: bool = False,
    batch_collate_fn: Callable | None = None,
    seed: int | None = None,
) -> wds.WebDataset:
    """Create a pipeline for loading the dataset.

    Parameters
    ----------
    path : str | AnyPathT
        Path to the directory containing the sharded tar files.
    file_pattern : str, optional
        Glob pattern to match shard files, by default ``"shard*tar"``.
    data_processor : Callable | None, optional
        Function to decode each sample, by default ``None``.
    shuffle_size : int | None, optional
        Sample shuffle buffer size. ``None`` disables sample shuffling.
    batch_size : int | None, optional
        If set, yield batches of this size instead of individual samples.
    shard_shuffle : bool, optional
        Whether to shuffle shard order, by default ``False``.
    shard_shuffle_size : int, optional
        Shard shuffle buffer size, by default ``1000``.
    split_by_worker : bool, optional
        Whether to split shards across DataLoader workers, by default ``False``.
    batch_collate_fn : Callable | None, optional
        Custom collation function for batched mode, by default ``None``.
    seed : int | None, optional
        Random seed for shuffling. ``None`` disables shuffling.

    Returns
    -------
    wds.WebDataset
        Configured WebDataset pipeline.

    Raises
    ------
    FileNotFoundError
        If no shard files are found in the specified path.
    """
    path = anypath(path)
    shard_files = [str(s) for s in path.glob(file_pattern)]

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {path}")

    webds_kwargs = {"shardshuffle": shard_shuffle_size if shard_shuffle else False}
    if shard_shuffle and seed is not None:
        webds_kwargs["seed"] = seed
    if split_by_worker:
        webds_kwargs["workersplitter"] = wds.split_by_worker

    webds = wds.WebDataset(shard_files, **webds_kwargs)

    if shuffle_size is not None and seed is not None:
        webds = webds.shuffle(shuffle_size, seed=seed)
    if data_processor:
        webds = webds.map(data_processor)
    if batch_size is not None:
        webds = webds.batched(batch_size, collation_fn=batch_collate_fn)

    return webds


class WebDatasetBackend(StreamingBackend):
    """WebDataset implementation of the StreamingBackend protocol.

    This backend wraps a WebDataset and provides a streaming interface
    for iterating over samples. It does not support random access or
    operations that require knowing the full dataset size.

    Parameters
    ----------
    dataset : wds.WebDataset
        The WebDataset to wrap
    """

    def __init__(self, dataset: wds.WebDataset) -> None:
        """Initialize the backend with a WebDataset.

        Parameters
        ----------
        dataset : wds.WebDataset
            The WebDataset to wrap
        """
        self._dataset = dataset
        self._columns: list[str] | None = None
        self._filter_funcs: list[Callable[[dict[str, Any]], bool]] = []
        self._map_funcs: list[Callable[[dict[str, Any]], dict[str, Any]]] = []

    def _copy(self) -> "WebDatasetBackend":
        """Create a shallow copy of this backend with copied operation lists.

        The underlying WebDataset is shared (not copied), but the filter and
        map function lists are copied so modifications don't affect the original.

        Returns
        -------
        WebDatasetBackend
            A new backend instance with copied operation lists
        """
        new_backend = WebDatasetBackend(self._dataset)
        new_backend._filter_funcs = self._filter_funcs.copy()
        new_backend._map_funcs = self._map_funcs.copy()
        new_backend._columns = None  # force re-compute through new map funcs
        return new_backend

    @classmethod
    def from_path(
        cls,
        path: str | AnyPathT,
        file_pattern: str = "shard*tar",
        data_processor: Callable | None = audio_decoder,
        shuffle_size: int | None = None,
        batch_size: int | None = None,
        shard_shuffle: bool = False,
        shard_shuffle_size: int = 1000,
        split_by_worker: bool = False,
        batch_collate_fn: Callable | None = None,
        seed: int | None = 42,
        streaming: bool = True,
    ) -> "WebDatasetBackend":
        """Read a WebDataset from the specified path and return a wrapped backend.

        Parameters
        ----------
        path : str | AnyPathT
            Path to the directory containing the sharded tar files.
        file_pattern : str, optional
            Glob pattern to match shard files, by default ``"shard*tar"``.
        data_processor : Callable | None, optional
            Function to decode each sample, by default ``None``.
        shuffle_size : int | None, optional
            Sample shuffle buffer size. ``None`` disables sample shuffling.
        batch_size : int | None, optional
            If set, yield batches of this size instead of individual samples.
        shard_shuffle : bool, optional
            Whether to shuffle shard order, by default ``False``.
        shard_shuffle_size : int, optional
            Shard shuffle buffer size, by default ``1000``.
        split_by_worker : bool, optional
            Whether to split shards across DataLoader workers, by default ``False``.
        batch_collate_fn : Callable | None, optional
            Custom collation function for batched mode, by default ``None``.
        seed : int | None, optional
            Random seed for shuffling. ``None`` disables shuffling, by default ``42``.
        streaming : bool, optional
            Accepted for interface uniformity with tabular backends. Always ``True``
            for WebDataset — this parameter has no effect.

        Returns
        -------
        WebDatasetBackend
            Wrapped WebDataset backend.
        """
        dataset = _load_webdataset(
            path,
            file_pattern=file_pattern,
            data_processor=data_processor,
            shuffle_size=shuffle_size,
            batch_size=batch_size,
            shard_shuffle=shard_shuffle,
            shard_shuffle_size=shard_shuffle_size,
            split_by_worker=split_by_worker,
            batch_collate_fn=batch_collate_fn,
            seed=seed,
        )
        return cls(dataset)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples as dictionaries.

        Yields
        ------
        dict[str, Any]
            Dictionary for each sample mapping field names to values
        """
        for sample in self._dataset:
            # Apply map functions first
            for map_fn in self._map_funcs:
                sample = map_fn(sample)
            # Then apply filters
            if self._filter_funcs and not all(fn(sample) for fn in self._filter_funcs):
                continue
            yield sample

    @property
    def columns(self) -> list[str]:
        """Get the list of column/field names.

        Note: This requires peeking at the first sample, which consumes it.
        The columns are cached after the first access.

        Returns
        -------
        list[str]
            List of column/field names
        """
        if self._columns is None:
            try:
                first_sample = next(iter(self))
            except StopIteration:
                return []
            self._columns = list(first_sample.keys())
        return self._columns

    def column_exists(self, column: str) -> bool:
        """Check if a column/field exists in the data.

        Parameters
        ----------
        column : str
            Column name to look for

        Returns
        -------
        bool
            True if column exists, False otherwise
        """
        return column in self.columns

    @property
    def unwrap(self) -> wds.WebDataset:
        """Get the underlying WebDataset object.

        Returns
        -------
        wds.WebDataset
            The underlying WebDataset
        """
        return self._dataset

    @property
    def is_streaming(self) -> bool:
        """Check if backend is in streaming mode.

        Returns
        -------
        bool
            Always True for WebDatasetBackend
        """
        return True

    def filter_isin(
        self,
        column: str,
        values: list[Any],
        *,
        negate: bool = False,
    ) -> "WebDatasetBackend":
        """Filter samples where column values are in (or not in) a list.

        This filter is applied lazily during iteration. Chaining multiple filters
        will combine them with logical AND.

        Returns a new WebDatasetBackend instance, leaving the original unchanged.

        Parameters
        ----------
        column : str
            Column name to filter on
        values : list[Any]
            List of values to match
        negate : bool, optional
            If True, keep rows NOT in values list, by default False

        Returns
        -------
        WebDatasetBackend
            New backend with filter configured (applied during iteration)
        """
        new_backend = self._copy()
        value_set = set(values)

        def filter_fn(sample: dict[str, Any]) -> bool:
            if column not in sample:
                return negate
            in_values = sample[column] in value_set
            return not in_values if negate else in_values

        new_backend._filter_funcs.append(filter_fn)
        return new_backend

    def dropna(
        self,
        subset: list[str] | None = None,
    ) -> "WebDatasetBackend":
        """Remove samples with missing values.

        This filter is applied lazily during iteration.

        Returns a new WebDatasetBackend instance, leaving the original unchanged.

        Parameters
        ----------
        subset : list[str] | None, optional
            Column names to consider for null detection.
            If None, check all columns, by default None

        Returns
        -------
        WebDatasetBackend
            New backend with dropna configured (applied during iteration)
        """
        new_backend = self._copy()

        def dropna_fn(sample: dict[str, Any]) -> bool:
            if subset is None:
                return all(sample.get(col) is not None for col in sample)
            else:
                return all(sample.get(col) is not None for col in subset)

        new_backend._filter_funcs.append(dropna_fn)
        return new_backend

    def map_column(
        self,
        column: str,
        mapping: dict[Any, Any],
        output_column: str,
        *,
        default: Any | None = None,  # noqa ANN401
    ) -> "WebDatasetBackend":
        """Create a new column by mapping values from an existing column.

        This transformation is applied lazily during iteration.

        Returns a new WebDatasetBackend instance, leaving the original unchanged.

        Parameters
        ----------
        column : str
            Source column name
        mapping : dict[Any, Any]
            Dictionary mapping source values to output values
        output_column : str
            Name of the new column to create
        default : Any, optional
            Value to use for unmapped keys, by default None

        Returns
        -------
        WebDatasetBackend
            New backend with mapping configured (applied during iteration)
        """
        new_backend = self._copy()

        def map_fn(sample: dict[str, Any]) -> dict[str, Any]:
            if column not in sample:
                sample[output_column] = default
            else:
                sample[output_column] = mapping.get(sample[column], default)
            return sample

        new_backend._map_funcs.append(map_fn)
        return new_backend

    def apply_fn(
        self,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> "WebDatasetBackend":
        """Apply a custom function to each sample during iteration.

        Returns a new WebDatasetBackend instance, leaving the original unchanged.

        Parameters
        ----------
        fn : Callable[[dict[str, Any]], dict[str, Any]]
            Function to apply. Should accept a sample dict and return
            a transformed sample dict.

        Returns
        -------
        WebDatasetBackend
            New backend with function configured (applied during iteration)
        """
        new_backend = self._copy()
        new_backend._map_funcs.append(fn)
        return new_backend

    def save_to(
        self,
        path: str | AnyPathT,
        format: str = "webdataset",
        encoder_fn: Callable | None = None,
        shard_pattern: str = "shard_%04d.tar",
        maxcount: int = 100_000,
        maxsize: float = 3e9,
    ) -> int:
        """Write the backend's samples to sharded tar files on disk or cloud storage.

        Applies all accumulated filters and maps before writing. Mirrors
        `from_path` to provide a symmetric load/save API on the backend.

        For cloud paths (GCS, R2), samples are written to a temporary local
        directory first, then uploaded to the destination.

        Parameters
        ----------
        path : str | AnyPathT
            Destination directory (local or cloud). Created if it does not exist.
        format : str, optional
            Output format. Only ``"webdataset"`` is supported, by default
            ``"webdataset"``.
        encoder_fn : Callable | None, optional
            Function ``dict[str, Any] -> dict[str, bytes]`` in WebDataset
            format. If ``None``, auto-detected from the first sample:
            samples with an ``"audio"`` key use `audio_encoder`, all others
            use `json_encoder`.
        shard_pattern : str, optional
            Printf-style shard file name pattern, by default
            ``"shard_%04d.tar"``.
        maxcount : int, optional
            Maximum samples per shard, by default 100 000.
        maxsize : float, optional
            Maximum shard size in bytes, by default 3 GB.

        Returns
        -------
        int
            Number of samples written.

        Raises
        ------
        ValueError
            If ``format`` is not ``"webdataset"``.
        """
        if format != "webdataset":
            raise ValueError(f"Unsupported format '{format}' for WebDatasetBackend")

        from .webdataset_utils import write_to_webdataset

        resolved = anypath(path)
        return write_to_webdataset(
            iter(self),
            resolved,
            encoder_fn=encoder_fn,
            shard_pattern=shard_pattern,
            maxcount=maxcount,
            maxsize=maxsize,
        )

    def __repr__(self) -> str:
        """Return string representation of the backend.

        Returns
        -------
        str
            String representation showing backend type
        """
        return "WebDatasetBackend(streaming=True)"
