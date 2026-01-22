"""WebDataset implementation of the StreamingBackend protocol."""

import io
import json
from typing import Any, Callable, Iterator

import numpy as np
import soundfile as sf
import webdataset as wds

from esp_data.io import AnyPathT, PureCloudPath, anypath, filesystem_from_path

from .protocol import StreamingBackend


def audio_decoder(data: dict, dtype: str = "float32", format: str = "FLAC") -> dict[str, Any]:
    """Decode audio data from a WebDataset sample.

    Parameters
    ----------
    data: dict
        The sample containing audio data in WebDataset format
    dtype: str
        The data type of the decoded audio data (default: "float32")
    format: str
        The format of the audio data (default: "FLAC")

    Returns
    -------
    dict
        Dictionary containing the decoded audio data and metadata.

    Raises
    ------
    ValueError
        If the sample does not contain an audio key ending with .flac, .wav, etc.
    """
    audio_key = next((k for k in data if k.endswith(f".{format.lower()}")), None)
    if not audio_key:
        raise ValueError("Sample must contain an audio key ending with .flac, .wav, etc.")

    audio_buffer = io.BytesIO(data[audio_key])
    audio_data, samplerate = sf.read(audio_buffer, dtype=dtype)

    # Reconstruct sample
    sample = {}
    sample["audio"] = audio_data
    sample["sample_rate"] = samplerate
    md = json.loads(data.get("metadata.json", "{}").decode("utf-8"))
    sample.update(md)

    return sample


def audio_encoder(
    sample: dict[str, Any],
    sample_rate: int = 16000,
    dtype: str = "float32",
    format: str = "FLAC",
) -> dict[str, Any]:
    """Encode audio data in the sample to a specific format.

    Parameters
    ----------
    sample: dict[str, Any]
        The sample containing audio data
    sample_rate: int
        The sample rate of the audio data
    dtype: str
        The data type of the audio data (default: "float32")
    format: str
        The format to encode the audio data to (e.g., "WAV", "FLAC", "OGG")
        Default is "FLAC".

    Returns
    -------
    dict
        Dictionary containing the encoded audio data and metadata
        in the WebDataset format.

    Raises
    ------
    ValueError
        If the sample does not contain an "audio" key with audio data.
    """
    if "audio" not in sample:
        raise ValueError("Sample must contain 'audio' key with audio data")

    data_out = {}
    audio_buffer = io.BytesIO()
    # Convert audio data to the specified format
    if isinstance(sample["audio"], (list, tuple)):
        # If audio is a list or tuple, convert to numpy array
        sample["audio"] = np.array(sample["audio"], dtype=dtype)
    elif isinstance(sample["audio"], np.ndarray):
        # If audio is already a numpy array, ensure it's the correct dtype
        sample["audio"] = sample["audio"].astype(dtype)

    sf.write(audio_buffer, sample["audio"], sample_rate, format=format)

    data_out[f"audio.{format.lower()}"] = audio_buffer.getvalue()

    # Add metadata (without audio)
    sample = {k: v for k, v in sample.items() if k != "audio"}  # Remove audio key from metadata
    data_out["metadata.json"] = json.dumps(sample, indent=2).encode("utf-8")
    return data_out


def json_encoder(
    sample: dict[str, Any],
    indent: int = 2,
) -> dict[str, Any]:
    """Encode a sample to JSON format.

    Parameters
    ----------
    sample: dict[str, Any]
        The sample to encode
    indent: int
        Indentation level for JSON (default: 2)

    Returns
    -------
    dict
        Dictionary containing the encoded sample in JSON format.
    """
    json_data = json.dumps(sample, indent=indent).encode("utf-8")
    return {"sample.json": json_data}


def json_decoder(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Decode a sample from JSON format.

    Parameters
    ----------
    data: dict[str, Any]
        The sample containing JSON data

    Returns
    -------
    dict
        Dictionary containing the decoded sample.

    Raises
    ------
    ValueError
        If the sample does not contain a "sample.json" key.
    """
    if "sample.json" not in data:
        raise ValueError("Sample must contain 'sample.json' key with JSON data")

    json_data = json.loads(data["sample.json"].decode("utf-8"))
    return json_data


def make_file_opener_for_wds(
    file_path: str | AnyPathT,
    mode: str = "wb",
    block_size: int = 1024 * 1024 * 100,
) -> Callable:
    """Make a file opener function for WebDataset.

    If local path, create parent dirs if needed.

    Arguments
    ---------
    file_path: str | AnyPathT
        The file path to open
    mode: str
        The mode in which to open the file (default: "wb")
    block_size: int
        Block size for WebDataset (default: 100 MB)

    Returns
    -------
    Callable
        A function that opens the file in the specified mode
        or a file object if the path is local.
    """
    path_obj = anypath(file_path)

    if not isinstance(path_obj, PureCloudPath):
        # Local filesystem - create parent dirs if needed
        parent_dir = path_obj.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        return open(str(path_obj), mode=mode)
    else:
        # Remote filesystem (GCS, R2, etc.)
        fs = filesystem_from_path(str(path_obj))
        return fs.open(str(path_obj.no_prefix), mode=mode, block_size=block_size)


def load_webdataset(
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

    Arguments
    ---------
    path: str | AnyPath
        Path to the directory where the sharded dataset will be stored or
        is already stored.
    file_pattern: str, optional
        Pattern to match the shard files.
    data_processor: Callable, optional
        Function to process the data.
    shuffle_size: int, optional
        Size of the shuffle buffer.
    batch_size: int, optional
        Batch size for processing audio files.
    shard_shuffle: bool, optional
        Whether to shuffle the shards.
    shard_shuffle_size: int, optional
        Size of the shuffle buffer for shards.
    split_by_worker: bool, optional
        Whether to split the dataset by worker.
    batch_collate_fn: Callable, optional
        Function to collate the batch.
    seed : int | None, optional
        Seed for shuffling. Defaults to True, random seed. If None, means no shuffling!

    Returns
    -------
    wds.WebDataset
        WebDataset object

    Raises
    ------
    FileNotFoundError
        If no shard files are found in the specified path.
    """
    path = anypath(path)
    shard_files = list([str(s) for s in path.glob(file_pattern)])

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
        new_backend._columns = self._columns
        return new_backend

    @classmethod
    def from_path(
        cls,
        path: str | AnyPathT,
        file_pattern: str = "shard*tar",
        data_processor: Callable | None = None,
        shuffle_size: int | None = None,
        batch_size: int | None = None,
        shard_shuffle: bool = False,
        shard_shuffle_size: int = 1000,
        split_by_worker: bool = False,
        batch_collate_fn: Callable | None = None,
        seed: int | None = 42,
    ) -> "WebDatasetBackend":
        """Read a WebDataset from the specified path and return a wrapped backend.

        Parameters
        ----------
        path: str | AnyPath
            Path to the directory where the sharded dataset will be stored or
            is already stored.
        file_pattern: str, optional
            Pattern to match the shard files.
        data_processor: Callable, optional
            Function to process the data.
        shuffle_size: int, optional
            Size of the shuffle buffer.
        batch_size: int, optional
            Batch size for processing audio files.
        shard_shuffle: bool, optional
            Whether to shuffle the shards.
        shard_shuffle_size: int, optional
            Size of the shuffle buffer for shards.
        split_by_worker: bool, optional
            Whether to split the dataset by worker.
        batch_collate_fn: Callable, optional
            Function to collate the batch.
        seed : int | None, optional
            Seed for shuffling. Defaults to True, random seed. If None, means no shuffling!

        Returns
        -------
        WebDatasetBackend
            Wrapped WebDataset backend
        """
        dataset = load_webdataset(
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
            # iter once and get keys from first sample
            first_sample = next(iter(self._dataset))
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

    def __repr__(self) -> str:
        """Return string representation of the backend.

        Returns
        -------
        str
            String representation showing backend type
        """
        return "WebDatasetBackend(streaming=True)"
