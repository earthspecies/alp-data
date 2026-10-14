"""SpanishCarrionCrowsChunks dataset"""

from __future__ import annotations

from typing import Any, Iterator

import librosa
import numpy as np

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_config, register_dataset
from alp_data.backends import BackendType, DataBackend
from alp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_config
class SpanishCarrionCrowsChunksConfig(DatasetConfig):
    """Configuration for the SpanishCarrionCrowsChunks dataset.

    Registering this makes the chunking parameters first-class config fields.
    Without it the base `DatasetConfig` is used, where they survive only as
    ``extra`` keys — present in a dump when the YAML happens to set them and
    absent when it does not — so a config omitting them would fail rather than
    fall back to these defaults.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset. Must be "spanish-carrion-crows-chunks".
    split : str
        The split to load. This dataset has only "all".
    output_take_and_give : dict[str, str] | None
        A dictionary mapping the original column names to the new column names.
        It acts as a filter as well.
    sample_rate : int
        The sample rate audio is resampled to, in Hz. Defaults to 32000, the
        rate BirdCODE expects; the chunk grid is laid out in samples at this
        rate, so it is not a free choice once a run has been made.
    chunk_duration_sec : float
        Duration of one dataset item, in seconds. Defaults to 900.
    chunk_overlap_sec : float
        Overlap between adjacent chunks, in seconds. Defaults to 5. Wants to
        be at least as long as the longest event of interest, so that an event
        near a boundary falls wholly inside one chunk, and ideally at least one
        model input window, so that it is surrounded by real audio rather than
        the zero padding at a chunk edge.
    data_root : str | AnyPathT | None
        The root directory for the dataset. This is optionally appended to the
        path item of a sample in the dataset.
    backend : BackendType
        The backend to use ("pandas" or "polars"), by default "polars"
    streaming : bool
        Whether to use streaming mode. Unsupported by this dataset, which needs
        every recording's duration before it can lay out the chunk grid.
    """

    dataset_name: str = "spanish-carrion-crows-chunks"
    split: str = "all"
    output_take_and_give: dict[str, str] | None = None
    sample_rate: int = 32000
    chunk_duration_sec: float = 900
    chunk_overlap_sec: float = 5
    data_root: str | AnyPathT | None = None
    backend: BackendType = "polars"
    streaming: bool = False


@register_dataset
class SpanishCarrionCrowsChunks(Dataset):
    """SpanishCarrionCrowsChunks Dataset

    Description
    -----------
    This class makes the Spanish carrion crow biologger dataset available
    (2018, 2019, 2021). Each entry is an audio recording, chunk start time,
    chunk stop time, sampling rate, year, territory, focal individual
    (individual wearing the biologger). Therefore, each dataset item
    corresponds to a chunk of time within an audio recording, and a single
    audio recording is represented by multiple chunks of duration
    chunk_duration_sec. Chunks overlap by chunk_overlap_sec seconds. This
    format is built for BirdCODE-like inference, overlapping chunks allow
    us to ignore events inferred at the edges of the chunks to ensure that
    we do not break up events.

    We omit selection tables and various other aspects of the full crow dataset.

    Should talk to collaborators (Daniela Canestrari and Vittorio Baglione)
    about usage in work to be published.

    """

    info = DatasetInfo(
        name="spanish-carrion-crows-chunks",
        owner="maddie",
        split_paths={
            "all": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/all__spanish-carrion-crows_info.csv"
        },
        version="0.1.0",
        description="Crow biologger audio chunked for BirdCODE detection",
        sources="University of Leon",
        license="private",
    )

    _sample_rate_paths: dict[int, str] = {}
    _originals_path_column = "audio_fp"

    #: Slack allowed between a chunk's nominal span and the audio actually
    #: decoded for it (see `_process`). ffmpeg's ``-t`` duration and the
    #: resampler each round by a sample or so; a larger discrepancy means the
    #: read returned the wrong amount of audio. Note this constrains the
    #: chunk's length only — a seek landing at the wrong offset but returning
    #: the right number of samples passes.
    #: The latter ability to seek the correct offset will be in the tests.
    _LENGTH_TOLERANCE_SAMPLES = 4

    def __init__(
        self,
        sample_rate: int = 32000,
        chunk_duration_sec: float = 900,
        chunk_overlap_sec: float = 5,
        output_take_and_give: dict[str, str] | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        sample_rate: int
            Audio sampling rate to conform to, in Hz
        chunk_duration_sec: int
            Duration of one dataset item, in seconds
        chunk_overlap_sec: int
            The overlap duration between adjacent chunks, in seconds
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns as well).
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row['audio_fp'].
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = "all"
        self._data = None
        self.sample_rate = sample_rate  # Hz
        self.chunk_duration_sec = chunk_duration_sec  # sec
        self.chunk_overlap_sec = chunk_overlap_sec  # sec

        self._load()

        if data_root is None:
            self.data_root = anypath("gs://")
        else:
            self.data_root = anypath(data_root)

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return ["all"]

    @property
    def available_sample_rates(self) -> list[int]:
        """Only allow 32 kHz which is needed for BirdCode."""
        return [self.sample_rate]

    def _load(self) -> None:
        """Load the recording table and expand it into one row per chunk.

        Raises
        ------
        NotImplementedError
            If the dataset is in streaming mode. Chunking needs every
            recording's duration up front to lay out the chunk grid, which a
            streaming backend cannot provide without consuming itself.
        """
        if self._streaming:
            raise NotImplementedError(
                f"{type(self).__name__} does not support streaming mode: the chunk layout "
                "requires every recording's duration before the first item is produced."
            )
        location = self.info.split_paths[self.split]
        recordings = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )
        self._data = self._expand_to_chunks(recordings)

    def _expand_to_chunks(self, recordings: DataBackend) -> DataBackend:
        """Expand a table of recordings into one row per chunk.

        Each recording's row is repeated once per chunk (see `_chunk_index`), and
        provided with a `chunk_id` column that identifies the chunk's position,
        span (in samples), and the sample rate (with which the sample span is
        defined). The chunk_id should identify a chunk uniquely, since BirdCODE
        will use this for storing its inference results. We remove the selection
        table; we will restitch the outputs of BirdCODE separately in order to
        perform evaluations.

        Parameters
        ----------
        recordings : DataBackend
            One row per recording, carrying at least `audio_duration` and
            `audio_fp`.

        Returns
        -------
        DataBackend
            One row per chunk, with the parent recording's remaining columns
            plus `chunk_idx`, `chunk_start_sample`, `chunk_end_sample`,
            `chunk_start_sec`, `chunk_end_sec` and `chunk_id`.
        """
        durations = np.array(
            [float(row["audio_duration"]) for row in recordings.select_columns(["audio_duration"])],
            dtype=np.float64,
        )
        index = self._chunk_index(durations)
        row_index = index["row_index"]

        # The `selection_table` column is dropped *before* the rows are repeated.
        # It holds the whole recording's events as TSV text, so carrying it
        # through the expansion would copy every recording's table once per
        # chunk, which we avoid here.
        kept_columns = [column for column in recordings.columns if column != "selection_table"]
        chunks = recordings.select_columns(kept_columns)[row_index.tolist()]

        # Built from the unexpanded paths and gathered with `row_index`, so the
        # 50k-row table is never iterated just to read one column back out.
        paths = [row["audio_fp"] for row in recordings.select_columns(["audio_fp"])]
        chunk_ids = [
            f"{paths[parent]}#{start}-{end}@{self.sample_rate}"
            for parent, start, end in zip(
                row_index, index["start_sample"], index["end_sample"], strict=True
            )
        ]

        chunks = chunks.add_column("chunk_idx", index["chunk_idx"].tolist())
        chunks = chunks.add_column("chunk_start_sample", index["start_sample"].tolist())
        chunks = chunks.add_column("chunk_end_sample", index["end_sample"].tolist())
        chunks = chunks.add_column("chunk_start_sec", index["start_sec"].tolist())
        chunks = chunks.add_column("chunk_end_sec", index["end_sec"].tolist())
        return chunks.add_column("chunk_id", chunk_ids)

    def _chunk_index(self, durations: np.ndarray) -> dict[str, np.ndarray]:
        """Compute the chunk layout for a set of recordings.

        Each recording is tiled with chunks of `chunk_duration_sec` seconds
        starting every ``chunk_duration_sec - chunk_overlap_sec`` seconds, so a
        recording of ``n`` samples yields
        ``ceil(max(n - chunk_samples, 0) / hop_samples) + 1`` chunks. A
        recording no longer than one chunk yields exactly one, and the final
        chunk of a longer recording is truncated at the end of the file.

        The layout is computed in whole samples at `sample_rate` — the rate the
        audio is resampled to, not the recording's native rate — so chunk
        starts are exact integers rather than a float step accumulated over a
        hundred-odd chunks, and the offsets can be embedded verbatim in each
        chunk's identifier. The seconds are derived from those samples and are
        what `read_audio` is given, since it takes a time range rather than a
        sample range.

        Parameters
        ----------
        durations : np.ndarray
            Duration in seconds of each recording, shape ``(n_recordings,)``.

        Returns
        -------
        dict[str, np.ndarray]
            Parallel arrays of length ``n_chunks_total``: `row_index` (index of
            each chunk's parent recording in `durations`), `chunk_idx` (the
            chunk's zero-based position within its recording), `start_sample`
            and `end_sample` (the chunk's half-open span in samples at
            `sample_rate`), and `start_sec` and `end_sec` (the same span in
            seconds).

        Raises
        ------
        ValueError
            If `chunk_overlap_sec` is not smaller than `chunk_duration_sec`,
            which would make the chunk hop non-positive.
        """
        hop_sec = self.chunk_duration_sec - self.chunk_overlap_sec
        if hop_sec <= 0:
            raise ValueError(
                f"chunk_overlap_sec ({self.chunk_overlap_sec}) must be smaller than "
                f"chunk_duration_sec ({self.chunk_duration_sec})"
            )

        chunk_samples = int(round(self.chunk_duration_sec * self.sample_rate))
        hop_samples = int(round(hop_sec * self.sample_rate))

        # Floor, so a chunk never claims audio past the end of the file.
        seconds = np.asarray(durations, dtype=np.float64)
        total_samples = np.floor(seconds * self.sample_rate).astype(np.int64)
        past_first_chunk = np.maximum(total_samples - chunk_samples, 0)
        n_chunks = np.ceil(past_first_chunk / hop_samples).astype(np.int64) + 1

        row_index = np.repeat(np.arange(len(total_samples), dtype=np.int64), n_chunks)
        # Position within the parent recording: a global arange rebased by the
        # global index at which each recording's chunks start.
        group_start = np.cumsum(n_chunks) - n_chunks
        global_idx = np.arange(int(n_chunks.sum()), dtype=np.int64)
        chunk_idx = global_idx - np.repeat(group_start, n_chunks)

        start_sample = chunk_idx * hop_samples
        end_sample = np.minimum(start_sample + chunk_samples, total_samples[row_index])

        return {
            "row_index": row_index,
            "chunk_idx": chunk_idx,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "start_sec": start_sample / self.sample_rate,
            "end_sec": end_sample / self.sample_rate,
        }

    def __len__(self) -> int:
        """Return the number of chunks in the dataset.

        One item is one chunk: `_load` expands the recording table into a row
        per chunk (see `_expand_to_chunks`), so this is simply the length of
        the expanded table.

        Returns
        -------
        int
            Total number of chunks across every recording in the split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Read one chunk's audio as mono at `sample_rate`.

        Only the chunk's own time range is read, rather than the whole
        recording — these run to several hours, and decoding one whole would
        cost tens of gigabytes at the native rate.

        The decoded length is checked against the chunk's nominal span. The
        recording durations are exact sample counts, so the expected length is
        known precisely; a read that lands somewhere other than where it was
        asked to — a bad seek, a truncated HTTP range, a file disagreeing with
        its own header — would otherwise produce a perfectly plausible chunk
        whose detections are silently offset.

        Parameters
        ----------
        row : dict[str, Any]
            One chunk's row, carrying `chunk_start_sec`, `chunk_end_sec`,
            `chunk_start_sample`, `chunk_end_sample` and `chunk_id` alongside
            the parent recording's own columns.

        Returns
        -------
        dict[str, Any]
            The row with `audio` (mono float32 at `sample_rate`) and
            `sample_rate` added, or the remapped subset of it when
            `output_take_and_give` is set.

        Raises
        ------
        ValueError
            If the decoded chunk's length differs from its nominal span by more
            than `_LENGTH_TOLERANCE_SAMPLES`.
        """
        # `anypath` renders a bare scheme as ``gs:///``, so joining onto it
        # leaves a stray separator and the path then reports an *empty* bucket.
        # `alp_data.io` scopes its short-lived GCS token to the bucket the path
        # names, so an empty one grants nothing: the ffmpeg range read is
        # refused with a 403 and `read_audio` quietly falls back to downloading
        # the whole recording, costing gigabytes per chunk instead of megabytes.
        joined = anypath(self.data_root) / row[self._originals_path_column]
        audio_path = anypath(str(joined).replace(":///", "://", 1))
        audio, sr = read_audio(
            audio_path,
            start_time=row["chunk_start_sec"],
            end_time=row["chunk_end_sec"],
        )
        # The reader returns the file's native channel layout as
        # ``(frames, channels)``; collapse it before resampling, which would
        # otherwise filter along the channel axis rather than time.
        audio = audio_stereo_to_mono(np.asarray(audio, dtype=np.float32))

        if sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="soxr_hq",
            )
            sr = self.sample_rate

        expected_samples = int(row["chunk_end_sample"]) - int(row["chunk_start_sample"])
        if abs(len(audio) - expected_samples) > self._LENGTH_TOLERANCE_SAMPLES:
            raise ValueError(
                f"chunk {row['chunk_id']}: decoded {len(audio)} samples at {sr} Hz, "
                f"expected {expected_samples} "
                f"(tolerance +/-{self._LENGTH_TOLERANCE_SAMPLES} samples)"
            )

        row["audio"] = np.ascontiguousarray(audio, dtype=np.float32)
        row["sample_rate"] = sr
        row["unique_id"] = row["chunk_id"]

        if self.output_take_and_give:
            item = {}
            for old_key, new_key in self.output_take_and_give.items():
                item[new_key] = row[old_key]
            return item

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific sample from the dataset.
        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the audio data, text label, label, and path.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["SpanishCarrionCrowsChunks", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters.

        Returns
        -------
        tuple[Dataset, dict[str, Any]]
            A tuple containing the dataset instance and metadata.
            If the dataset_config contains transformations, they will be applied
            and the metadata will be returned as dict, otherwise an empty dict.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            sample_rate=cfg["sample_rate"],
            chunk_duration_sec=cfg["chunk_duration_sec"],
            chunk_overlap_sec=cfg["chunk_overlap_sec"],
            output_take_and_give=cfg["output_take_and_give"],
            data_root=cfg["data_root"],
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )

        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta

        return ds, {}

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
