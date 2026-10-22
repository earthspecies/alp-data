"""High Sierra Nevada soundscapes dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import pandas as pd

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

# _RAW_ROOT = f"{DATA_HOME}/sierra_nevada_soundscapes/v0.1.0/formatted"

# temporary override; remove before merge
_RAW_ROOT = "gs://esp-data-ingestion/sierra_nevada_soundscapes/formatted"


@register_dataset
class HighSierraNevada(Dataset):
    """HighSierraNevada Dataset

    Description
    -----------
    Fully annotated 10-minute morning soundscapes from high-elevation lake
    basins in Sequoia and Kings Canyon National Parks, California, USA,
    provided by the Cornell Lab of Ornithology.

    Description from the Zenodo:

    "This collection contains 100 soundscape recordings of 10 minutes
    duration, which have been annotated with 10,296 bounding box labels for 21
    different bird species from the Western United States. The data were
    recorded in 2015 in the southern end of the Sierra Nevada mountain range in
    California, USA. This collection has been featured as test data in the
    2020 BirdCLEF and Kaggle Birdcall Identification competition and can
    primarily be used for training and evaluation of machine learning
    algorithms.

    Data collection

    The recordings were made in Sequoia and Kings Canyon National Parks, two
    contiguous national parks in the southern Sierra Nevada mountain range in
    California, USA. The focus of the acoustic study was the high-elevation
    region of the Parks; specifically, the headwater lake basins above 3,000
    km in elevation. [...] Soundscapes were recorded for 24 h continuously at
    10 lakes (5 fishless, 5 fish-containing) throughout Sequoia and Kings
    Canyon National Parks during June-September 2015. Song Meter SM2+ units
    (Wildlife Acoustics, USA) [...] continuously recorded mono-channel,
    16-bits uncompressed WAVE files at 48 kHz sampling rate. For this
    collection, recordings were resampled at 32 kHz and converted to FLAC.

    Sampling and annotation protocol

    A total of 100 10-minute segments of audio between July 9 and 12, 2015
    from morning hours (06:10-09:10 PDT) from all 10 sites were selected at
    random. Annotators were asked to box every bird call they could
    recognize, ignoring those that are too faint or unidentifiable. Every
    sound that could not be confidently assigned an identity was reviewed with
    1-2 other experts in bird identification. To minimize observer bias, all
    identifying information about the location, date and time of the
    recordings was hidden from the annotator. Raven Pro software was used to
    annotate the data. Provided labels contain full bird calls that are boxed
    in time and frequency. In this collection, we use eBird species codes as
    labels, following the 2021 eBird taxonomy (Clements list). Unidentifiable
    calls have been marked with "????" and were added as bounding box labels
    to the ground truth annotations."

    Recording filenames contain a sequential ID, date, and start time in PDT.
    18 recordings are 3-4 s shorter than 10 minutes.

    Labels
    ------
    The original eBird codes are mapped to scientific names with the 2021
    eBird taxonomy, then to accepted GBIF canonical names. Calls the annotators
    could not identify (``????`` in the source) are labelled ``Unknown``.
    Zero-length events are dropped.

    Splits
    ------
    - ``all``: all 100 recordings, ``Unknown`` events removed.
    - ``all_with_unknowns``: all 100 recordings, ``Unknown`` events kept.

    Each entry consists of:
    - an audio recording
    - a selection table (Raven format), with Species labels

    References
    ----------
    Dataset: [Zenodo](https://zenodo.org/records/7525805)

    A derived version of this dataset, consisting of 5-sec clips with clip-level labels,
    appears as part of BirdSet (HSN)

    """

    info = DatasetInfo(
        name="high_sierra_nevada",
        owner="benjamin",
        split_paths={
            "all": f"{_RAW_ROOT}/all.csv",
            "all_with_unknowns": f"{_RAW_ROOT}/all_with_unknowns.csv",
        },
        version="0.1.0",
        description=(
            "100 10-minute morning soundscapes from the southern Sierra Nevada "
            "(California, USA), annotated with time-frequency boxes for 21 bird species."
        ),
        sources="Zenodo",
        license="CC-BY-4.0",
    )

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: Dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load (key in info.split_paths).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns as well).
        sample_rate : int | None
            If set, audio is resampled to this rate. Defaults to None. The native
            rate is 32000.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row['audio_path'].
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.annotation_columns = ["Species"]

        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None

        # Load split CSV
        self._load()

        # If no explicit data_root, assume parent dir of the split path
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        """Load the CSV of the current split.

        Raises
        ------
        LookupError
            If `self.split` is not one of `available_splits`.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            location, streaming=self._streaming, keep_default_na=False, na_values=[""]
        )

    def __len__(self) -> int:
        """Number of recordings in the loaded split.

        Returns
        -------
        int
            Number of recordings.

        Raises
        ------
        RuntimeError
            If no split has been loaded.
        NotImplementedError
            If the dataset is in streaming mode.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of the dataset.

        Parameters
        ----------
        row : dict[str, Any]
            A dictionary representing a single row of the dataset.

        Returns
        -------
        dict[str, Any]
            The processed row.
        """

        # Resolve audio path
        audio_path = (
            (self.data_root / row["audio_path"]) if self.data_root else anypath(row["audio_path"])
        )

        # Read audio
        audio, sample_rate = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        # Resample if necessary
        if self.sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sample_rate = self.sample_rate

        # Selection table
        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")

        # Clip events outside audio (keep only events that begin before audio end)
        audio_dur = len(audio) / float(sample_rate)
        st = st[st["Begin Time (s)"] < audio_dur].copy()

        # Build output
        row["audio"] = audio
        row["sample_rate"] = sample_rate
        row["selection_table"] = st

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
            A dictionary containing the processed data.
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
    ) -> tuple["HighSierraNevada", dict[str, Any]]:
        """Create the dataset from a `DatasetConfig`.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration with the constructor arguments and optional transformations.

        Returns
        -------
        tuple[HighSierraNevada, dict[str, Any]]
            The dataset and the metadata returned by its transformations (empty if none).
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            data_root=cfg["data_root"],
            sample_rate=cfg["sample_rate"],
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def get_available_labels(self, anno_column: str = "Species") -> List[str]:
        """Return all possible labels for a given annotation column.

        Parameters
        ----------
        anno_column : str, optional
            Selection table column to collect labels from, by default "Species".

        Returns
        -------
        list[str]
            Sorted unique labels found in `anno_column` across all selection tables.
        """
        available_labels = set()
        for row in self._data:
            st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
            available_labels.update(st[anno_column].astype(str).tolist())
        return sorted(available_labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
