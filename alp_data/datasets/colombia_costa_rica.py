"""Colombia and Costa Rica coffee farm soundscapes dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import pandas as pd

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import DATA_HOME, AnyPathT, anypath, audio_stereo_to_mono, read_audio

_RAW_ROOT = f"{DATA_HOME}/colombia_costa_rica_soundscapes/v0.1.0/formatted"


@register_dataset
class ColombiaCostaRica(Dataset):
    """ColombiaCostaRica Dataset

    Description
    -----------
    Fully annotated dawn and dusk soundscapes from neotropical coffee farms in
    Jardín, Colombia and San Ramón, Costa Rica, provided by the Cornell Lab of
    Ornithology.

    Description from the Zenodo:

    "This collection contains 34 hour-long soundscape recordings, which have
    been annotated by expert ornithologists who provided 6,952 bounding box
    labels for 89 different bird species from Colombia and Costa Rica. The
    data were recorded in 2019 at two highly diverse neotropical coffee farm
    landscapes from the towns of Jardín, Colombia and San Ramon, Costa Rica.
    This collection has partially been featured as test data in the 2021
    BirdCLEF competition and can primarily be used for training and evaluation
    of machine learning algorithms.

    Data collection

    [...] Soundscapes for this collection were recorded using SWIFT recorders,
    positioned 3m above the ground. We recorded 48kHz one-hour long sound files
    from 4:30 to 7:30 to capture the most active time frame of the avian dawn
    chorus. In the same way, we recorded from 16:00 to 19:00, to capture the
    second avian bioacoustics activity peak that occurs before sunset, and to
    include nocturnal species. All audio was unified, converted to FLAC, and
    resampled to 32 kHz for this collection.

    Sampling and annotation protocol

    We subsampled data for this collection by randomly selecting recordings
    coming from different farm locations and dates. Using Raven Pro,
    annotators were asked to create a selection box around every bird call
    they could recognize, ignoring those that were too faint or
    unidentifiable. We allowed overlapping selections. Provided labels contain
    full bird calls that are boxed in time and frequency. Annotators were
    allowed to combine multiple consecutive calls of the same species into one
    bounding box label if pauses between calls were shorter than 5 seconds. We
    converted labels to eBird species codes, following the 2021 eBird taxonomy
    (Clements list). Unidentifiable calls have been marked with "????" and were
    added as bounding box labels to the ground truth annotations."

    Recording filenames contain a sequential ID, site ID (``S01``: Jardín,
    Colombia; ``S02``: San Ramón, Costa Rica), date, and local start time.
    The audio is losslessly re-encoded from the Zenodo FLACs, 14 of which
    libsndfile cannot decode as distributed.

    Labels
    ------
    Labels come from updated annotations provided by Alvaro Vega-Hidalgo in
    October 2026, which correct two single-event labels in the Zenodo release:
    a Russet-crowned Warbler (``rucwar1``, an Andean species) in
    ``NES_017_S02`` is relabelled Rufous-capped Warbler (``rucwar``), and a
    Melodious Warbler (``melwar1``, a European species) in ``NES_019_S02`` is
    relabelled Melodious Blackbird (``melbla1``).

    The eBird codes are mapped to scientific names with the 2021 eBird
    taxonomy, then to accepted GBIF canonical names. Calls the annotators could
    not identify (``????`` in the source) are labelled ``Unknown``. This leaves
    87 species. Zero-length events are dropped.

    Splits
    ------
    - ``all``: all 34 recordings, ``Unknown`` events removed.
    - ``all_with_unknowns``: all 34 recordings, ``Unknown`` events kept.

    Each entry consists of:
    - an audio recording
    - a selection table (Raven format), with Species labels

    References
    ----------
    Dataset: [Zenodo](https://zenodo.org/records/7525349)

    """

    info = DatasetInfo(
        name="colombia_costa_rica",
        owner="benjamin",
        split_paths={
            "all": f"{_RAW_ROOT}/all.csv",
            "all_with_unknowns": f"{_RAW_ROOT}/all_with_unknowns.csv",
        },
        version="0.1.0",
        description=(
            "34 hour-long dawn and dusk soundscapes from coffee farms in Colombia and "
            "Costa Rica, annotated with time-frequency boxes for 87 bird species."
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
    ) -> tuple["ColombiaCostaRica", dict[str, Any]]:
        """Create the dataset from a `DatasetConfig`.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration with the constructor arguments and optional transformations.

        Returns
        -------
        tuple[ColombiaCostaRica, dict[str, Any]]
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
