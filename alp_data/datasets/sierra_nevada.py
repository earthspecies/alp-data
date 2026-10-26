"""Sierra Nevada soundscapes dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import pandas as pd

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import DATA_HOME, AnyPathT, anypath, audio_stereo_to_mono, read_audio

_RAW_ROOT = f"{DATA_HOME}/western_us_soundscapes/v0.1.0/formatted"


@register_dataset
class SierraNevada(Dataset):
    """SierraNevada Dataset

    Description
    -----------
    Fully annotated morning and evening soundscapes from the Lassen and Plumas
    National Forests in the Sierra Nevada, California, USA, provided by the
    Cornell Lab of Ornithology.

    Description from the Zenodo:

    "This collection contains 33 hour-long soundscape recordings, which have
    been annotated with 20,147 bounding box labels for 56 different bird
    species from the Western United States. The data were recorded in 2018 in
    the Sierra Nevada, California, USA. This collection has partially been
    featured as test data in the 2021 BirdCLEF competition and can primarily
    be used for training and evaluation of machine learning algorithms.

    Data collection

    Measuring the effects of forest management activities in the Sierra
    Nevada, California, USA can reveal a potential correlation with avian
    population density and diversity. For this dataset, passive acoustic
    surveys were conducted in the Lassen and Plumas National Forests in
    May-August 2018. Survey grid cells (4 km2) were randomly selected from a
    6,000-km2 area, and SWIFT recording units were deployed at locations
    conducive to sound propagation (e.g., ridges rather than gullies) within
    those cells. [...] Recording units recorded continuously 17:00 - 23:59,
    0:00 - 10:00, one-hour files were stored as uncompressed WAVE sampled at
    32 kHz and later converted to FLAC.

    Sampling and annotation protocol

    We subsampled data for this collection by selecting locations that
    spanned the full elevational and latitudinal gradients of our study area
    (~840 – 1700 m asl and 39.41 – 40.71°N), and thus represent a broad range
    of plant communities. A single annotator boxed every bird call he could
    recognize, ignoring those that are too faint or unidentifiable. Raven Pro
    software was used to annotate the data. Provided labels contain full bird
    calls that are boxed in time and frequency. The annotator was allowed to
    combine multiple consecutive calls of one species into one bounding box
    label if pauses between calls were shorter than five seconds. We use eBird
    species codes as labels, following the 2021 eBird taxonomy (Clements
    list)."

    Recording filenames contain a sequential ID, date, and start time in PDT;
    recordings start between 04:00 and 09:00 or between 17:00 and 19:00. 4
    recordings are up to ~5 s shorter than an hour. The audio is losslessly
    re-encoded from the Zenodo FLACs, 5 of which libsndfile cannot decode as
    distributed.

    Labels
    ------
    The original eBird codes are mapped to scientific names with the 2021
    eBird taxonomy, then to accepted GBIF canonical names. Unlike other
    collections in this series, there are no unidentified (``????``) calls.
    Zero-length events are dropped.

    Splits
    ------
    - ``all``: all 33 recordings.

    Each entry consists of:
    - an audio recording
    - a selection table (Raven format), with Species labels

    References
    ----------
    Dataset: [Zenodo](https://zenodo.org/records/7050014)

    A derived version of this dataset, consisting of 5-sec clips with clip-level labels,
    appears as part of BirdSet (SNE)

    """

    info = DatasetInfo(
        name="sierra_nevada",
        owner="benjamin",
        split_paths={
            "all": f"{_RAW_ROOT}/all.csv",
        },
        version="0.1.0",
        description=(
            "33 hour-long soundscapes from the Sierra Nevada (California, USA), "
            "annotated with time-frequency boxes for 56 bird species."
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["SierraNevada", dict[str, Any]]:
        """Create the dataset from a `DatasetConfig`.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration with the constructor arguments and optional transformations.

        Returns
        -------
        tuple[SierraNevada, dict[str, Any]]
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
