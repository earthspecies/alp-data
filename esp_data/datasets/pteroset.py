"""PteroSet dataset."""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/pteroset/v0.1.0"
SPECIES_INFO_PATH = f"{_GCS_ROOT}/species_labels.csv"


@register_dataset
class PteroSet(Dataset):
    """PteroSet Dataset.

    Description
    -----------
    A strongly annotated passive-acoustic-monitoring dataset for tropical bird
    monitoring from two Colombian regions (Putumayo and Magdalena). Each entry
    is a passive-acoustic recording plus a selection table; each row of the
    selection table is a time/frequency-bounded acoustic event annotated by
    experts in Raven Pro. Every event is a bird vocalisation
    (``Identification`` = ``AVEVOC``); the subset identified to species carries
    a GBIF canonical name in the ``Species`` column (others are ``Unknown``).
    The original PteroSet species code is retained in ``Species Code``.

    The collection contains 563 recordings (~73.6 hours, 15,372 annotated
    events, 6,702 identified to 168 species), recorded as 10-second clips every
    30 minutes over 24-hour cycles across five monitoring projects in 2023-2025.

    Splits
    ------
    Recordings are grouped by monitoring project, each exposed as a split, plus
    a combined ``all`` split:

        - ``map1`` : Magdalena (46 recordings)
        - ``ppa1`` : Putumayo, project 1 (108)
        - ``ppa2`` : Putumayo, project 2 (137)
        - ``ppa3`` : Putumayo, project 3 (151)
        - ``ppa4`` : Putumayo, project 4 (121)
        - ``all``  : all five combined (563)

    Pre-resampled Audio
    -------------------
    Originals are 192 kHz mono WAV. Pre-resampled audio is available at 16 kHz
    and 32 kHz (16-bit PCM WAV). When ``sample_rate`` matches one of these
    rates, the pre-resampled files are loaded directly (no on-the-fly
    resampling). For any other target rate, audio is resampled on-the-fly using
    librosa's ``kaiser_best`` method.

    References
    ----------
    https://zenodo.org/records/19137071
    https://github.com/microsoft/PteroSet
    arXiv:2605.20578
    """

    info = DatasetInfo(
        name="pteroset",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/all.csv",
            "map1": f"{_GCS_ROOT}/map1.csv",
            "ppa1": f"{_GCS_ROOT}/ppa1.csv",
            "ppa2": f"{_GCS_ROOT}/ppa2.csv",
            "ppa3": f"{_GCS_ROOT}/ppa3.csv",
            "ppa4": f"{_GCS_ROOT}/ppa4.csv",
        },
        version="0.1.0",
        description=(
            "PteroSet — a strongly annotated passive-acoustic dataset for "
            "tropical bird monitoring from Colombia. 563 recordings (~73.6 h, "
            "15,372 expert Raven-annotated events, 6,702 to 168 species) across "
            "five monitoring projects, with time-frequency selection tables. "
            "Species coerced to GBIF canonical names; raw PteroSet codes "
            "retained. Originals at 192 kHz; pre-resampled 16 kHz and 32 kHz "
            "versions available."
        ),
        sources=["https://zenodo.org/records/19137071"],
        license="CC-BY-4.0",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_fp"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load (key in info.split_paths): ``all`` or one of
            ``map1``/``ppa1``/``ppa2``/``ppa3``/``ppa4``.
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns as well).
        sample_rate : int | None
            If set, audio is resampled to this rate.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row's audio path.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "pandas".
        streaming : bool, optional
            Whether to use streaming mode, by default False.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.annotation_columns = ["Species"]
        self.unknown_label = "Unknown"
        self.sample_rate = sample_rate

        self.full_dataset_available_labels = None  # placeholder for labels if split == all

        self._load()

        if data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent
        else:
            self.data_root = anypath(data_root)

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Return pre-resampled sample rates whose path columns exist in the data."""
        return [sr for sr, col in self._sample_rate_paths.items() if col in self._data.columns]

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
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

        When the row contains ``window_start_sec`` / ``window_end_sec`` (set by
        the ``window_annotations`` transform), only the corresponding audio
        segment is loaded from disk/GCS instead of the full recording.

        Parameters
        ----------
        row : dict[str, Any]
            A dictionary representing a single row of the dataset.

        Returns
        -------
        dict[str, Any]
            The processed row.
        """
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if path_column in row and row[path_column] is not None and row[path_column] != "":
                audio_path = anypath(self.data_root) / row[path_column]
                use_presampled = True

        if not use_presampled:
            audio_path = anypath(self.data_root) / row[self._originals_path_column]

        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")

        if window_start is not None and window_end is not None:
            audio, sr = read_audio(
                audio_path, start_time=float(window_start), end_time=float(window_end)
            )
        else:
            audio, sr = read_audio(audio_path)

        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        if not use_presampled and self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sr = self.sample_rate

        row["audio"] = audio
        row["sample_rate"] = sr

        raw_st = row.get("selection_table")
        if raw_st is not None:
            if isinstance(raw_st, str):
                st = pd.read_csv(StringIO(raw_st), sep="\t")
            elif isinstance(raw_st, pd.DataFrame):
                st = raw_st
            else:
                st = pd.DataFrame()

            audio_dur = len(audio) / float(sr)
            if "Begin Time (s)" in st.columns:
                st = st[st["Begin Time (s)"] < audio_dur].copy()
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
            A dictionary containing the audio, sample rate, and selection table.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        ------
        dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["PteroSet", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters.

        Returns
        -------
        tuple[PteroSet, dict[str, Any]]
            A tuple containing the dataset instance and metadata. If the
            dataset_config contains transformations, they will be applied and
            the metadata will be returned as dict, otherwise an empty dict.
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

    def get_available_labels(self, anno_column: str | None = "Species") -> list[str]:
        """Return all possible species labels.

        Returns
        -------
        list[str]
            A list of all the available labels for ``anno_column``.
        """
        if self.split == "all":
            if self.full_dataset_available_labels is None:
                self.full_dataset_available_labels = pd.read_csv(SPECIES_INFO_PATH)[
                    "Species"
                ].to_list()
            return self.full_dataset_available_labels
        else:
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
