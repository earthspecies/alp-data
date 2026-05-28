"""Weldy NW Dawn Chorus dataset."""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/weldy_dawn_chorus/v0.1.0"
SPECIES_INFO_PATH = f"{_GCS_ROOT}/species_labels.csv"


@register_dataset
class WeldyDawnChorus(Dataset):
    """Weldy et al. 2024 — NW US dawn-chorus audio-tagging dataset.

    Description
    -----------
    1,575 × 5-min mono soundscape recordings (32 kHz, two omnidirectional mics
    averaged) from 525 sites across federal forests in California, Oregon and
    Washington (May–August 2022 dawn chorus). Annotations are **multi-label
    per 2-s window**, sonotype-aware (``song``/``call``/``drum``/...), covering
    58 birds + 2 mammals + 6 biotic-aggregate + 8 non-biotic sound types,
    expressed as eBird-2021 codes with sonotype suffixes (e.g. ``herthr_song_1``
    for Hermit Thrush song variant 1).

    Each row carries an embedded tab-separated ``selection_table`` listing every
    annotated window's bounds and labels. Per-window columns include the GBIF
    canonical name, eBird code, sonotype, category (species/non-biotic/
    biotic-aggregate/method) and ``clip_complete`` flag.

    Splits
    ------
        - ``all``       : every recording (1,575)
        - ``complete``  : fully-annotated subset (156 files, ~12 h)
        - ``partial``   : partially-annotated subset (215 files)
        - ``labeled``   : ``complete`` ∪ ``partial`` (371)
        - ``unlabeled`` : remaining ~1,204 recordings (useful negatives / SSL)

    Pre-resampled Audio
    -------------------
    Originals are 32 kHz stereo WAV; pre-resampled 16 kHz and 32 kHz mono WAVs
    are available at ``audio_16k/`` / ``audio_32k/`` (PCM16). Loaded directly
    when ``sample_rate`` matches; otherwise resampled on-the-fly with librosa
    ``kaiser_best``.

    References
    ----------
    https://zenodo.org/records/10895837
    Weldy et al. 2024, *Audio tagging of avian dawn chorus recordings in
    California, Oregon, and Washington* (Northwest Forest Plan Monitoring).
    """

    info = DatasetInfo(
        name="weldy_dawn_chorus",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/all.csv",
            "complete": f"{_GCS_ROOT}/complete.csv",
            "partial": f"{_GCS_ROOT}/partial.csv",
            "labeled": f"{_GCS_ROOT}/labeled.csv",
            "unlabeled": f"{_GCS_ROOT}/unlabeled.csv",
        },
        version="0.1.0",
        description=(
            "Weldy et al. 2024 NW US dawn-chorus dataset: 1,575 × 5-min "
            "soundscape recordings (32 kHz) from CA/OR/WA federal forests, "
            "with multi-label sonotype-aware annotations on 2-s windows over "
            "the fully-annotated 156 files + 5,500 partial annotations on 215 "
            "more. Species coerced to GBIF canonical names; raw Weldy labels + "
            "sonotype + category retained."
        ),
        sources=["https://zenodo.org/records/10895837"],
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
            Split to load (key in info.split_paths).
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
        self.full_dataset_available_labels = None

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
        self._data = self._backend_class.from_csv(
            self.info.split_paths[self.split],
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

        Loads audio (presampled or original), downmixes to mono, optionally
        resamples, and parses the embedded ``selection_table`` TSV string into
        a DataFrame.

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
            return {new: row[old] for old, new in self.output_take_and_give.items()}
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["WeldyDawnChorus", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters.

        Returns
        -------
        tuple[WeldyDawnChorus, dict[str, Any]]
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
        if self.split in ("all", "labeled", "complete"):
            if self.full_dataset_available_labels is None:
                self.full_dataset_available_labels = pd.read_csv(SPECIES_INFO_PATH)[
                    "Species"
                ].to_list()
            return self.full_dataset_available_labels
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
