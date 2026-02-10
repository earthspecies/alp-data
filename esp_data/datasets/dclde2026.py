"""DCLDE 2026 Killer Whale dataset.

Palmer et al. (2025) — A Public Dataset of Annotated Orcinus orca Acoustic Signals
for Detection and Ecotype Classification. doi:10.1038/s41597-025-05281-5
"""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

# All species that appear in the processed annotations
SPECIES_LABELS = [
    "Killer whale",
    "Humpback whale",
    "Pacific white-sided dolphin",
    "Bowhead whale",
    "Unknown biological",
    "Fish",
    "Vessel noise",
    "Background",
    "Odontocete",
    "Risso's dolphin",
    "Gray whale",
    "Sperm whale",
    "Mooring noise",
    "Seal",
]

# Killer whale ecotype labels
ECOTYPE_LABELS = [
    "SRKW",  # Southern Resident
    "TKW",  # Transient (Bigg's)
    "NRKW",  # Northern Resident
    "SAR",  # Southern Alaskan Resident
    "OKW",  # Offshore
]


@register_dataset
class DCLDE2026(Dataset):
    """DCLDE 2026 Killer Whale Dataset.

    Description
    -----------
    Multi-provider annotated acoustic recordings of killer whales, humpback
    whales, and bowhead whales from Alaska, British Columbia, and Washington
    (2011–2024). Each entry is an audio file plus an enriched selection table
    containing detection/call-level annotations with species, ecotype, call
    type, pod, clan, and acoustic behavior labels — all human-annotated.

    The selection table columns are:
        Begin Time (s), End Time (s), Low Freq (Hz), High Freq (Hz),
        species, canonical_name, sound_detail, ecotype, call_type,
        acoustic_behavior, pod, clan, annotation_level, confidence

    Available tasks:
        - Species classification: Killer whale / Humpback whale / Bowhead whale / Unknown biological
        - KW detection (binary): presence / absence of killer whale
        - Ecotype classification: SRKW / TKW / NRKW / SAR / OKW
        - Call type classification: S-series, N-series, T-series, OFF-series, BP, W, etc.
        - Acoustic behavior: burst_pulse / pulsed_call / whistle / click / buzzer / echolocation
        - Pod identification: J / K / L pods (Southern Resident)
        - Clan identification: A / G clans (Northern Resident)

    Data providers: DFO_CRP, JASCO_VFPA, DFO_WDLP, UAF_NGOS, SIMRES, SIO,
    ONC, OrcaSound, JASCO_VFPA_ONC, SMRUConsulting.

    References
    ----------
    Palmer et al. (2025) doi:10.1038/s41597-025-05281-5
    License: CC-BY-4.0
    """

    info = DatasetInfo(
        name="dclde2026",
        owner="david",
        split_paths={
            "all": "gs://esp-ml-datasets/dclde2026/v0.1.0/raw/2026/dclde_2026_killer_whales/processed.csv",
        },
        version="0.1.0",
        description="DCLDE 2026 killer whale dataset with species, ecotype, call type, "
        "pod, clan, and acoustic behavior annotations across 10 providers",
        sources="Palmer et al. (2025) doi:10.1038/s41597-025-05281-5",
        license="CC-BY-4.0",
    )

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
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
            If set, audio is resampled to this rate.
        data_root : str | AnyPathT | None
            Root directory containing provider audio subdirectories.
            If None, defaults to the parent directory of the split CSV path.
        backend : BackendType
            The backend to use ("pandas" or "polars"), by default "polars".
        streaming : bool
            Whether to use streaming mode, by default False.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None
        self.annotation_columns = [
            "species",
            "ecotype",
            "call_type",
            "acoustic_behavior",
            "pod",
            "clan",
        ]

        # Load split CSV
        self._load()

        # If no explicit data_root, assume parent dir of the split path
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

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
        """Return the number of audio files in the dataset.

        Returns
        -------
        int
            Number of audio files in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
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
        # Resolve audio path
        audio_fp = self.data_root / row["audio_path"]

        # Read audio
        audio, sr = read_audio(audio_fp)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        # Resample if necessary
        target_sr = self.sample_rate
        if target_sr is not None and sr != target_sr:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=target_sr,
                scale=True,
                res_type="kaiser_best",
            )
            sr = target_sr

        # Parse selection table from serialized TSV
        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t", keep_default_na=False)

        # Clip events outside audio (keep only events that begin before audio end)
        audio_dur = len(audio) / float(sr)
        st = st[st["Begin Time (s)"] < audio_dur].copy()

        # Build output
        row["audio"] = audio
        row["sample_rate"] = sr
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
            A dictionary containing the processed data.
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["DCLDE2026", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters.

        Returns
        -------
        tuple[DCLDE2026, dict[str, Any]]
            A tuple containing the dataset instance and metadata.
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

    def get_available_labels(self, annotation_column: str = "species") -> list[str]:
        """Return all possible labels for a given annotation column.

        Parameters
        ----------
        annotation_column : str
            Which annotation column to get labels for.
            Predefined label sets exist for: ``species``, ``ecotype``.

        Returns
        -------
        list[str]
            All possible label values for the given column.

        Raises
        ------
        ValueError
            If ``annotation_column`` does not have a predefined label set.
        """
        if annotation_column == "species":
            return SPECIES_LABELS
        elif annotation_column == "ecotype":
            return ECOTYPE_LABELS
        else:
            raise ValueError(
                f"No predefined label set for '{annotation_column}'. "
                f"Columns with predefined labels: species, ecotype"
            )

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{base}\n"
            f"Audio files: {n}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
