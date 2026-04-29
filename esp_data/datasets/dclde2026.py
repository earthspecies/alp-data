"""DCLDE 2026 dataset"""

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

COARSE_CALL_TYPE_LABELS = ["call", "whistle", "click", "burst_pulse"]

# Data providers in the DCLDE 2026 dataset.
# UAF_NGOS is excluded
PROVIDERS = [
    "DFO_CRP",
    "JASCO_VFPA",
    "DFO_WDLP",
    "SIMRES",
    "SIO",
    "ONC",
    "OrcaSound",
    "JASCO_VFPA_ONC",
    "SMRUConsulting",
]

# Per-row provenance columns (expected in the CSV).
PROVENANCE_COLUMNS = [
    "provider",
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

    Columns
    -------
    audio_path : str
        Relative path to source audio.
    selection_table : str
        TSV-serialised selection table with columns:
        ``Begin Time (s)``, ``End Time (s)``, ``Low Freq (Hz)``,
        ``High Freq (Hz)``, ``species``, ``canonical_name``,
        ``sound_detail``, ``ecotype``, ``call_type``,
        ``acoustic_behavior``, ``pod``, ``clan``,
        ``annotation_level``, ``confidence``, ``coarse_call_type``.
    provider : str
        Data provider name (see :data:`PROVIDERS`).
    16khz_path, 32khz_path : str | None
        Paths to pre-resampled audio (when available).

    Splits
    ---------
    - "all": All data (default)

    Available tasks
    ---------------
    - Species classification: Killer whale / Humpback whale / Bowhead whale / Unknown biological
    - KW detection (binary): presence / absence of killer whale
    - Ecotype classification: SRKW / TKW / NRKW / SAR / OKW
    - Call type classification (fine-grained): S04, N24ii, T01, whistle, BP, EL, etc.
    - Call type classification (coarse): call / whistle / click / burst_pulse
      (see :data:`COARSE_CALL_TYPE_LABELS` below; aligned with Watkins taxonomy)
    - Pod identification: J / K / L pods (Southern Resident)
    - Clan identification: A / G clans (Northern Resident)

    Provider Notes
    --------------
    Data providers: DFO_CRP, JASCO_VFPA, DFO_WDLP, SIMRES, SIO,
    ONC, OrcaSound, JASCO_VFPA_ONC, SMRUConsulting.

    **UAF_NGOS** (University of Alaska Fairbanks / North Gulf Oceanic Society)
    is excluded from the dataset.

    Providers differ in annotation precision and coverage. Combining data
    from multiple providers should be done carefully — consider filtering
    by provider using Transforms (e.g. ``filter_isin`` on the ``provider``
    column) when training or evaluating.

    Per-provider observations (detection focus):

    - SMRU: Not very temporally precise; selections sometimes cover large
      segments around the call. Multiple faint calls may be grouped.
    - SIMRES: Consistent annotations.
    - VFPA: Generally good; a few missed calls and slightly less
      consistency with faint calls.

    Coarse call types (aligned with Watkins taxonomy):

    Mapping rules:
    call         — Discrete pulsed calls (S-series, N-series, T-series,
                     OFF-series, NS) and variable vocalizations (tone, moan,
                     upsweep, chirp, groan, knock, shriek, whup, creak,
                     grunt, scream, rasp, growl).
    whistle      — Whistle-labeled signals (whistle, whistle/tone, W).
    click        — Echolocation clicks (EL) and rapid click trains (buzz, BZ).
    burst_pulse  — Burst-pulse signals (BP).

    Unknown / ambiguous labels (Unk, Multiple overlapping, etc.) map to empty
    string and are dropped when ``drop_empty_windows=True`` in windowing.


    Examples
    --------
    >>> from esp_data.datasets import DCLDE2026
    >>> dataset = DCLDE2026(split="all")
    >>> print(dataset.info.name)
    dclde2026

    References
    ----------
    Palmer et al. (2025) doi:10.1038/s41597-025-05281-5
    License: CC-BY-4.0
    """

    info = DatasetInfo(
        name="dclde2026",
        owner="david",
        split_paths={
            "all": "gs://esp-ml-datasets/dclde2026/v0.1.0/raw/2026/dclde_2026_killer_whales/processed_enriched_v2.csv",
        },
        version="0.1.0",
        description="DCLDE 2026 killer whale dataset with species, ecotype, call type, "
        "pod, clan, and acoustic behavior annotations across 9 providers",
        sources="Palmer et al. (2025) doi:10.1038/s41597-025-05281-5",
        license="CC-BY-4.0",
    )

    _sample_rate_paths: dict[int, str] = {
        16000: "16khz_path",
        32000: "32khz_path",
    }

    # Subdirectories under data_root where pre-resampled audio lives.
    _sample_rate_subdirs: dict[int, str] = {
        16000: "audio_16k",
        32000: "audio_32k",
    }

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
            If set, audio is resampled to this rate.  Pre-resampled paths
            (``16khz_path``, ``32khz_path``) are preferred when present in the
            CSV; otherwise audio is resampled on-the-fly.
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
        self.annotation_columns = [
            "species",
            "ecotype",
            "call_type",
            "acoustic_behavior",
            "pod",
            "clan",
        ]
        self.data_root = anypath(data_root) if data_root is not None else None

        self._load()

        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        """Load the split CSV into the configured backend.

        Raises
        ------
        LookupError
            If the requested split is not in ``info.split_paths``.
        """
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

    # ------------------------------------------------------------------
    # Audio path resolution
    # ------------------------------------------------------------------

    def _resolve_audio_path(self, row: dict[str, Any]) -> tuple[AnyPathT, bool]:
        """Return ``(full_audio_path, is_presampled)``.

        If the CSV contains a pre-resampled path column for the requested
        sample rate (e.g. ``16khz_path``) and the value is non-empty, that
        path is used and ``is_presampled=True``.  The resampled file is
        located under ``data_root / <sr_subdir> / <16khz_path>``, where
        ``<sr_subdir>`` comes from :attr:`_sample_rate_subdirs` (e.g.
        ``audio_16k``).  Otherwise falls back to the original ``audio_path``.

        Returns
        -------
        tuple[AnyPathT, bool]
            ``(full_audio_path, is_presampled)`` — the resolved path and
            whether it points to a pre-resampled file.
        """
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            col = self._sample_rate_paths[self.sample_rate]
            if col in row and row[col] is not None and str(row[col]).strip():
                subdir = self._sample_rate_subdirs.get(self.sample_rate, "")
                if subdir:
                    return self.data_root / subdir / row[col], True
                return self.data_root / row[col], True
        return self.data_root / row["audio_path"], False

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        audio_fp, is_presampled = self._resolve_audio_path(row)

        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")

        if window_start is not None and window_end is not None:
            audio, sr = read_audio(
                audio_fp, start_time=float(window_start), end_time=float(window_end)
            )
        else:
            audio, sr = read_audio(audio_fp)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        # Resample on-the-fly only when no pre-resampled file was used
        if not is_presampled and self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sr = self.sample_rate

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
            sample_rate=cfg["sample_rate"],
            data_root=cfg["data_root"],
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
