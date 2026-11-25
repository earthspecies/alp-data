"""Mediterranean cetacean PAM SED dataset — strong detection (Raven boxes)."""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/mediterranean-cetaceans/v0.1.0"
_ST_COLUMNS = [
    "Selection",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
    "signalType",
]


@register_dataset
class MediterraneanCetaceans(Dataset):
    """Mediterranean cetacean passive-acoustic detection (Raven boxes).

    Description
    -----------
    Jankauskaite et al. (2025), "Multi-platform low-cost passive acoustic
    monitoring of cetaceans in the western Mediterranean" (Zenodo 17282717,
    CC-BY-4.0). Strong/detection ingest: Raven time+frequency selection tables
    on the FULL-LENGTH HydroMoth recordings, split to one selection row per
    ~1-minute WAV file the event covers. Six labels — five odontocete species
    (*Stenella coeruleoalba*, *Tursiops truncatus*, *Grampus griseus*,
    *Globicephala melas*, *Physeter macrocephalus*) plus family-level
    unidentified *Delphinidae* — each box carrying a ``signalType`` call type
    (``clicks`` / ``whistles``). WABAD-shaped: one row per recording with a
    ``selection_table`` (``Selection, Begin Time (s), End Time (s), Low Freq
    (Hz), High Freq (Hz), Species, signalType``). Recordings with no overlapping
    event are retained as pure-negative examples.

    This is the strong/detection complement to the weak clip-level SuperWhales
    entry (``zenodo_17282717_mediterranean_cetacean_clips``), built from the
    same release.

    Nyquist caveat
    --------------
    Source audio is high-rate (192 kHz HydroMoth). Against the 32 kHz / 16 kHz-
    Nyquist stack, essentially every box straddles or exceeds 16 kHz (whistles
    reach ~27 kHz, clicks 13-59 kHz), so frequency-bbox targets are not viable
    and click detection is degraded (its energy is largely ultrasonic). Time-
    level presence detection of whistles (low contour in band) and species
    presence remain usable. 16 kHz + 32 kHz mirrors are provided (32 kHz
    recommended); ``High Freq (Hz)`` is retained but should not be used as a
    regression target without a ``<= 16000`` filter.

    Splits
    ------
    Single ``all`` split (deployment/recording-level holdouts can be carved
    later).

    Loader behaviour
    ----------------
    Windowed reads via ``window_start_sec`` / ``window_end_sec`` (set by the
    ``window_annotations`` transform, which reads the ``audio_duration``
    column); selection tables are re-clipped to the windowed audio end.
    ``annotation_columns = ["Species"]``.

    References
    ----------
    https://zenodo.org/records/17282717 . License: CC-BY-4.0.
    """

    info = DatasetInfo(
        name="mediterranean_cetaceans",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/mediterranean_cetaceans_all.csv",
        },
        version="0.1.0",
        description=(
            "Western Mediterranean cetacean PAM detection: Raven time+freq boxes "
            "on full HydroMoth recordings, 6 labels (5 odontocetes + Delphinidae) "
            "with clicks/whistles signalType, split per ~1-min file. WABAD-shaped."
        ),
        sources=["https://zenodo.org/records/17282717"],
        license="CC-BY-4.0",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_fp"
    _mixup_group = "marine_mammal"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """Initialise the MediterraneanCetaceans dataset.

        Parameters
        ----------
        split : str
            Split to load (key in :attr:`info.split_paths`).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original -> new output keys (filters columns).
        sample_rate : int | None
            Target sample rate. 16 kHz / 32 kHz load the pre-resampled mirror
            directly; other rates resample on the fly. ``None`` returns source.
        data_root : str | AnyPathT | None
            Root prepended to each row's relative audio path. Defaults to the
            manifest's parent directory on GCS.
        backend : BackendType
            ``"pandas"`` or ``"polars"``.
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.annotation_columns = ["Species"]
        self.unknown_label = "Unknown"
        self.sample_rate = sample_rate

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
        """Return pre-resampled sample rates whose path columns exist."""
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
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Load audio + parsed selection table for one recording.

        When ``window_start_sec`` / ``window_end_sec`` are present (set by the
        ``window_annotations`` transform), only that segment is read.

        Returns
        -------
        dict[str, Any]
            The row with ``audio``, ``sample_rate`` and a parsed
            ``selection_table`` DataFrame.
        """
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if path_column in row and row[path_column] not in (None, ""):
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
        row["mixup_group"] = self._mixup_group

        # Selection table parsed with pandas (polars' rayon pool deadlocks
        # after fork() in DataLoader workers).
        raw_st = row.get("selection_table")
        if raw_st is None or raw_st == "":
            st = pd.DataFrame(columns=_ST_COLUMNS)
        elif isinstance(raw_st, str):
            st = pd.read_csv(StringIO(raw_st), sep="\t", keep_default_na=False, na_values=[""])
        elif isinstance(raw_st, pd.DataFrame):
            st = raw_st
        else:
            st = pd.DataFrame(columns=_ST_COLUMNS)

        audio_dur = len(audio) / float(sr)
        if "Begin Time (s)" in st.columns:
            st = st[st["Begin Time (s)"] < audio_dur].reset_index(drop=True)
        row["selection_table"] = st

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["MediterraneanCetaceans", dict[str, Any]]:
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

    def get_available_labels(self, anno_column: str = "Species") -> list[str]:
        """Return the species label vocabulary present in the manifest.

        Returns
        -------
        list[str]
            Sorted unique ``Species`` values across all selection tables.
        """
        labels: set[str] = set()
        for row in self._data:
            raw = row.get("selection_table")
            if not raw:
                continue
            st = pd.read_csv(StringIO(raw), sep="\t", keep_default_na=False, na_values=[""])
            if "Species" in st.columns:
                labels.update(str(s) for s in st["Species"].dropna().unique())
        return sorted(labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{base}\n"
            f"Recordings: {n}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
