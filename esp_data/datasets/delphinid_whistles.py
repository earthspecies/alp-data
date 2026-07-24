"""Delphinid whistle detection dataset — bottlenose dolphin whistles (bbox)."""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/delphinid-whistles/v0.1.0"
SPECIES = "Tursiops truncatus"
_ST_COLUMNS = [
    "Selection",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
]


@register_dataset
class DelphinidWhistles(Dataset):
    """Delphinid whistle detection — bottlenose dolphin (bbox selection tables).

    Description
    -----------
    Ferguson et al. (2025) "Bounding-box detection data for delphinid whistles"
    (Dryad DOI 10.5061/dryad.z34tmpgq6, CC0-1.0). Time+frequency bounding-box
    annotations of bottlenose dolphin (*Tursiops truncatus*) whistles across
    four acoustic environments — two aquarium (IMMS Gulfport; Oceanogràfic
    Valencia) and two open ocean (DCLDE 2011; NOAA SWFSC towed array). WABAD-
    shaped: one row per audio recording plus a ``selection_table`` with
    ``Selection, Begin Time (s), End Time (s), Low Freq (Hz), High Freq (Hz),
    Species`` (Species is always ``Tursiops truncatus``). 8,576 whistle boxes
    over 96 recordings; open-ocean recordings with no whistles are retained as
    pure-negative examples.

    Audio is 48 kHz mono; whistle boxes were annotated on a 0–24 kHz view.
    Pre-resampled 16 kHz and 32 kHz mirrors are provided (32 kHz recommended).
    At the 32 kHz / 16 kHz-Nyquist stack, ~99% of whistles have their lower
    contour in band and ~47% of boxes are fully below 16 kHz (aquarium ~62%,
    open ocean ~30%); ~52% straddle the ceiling (top clipped). Time-only
    detection is robust everywhere; frequency-bbox tasks should filter to
    ``High Freq (Hz) <= 16000`` to avoid ceiling-pinned targets.

    Splits
    ------
    ``train`` / ``val`` / ``test`` / ``all``. ``test`` is the dataset's native
    held-out test set. ``val`` is a seeded 15% recording-level holdout carved
    from the open-ocean training files (aquarium has too few merged recordings
    to split, so it stays in ``train``).

    Loader behaviour
    ----------------
    Windowed reads via ``window_start_sec`` / ``window_end_sec`` (set by the
    ``window_annotations`` transform); selection tables are re-clipped to the
    windowed audio end. ``annotation_columns = ["Species"]``.

    References
    ----------
    https://doi.org/10.5061/dryad.z34tmpgq6 ; primary article JASA 157(6):4613
    (https://doi.org/10.1121/10.0036942). License: CC0-1.0.
    """

    info = DatasetInfo(
        name="delphinid_whistles",
        owner="david",
        split_paths={
            "train": f"{_GCS_ROOT}/delphinid_whistles_train.csv",
            "val": f"{_GCS_ROOT}/delphinid_whistles_val.csv",
            "test": f"{_GCS_ROOT}/delphinid_whistles_test.csv",
            "all": f"{_GCS_ROOT}/delphinid_whistles_all.csv",
        },
        version="0.1.0",
        description=(
            "Delphinid (Tursiops truncatus) whistle detection with time+frequency "
            "bounding boxes across 2 aquarium + 2 open-ocean sites; 8,576 boxes / "
            "96 recordings, 48 kHz, WABAD-shaped selection tables."
        ),
        sources=["https://doi.org/10.5061/dryad.z34tmpgq6"],
        license="CC0-1.0",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_fp"
    _mixup_group = "marine_mammal"

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """Initialise the DelphinidWhistles dataset.

        Parameters
        ----------
        split : str
            Split to load (key in :attr:`info.split_paths`).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original -> new output keys (filters columns).
        sample_rate : int | None
            Target sample rate. 16 kHz / 32 kHz load the pre-resampled mirror
            directly; other rates resample on the fly. ``None`` returns 48 kHz.
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
    ) -> tuple["DelphinidWhistles", dict[str, Any]]:
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
        """Return the (single-species) label vocabulary.

        Returns
        -------
        list[str]
            ``["Tursiops truncatus"]``.
        """
        return [SPECIES]

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
