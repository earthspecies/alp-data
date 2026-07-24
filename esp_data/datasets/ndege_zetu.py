"""Ndege Zetu — Mt Kenya bird soundscape (weak clip-level multi-label)."""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/ndege-zetu/v0.1.0"
LABELS_PATH = f"{_GCS_ROOT}/ndege_zetu_labels.csv"
_ST_COLUMNS = [
    "Selection",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
    "Presence",
]


@register_dataset
class NdegeZetu(Dataset):
    """Ndege Zetu — Mt Kenya ARU bird soundscape, weak multi-label.

    Description
    -----------
    wa Maina / DeKUT-DSAIL (2025) "Ndege Zetu" (Dryad DOI 10.5061/dryad.d51c5b0c7,
    CC0-1.0; Phil Trans R Soc B 2024.0057). ~1-minute autonomous-recording-unit
    soundscapes from two Mt Kenya sites (Dedan Kimathi University Wildlife
    Conservancy; Mt Kenya National Park), annotated by expert ornithologists with
    **weak, clip-level multi-label** species presence — foreground and background
    species per recording, with **no time localization**. 3,893 recordings, 100
    species (common names mapped to scientific via the dataset's Kenya species
    list); ~30% carry at least one species, the rest are negatives.

    The weak label is exposed both as ``foreground_species`` /
    ``background_species`` columns and as a full-clip ``selection_table`` (one row
    per species spanning the whole recording, with a ``Presence`` column). This
    lets it drive weak species tasks and, with the caveat that windows inherit
    the whole-clip label (no localization), the ``window_annotations`` path.

    Audio
    -----
    16 kHz mono MP3 originals (~60 s; **Nyquist 8 kHz**, so high-pitched species
    are frequency-capped). Pre-resampled 16 kHz (native) and 32 kHz (upsampled)
    WAV mirrors are provided; an ``audio_duration_sec`` column gives the exact
    per-recording duration.

    Splits
    ------
    ``train`` / ``val`` / ``test`` / ``all`` — deterministic per-recording
    ~80/10/10 hashed split, proportional across the two sites.

    References
    ----------
    https://doi.org/10.5061/dryad.d51c5b0c7 ; https://doi.org/10.1098/rstb.2024.0057
    License: CC0-1.0.
    """

    info = DatasetInfo(
        name="ndege_zetu",
        owner="david",
        split_paths={
            "train": f"{_GCS_ROOT}/ndege_zetu_train.csv",
            "val": f"{_GCS_ROOT}/ndege_zetu_val.csv",
            "test": f"{_GCS_ROOT}/ndege_zetu_test.csv",
            "all": f"{_GCS_ROOT}/ndege_zetu_all.csv",
        },
        version="0.1.0",
        description=(
            "Ndege Zetu Mt Kenya ARU bird soundscape: weak clip-level multi-label "
            "(foreground/background species) over 3,893 ~1-min 16 kHz recordings, "
            "100 species, 2 sites."
        ),
        sources=["https://doi.org/10.5061/dryad.d51c5b0c7"],
        license="CC0-1.0",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_fp"
    _mixup_group = "bird"

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """Initialise the NdegeZetu dataset.

        Parameters
        ----------
        split : str
            Split to load (key in :attr:`info.split_paths`).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original -> new output keys (filters columns).
        sample_rate : int | None
            Target sample rate. 16 kHz / 32 kHz load the pre-resampled mirror
            directly; other rates resample on the fly. ``None`` returns the
            native 16 kHz. (Source is 16 kHz, so 32 kHz is upsampled.)
        data_root : str | AnyPathT | None
            Root prepended to relative audio paths. Defaults to the manifest's
            parent directory on GCS.
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
        self._available_labels: list[str] | None = None

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
        """Load audio + parsed weak selection table for one recording.

        When ``window_start_sec`` / ``window_end_sec`` are present (set by the
        ``window_annotations`` transform), only that segment is read; the
        (full-clip) weak labels are inherited by the window.

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["NdegeZetu", dict[str, Any]]:
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
        """Return the species vocabulary (100 scientific names).

        Returns
        -------
        list[str]
            Sorted scientific names from ``ndege_zetu_labels.csv``.
        """
        if self._available_labels is None:
            self._available_labels = pd.read_csv(LABELS_PATH)["Species"].astype(str).tolist()
        return self._available_labels

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
