"""Baringo (western Kenya) soundscapes — centerpoint bird-call detection."""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/baringo-soundscapes/v0.1.0"
LABELS_PATH = f"{_GCS_ROOT}/baringo_soundscapes_labels.csv"
_ST_COLUMNS = [
    "Selection",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
    "eBird_Code",
]


@register_dataset
class BaringoSoundscapes(Dataset):
    """Western-Kenya (Baringo) soundscapes — centerpoint bird-call detection.

    Description
    -----------
    Kahl, Reers, Cherutich, Jacot, Klinck (2024) "A collection of annotated
    soundscape recordings from western Kenya" (Zenodo DOI 10.5281/zenodo.10943500,
    CC-BY-4.0). 35 ~1-hour soundscape recordings (32 h total) from Baringo County,
    Kenya, annotated by expert ornithologists with **centerpoint** labels — each
    of 10,294 bird calls is marked by its *center time* (zero-width event), for
    176 species (eBird codes mapped to scientific names). Audio is 32 kHz FLAC
    (Nyquist 16 kHz — a native fit for the stack).

    WABAD-shaped: one row per recording plus a ``selection_table`` whose events
    have ``Begin Time (s) == End Time (s) == center`` (no onset/offset). This
    drives ``window_annotations`` (a species is "in" a window when its center
    falls inside); center->box expansion, if needed for timestamp tasks, is left
    to the consumer.

    Provenance note: partly used as 2023 BirdCLEF test data and broadly
    overlapping (genus level) with the BEANS-Zero held-out taxa, so it is exposed
    as a single ``all`` split intended as a held-out soundscape SED eval rather
    than training data.

    Audio
    -----
    Served as clean 16 kHz + 32 kHz WAV mirrors (``16khz_path`` / ``32khz_path``;
    ``audio_fp`` points to the 32 kHz WAV). The source 32 kHz FLACs trip a
    libsndfile decoder bug, so they are re-encoded at build time rather than
    served directly. An ``audio_duration`` column gives the exact
    per-recording duration.

    References
    ----------
    https://doi.org/10.5281/zenodo.10943500 . License: CC-BY-4.0.
    """

    info = DatasetInfo(
        name="baringo_soundscapes",
        owner="david",
        split_paths={"all": f"{_GCS_ROOT}/baringo_soundscapes_all.csv"},
        version="0.1.0",
        description=(
            "Western Kenya (Baringo) soundscapes: 35 ~1-hour 32 kHz recordings, "
            "10,294 centerpoint bird-call labels, 176 species; single 'all' split "
            "(intended as a held-out soundscape SED eval)."
        ),
        sources=["https://doi.org/10.5281/zenodo.10943500"],
        license="CC-BY-4.0",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_fp"
    _mixup_group = "bird"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """Initialise the BaringoSoundscapes dataset.

        Parameters
        ----------
        split : str
            Split to load (only ``"all"``).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original -> new output keys (filters columns).
        sample_rate : int | None
            Target sample rate. 16 kHz / 32 kHz load the pre-resampled WAV mirror
            directly; other rates resample on the fly.
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
        """Load audio + parsed centerpoint selection table for one recording.

        Honours ``window_start_sec`` / ``window_end_sec`` (from
        ``window_annotations``) for windowed reads.

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
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["BaringoSoundscapes", dict[str, Any]]:
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
        """Return the species vocabulary (176 scientific names).

        Returns
        -------
        list[str]
            Sorted scientific names from ``baringo_soundscapes_labels.csv``.
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
