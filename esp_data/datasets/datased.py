"""DataSED dataset — Sound Event Detection of environmental noise."""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/datased/v0.1.0"
LABELS_PATH = f"{_GCS_ROOT}/datased_labels.csv"


@register_dataset
class DataSED(Dataset):
    """DataSED — strongly-labelled environmental-noise soundscapes.

    Description
    -----------
    DataSED (Fredianelli et al. 2025) is an open collection of 717
    non-synthesised WAV recordings (44.1 kHz, mono, ~17 h total) gathered
    from sound-level measurements and online repositories across urban to
    rural environments in Italy. Every recording carries strong
    (time-localised) sound-event annotations in the esp-data WABAD-shaped
    format: one row per recording plus a ``selection_table`` with begin /
    end times and an event ``Label``.

    Two annotation schemes are exposed as separate splits:

    - **poly** (headline): overlapping multi-class events representing
      realistic conditions — 4,034 events / 703 recordings / 21 classes
      (excludes ``Wind turbine``).
    - **mono**: non-overlapping single-class-at-a-time annotation —
      4,309 events / 717 recordings / 22 classes.

    Recordings are **not exhaustively labelled** — spans with no marked
    event are *not* guaranteed to be silent / negative. Treat only
    annotated spans as positives.

    Pre-resampled Audio
    -------------------
    16 kHz and 32 kHz mirrors are pre-computed. When ``sample_rate``
    matches one of these, the mirror is loaded directly; otherwise audio
    is resampled on the fly with librosa ``kaiser_best``. The 44.1 kHz
    broadband source makes 32 kHz the recommended default.

    Splits
    ------
    ``poly_all`` / ``poly_train`` / ``poly_val`` and
    ``mono_all`` / ``mono_train`` / ``mono_val``. The train/val split is
    recording-level (seeded 90/10) and shared across schemes, so a
    recording never crosses between train and val.

    References
    ----------
    https://zenodo.org/records/15346092 (DOI 10.5281/zenodo.15346092)

    License: CC-BY-NC-SA-4.0 (non-commercial, share-alike).
    """

    info = DatasetInfo(
        name="datased",
        owner="david",
        split_paths={
            "poly_all": f"{_GCS_ROOT}/datased_poly_all.csv",
            "poly_train": f"{_GCS_ROOT}/datased_poly_train.csv",
            "poly_val": f"{_GCS_ROOT}/datased_poly_val.csv",
            "mono_all": f"{_GCS_ROOT}/datased_mono_all.csv",
            "mono_train": f"{_GCS_ROOT}/datased_mono_train.csv",
            "mono_val": f"{_GCS_ROOT}/datased_mono_val.csv",
        },
        version="0.1.0",
        description=(
            "DataSED: strongly-labelled environmental-noise soundscapes "
            "(717 recordings, 44.1 kHz, ~17 h) for sound event detection; "
            "polyphonic (21 classes) and monophonic (22 classes) schemes."
        ),
        sources=["https://zenodo.org/records/15346092"],
        license="CC-BY-NC-SA-4.0",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_fp"

    def __init__(
        self,
        split: str = "poly_train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """Initialise the DataSED dataset.

        Parameters
        ----------
        split : str
            Split to load (key in :attr:`info.split_paths`).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original -> new output keys (filters columns).
        sample_rate : int | None
            Target sample rate. 16 kHz / 32 kHz load the pre-resampled
            mirror directly; other rates resample on the fly. ``None``
            returns the file's native 44.1 kHz.
        data_root : str | AnyPathT | None
            Root prepended to each row's relative audio path. Defaults to
            the manifest's parent directory on GCS.
        backend : BackendType
            ``"pandas"`` or ``"polars"``.
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.annotation_columns = ["Label"]
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
        """Load + return audio and parsed selection table for one recording.

        When ``window_start_sec`` / ``window_end_sec`` are present (set by
        the ``window_annotations`` transform), only that segment is read.

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

        # Selection table parsed with pandas (polars' rayon pool deadlocks
        # after fork() in DataLoader workers).
        raw_st = row.get("selection_table")
        if raw_st is None or raw_st == "":
            st = pd.DataFrame(columns=["Selection", "Begin Time (s)", "End Time (s)", "Label"])
        elif isinstance(raw_st, str):
            st = pd.read_csv(StringIO(raw_st), sep="\t", keep_default_na=False, na_values=[""])
        elif isinstance(raw_st, pd.DataFrame):
            st = raw_st
        else:
            st = pd.DataFrame(columns=["Selection", "Begin Time (s)", "End Time (s)", "Label"])

        # Drop events beginning at/after the (windowed) audio end.
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["DataSED", dict[str, Any]]:
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

    def get_available_labels(self, anno_column: str = "Label") -> list[str]:
        """Return the DataSED event-class vocabulary (22 classes).

        Returns
        -------
        list[str]
            Sorted list of event labels from ``datased_labels.csv``.
        """
        if self._available_labels is None:
            self._available_labels = pd.read_csv(LABELS_PATH)["Label"].astype(str).tolist()
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
