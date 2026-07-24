"""CEB: Central-European strongly-labeled bird soundscapes + Xeno-Canto.

Event-level bird annotations from Martin, Reers, … Kahl, Rauch, Scholz, Sick,
Tomforde et al. (Zenodo record 20762099). Each row is one audio file with all of
its events embedded in an inline ``selection_table`` TSV, matching
:class:`esp_data.datasets.WABAD` / :class:`esp_data.datasets.XenoCantoStrong` so
the existing ``window_annotations`` + ``annotation_features`` transform chain
works unchanged.

Three subsets, exposed as splits:

- ``train_xenocanto`` — 2,210 focal Xeno-Canto files, 15,495 **strong** events
  (tight time+frequency bounding boxes), single species per event, with XC
  provenance columns.
- ``train_soundscape`` — 18,469 soundscape files, 62,298 **weak** events
  (center ±2.5 s = 5 s windows, no frequency bounds, often no voc type); ~8% of
  events are multi-species (exploded to one selection row per species).
- ``test_soundscape`` — 147 soundscape files, 15,064 **strong** events; the
  intended strong evaluation split.

Built by ``scripts/data_preprocessing_scripts/ceb/build_ceb.py`` from the raw
Zenodo CSVs staged at ``gs://esp-data-ingestion/ceb/v0.1.0/raw/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
from typing import Any

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_ROOT = "gs://esp-data-ingestion/ceb/v0.1.0"
_AUDIO_ROOT = f"{_ROOT}/audio"


@register_dataset
class CEB(Dataset):
    """CEB event-level bird dataset (WABAD-shaped, strong + weak subsets).

    Columns
    -------
    filepath : str
        Relative path within the subset archive (e.g. ``TE001/000001_morning.flac``).
    audio_fp : str
        Path relative to :attr:`data_root` (``<subset>/<filepath>``).
    subset : str
        One of ``train_xenocanto`` / ``train_soundscape`` / ``test_soundscape``.
    label_quality : str
        ``strong`` (time+freq boxes) or ``weak`` (5 s center windows).
    n_events : int
        Number of events embedded in ``selection_table``.
    selection_table : str
        TSV string with columns ``Begin Time (s)``, ``End Time (s)``,
        ``Low Freq (Hz)``, ``High Freq (Hz)``, ``Species`` (GBIF scientific
        name), ``common_name``, ``ebird_code``, ``sound_type`` (vocalization
        type), ``sex``.
    dataset_name, lat, long, license : str / float
        Per-file metadata (lat/long rounded to whole degrees; empty for
        ``train_xenocanto``).
    xc_id, xc_url, xc_recordist, xc_original_scientific_name,
    xc_original_common_name : str
        Xeno-Canto provenance (``train_xenocanto`` only).
    source_dataset : str
        Constant ``"ceb"``.

    Splits
    ------
    ``train_xenocanto``, ``train_soundscape``, ``test_soundscape``.

    References
    ----------
    Zenodo 10.5281/zenodo.20762099. Soundscapes + metadata CC-BY-4.0;
    ``train_xenocanto`` audio keeps its original per-row Xeno-Canto license.
    """

    info = DatasetInfo(
        name="ceb",
        owner="david",
        split_paths={
            "train_xenocanto": f"{_ROOT}/ceb_train_xenocanto_with_selection_table.csv",
            "train_soundscape": f"{_ROOT}/ceb_train_soundscape_with_selection_table.csv",
            "test_soundscape": f"{_ROOT}/ceb_test_soundscape_with_selection_table.csv",
        },
        version="0.1.0",
        description=(
            "CEB: Central-European strongly-labeled bird soundscapes and "
            "Xeno-Canto recordings with bounding boxes and vocalization types "
            "(92,857 events / 20,826 files / 256 eBird species / 21 voc types), "
            "shaped as a WABAD-style manifest with an inline selection_table per file."
        ),
        sources=(
            "Zenodo 20762099 (Martin, Reers, Kahl, Rauch, Scholz, Sick, Tomforde "
            "et al.); staged at gs://esp-data-ingestion/ceb/v0.1.0/raw/"
        ),
        license="CC-BY-4.0 (soundscapes); Xeno-Canto originals for train_xenocanto",
    )

    # No pre-resampled mirrors — audio is read from FLAC and resampled on the fly.
    _sample_rate_paths: dict[int, str] = {}
    _originals_path_column = "audio_fp"

    def __init__(
        self,
        split: str = "test_soundscape",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """Initialise the CEB dataset.

        Parameters
        ----------
        split : str
            Split key in :attr:`info.split_paths`.
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        sample_rate : int | None
            Target sample rate; audio is resampled on load. Defaults to 32 kHz.
        data_root : str | AnyPathT | None
            Root prepended to ``audio_fp``. Defaults to the GCS CEB audio root.
        backend : BackendType
            ``"polars"`` or ``"pandas"``.
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        self.annotation_columns = ["Species"]
        self._load()
        self.data_root = anypath(data_root) if data_root is not None else anypath(_AUDIO_ROOT)

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """No pre-resampled mirrors; loading resamples on the fly."""
        return []

    def _load(self) -> None:
        """Load the split manifest CSV.

        Raises
        ------
        LookupError
            If the split is not valid.
        """
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
        """Return the number of files in the split.

        Returns
        -------
        int
            Number of files in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Read audio (optionally windowed) and parse the selection table.

        If a transform has set ``window_start_sec`` / ``window_end_sec`` (e.g.
        ``window_annotations``), only that segment is read.

        Parameters
        ----------
        row : dict[str, Any]
            A single manifest row.

        Returns
        -------
        dict[str, Any]
            The row with ``audio``, ``sample_rate`` and a parsed
            ``selection_table`` DataFrame.
        """
        audio_path = anypath(self.data_root) / str(row[self._originals_path_column])

        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")
        if window_start is not None and window_end is not None:
            audio, sr = read_audio(
                audio_path, start_time=float(window_start), end_time=float(window_end)
            )
        else:
            audio, sr = read_audio(audio_path)

        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
        if self.sample_rate is not None and sr != self.sample_rate:
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
                st = pd.read_csv(StringIO(raw_st), sep="\t", keep_default_na=False)
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
        """Return a single processed row.

        Returns
        -------
        dict[str, Any]
            The processed row (audio + selection_table + metadata).
        """
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over processed rows.

        Yields
        ------
        dict[str, Any]
            Each processed row.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["CEB", dict[str, Any]]:
        """Create a CEB instance from a configuration.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration.

        Returns
        -------
        tuple[CEB, dict[str, Any]]
            The dataset and any transformation metadata.
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

    def get_available_labels(self, annotation_column: str = "Species") -> list[str]:
        """Return all per-event labels found in the selection tables.

        Parameters
        ----------
        annotation_column : str
            Column inside each row's selection_table TSV. Defaults to ``"Species"``.

        Returns
        -------
        list[str]
            Sorted unique label values across the dataset.

        Raises
        ------
        ValueError
            If ``annotation_column`` is not present in the selection_table.
        """
        labels: set[str] = set()
        for row in self._data:
            raw = row.get("selection_table")
            if not raw:
                continue
            st = pd.read_csv(StringIO(raw), sep="\t", keep_default_na=False)
            if annotation_column not in st.columns:
                raise ValueError(
                    f"Column '{annotation_column}' not in selection_table; "
                    f"available: {list(st.columns)}"
                )
            labels.update(st[annotation_column].astype(str).tolist())
        return sorted(labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{base}\n"
            f"Files: {n} (split={self.split})\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
