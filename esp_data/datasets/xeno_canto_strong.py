"""Xeno-canto strongly-annotated subset (event-level human annotations).

The XC project hosts a curated set of soundscape recordings where human
annotators have marked the start/end of individual vocalizations and
labelled each event with the segment-level ``scientific_name``. This
dataset is shaped to match :class:`esp_data.datasets.WABAD` exactly —
one row per audio file, with all events for that file embedded in an
inline ``selection_table`` TSV — so the existing
``window_annotations`` + ``annotation_features`` transform chain works
unchanged.

Built from three GCS-resident CSVs in the xeno-canto v0.1.0 pipeline:
``xc_annotation_segments.csv`` (per-event), ``xc_annotated_extras.csv``
(per-recording GBIF + paths), ``xc_annotated_recordings.csv`` (per-rec
duration). See ``scripts/data_preprocessing_scripts/xeno_canto_strong/
build_xc_strong.py``.
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

_DATA_ROOT = "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw"


@register_dataset
class XenoCantoStrong(Dataset):
    """Xeno-canto event-level strongly-annotated dataset (WABAD-shaped).

    Description
    -----------
    21,300 audio files from xeno-canto's human-annotated subset (mostly
    soundscape recordings; XC's per-file focal ``recording_scientific_name``
    is the catch-all ``Sonus naturalis`` for ~99% of these files, but
    each file carries 1-73 annotated events with true species labels in
    its ``selection_table`` TSV — 68,755 events / 811 unique segment-
    level species in total).

    The schema mirrors :class:`esp_data.datasets.WABAD` so the existing
    ``window_annotations`` + ``annotation_features`` chat-task pipeline
    works without modification: each row is one audio file with
    ``audio_fp``, ``16khz_path``, ``32khz_path``, ``audio_duration``,
    and an inline ``selection_table`` TSV whose columns are
    ``Begin Time (s)``, ``End Time (s)``, ``Low Freq (Hz)``,
    ``High Freq (Hz)``, ``Species``, ``sound_type``, ``sex``,
    ``life_stage``, ``annotator``.

    Columns
    -------
    xc_id : int
        Xeno-canto recording id (numeric, no XC prefix).
    audio_fp : str
        Relative path to the original audio (e.g. ``audio/XC65654.mp3``).
    16khz_path, 32khz_path : str
        Relative paths to pre-resampled mirrors.
    audio_duration : float
        Clip duration in seconds (parsed from XC ``length``).
    selection_table : str
        TSV string holding all event annotations for this file.
    n_events : int
        Number of events embedded in ``selection_table``.
    recording_scientific_name : str
        File-level focal species (mostly ``Sonus naturalis``); the true
        per-event labels live inside ``selection_table.Species``.
    recording_species_common : str
        File-level common name (mostly empty for soundscapes).
    kingdom, phylum, class, order, family, genus : str
        File-level (focal) GBIF taxonomy.
    gbifID, taxonKey, speciesKey : str
        File-level GBIF identifiers.
    latitudeDecimal, longitudeDecimal, country_code, locality, continent :
        Geographic context for the recording.
    eventDate, year, month, day : str
        Recording date.
    recordedBy, rightsHolder : str
        Attribution.
    license, media_license, license_url, media_license_url : str
        Per-recording licensing. Mostly CC-BY-NC-SA-4.0.
    source_dataset : str
        Constant ``"xeno_canto_strong"``.

    Splits
    ------
    - ``all``: every file (21,300 rows). No held-out subset yet — the
      caller is expected to apply leakage filters via the YAML chain
      (e.g. ``drop_where_column_contains`` for BEANS-Zero held-out taxa,
      following the iNaturalist / xeno-canto pattern).

    Available tasks
    ---------------
    Same as WABAD when combined with ``window_annotations`` +
    ``annotation_features``: multilabel species, species count,
    per-species call counts, frequency-range summaries, etc.

    References
    ----------
    Xeno-canto annotated audio collection (see
    ``configs/xc_annotated_audio.yaml`` and ``jobs/run_xc_annotated_audio.sh``
    in the esp-data pipeline). License: CC-BY-NC-SA-4.0 dominates.
    """

    info = DatasetInfo(
        name="xeno_canto_strong",
        owner="david",
        split_paths={
            "all": f"{_DATA_ROOT}/xc_strong_with_selection_table.csv",
        },
        version="0.1.0",
        description=(
            "Xeno-canto event-level human annotations: 21,300 files with "
            "68,755 events spanning 811 species, shaped as a WABAD-style "
            "manifest with inline selection_table TSV per file."
        ),
        sources=(
            "xeno-canto.org annotated recordings (gs://esp-data-ingestion/"
            "xeno-canto/v0.1.0/raw/xc_annotation_segments.csv)"
        ),
        license="CC-BY-NC-SA-4.0 (mostly)",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_fp"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """Initialise the XenoCantoStrong dataset.

        Parameters
        ----------
        split : str
            Split key in :attr:`info.split_paths`. Only ``"all"`` exists.
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        sample_rate : int | None
            Target rate. Pre-resampled 16 kHz / 32 kHz mirrors used when
            available. Defaults to 32 kHz to match the xeno-canto-focused
            entries elsewhere in Stage 2.
        data_root : str | AnyPathT | None
            Root prepended to relative audio paths. Defaults to the GCS
            xeno-canto raw root.
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
        return [
            sr
            for sr, col in self._sample_rate_paths.items()
            if col in (self._data.columns if self._data is not None else [])
        ]

    def _load(self) -> None:
        """Load the split CSV.

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
        """Process a single row.

        Mirrors :meth:`esp_data.datasets.WABAD._process`. If a transform
        has set ``window_start_sec`` / ``window_end_sec`` (e.g.
        ``window_annotations``), only that audio segment is read.

        Parameters
        ----------
        row : dict[str, Any]
            A dictionary representing a single row of the dataset.

        Returns
        -------
        dict[str, Any]
            The processed row with ``audio``, ``sample_rate`` populated.
        """
        use_presampled = False
        audio_path = None
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            col = self._sample_rate_paths[self.sample_rate]
            val = row.get(col)
            if val is not None:
                s = str(val).strip()
                if s and s.lower() != "nan":
                    audio_path = anypath(self.data_root) / s
                    use_presampled = True
        if audio_path is None:
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["XenoCantoStrong", dict[str, Any]]:
        """Create an XenoCantoStrong instance from a configuration.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration.

        Returns
        -------
        tuple[XenoCantoStrong, dict[str, Any]]
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
        """Return all per-event species labels found in selection_tables.

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
            f"Files: {n}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
