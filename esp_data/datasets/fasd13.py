"""FASD13: Fewshot Animal Sound Detection 13 benchmark."""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_BASE = "gs://esp-data-ingestion/fasd13/v0.1.0"

# Sub-dataset codes, in the order of the FASD13 summary table.
SUBDATASET_CODES = (
    "AS",
    "CC",
    "GS",
    "HA",
    "HG",
    "HW",
    "JS",
    "KD",
    "MS",
    "PM",
    "RG",
    "RS",
    "RW",
)


@register_dataset
class FASD13(Dataset):
    """FASD13 few-shot bioacoustic sound event detection benchmark.

    Description
    -----------
    Each entry is a full recording plus a selection table of the onsets and
    offsets of a single predetermined target category. 13 sub-datasets, 109
    recordings, ~143 h, spanning anurans, birds, primates, whales, insects,
    spiders and gunshots, from terrestrial and underwater passive acoustic
    monitoring, on-body recorders, substrate-borne recording and the lab.

    This is the *evaluation* benchmark from the DRASDIC paper. It is unrelated
    in content to :class:`esp_data.datasets.drasdic.DRASDIC`, which holds the
    synthetic *training* corpus from the same work.

    N-shot protocol
    ---------------
    Events are ordered by end time and ``UNK`` events are excluded from shot
    selection. An N-shot system is given the audio up through the Nth event
    (``shot_end_times``) and must detect events in the remainder. Each row
    carries ``shot_end_times`` so a downstream transform can split the
    recording into its support and query regions for any N up to 5.

    Selection table
    ---------------
    Raven/WABAD-shaped TSV with ``Begin Time (s)``, ``End Time (s)``, ``Q``
    (``POS``/``UNK``/``NEG``), ``Label`` and ``event_index``. The positive class
    is nameless in FASD13 -- for few-shot detection it is defined by the support
    examples rather than by a name -- so ``Label`` is the constant ``"target"``
    and the ``detection_target`` column records what kind of category it is
    (Species, Call Type, Sound Type, Production Mechanism, Species+Life Stage).
    Per the official protocol, ``UNK`` events are neither positives nor
    negatives and must be masked out of scoring.

    Multi-audio mode
    ----------------
    When a row carries ``support_16khz_paths`` / ``support_32khz_paths`` (set by
    the few-shot episode transform), the dataset returns an ``audios`` list --
    the support clips followed by the query window -- instead of a single
    ``audio`` array, matching the multi-audio convention used by
    :class:`esp_data.datasets.beans_pro_multi_audio.BeansProMultiAudio`.

    Pre-resampled audio
    -------------------
    16 kHz and 32 kHz mono mirrors are available. When ``sample_rate`` matches
    one of these, the pre-resampled files are read directly; any other rate is
    resampled on the fly with librosa ``kaiser_best``. Native rates vary by
    sub-dataset (16 kHz to 96 kHz) and some sources are stereo, so the mirrors
    are the only mono-normalised view.

    Licensing
    ---------
    Per sub-dataset, not uniform: CC-BY-1.0, CC-BY-4.0, CC-BY-SA-4.0,
    CC-BY-NC-4.0, CC-BY-NC-SA-4.0, public domain, and one custom
    citation-required entry (MS). Carried per row in ``license`` / ``citation``;
    see also ``LICENSE.txt`` alongside the manifests on GCS.

    References
    ----------
    https://zenodo.org/records/15843741
    https://arxiv.org/abs/2503.00296
    """

    info = DatasetInfo(
        name="fasd13",
        owner="david",
        split_paths={
            "all": f"{_GCS_BASE}/fasd13_all.csv",
            "KD_slow6": f"{_GCS_BASE}/fasd13_KD_slow6.csv",
            "KD_slow6_native": f"{_GCS_BASE}/fasd13_KD_slow6_native.csv",
            "KD_slow3_native": f"{_GCS_BASE}/fasd13_KD_slow3_native.csv",
            "KD_slow3": f"{_GCS_BASE}/fasd13_KD_slow3.csv",
            **{code: f"{_GCS_BASE}/fasd13_{code}.csv" for code in SUBDATASET_CODES},
        },
        version="0.1.0",
        description=(
            "FASD13 few-shot bioacoustic sound event detection benchmark: 13 sub-datasets, "
            "109 recordings, ~143 h, each with onset/offset annotations of one predetermined "
            "target category. Evaluation-only benchmark from Hoffman et al. 2025."
        ),
        sources=["https://zenodo.org/records/15843741", "https://arxiv.org/abs/2503.00296"],
        license="Mixed per sub-dataset; see the per-row license/citation columns.",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _support_paths_columns: dict[int, str] = {
        16000: "support_16khz_paths",
        32000: "support_32khz_paths",
    }
    _originals_path_column = "audio_fp"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        split : str
            ``"all"`` or one of the 13 sub-dataset codes (``AS``, ``CC``,
            ``GS``, ``HA``, ``HG``, ``HW``, ``JS``, ``KD``, ``MS``, ``PM``,
            ``RG``, ``RS``, ``RW``).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original -> new output keys (filters columns too).
        sample_rate : int | None
            Target sample rate. 16000 and 32000 read pre-resampled mirrors.
        data_root : str | AnyPathT | None
            Root to resolve relative audio paths against. Defaults to the
            manifest's parent directory on GCS.
        backend : BackendType
            Tabular backend ("polars" or "pandas").
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.annotation_columns = ["Label"]
        self.sample_rate = sample_rate

        self._load()

        if data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent
        else:
            self.data_root = anypath(data_root)

    def _load(self) -> None:
        """Load the manifest CSV for the configured split.

        Raises
        ------
        LookupError
            If ``split`` is not a valid split name.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths)}"
            )
        self._data = self._backend_class.from_csv(
            self.info.split_paths[self.split],
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

    @property
    def columns(self) -> list[str]:
        """Return column names of the loaded data."""
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        """Return all valid split names."""
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Return pre-resampled sample rates whose path columns exist in the data."""
        return [sr for sr, col in self._sample_rate_paths.items() if col in self._data.columns]

    def __len__(self) -> int:
        """Return the number of recordings in the current split.

        Returns
        -------
        int

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        NotImplementedError
            In streaming mode.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _resolve_audio_path(self, row: dict[str, Any]) -> tuple[AnyPathT, bool]:
        """Pick the pre-resampled mirror for the target rate, else the original.

        Returns
        -------
        tuple[AnyPathT, bool]
            ``(path, use_presampled)``.
        """
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            column = self._sample_rate_paths[self.sample_rate]
            if row.get(column):
                return anypath(self.data_root) / row[column], True
        return anypath(self.data_root) / row[self._originals_path_column], False

    def _read(
        self,
        path: AnyPathT,
        use_presampled: bool,
        start: float | None = None,
        end: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Read (optionally a slice of) one audio file as mono float32.

        Returns
        -------
        tuple[np.ndarray, int]
            ``(audio, sample_rate)``.
        """
        if start is not None and end is not None:
            audio, sr = read_audio(path, start_time=float(start), end_time=float(end))
        else:
            audio, sr = read_audio(path)
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
        return audio, sr

    @staticmethod
    def _as_path_list(raw: str | list[str] | None) -> list[str]:
        """Coerce a support-path cell into a list of paths.

        Accepts a list (the usual polars representation), a ``|``-joined string
        (for manifests that flatten it), or ``None``.

        Returns
        -------
        list[str]
        """
        if raw is None:
            return []
        if isinstance(raw, str):
            return [p for p in raw.split("|") if p]
        return [str(p) for p in raw if p]

    def _support_paths(self, row: dict[str, Any]) -> tuple[list[str], int | None]:
        """Return the support-clip paths and the rate they are stored at.

        Prefers the mirror matching ``sample_rate`` but falls back to the other
        one, resampling on read. The fallback matters: the chat-task evaluator
        rewrites every dataset's ``sample_rate`` to the model's own, so a row
        built for 32 kHz is routinely loaded by a 16 kHz model. Resolving only
        the exact rate would find nothing, drop the row to single-audio mode,
        and surface much later as an opaque embedding shape mismatch.

        Returns
        -------
        tuple[list[str], int | None]
            ``(paths, source_rate)``; ``([], None)`` in single-audio mode.
        """
        preferred = self.sample_rate if self.sample_rate in self._support_paths_columns else None
        order = ([preferred] if preferred is not None else []) + [
            rate for rate in self._support_paths_columns if rate != preferred
        ]
        for rate in order:
            paths = self._as_path_list(row.get(self._support_paths_columns[rate]))
            if paths:
                return paths, rate
        return [], None

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Load audio and parse the selection table for one row.

        When ``window_start_sec`` / ``window_end_sec`` are set (by the
        ``window_annotations`` transform) only that segment is read from
        disk/GCS. When support-clip paths are present the row is returned in
        multi-audio form: an ``audios`` list of the support clips followed by
        the query window.

        Parameters
        ----------
        row : dict[str, Any]
            A single manifest row.

        Returns
        -------
        dict[str, Any]
            The processed row.
        """
        audio_path, use_presampled = self._resolve_audio_path(row)
        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")
        query, sr = self._read(
            audio_path,
            use_presampled,
            start=window_start if window_end is not None else None,
            end=window_end if window_start is not None else None,
        )

        support_paths, support_rate = self._support_paths(row)
        if support_paths:
            # Only treat the clips as pre-sampled when their mirror actually
            # matches the target rate, otherwise _read must resample them.
            supports_presampled = support_rate == self.sample_rate
            supports = [
                self._read(anypath(self.data_root) / p, supports_presampled)[0]
                for p in support_paths
            ]
            row["audios"] = [*supports, query]
            row["audio_paths"] = [*support_paths, str(audio_path)]
        else:
            row["audio"] = query
        row["sample_rate"] = sr

        raw_st = row.get("selection_table")
        if raw_st is not None:
            if isinstance(raw_st, str):
                st = pd.read_csv(StringIO(raw_st), sep="\t")
            elif isinstance(raw_st, pd.DataFrame):
                st = raw_st
            else:
                st = pd.DataFrame()
            row["selection_table"] = st

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one processed row.

        Returns
        -------
        dict[str, Any]
        """
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over processed rows.

        Yields
        ------
        dict[str, Any]
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["FASD13", dict[str, Any]]:
        """Create a FASD13 instance from a dataset config.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration with ``split``, ``sample_rate``, etc.

        Returns
        -------
        tuple[FASD13, dict[str, Any]]
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

    def get_available_labels(self, anno_column: str | None = "Label") -> list[str]:
        """Return the label vocabulary of the selection tables.

        FASD13 has a single, nameless positive class per recording, so this is
        always ``["target"]``. Use the ``detection_target`` column for the kind
        of category each sub-dataset annotates.

        Parameters
        ----------
        anno_column : str | None
            Annotation column; only ``"Label"`` is meaningful here.

        Returns
        -------
        list[str]
        """
        labels = set()
        for row in self._data:
            st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
            if anno_column in st.columns:
                labels.update(st[anno_column].astype(str).tolist())
        return sorted(labels)

    def __str__(self) -> str:
        """Return a human-readable summary.

        Returns
        -------
        str
        """
        base = f"{self.info.name} (v{self.info.version}), split: {self.split}"
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{base}, {n} recordings\n"
            f"Description: {self.info.description}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
