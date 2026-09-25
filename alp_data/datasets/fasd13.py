"""FASD13 dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import DATA_HOME, AnyPathT, anypath, audio_stereo_to_mono, read_audio

# The manifests sit directly under v0.1.0/, alongside audio_native/, audio_16k/
# and audio_32k/ -- not under a raw/ subdirectory, which here holds the Zenodo
# provenance CSVs. Audio paths in the manifests are relative to this directory.
_RAW_ROOT = f"{DATA_HOME}/fasd13/v0.1.0"

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
    """FASD13: Fewshot Animal Sound Detection 13.

    Description
    -----------
    The evaluation benchmark introduced by *Synthetic data enables
    context-aware bioacoustic sound event detection* (Hoffman et al., 2025):
    13 sub-datasets, 109 recordings and roughly 143 hours of audio spanning
    anurans, birds, primates, whales, insects, spiders and gunshots, recorded
    through terrestrial and underwater passive acoustic monitoring, on-body
    recorders, substrate-borne recording and in the lab.

    Each entry is a full recording plus a selection table, in the same shape as
    the WABAD dataset. The benchmark is evaluation-only. DRASDIC ("Domain
    Randomization for Animal Sound Detection In-Context") is that paper's
    *model*, trained on synthetic scenes and evaluated on FASD13; FASD13 shares
    no audio with its training data.

    Sub-datasets
    ------------
    code  name            files  h      events  detection target      license
    ----  --------------  -----  -----  ------  --------------------  ------------------------
    AS    AnuraSet        12     0.2    162     Species               CC-BY-1.0
    CC    Carrion Crow    10     10.0   2200    Species+Life Stage    CC-BY-SA-4.0
    GS    Gunshot         7      38.33  85      Production Mechanism  CC-BY-NC-4.0
    HA    Hawaiian Birds  12     1.1    628     Species               CC-BY-4.0
    HG    Hainan Gibbons  9      72.0   483     Species               CC-BY-NC-SA-4.0
    HW    Humpback Whale  10     2.79   1565    Species               public-domain
    JS    Jumping Spider  4      0.23   924     Sound Type            CC-BY-SA-4.0
    KD    Katydid         12     2.0    883     Species               public-domain
    MS    Marmoset        10     1.67   1369    Call Type             custom-citation-required
    PM    Powdermill      4      6.42   2032    Species               public-domain
    RG    Ruffed Grouse   2      1.5    34      Species               public-domain
    RS    Rana Sierrae    7      1.87   552     Species               public-domain
    RW    Right Whale     10     5.0    398     Species               CC-BY-4.0

    `CC` and `JS` are published for the first time in FASD13; the other 11 are
    re-releases. Sources, by first author:

    AS Cañas 2023; CC Hoffman 2025; GS Gottesman 2024; HA Navine 2022; HG Dufourq 2020
    HW Allen 2021; JS Hoffman 2025; KD Madhusudhana 2024; MS Sarkar 2023; PM Chronister 2021
    RG Lapp 2022; RS Lapp 2023; RW Simard 2020

    Full citations are in the row-level `citation` column, and `LICENSE.txt`
    sits alongside the manifests. Note the duration imbalance: `HG` (72 h) and
    `GS` (38 h) are 77 % of the benchmark between them, so cap or stratify per
    sub-dataset before pooling.

    Labels
    ------
    Every recording is annotated for a single, *nameless* target category -- in
    few-shot detection the target is defined by the support examples, not by a
    name -- so there is no species-style label column. The row-level
    `detection_target` records what kind of category a sub-dataset annotates
    (Species, Call Type, Sound Type, Production Mechanism, Species+Life Stage).

    Selection tables carry `Selection`, `Begin Time (s)`, `End Time (s)`, `Q`
    and `event_index`. `Q` is the per-event status and is the only annotation
    column: `POS` for a target event, `UNK` where annotators could not
    determine presence. Per the protocol `UNK` events are neither positives nor
    negatives and **must be masked out of scoring** rather than counted as
    detections; 1,029 of the 12,344 events are `UNK`, concentrated in `MS`
    (32 %), `RS` (19 %) and `CC` (11 %). `event_index` numbers the non-`UNK`
    events in end-time order and is `-1` for `UNK`. There are no frequency
    bounds -- FASD13 is time-only.

    N-shot protocol
    ---------------
    Following Nolasco et al. 2023, events are ordered by end time and `UNK`
    events are excluded from shot selection. An N-shot system is given the
    audio up through the end of the Nth event, and must detect events in the
    remainder, which is the only region scored.

    The boundaries are not stored on the row -- a stored copy goes stale under
    windowed reads, where table times are re-based but a column would not be.
    Derive them from the selection table instead, which is three lines:

        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
        known = st[st["Q"] != "UNK"]
        shot_ends = sorted(known["End Time (s)"])[:n]     # recording-absolute

    `shot_ends[n - 1]` ends the support region for an n-shot episode;
    everything after it is query. Derive this from an **unwindowed** row so the
    times stay recording-absolute, and when splitting a recording into several
    query windows take it once and reuse it -- deriving it per window would
    draw support from query audio.

    Windowed reads
    --------------
    A row carrying `window_start_sec` and `window_end_sec` reads only that
    segment, streamed from cloud storage rather than downloaded whole, and its
    selection table is re-based onto the window: non-overlapping events are
    dropped and the rest shifted and clipped to line up with the returned
    audio. `event_index` is preserved, so a caller can still tell which of the
    recording's events a row refers to.

    This is what makes the benchmark usable without loading whole recordings:
    several sub-datasets ship 8-hour files, so cutting a support clip or
    chunking a query region has to happen at read time.

    Pre-resampled Audio
    -------------------
    Three views of every recording are stored. `audio_fp` is the original as
    Zenodo ships it, at its native rate and channel count; `16khz_path` and
    `32khz_path` are pre-resampled mono mirrors. When `sample_rate` is 16000 or
    32000 the matching mirror is read directly. Any other rate -- including
    `None`, which returns native audio -- reads the original and resamples with
    librosa's `kaiser_best` where needed. 32 kHz is the default.

    Native rates span 8 kHz to 187.5 kHz, and what the mirrors cost differs by
    sub-dataset:

    - `KD` (katydids, 96 kHz native) carries genuine content above the mirrors'
      16 kHz ceiling. Measured against adjacent background, its events show
      more excess energy in 16-32 kHz than in either band below it, so the
      32 kHz mirror discards part of the signal. Read the originals for `KD`.
    - `CC` (46.875 / 187.5 kHz) and `MS` (44.1 kHz) have the headroom but do
      not use it -- their event energy sits below 16 kHz, so the mirrors lose
      nothing measurable.
    - `GS` (8 kHz) and `HG` (9.6 kHz) are *upsampled* into the mirrors. They
      are 77 % of the benchmark by duration, so a 32 kHz read of most of FASD13
      carries nothing above ~4-5 kHz; do not mistake the rate for bandwidth.
    - `AS` and `HG` are stereo at source. Reads are mono-averaged whatever the
      view, as in WABAD; the originals keep their channels.

    Two `HA` recordings are FLAC bitstreams under a `.wav` extension that
    libsndfile cannot decode in full, so their originals are published
    transcoded to PCM; `HA` is 32 kHz mono at source, so no audio is altered.

    Licensing
    ---------
    Licensing is per sub-dataset rather than uniform (see the table above), and
    `MS` requires citing two specific papers. It is carried per row in the
    `license` and `citation` columns.

    References
    ----------
    [Zenodo](https://zenodo.org/records/15843741)
    [arXiv](https://arxiv.org/abs/2503.00296)

    """

    info = DatasetInfo(
        name="fasd13",
        owner="david; benjamin",
        split_paths={
            "all": f"{_RAW_ROOT}/fasd13_all.csv",
            **{code: f"{_RAW_ROOT}/fasd13_{code}.csv" for code in SUBDATASET_CODES},
        },
        version="0.1.0",
        description="FASD13 few-shot bioacoustic sound event detection benchmark: 13 "
        "sub-datasets, 109 recordings and roughly 143 hours of audio, each recording "
        "annotated with the onsets and offsets of a single predetermined target "
        "category. Spans anurans, birds, primates, whales, insects, spiders and "
        "gunshots. Evaluation-only benchmark from Hoffman et al. 2025.",
        sources=["https://zenodo.org/records/15843741", "https://arxiv.org/abs/2503.00296"],
        license="Mixed per sub-dataset; see the per-row license/citation columns.",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
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
        """
        Parameters
        ----------
        split : str
            Split to load (key in info.split_paths): `"all"` or one of the 13
            sub-dataset codes (`AS`, `CC`, `GS`, `HA`, `HG`, `HW`, `JS`, `KD`,
            `MS`, `PM`, `RG`, `RS`, `RW`).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns as well).
        sample_rate : int | None
            If set, audio is resampled to this rate. 16000 and 32000 read the
            pre-resampled mirrors directly.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row's audio path.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.annotation_columns = ["Q"]
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
        """Return pre-resampled sample rates whose path columns exist in the data."""
        return [sr for sr, col in self._sample_rate_paths.items() if col in self._data.columns]

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
        """Return the number of recordings in the dataset.

        Returns
        -------
        int
            Number of recordings in the current split.

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
                "Length is not available in streaming mode.Iterate over the dataset instead."
            )
        return len(self._data)

    @staticmethod
    def _window_selection_table(
        st: pd.DataFrame, window_start: float, duration: float
    ) -> pd.DataFrame:
        """Re-base a recording-absolute selection table onto a windowed read.

        Events that do not overlap the window are dropped; the rest are shifted
        so their times are relative to the start of the returned audio, and
        clipped to it. `event_index` is left untouched so a caller can still
        tell which of the recording's events a row refers to, which is what the
        N-shot protocol needs.

        Parameters
        ----------
        st : pd.DataFrame
            Selection table with recording-absolute `Begin Time (s)` and
            `End Time (s)`.
        window_start : float
            Start of the window, in seconds from the start of the recording.
        duration : float
            Duration of the audio actually returned, in seconds. Used instead
            of the requested window end so that a window running past the end
            of the file is handled correctly.

        Returns
        -------
        pd.DataFrame
            The windowed selection table, with times relative to the window.
        """
        window_end = window_start + duration
        st = st[(st["End Time (s)"] > window_start) & (st["Begin Time (s)"] < window_end)].copy()
        st["Begin Time (s)"] = (st["Begin Time (s)"] - window_start).clip(lower=0.0)
        st["End Time (s)"] = (st["End Time (s)"] - window_start).clip(upper=duration)
        return st

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of the dataset.

        When the row carries `window_start_sec` and `window_end_sec`, only that
        segment is read and the selection table is re-based onto it. See the
        class docstring.

        Parameters
        ----------
        row : dict[str, Any]
            A dictionary representing a single row of the dataset.

        Returns
        -------
        dict[str, Any]
            The processed row.
        """
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if path_column in row and row[path_column] is not None and row[path_column] != "":
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

        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
        audio_duration = len(audio) / float(sr)
        if window_start is not None and window_end is not None:
            st = self._window_selection_table(st, float(window_start), audio_duration)
        else:
            # Same guard WABAD and DCLDE2026 apply: drop events that begin past
            # the end of the audio. A no-op on the shipped manifests -- no event
            # in the benchmark overhangs its recording -- but it keeps the
            # unwindowed path identical to its siblings.
            st = st[st["Begin Time (s)"] < audio_duration].copy()

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
            A dictionary containing the audio, sample rate, selection table and
            the recording's metadata columns.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["FASD13", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters.

        Returns
        -------
        tuple[Dataset, dict[str, Any]]
            A tuple containing the dataset instance and metadata.
            If the dataset_config contains transformations, they will be applied
            and the metadata will be returned as dict, otherwise an empty dict.
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

    def get_available_labels(self, anno_column: str | None = "Q") -> list[str]:
        """Return the event-status vocabulary of the selection tables.

        FASD13's target class is nameless, so there is no species-style label
        column; `Q` is the annotation column. This returns the statuses present
        in the split, normally `["POS", "UNK"]`, or `["POS"]` for the nine
        sub-datasets with no `UNK` events. Use the row-level `detection_target`
        for the kind of category a sub-dataset annotates.

        Parameters
        ----------
        anno_column : str | None
            Selection-table column to read; only `"Q"` is meaningful here.

        Returns
        -------
        list[str]
            A sorted list of the values present in anno_column.
        """
        available_labels = set()
        for row in self._data:
            st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
            if anno_column in st.columns:
                available_labels.update(st[anno_column].astype(str).tolist())
        return sorted(available_labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
