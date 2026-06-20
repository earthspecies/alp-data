"""PIFSC PIPAN — NOAA Pacific Islands Passive Acoustic Network annotations.

Phase 1 ingests the Allen et al. (2021) annotations from the NOAA NCEI
public archive (DOI: 10.25921/Z787-9Y54). Each row is one annotation
event with strong (tight time bounds) or weak (subchunk-level) labels
for humpback song (``Mn``) plus negative / non-target classes
(``Background``, ``Other``, ``Vessel``, ``Fish``, ``Device``).

The underlying audio is decimated to 10 kHz from HARP hydrophone
recordings; XWAV foreign-metadata in each FLAC stores the deployment-
level subchunk table (75-second sub-chunks separated by duty-cycle
gaps). The build script (``scripts/data_preprocessing_scripts/
pifsc_pipan/build_pifsc_pipan.py``) flattens annotations to events and
persists exact in-file offsets so consumers never need to read XWAV at
runtime.

Phase 2 (deferred): Allen et al. (2024) Bryde's whale biotwang
annotations + the additional NOAA Perch 2.0 multi-species labels
(orca, common minke, sei, blue, fin, anthropogenic, unknown).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/pifsc-pipan/v0.1.0"

# All annotation labels from the Allen 2021 vocabulary.
LABELS = ["Mn", "Background", "Other", "Vessel", "Fish", "Device"]

# Deployments referenced by the source CSV (subset of all PIPAN sites).
DEPLOYMENTS = [
    "crosssm",
    "equator",
    "hawaii",
    "howland",
    "kauai",
    "kingman",
    "laddsm_d",
    "laddsm_s",
    "pagan",
    "palmyra_ns",
    "palmyra_wt",
    "phr_a",
    "phr_b",
    "saipan",
    "tinian",
    "wake",
]


@register_dataset
class PIFSCPipan(Dataset):
    """PIFSC PIPAN — Pacific Islands Passive Acoustic Network annotations.

    Description
    -----------
    Allen et al. (2021) annotations on the NOAA PIFSC PIPAN 10 kHz
    decimated archive: humpback whale song plus negative/noise classes,
    drawn from 16 long-term HARP hydrophone deployments in the Pacific
    (2005–2018, Wake Atoll / Saipan / Hawaiian Islands / Palmyra Atoll /
    Cross Seamount / etc.). Each row is one analyst-marked event with
    strong (tight time bounds) or weak (subchunk-level) labels.

    Audio is the original 10 kHz decimated FLAC at ``audio_path``;
    pre-resampled 16 kHz / 32 kHz mirrors exist for a subset of files
    (``16khz_path`` / ``32khz_path`` are empty when unavailable, in
    which case the loader resamples on the fly via librosa).

    Columns
    -------
    audio_path : str
        Absolute ``gs://`` path to the source 10 kHz FLAC.
    16khz_path, 32khz_path : str
        Absolute ``gs://`` paths to pre-resampled WAVs (empty if not
        pre-resampled — most files are not).
    deployment : str
        Deployment slug (see :data:`DEPLOYMENTS`).
    xwav_subchunk_index : int
        0-based subchunk inside the source XWAV file.
    begin_in_subchunk_s, end_in_subchunk_s : float
        Event time relative to the subchunk's audio start.
    begin_in_file_s, end_in_file_s : float
        Event time in the decoded continuous audio stream (used as
        ``window_start_sec`` / ``window_end_sec`` by the loader).
    begin_utc, end_utc : str
        ISO-8601 UTC datetimes.
    label : str
        Annotation label (see :data:`LABELS`).
    label_is_strong : bool
        True when ``begin_in_subchunk_s`` / ``end_in_subchunk_s`` give
        tight bounds; False for subchunk-level (weak) labels.
    implicit_negatives : bool
        True if unlabelled time in the same subchunk is *known* not to
        contain this label (useful for choosing negatives).
    audit_name : str
        Annotation effort that produced the event (``initial``,
        ``validation``, ``postpub``, ``model{1..4}``, …).
    coarse_call_type : str
        ``song`` for humpback, ``noise`` / ``fish`` / ``background`` /
        ``other`` otherwise.
    species : str
        GBIF canonical name (``Megaptera novaeangliae``) for ``Mn``,
        empty for non-species labels.
    canonical_name, gbifID, kingdom, phylum, class, order, family,
    genus, species_common
        GBIF taxonomy fields, populated only for ``Mn``.

    Splits
    ------
    - ``all`` : every annotation event (~38,857 rows from 16 deployments)
    - ``train`` / ``val`` : deployment-stratified 90/10 split at the
      audio-file level (annotations from one FLAC stay in one split)
    - ``humpback-detection-val`` : balanced binary humpback-presence eval
      derived from ``val`` (902 clips, 451 positive / 451 negative).
      Positives are *strong* ``Mn`` events centred in a fixed 10 s window;
      negatives are all non-``Mn`` labels (mostly ``Background``). Carries an
      extra ``det_target`` column (``Megaptera novaeangliae`` / ``None``).
      Built by ``scripts/data_preprocessing_scripts/pifsc_pipan/
      build_pifsc_pipan_humpback_detection.py``.

    Loader behaviour
    ----------------
    - Reads the audio window ``[begin_in_file_s, end_in_file_s]`` from
      the resolved 16 kHz path if present, otherwise from the original
      FLAC (resampling 10→16 kHz on the fly with kaiser_best).
    - Zero-width annotations (``begin == end``) — common for weak
      subchunk-level labels — are widened to the full subchunk's audio
      window by the caller via the ``subchunk_duration_s`` derived
      column.

    Available tasks
    ---------------
    - Humpback presence detection (binary: ``Mn`` vs others)
    - Multi-class noise / target classification
    - Weak-label aggregation: per-subchunk multi-label clips

    References
    ----------
    - Allen et al. (2021) "A Convolutional Neural Network for Automated
      Detection of Humpback Whale Song in a Diverse, Long-Term Passive
      Acoustic Dataset", Front. Mar. Sci. doi:10.3389/fmars.2021.607321
    - NOAA Pacific Islands Fisheries Science Center (2021), DOI
      10.25921/Z787-9Y54.

    License: CC0-1.0
    """

    info = DatasetInfo(
        name="pifsc_pipan",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/pifsc_pipan_all.csv",
            "train": f"{_GCS_ROOT}/pifsc_pipan_train.csv",
            "val": f"{_GCS_ROOT}/pifsc_pipan_val.csv",
            "humpback-detection-val": (f"{_GCS_ROOT}/pifsc_pipan_humpback_detection_val.csv"),
        },
        version="0.1.0",
        description=(
            "PIFSC PIPAN Allen-2021 annotations: humpback song plus "
            "negative/noise classes from 16 NOAA PIFSC HARP deployments "
            "across the Pacific (2005–2018), one row per analyst event "
            "(strong or weak), 38,857 events on 5,489 unique files at "
            "10 kHz decimated."
        ),
        sources=(
            "Allen et al. (2021) doi:10.3389/fmars.2021.607321; "
            "NOAA PIFSC PIPAN 10kHz Data DOI:10.25921/Z787-9Y54"
        ),
        license="CC0-1.0",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_path"
    _mixup_group = "marine_mammal"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialise the PIFSC PIPAN dataset.

        Parameters
        ----------
        split : str
            Split to load (key in :attr:`info.split_paths`).
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        sample_rate : int | None
            Target sample rate. Pre-resampled 16 kHz / 32 kHz files used
            when ``{sr}khz_path`` is populated, otherwise FLAC resampled
            on-the-fly. ``None`` returns the file's native rate.
        data_root : str | AnyPathT | None
            Unused — ``audio_path`` columns hold absolute ``gs://`` URIs.
            Accepted for API parity with other datasets.
        backend : BackendType
            ``"polars"`` or ``"pandas"``.
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        # Honour data_root if supplied, but absolute paths in the CSV
        # take precedence at load time.
        self.data_root = anypath(data_root) if data_root else None
        self._load()

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

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _resolve_audio_path(self, row: dict[str, Any]) -> tuple[AnyPathT, bool]:
        """Return ``(absolute_path, is_presampled)`` for the row.

        Returns
        -------
        tuple[AnyPathT, bool]
            The audio path and whether it is already at ``self.sample_rate``.
        """
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            col = self._sample_rate_paths[self.sample_rate]
            val = row.get(col)
            if val is not None:
                s = str(val).strip()
                if s and s.lower() != "nan":
                    return anypath(s), True
        return anypath(str(row[self._originals_path_column])), False

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Load + return the audio window for one annotation event.

        Returns
        -------
        dict[str, Any]
            The row with ``audio`` and ``sample_rate`` populated.
        """
        audio_fp, is_presampled = self._resolve_audio_path(row)

        # Window into the file at the event's bounds. Zero-width
        # annotations are widened to ~10 s so the loader returns audio.
        ws = row.get("begin_in_file_s")
        we = row.get("end_in_file_s")
        if ws is not None and we is not None:
            ws_f = float(ws)
            we_f = float(we)
            if we_f <= ws_f:
                we_f = ws_f + 10.0
            audio, sr = read_audio(audio_fp, start_time=ws_f, end_time=we_f)
        else:
            audio, sr = read_audio(audio_fp)

        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
        if not is_presampled and self.sample_rate is not None and sr != self.sample_rate:
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

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __len__(self) -> int:
        """Return the number of annotation events in the split.

        Returns
        -------
        int
            Number of annotation events in the current split.

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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single processed annotation event.

        Returns
        -------
        dict[str, Any]
            The processed row (audio + event metadata).
        """
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over processed annotation events.

        Yields
        ------
        dict[str, Any]
            Each processed annotation event.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple[PIFSCPipan, dict[str, Any]]:
        """Create a PIFSCPipan instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration.

        Returns
        -------
        tuple[PIFSCPipan, dict[str, Any]]
            The dataset instance and any transformation metadata.
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

    def get_available_labels(self, annotation_column: str = "label") -> list[str]:
        """Return all possible label values for a column.

        Parameters
        ----------
        annotation_column : str
            One of ``"label"`` (the annotation enum) or ``"deployment"``.

        Returns
        -------
        list[str]
            The available label values.

        Raises
        ------
        ValueError
            If ``annotation_column`` has no predefined label set.
        """
        if annotation_column == "label":
            return LABELS
        if annotation_column == "deployment":
            return DEPLOYMENTS
        raise ValueError(
            f"No predefined label set for '{annotation_column}'. "
            f"Columns with predefined labels: label, deployment"
        )

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{base}\n"
            f"Annotation events: {n}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
