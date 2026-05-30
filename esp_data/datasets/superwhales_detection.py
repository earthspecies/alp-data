"""SuperWhale Detection dataset — aggregate of 13 marine mammal detection datasets.

Each row represents one audio file and carries a ``selection_table`` column: a
TSV-encoded blob listing per-event annotations (begin/end times, frequencies,
species, call type, etc.).  After the GBIF-linking pipeline step the TSV events
also contain ``canonical_name``, ``genus``, ``family``, ``species_common``, and
``gbifID`` columns.

Component Datasets
------------------
The merged detection CSV is built from 13 source datasets spanning baleen
whales, odontocetes, and mixed-species recordings.  Each source dataset is
identified by its ``source_dataset`` column value in the CSV.  See
``DETECTION_CATALOG`` for per-dataset documentation.

Filtering
---------
Use ``include_datasets`` to restrict to a subset of source datasets.

Negative-Clip Control
---------------------
For each source dataset the ``positives_only`` mapping controls whether rows
whose selection table contains *no* events (header only) are kept or dropped.
By default every source dataset is ``positives_only=True`` — only rows with at
least one annotated event are returned.  Override per source dataset::

    config = SuperWhaleDetectionConfig(
        include_datasets=["dclde_2013_nefsc_sbnms_allbaleen"],
        positives_only={"dclde_2013_nefsc_sbnms_allbaleen": False},
    )
"""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd
from pydantic import Field

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.dataset import register_config
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio, read_text

# ── Component Dataset Catalog ──────────────────────────────────────────────
#
# Each entry documents one source detection dataset.  The key is the
# ``source_dataset`` value that appears in the merged CSV.
#
# Fields:
#   description   – short human-readable description
#   species       – list of species present (scientific names)
#   call_types    – list of coarse call types annotated
#   license       – licence string
#   rows          – approximate number of audio-file rows
#   source_url    – canonical URL for the dataset

DETECTION_CATALOG: dict[str, dict[str, Any]] = {
    "biodcase_2025_task2_whale_calls": {
        "description": (
            "BioDCASE 2025 Task 2: strongly-labelled blue & fin whale "
            "calls from OHASISBIO hydrophones in the Indian Ocean."
        ),
        "species": ["Balaenoptera musculus", "Balaenoptera physalus"],
        "call_types": ["call"],
        "license": "CC-BY-4.0",
        "rows": 4722,
        "source_url": "https://zenodo.org/records/14887842",
    },
    "bottlenose_dolphin_trawl_figshare_6313308": {
        "description": (
            "Acoustic emissions of bottlenose dolphins recorded during "
            "trawl fishing in Moreton Bay, Australia."
        ),
        "species": ["Tursiops truncatus"],
        "call_types": ["click", "whistle"],
        "license": "CC-BY-4.0",
        "rows": 1,
        "source_url": "https://figshare.com/articles/dataset/6313308",
    },
    "cetusid_dolphin_vocalisations": {
        "description": (
            "CetusID dolphin vocalisations: training/testing annotations "
            "for common dolphins, orcas, Indo-Pacific humpback dolphins, "
            "and Indo-Pacific bottlenose dolphins."
        ),
        "species": [
            "Delphinus delphis",
            "Orcinus orca",
            "Sousa plumbea",
            "Tursiops aduncus",
        ],
        "call_types": ["call", "whistle"],
        "license": "CC-BY-4.0",
        "rows": 13,
        "source_url": "https://zenodo.org/records/11100712",
    },
    "dclde_2013_nefsc_sbnms_allbaleen": {
        "description": (
            "DCLDE 2013: continuous HARP recordings from NEFSC at "
            "Stellwagen Bank with blue, fin, sei, right, and humpback "
            "whale annotations."
        ),
        "species": [
            "Balaenoptera borealis",
            "Balaenoptera musculus",
            "Balaenoptera physalus",
            "Eubalaena glacialis",
            "Megaptera novaeangliae",
        ],
        "call_types": ["call", "song"],
        "license": "CC0-1.0",
        "rows": 668,
        "source_url": "https://www.cetus.ucsd.edu/dclde/datasetDocumentation.html",
    },
    "dclde_2015_annotations": {
        "description": (
            "DCLDE 2015: long continuous HARP recordings (Channel "
            "Islands, Gulf of Maine) with blue and fin whale call "
            "annotations at 2 kHz sample rate."
        ),
        "species": ["Balaenoptera musculus", "Balaenoptera physalus"],
        "call_types": ["call"],
        "license": "CC0-1.0",
        "rows": 23,
        "source_url": "https://www.cetus.ucsd.edu/dclde/datasetDocumentation.html",
    },
    "dclde_2018_hf_annotations": {
        "description": (
            "DCLDE 2018 HF Odontocete: high-frequency HARP recordings "
            "(200 kHz) of marine mammal echolocation encounters from "
            "Gulf of Mexico (GofMX-DT), Cape Hatteras (HAT-A), "
            "Jacksonville (JAX-D), and Atlantic (WAT-Hz, WAT-NC) sites. "
            "1,213 audio files (~758 hours) with encounter-level start/"
            "end annotations covering Cuvier's beaked whale, Gervais' and "
            "Sowerby's beaked whales, Risso's dolphin, short-finned pilot "
            "whale, Fraser's dolphin, Stenella spp., plus suborder-level "
            "Odontoceti for unidentified clicks. Pre-resampled 16/32 kHz "
            "WAV available for ~98% of files."
        ),
        "species": [
            "Ziphius cavirostris",
            "Mesoplodon europaeus",
            "Mesoplodon bidens",
            "Grampus griseus",
            "Globicephala macrorhynchus",
            "Lagenodelphis hosei",
            "Stenella",
        ],
        "call_types": ["click"],
        "license": "CC0-1.0",
        "rows": 1213,
        "source_url": "https://doi.org/10.25921/400a-tf35",
    },
    "zenodo_17282717_mediterranean_cetacean_clips": {
        "description": (
            "Mediterranean cetacean PAM clips (Jankauskaite et al. 2025): "
            "low-cost HydroMoth deployments on fishing boats and boat-based "
            "surveys in the Western Mediterranean Sea (2022-2024). 777 "
            "expert-annotated WAV clips at heterogeneous sample rates "
            "(48 / 96 / 192 / 384 kHz) cut from the deployment recordings, "
            "one clip per acoustic event with signalType (clicks / whistles "
            "/ mixed) and quality (low/medium/high) metadata. Adds "
            "Mediterranean geographic coverage and two species novel to "
            "SuperWhales: Stenella coeruleoalba (striped dolphin) and "
            "Globicephala melas (long-finned pilot whale)."
        ),
        "species": [
            "Physeter macrocephalus",
            "Tursiops truncatus",
            "Stenella coeruleoalba",
            "Grampus griseus",
            "Globicephala melas",
            "Delphinidae",
        ],
        "call_types": ["click", "whistle", "mixed"],
        "license": "CC-BY-4.0",
        "rows": 777,
        "source_url": "https://zenodo.org/records/17282717",
    },
    "dolphinfree": {
        "description": (
            "DolphinFree: short-duration common dolphin whistle "
            "annotations from the Portuguese coast."
        ),
        "species": ["Delphinus delphis"],
        "call_types": ["whistle"],
        "license": "CC-BY-4.0",
        "rows": 275,
        "source_url": "https://zenodo.org/records/4535615",
    },
    "narw_upcalls_variable_snr_kirsebom_2020": {
        "description": (
            "North Atlantic right whale upcall detections at variable "
            "SNR from the Gulf of St Lawrence (Kirsebom et al. 2020)."
        ),
        "species": ["Eubalaena glacialis"],
        "call_types": ["call"],
        "license": "CC-BY-4.0",
        "rows": 12000,
        "source_url": "https://doi.org/10.6084/m9.figshare.12436199",
    },
    "nefsc_baleen_narw_detections": {
        "description": (
            "NEFSC baleen whale / NARW continuous HARP recordings "
            "with right whale upcall annotations."
        ),
        "species": ["Eubalaena glacialis"],
        "call_types": ["call"],
        "license": "CC0-1.0",
        "rows": 133,
        "source_url": "https://www.ncei.noaa.gov/products/passive-acoustic-data",
    },
    "pifsc_pipan_humpback_annotations": {
        "description": (
            "PIFSC PIPAN: humpback whale song annotations from "
            "hydrophone deployments around Wake Atoll and Saipan."
        ),
        "species": ["Megaptera novaeangliae"],
        "call_types": ["song"],
        "license": "CC0-1.0",
        "rows": 2310,
        "source_url": "https://www.ncei.noaa.gov/products/passive-acoustic-data",
    },
    "wio_blue_whale_zenodo_3624145": {
        "description": (
            "Western Indian Ocean blue/fin whale dataset "
            "(Bouffaut et al.): MACS-B and MACS-D call annotations."
        ),
        "species": ["Balaenoptera musculus", "Balaenoptera physalus"],
        "call_types": ["call"],
        "license": "CC-BY-4.0",
        "rows": 4,
        "source_url": "https://zenodo.org/records/3624145",
    },
    "delphinid_whistle_bbox_dryad_ferguson": {
        "description": (
            "Delphinid whistle bounding-box annotations: bottlenose "
            "dolphin whistle detections with frequency bounds "
            "(Ferguson et al., Dryad)."
        ),
        "species": ["Tursiops truncatus"],
        "call_types": ["whistle"],
        "license": "CC0-1.0",
        "rows": 96,
        "source_url": "https://datadryad.org/stash/dataset/doi:10.5061/dryad.v15dv422r",
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────


def _selection_table_has_events(tsv: str) -> bool:
    """Return True if the selection-table TSV blob has ≥1 event row.

    Returns
    -------
    bool
        True when ``tsv`` contains at least one data row after the header line.
    """
    if not tsv:
        return False
    lines = tsv.strip().split("\n")
    return len(lines) > 1


# ── Selection-table column names ───────────────────────────────────────────

SELECTION_TABLE_COLUMNS = [
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "species",
    "taxon",
    "taxon_rank",
    "call_type",
    "coarse_call_type",
    "confidence",
    # Added by link_gbif step:
    "canonical_name",
    "genus",
    "family",
    "species_common",
    "gbifID",
]

PROVENANCE_COLUMNS = [
    "source_dataset",
    "source_url",
    "license",
    "source_paper_doi",
    "all_cetaceans_labeled",
]


# ── Config ─────────────────────────────────────────────────────────────────


@register_config
class SuperWhaleDetectionConfig(DatasetConfig):
    """Configuration for the SuperWhale Detection dataset.

    Parameters
    ----------
    dataset_name : str
        Must be ``"superwhale_detection"``.
    split : str
        Split to load (default ``"train"``).
    include_datasets : list[str] | None
        Source-dataset names to include.  ``None`` means *all* 11 datasets.
        Names must match the ``source_dataset`` column in the CSV — see
        ``DETECTION_CATALOG`` keys for the full list.
    positives_only : dict[str, bool]
        Per-source-dataset control over whether to keep only rows whose
        selection table contains at least one event.  Keys are
        ``source_dataset`` names; missing keys default to **True**
        (negatives dropped).  Set a key to ``False`` to include negative
        clips from that dataset.
    sample_rate : int | None
        Target sample rate for resampling loaded audio.
    data_root : str | None
        Root directory prepended to ``audio_path`` for resolving files.
    """

    dataset_name: str = "superwhale_detection"
    split: str = "train"
    include_datasets: list[str] | None = Field(
        default=None,
        description=(
            "Source-dataset names to include.  None = all.  See DETECTION_CATALOG for valid names."
        ),
    )
    positives_only: dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "Per-source-dataset flag: True keeps only rows with ≥1 event "
            "in the selection table.  Missing keys default to True."
        ),
    )
    sample_rate: int | None = 16000
    data_root: str | AnyPathT | None = None
    output_take_and_give: dict[str, str] | None = None
    backend: BackendType = "polars"
    streaming: bool = False


# ── Dataset ────────────────────────────────────────────────────────────────

# GCS bucket root
_GCS_ROOT = "gs://esp-data-ingestion/superwhale/v0.1.0/raw"


@register_dataset
class SuperWhaleDetection(Dataset):
    """SuperWhale Detection: an aggregate of 11 marine mammal detection datasets.

    Each row is an audio file paired with a ``selection_table`` TSV blob
    containing per-event annotations.  The dataset covers baleen whales (blue,
    fin, sei, right, humpback), and several delphinid species.

    Component Datasets
    ------------------
    See ``DETECTION_CATALOG`` for per-dataset documentation, or call
    ``SuperWhaleDetection.describe_catalog()`` to print a summary.

    Negative-Clip Control
    ---------------------
    Some source datasets may include audio rows with no annotated events
    (negative clips).  By default only positive rows (≥1 event) are returned.
    Use ``positives_only`` in the config to override per source dataset.

    Examples
    --------
    >>> from esp_data.datasets import SuperWhaleDetection
    >>> ds = SuperWhaleDetection(split="train")
    >>> print(len(ds))

    >>> # Include only NARW and allow negatives:
    >>> ds = SuperWhaleDetection(
    ...     include_datasets=["narw_upcalls_variable_snr_kirsebom_2020"],
    ...     positives_only={"narw_upcalls_variable_snr_kirsebom_2020": False},
    ... )
    """

    info = DatasetInfo(
        name="superwhale_detection",
        owner="david",
        split_paths={
            "train": f"{_GCS_ROOT}/superwhale_detection.csv",
        },
        version="0.1.0",
        description=(
            "Aggregate detection dataset of 11 marine mammal acoustic "
            "datasets.  Each row is an audio file with a selection-table "
            "TSV containing per-event annotations (species, call type, "
            "time/frequency bounds).  Covers baleen whales and delphinids.  "
            "See DETECTION_CATALOG for per-dataset documentation."
        ),
        sources=[info["source_url"] for info in DETECTION_CATALOG.values()],
        license="Mixed (CC-BY-4.0, CC0-1.0; see component datasets)",
    )

    # Pre-resampled path columns
    _sample_rate_paths = {
        32000: "32khz_path",
        16000: "16khz_path",
    }

    def __init__(
        self,
        split: str = "train",
        include_datasets: list[str] | None = None,
        positives_only: dict[str, bool] | None = None,
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialise the SuperWhale Detection dataset.

        Parameters
        ----------
        split : str
            The split to load (default ``"train"``).
        include_datasets : list[str] | None
            Source-dataset names to include.  ``None`` means all.
        positives_only : dict[str, bool] | None
            Per-source-dataset flag controlling negative-clip inclusion.
            Missing keys default to ``True`` (drop negatives).
        output_take_and_give : dict[str, str] | None
            Column renaming / selection mapping.
        sample_rate : int | None
            Target sample rate for audio resampling at access time.
        data_root : str | AnyPathT | None
            Root directory prepended to ``audio_path``.
        backend : BackendType
            DataFrame backend (``"polars"`` or ``"pandas"``).
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.include_datasets = include_datasets
        self.positives_only_map: dict[str, bool] = positives_only or {}
        self.sample_rate = sample_rate
        self._data = None
        self.data_root = anypath(data_root) if data_root is not None else None

        self._load()

        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

    # ── Loading & filtering ────────────────────────────────────────────

    def _load(self) -> None:
        """Load the merged detection CSV, then apply dataset & positives filters.

        Raises
        ------
        LookupError
            If ``self.split`` is not a key in ``info.split_paths``.
        ValueError
            If ``include_datasets`` contains a name not present in
            ``DETECTION_CATALOG``.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )

        location = self.info.split_paths[self.split]
        csv_text = read_text(location, encoding="utf-8")
        df = pd.read_csv(StringIO(csv_text), dtype=str, keep_default_na=False)

        # ── Filter to selected source datasets ─────────────────────────
        if self.include_datasets is not None:
            unknown = set(self.include_datasets) - set(DETECTION_CATALOG.keys())
            if unknown:
                raise ValueError(
                    f"Unknown source datasets: {unknown}. "
                    f"Valid names: {sorted(DETECTION_CATALOG.keys())}"
                )
            df = df[df["source_dataset"].isin(self.include_datasets)]

        # ── Filter negatives per source dataset ────────────────────────
        keep_mask = pd.Series(True, index=df.index)
        for source_ds in df["source_dataset"].unique():
            is_positives_only = self.positives_only_map.get(source_ds, True)
            if is_positives_only:
                ds_rows = df["source_dataset"] == source_ds
                has_events = df.loc[ds_rows, "selection_table"].apply(_selection_table_has_events)
                keep_mask.loc[ds_rows] = has_events

        df = df[keep_mask].reset_index(drop=True)

        # ── Wrap in backend ────────────────────────────────────────────
        self._data = self._backend_class.from_csv(
            StringIO(df.to_csv(index=False)),
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls, dataset_config: SuperWhaleDetectionConfig
    ) -> tuple["SuperWhaleDetection", dict[str, Any]]:
        """Create a SuperWhaleDetection instance from a config.

        Returns
        -------
        tuple[SuperWhaleDetection, dict[str, Any]]
            The instantiated dataset and a metadata dict from
            ``apply_transformations`` (empty when no transforms are
            configured).
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            include_datasets=cfg.get("include_datasets"),
            positives_only=cfg.get("positives_only"),
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

    # ── Iteration / indexing ───────────────────────────────────────────

    def _resolve_audio_path(self, row: dict[str, Any]) -> tuple[AnyPathT, bool]:
        """Resolve the audio path for ``row``, preferring a pre-resampled file.

        Returns
        -------
        tuple[AnyPathT, bool]
            ``(audio_path, is_presampled)``. ``is_presampled`` is True when the
            row carries a non-empty path column matching ``self.sample_rate``;
            otherwise the original ``audio_path`` is returned and the caller is
            expected to resample on the fly.
        """
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            col = self._sample_rate_paths[self.sample_rate]
            if col in row and row[col] is not None and str(row[col]).strip():
                return self.data_root / row[col], True
        return self.data_root / row["audio_path"], False

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row: load audio, parse selection table.

        Returns
        -------
        dict[str, Any]
            The row dict with ``audio`` (numpy float32 array), ``sample_rate``
            (int), and ``selection_table`` (parsed ``pd.DataFrame`` clipped to
            the loaded audio's duration) populated. When ``output_take_and_give``
            is configured the dict is reduced/renamed accordingly.
        """
        audio_path, is_presampled = self._resolve_audio_path(row)
        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")

        if window_start is not None and window_end is not None:
            audio, sr = read_audio(
                audio_path,
                start_time=float(window_start),
                end_time=float(window_end),
            )
        else:
            audio, sr = read_audio(audio_path)
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

        raw_st = row.get("selection_table")
        if raw_st is not None:
            if isinstance(raw_st, str):
                st = pd.read_csv(StringIO(raw_st), sep="\t", keep_default_na=False)
            elif isinstance(raw_st, pd.DataFrame):
                st = raw_st
            else:
                st = pd.DataFrame()

            for col in (
                "species",
                "taxon",
                "taxon_rank",
                "call_type",
                "coarse_call_type",
                "confidence",
            ):
                if col in st.columns:
                    st[col] = st[col].fillna("")

            audio_dur = len(audio) / float(sr)
            if "Begin Time (s)" in st.columns:
                st = st[st["Begin Time (s)"] < audio_dur].copy()

            row["selection_table"] = st

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No data loaded.")
        if self._streaming:
            raise NotImplementedError("Length unavailable in streaming mode.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    def __str__(self) -> str:
        n = len(self) if self._data is not None and not self._streaming else "?"
        ds_count = len(self.include_datasets) if self.include_datasets else len(DETECTION_CATALOG)
        return (
            f"{self.info.name} (v{self.info.version}), split='{self.split}'\n"
            f"  Rows: {n}  |  Source datasets: {ds_count}\n"
            f"  Description: {self.info.description}\n"
            f"  License: {self.info.license}\n"
            f"  Available splits: {', '.join(self.info.split_paths.keys())}"
        )

    # ── Catalog helpers ────────────────────────────────────────────────

    @staticmethod
    def describe_catalog() -> str:
        """Return a human-readable summary of all component detection datasets.

        Returns
        -------
        str
            Multi-line summary suitable for printing.
        """
        lines = ["SuperWhale Detection — Component Datasets", "=" * 50]
        for key, meta in DETECTION_CATALOG.items():
            lines.append(f"\n{key}")
            lines.append(f"  {meta['description']}")
            lines.append(f"  Species:    {', '.join(meta['species'])}")
            lines.append(f"  Call types: {', '.join(meta['call_types'])}")
            lines.append(f"  License:    {meta['license']}")
            lines.append(f"  Rows:       ~{meta['rows']}")
            lines.append(f"  URL:        {meta['source_url']}")
        return "\n".join(lines)

    def get_available_labels(self, annotation_column: str = "species") -> list[str]:
        """Return distinct labels observed across all selection tables.

        This requires iterating the serialised TSV, so it is relatively
        expensive on first call.

        Returns
        -------
        list[str]
            Sorted distinct non-null values seen in ``annotation_column``
            across every row's selection table.

        Raises
        ------
        RuntimeError
            If the dataset has not been loaded (``_load`` has not been called).
        """
        if self._data is None:
            raise RuntimeError("No data loaded.")

        labels: set[str] = set()
        for row in self._data:
            try:
                st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
                if annotation_column in st.columns:
                    labels.update(st[annotation_column].dropna().unique())
            except Exception:
                continue
        return sorted(labels)
