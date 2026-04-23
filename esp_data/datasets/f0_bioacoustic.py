"""F0 Bioacoustic Benchmark dataset.

Musikhin et al. (2025) "F0 estimation for bioacoustics: A benchmark/training
dataset of non-human vocalisations with annotated frequency contours"
doi:10.1080/09524622.2025.2500380
"""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

# ── Taxa in the dataset ──────────────────────────────────────────────────

TAXA = [
    "canids",
    "disk-winged_bats",
    "dolphins",
    "hummingbirds",
    "La_Palma_chaffinches",
    "lions",
    "little_owls",
    "long-billed_hermits",
    "monk_parakeets",
    "orangutans",
    "Reunion_grey_white_eyes",
    "rodents",
    "spotted_hyenas",
]

# Species that appear in the processed CSV (after GBIF linking in the
# ingestion pipeline).  This covers the primary binomials; canids and
# hummingbirds contribute many additional species.
SPECIES_LABELS = [
    "Athene noctua",
    "Canis aureus",
    "Canis latrans",
    "Canis lupus",
    "Canis lupus familiaris",
    "Canis lupus hallstromi",
    "Canis rufus",
    "Crocuta crocuta",
    "Fringilla coelebs",
    "Myiopsitta monachus",
    "Odontoceti",
    "Panthera leo",
    "Phaethornis longirostris",
    "Pongo pygmaeus",
    "Thyroptera tricolor",
    "Zosterops borbonicus",
]


# ── Helpers ───────────────────────────────────────────────────────────────


def _parse_f0_contour(tsv: str) -> pd.DataFrame:
    """Parse an inline-TSV F0 contour into a DataFrame.

    Parameters
    ----------
    tsv : str
        TSV string with ``Time`` and ``Freq`` columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``time_s`` (float), ``freq_hz`` (float).
    """
    if not tsv or not isinstance(tsv, str):
        return pd.DataFrame(columns=["time_s", "freq_hz"])
    try:
        df = pd.read_csv(StringIO(tsv), sep="\t")
        df = df.rename(columns={"Time": "time_s", "Freq": "freq_hz"})
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        df["freq_hz"] = pd.to_numeric(df["freq_hz"], errors="coerce")
        return df.dropna()
    except Exception:
        return pd.DataFrame(columns=["time_s", "freq_hz"])


@register_dataset
class F0Bioacoustic(Dataset):
    """F0 Bioacoustic Benchmark Dataset.

    Description
    -----------
    Multi-taxon dataset of ~250,000 non-human vocalizations from 13 taxa,
    each with a waveform and a ground-truth fundamental frequency (F0)
    contour. Vocalizations span a wide frequency range (50 Hz lions to
    26 kHz bats) and diverse acoustic behaviors (howls, whistles, calls,
    songs, ultrasonic vocalizations).

    Columns
    -------
    audio_path : str
        Relative path to source audio (e.g. ``dolphins/cut_no_overlap/file.wav``).
    f0_contour : str
        TSV-serialised F0 annotation with ``Time`` (seconds) and ``Freq`` (Hz) columns.
    taxon : str
        Taxon group name (e.g. ``dolphins``, ``canids``, ``hummingbirds``).
    species : str
        Resolved species binomial (after GBIF linking).
    canonical_name : str
        GBIF canonical name (after GBIF linking in ingestion pipeline).
    subspecies_detail : str
        Finer taxonomic detail when available (e.g. canid subspecies).
    16khz_path, 32khz_path : str | None
        Paths to pre-resampled audio (when available).
    mean_f0_hz : float | None
        Mean fundamental frequency in Hz across the contour; None if contour is empty.
    min_f0_hz : float | None
        Minimum fundamental frequency in Hz across the contour; None if contour is empty.
    max_f0_hz : float | None
        Maximum fundamental frequency in Hz across the contour; None if contour is empty.

    Available tasks
    ---------------
    - **F0 estimation**: Predict the fundamental frequency contour from audio.
    - **Species classification**: Classify vocalizations by species/taxon.
    - **F0 tracking**: Track F0 through time in the presence of noise.
    - **Cross-taxon generalization**: Train on some taxa, evaluate on held-out taxa.

    Notes
    -----
    - Sample rates vary enormously across taxa (4 kHz → 375 kHz).
      Pre-resampled 16 kHz / 32 kHz versions are available.
    - For ultrasonic taxa (disk-winged bats at 375 kHz), downsampling to
      16/32 kHz will alias the F0 — use 32 kHz or original audio for these.
    - F0 contours are ground-truth annotations, not algorithm predictions.
      The companion ``F0_predictions`` folder (not included here) contains
      algorithm benchmarks.

    Examples
    --------
    >>> from esp_data.datasets import F0Bioacoustic
    >>> ds = F0Bioacoustic(split="all")
    >>> sample = ds[0]
    >>> sample["audio"].shape, sample["f0_contour"].columns.tolist()
    ((N,), ['time_s', 'freq_hz'])

    >>> ds_birds = F0Bioacoustic(split="all", taxa=["hummingbirds", "La_Palma_chaffinches"])
    >>> len(ds_birds)  # Only bird taxa

    References
    ----------
    Musikhin et al. (2025) doi:10.1080/09524622.2025.2500380
    Data: https://doi.org/10.5061/dryad.prr4xgxw8
    License: CC0-1.0
    """

    info = DatasetInfo(
        name="f0_bioacoustic",
        owner="david",
        split_paths={
            "all": "gs://esp-data-ingestion/f0-prediction/f0_bioacoustic_normalized.csv",
            "train": "gs://esp-data-ingestion/f0-prediction/f0_bioacoustic_train.csv",
            "val": "gs://esp-data-ingestion/f0-prediction/f0_bioacoustic_val.csv",
            "train_unseen": "gs://esp-data-ingestion/f0-prediction/f0_bioacoustic_train_unseen.csv",
        },
        version="0.1.0",
        description=("Are with ground-truth fundamental frequency contours"),
        sources="Musikhin et al. (2025) doi:10.1080/09524622.2025.2500380",
        license="CC0-1.0",
    )

    _sample_rate_paths: dict[int, str] = {
        16000: "16khz_path",
        32000: "32khz_path",
    }

    _sample_rate_subdirs: dict[int, str] = {
        16000: "audio_16k",
        32000: "audio_32k",
    }

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        taxa: list[str] | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load (key in ``info.split_paths``).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys.
        sample_rate : int | None
            Target sample rate. Pre-resampled paths preferred when available;
            otherwise audio is resampled on-the-fly.
        taxa : list[str] | None
            Subset of taxa to include (e.g. ``["dolphins", "canids"]``).
            If None, all taxa are loaded. See :data:`TAXA` for the full list.
        data_root : str | AnyPathT | None
            Root directory containing audio files. If None, defaults to
            the parent directory of the split CSV path.
        backend : BackendType
            Backend to use (``"pandas"`` or ``"polars"``), default ``"polars"``.
        streaming : bool
            Whether to use streaming mode, default False.

        Raises
        ------
        ValueError
            If ``taxa`` contains unknown taxon names.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None

        self._load()

        # Filter to requested taxa
        if taxa is not None:
            unknown = set(taxa) - set(TAXA)
            if unknown:
                raise ValueError(f"Unknown taxa: {sorted(unknown)}. Valid taxa: {TAXA}")
            self._data = self._data.filter_isin("taxon", taxa)

        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    @property
    def available_taxa(self) -> list[str]:
        """Return the taxa present in the currently-loaded data.

        Returns
        -------
        list[str]
            Sorted list of unique taxon names.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet.")
        return sorted(self._data.get_unique("taxon"))

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
        if self._data is None:
            raise RuntimeError("No split has been loaded yet.")
        if self._streaming:
            raise NotImplementedError("Length not available in streaming mode.")
        return len(self._data)

    # ------------------------------------------------------------------
    # Audio path resolution
    # ------------------------------------------------------------------

    def _resolve_audio_path(self, row: dict[str, Any]) -> tuple[AnyPathT, bool]:
        """Return ``(full_audio_path, is_presampled)``.

        Prefers pre-resampled paths when available for the requested
        sample rate. Falls back to ``gcs_path`` if present (train_unseen
        split), then to ``data_root / audio_path``.
        """
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            col = self._sample_rate_paths[self.sample_rate]
            if col in row and row[col] is not None and str(row[col]).strip():
                subdir = self._sample_rate_subdirs.get(self.sample_rate, "")
                if subdir:
                    return self.data_root / subdir / row[col], True
                return self.data_root / row[col], True
        if row.get("gcs_path"):
            return anypath(row["gcs_path"]), False
        return self.data_root / row["audio_path"], False

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        # Parse F0 contour from serialised TSV
        f0 = _parse_f0_contour(row.get("f0_contour", ""))

        if self.sample_rate is not None:
            # Load and resample audio
            audio_fp, is_presampled = self._resolve_audio_path(row)
            audio, sr = read_audio(audio_fp)
            audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

            if not is_presampled and sr != self.sample_rate:
                audio = librosa.resample(
                    y=audio,
                    orig_sr=sr,
                    target_sr=self.sample_rate,
                    scale=True,
                    res_type="kaiser_best",
                )
                sr = self.sample_rate

            # Clip F0 annotations to audio duration
            audio_dur = len(audio) / float(sr)
            if not f0.empty:
                f0 = f0[f0["time_s"] <= audio_dur].copy()

            # Flag frequencies above Nyquist for the current sample rate
            nyquist = self.sample_rate / 2.0
            f0["above_nyquist"] = f0["freq_hz"] > nyquist

            row["audio"] = audio
            row["sample_rate"] = sr

        # Compute F0 summary statistics (None when contour is empty)
        valid_f0 = f0["freq_hz"] if not f0.empty else pd.Series([], dtype=float)
        row["mean_f0_hz"] = float(valid_f0.mean()) if not valid_f0.empty else None
        row["min_f0_hz"] = float(valid_f0.min()) if not valid_f0.empty else None
        row["max_f0_hz"] = float(valid_f0.max()) if not valid_f0.empty else None

        # Build output
        row["f0_contour"] = f0

        if self.output_take_and_give:
            item = {}
            for old_key, new_key in self.output_take_and_give.items():
                item[new_key] = row[old_key]
            return item

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["F0Bioacoustic", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary. Accepts optional ``taxa`` key
            (list of taxon names to include).
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            taxa=cfg.get("taxa"),
            data_root=cfg.get("data_root"),
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )

        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta

        return ds, {}

    def get_available_labels(self, annotation_column: str = "taxon") -> list[str]:
        """Return all possible labels for a given annotation column.

        Parameters
        ----------
        annotation_column : str
            ``"taxon"`` or ``"species"``.

        Returns
        -------
        list[str]
            Label values.
        """
        if annotation_column == "taxon":
            return TAXA
        elif annotation_column == "species":
            return SPECIES_LABELS
        else:
            raise ValueError(
                f"No predefined label set for '{annotation_column}'. "
                f"Columns with predefined labels: taxon, species"
            )

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        n = len(self) if self._data is not None and not self._streaming else "?"
        taxa = ", ".join(self.available_taxa) if self._data is not None else "?"
        return (
            f"{base}\n"
            f"Vocalizations: {n}\n"
            f"Taxa: {taxa}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
