"""Watkins Marine Mammal Sound Database dataset.

The Watkins Marine Mammal Sound Database (WMMSDB) is the most comprehensive
collection of marine mammal vocalisations, originally curated at Woods Hole
Oceanographic Institution.  This dataset wraps the 2018 remastered release
with GBIF-resolved taxonomy.

Each row is a single audio clip with a species label and taxonomic metadata.
The dataset covers ~50 species of cetaceans and pinnipeds.
"""

from typing import Any, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_RAW_ROOT = "gs://esp-data-ingestion/superwhale/v0.1.0/raw"


@register_dataset
class Watkins(Dataset):
    """Watkins Marine Mammal Sound Database (2018 remaster).

    Description
    -----------
    The Watkins Marine Mammal Sound Database is the largest publicly available
    collection of marine mammal vocalisations, originally compiled by William
    A. Watkins at Woods Hole Oceanographic Institution.  This dataset uses the
    2018 remastered FLAC release and includes GBIF-resolved taxonomic metadata.

    The dataset spans ~50 species across cetaceans (baleen whales, toothed
    whales, dolphins) and pinnipeds (seals, sea lions, walrus), with ~13,700
    audio clips at variable original sample rates.

    Available Metadata Fields
    -------------------------
    **Taxonomic Information:**
        - ``species``: Scientific species name (as labelled in the original dataset)
        - ``canonical_name``: GBIF-resolved canonical species name
        - ``species_common``: Common/vernacular species name
        - ``genus``, ``family``, ``order``, ``class``, ``phylum``, ``kingdom``:
          Taxonomic hierarchy
        - ``gbifID``: GBIF identifier

    **Vocalisation Labels:**
        - ``call_type``: Semicolon-separated fine-grained vocalisation types
          (e.g. ``"click;whistle"``, ``"moan"``, ``"pulsed_click;click;squeal"``).
          Populated for ~84% of rows.
        - ``coarse_call_type``: Semicolon-separated coarse categories
          (e.g. ``"click;whistle"``, ``"call"``, ``"burst_pulse;click"``).

    **Audio File Paths:**
        - ``audio_path``: Path to original FLAC audio relative to data_root
          (variable sample rate)
        - ``16khz_path``: Path to pre-resampled 16kHz WAV audio
        - ``32khz_path``: Path to pre-resampled 32kHz WAV audio

    **Audio Metadata:**
        - ``sample_rate_hz``: Original sample rate of the recording (Hz)
        - ``duration_s``: Duration of the recording (seconds)

    References
    ----------
    Watkins Marine Mammal Sound Database:
        https://cis.whoi.edu/science/B/whalesounds/index.cfm
    DOI: 10.1575/1912/7270

    Examples
    --------
    >>> from esp_data.datasets import Watkins
    >>> ds = Watkins(split="train")
    >>> print(len(ds))
    13693

    >>> ds_16k = Watkins(split="train", sample_rate=16000)
    """

    info = DatasetInfo(
        name="watkins",
        owner="david",
        split_paths={
            "train": "gs://esp-data-ingestion/watkins/v0.1.0/watkins.csv",
        },
        version="0.1.0",
        description=(
            "Watkins Marine Mammal Sound Database — 2018 remastered release.  "
            "~13,700 audio clips spanning ~50 species of cetaceans and "
            "pinnipeds with GBIF-resolved taxonomy.  Original audio at "
            "variable sample rates; pre-resampled 16kHz and 32kHz versions "
            "available."
        ),
        sources=["https://cis.whoi.edu/science/B/whalesounds/index.cfm"],
        license="LicenseRef-WHOI-Public",
    )

    _sample_rate_paths = {
        16000: "16khz_path",
        32000: "32khz_path",
    }

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialise the Watkins dataset.

        Parameters
        ----------
        split : str
            The split to load (default ``"train"``).
        output_take_and_give : dict[str, str] | None
            Column renaming / selection mapping.
        sample_rate : int | None
            Target sample rate.  If a pre-resampled version exists (16kHz or
            32kHz), it will be loaded directly; otherwise audio is resampled
            on-the-fly.  ``None`` returns original sample rate.
        data_root : str | AnyPathT | None
            Root directory prepended to ``audio_path``.  Defaults to the GCS
            bucket holding the original FLAC files.
        backend : BackendType
            DataFrame backend (``"polars"`` or ``"pandas"``).
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None

        if data_root is None:
            self.data_root = anypath(_GCS_RAW_ROOT)
            self._data_root_16k = anypath(f"{_GCS_RAW_ROOT}/audio_16k")
            self._data_root_32k = anypath(f"{_GCS_RAW_ROOT}/audio_32k")
        else:
            self.data_root = anypath(data_root)
            self._data_root_16k = anypath(data_root)
            self._data_root_32k = anypath(data_root)

        self._load()

    # ── Loading ────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the Watkins CSV from the configured split path.

        Raises
        ------
        LookupError
            If the split is not valid.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(location, streaming=self._streaming)

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset."""
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Pre-resampled sample rates available in the loaded data."""
        available = []
        if self._data is not None:
            for sr, col in self._sample_rate_paths.items():
                if col in self._data.columns:
                    available.append(sr)
        return available

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["Watkins", dict[str, Any]]:
        """Create a Watkins instance from a config.

        Returns
        -------
        tuple[Watkins, dict[str, Any]]
            The dataset instance and transformation metadata (empty if none).
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

    # ── Iteration / indexing ───────────────────────────────────────────

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row: load audio and optionally resample.

        Returns
        -------
        dict[str, Any]
            The processed row with ``audio`` and ``sample_rate`` keys added.
        """
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            col = self._sample_rate_paths[self.sample_rate]
            if col in row and row[col] is not None and str(row[col]).strip():
                if self.sample_rate == 16000:
                    audio_path = self._data_root_16k / row[col]
                else:
                    audio_path = self._data_root_32k / row[col]
                use_presampled = True

        if not use_presampled:
            audio_path = self.data_root / row["audio_path"]

        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

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
        return (
            f"{self.info.name} (v{self.info.version}), split='{self.split}'\n"
            f"  Rows: {n}\n"
            f"  Description: {self.info.description}\n"
            f"  License: {self.info.license}\n"
            f"  Available splits: {', '.join(self.info.split_paths.keys())}"
        )
