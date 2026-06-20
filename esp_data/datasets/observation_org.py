"""Observation.org dataset"""

from typing import Any, Dict, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class ObservationOrg(Dataset):
    """Observation.org audio dataset.

    Description
    -----------
    Observation.org is a citizen-science biodiversity platform. This dataset
    contains audio recordings sourced from Observation.org via GBIF, linked to
    taxonomic information following ESP's taxonomy app (GBIF backbone): species
    scientific and common names, genus, family, order, class. Additional metadata
    includes location, date, life stage, sex, and recordist information.
    The current version 0.1.0 includes Observation.org sound records up to June 2026.

    Only recordings under permissive licenses are included (the bulk of
    Observation.org audio is CC-BY-NC-ND, which is excluded), so this is a small,
    bird-dominated subset (~7.4k recordings, ~440 species).

    Available Metadata Fields
    -------------------------
    **Taxonomic Information:**
        - ``canonical_name``: Canonical species name (primary identifier)
        - ``scientific_name_unified``: Scientific (binomial) species name
        - ``species_common``: Common name for the species
        - ``genus``, ``family``, ``order``, ``class``, ``phylum``: Taxonomic hierarchy
        - ``gbifID``: GBIF (Global Biodiversity Information Facility) identifier

    **Audio File Paths:**
        - ``relative_path``: Path to original audio relative to the ``audio/`` dir
          (variable sample rate)
        - ``32khz_path``: Path to pre-resampled 32kHz audio
        - ``16khz_path``: Path to pre-resampled 16kHz audio

    **Recording Metadata:**
        - ``eventDate``: When the recording was made
        - ``lifeStage``, ``sex``: Biological context (``behavior`` is not provided
          by Observation.org)

    **Location:**
        - ``latitudeDecimal``, ``longitudeDecimal``: GPS coordinates
        - ``country_code``, ``locality``: Geographic location names

    **Rights & Attribution:**
        - ``recordist``: Person who made the recording
        - ``rightsHolder``: Copyright holder
        - ``license``: Occurrence license
        - ``media_license``: Media-specific (audio) license
        - ``media_url``: Original Observation.org sound URL

    Available Splits
    ----------------
    - ``train``: Training set (random split)
    - ``val``: Validation set (random split)
    - ``all``: Complete dataset (train + val)
    - ``train_unseen``: Training set excluding unseen taxa evaluated in BEANS-Zero
    - ``val_unseen``: Validation set excluding unseen taxa evaluated in BEANS-Zero
    - ``all_unseen``: Complete dataset excluding BEANS-Zero unseen taxa

    The ``_unseen`` splits are designed for training models that will be evaluated
    on BEANS-Zero's unseen taxa benchmark, ensuring no test taxa leak into training.

    References
    ----------
    Observation.org: https://observation.org/

    Examples
    --------
    >>> from esp_data.datasets import ObservationOrg
    >>> dataset = ObservationOrg(
    ...     split="train",
    ...     output_take_and_give={"canonical_name": "species"}
    ... )
    >>> print(dataset.info.name)
    observation_org
    >>> print(dataset.available_sample_rates)
    [32000, 16000]

    Load with pre-resampled 32kHz audio (no on-the-fly resampling needed)
    >>> dataset_32k = ObservationOrg(split="train", sample_rate=32000)

    Load with pre-resampled 16kHz audio (no on-the-fly resampling needed)
    >>> dataset_16k = ObservationOrg(split="train", sample_rate=16000)
    """

    _DATA_ROOT = "gs://esp-data-ingestion/observation-org/v0.1.0/raw/"

    info = DatasetInfo(
        name="observation_org",
        owner="david",
        split_paths={
            "train": f"{_DATA_ROOT}train.csv",
            "train_unseen": f"{_DATA_ROOT}train_unseen.csv",
            "val": f"{_DATA_ROOT}val.csv",
            "val_unseen": f"{_DATA_ROOT}val_unseen.csv",
            "all": f"{_DATA_ROOT}all.csv",
            "all_unseen": f"{_DATA_ROOT}all_unseen.csv",
        },
        version="0.1.0",
        description="Observation.org audio dataset with taxonomic metadata, sourced via "
        "GBIF. Available at original (variable) sample rates and 16kHz/32kHz "
        "(pre-resampled with librosa's kaiser_best method).",
        sources=["Observation.org"],
        license="CC BY-NC 4.0, CC BY-SA 4.0, CC BY 4.0, CC0 1.0",
    )

    # Mapping of sample rates to their corresponding path columns.
    # Pre-resampled paths include the "audio_16k/" / "audio_32k/" prefix and are
    # therefore relative to the dataset data_root (the "raw/" dir).
    _sample_rate_paths = {
        32000: "32khz_path",  # Pre-resampled to 32kHz
        16000: "16khz_path",  # Pre-resampled to 16kHz
    }

    # Column for original variable-rate audio. relative_path is relative to the
    # "audio/" dir (e.g. "488/obs_123.mp3"), so the "audio/" prefix is added below.
    _originals_path_column = "relative_path"

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the Observation.org dataset.

        Parameters
        ----------
        split : str, default="train"
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str], optional
            A dictionary mapping original column names to new column names.
        sample_rate : int, optional
            The sample rate to which audio files should be resampled. If the requested
            sample rate is available as pre-resampled audio (see `available_sample_rates`),
            the pre-resampled version is loaded directly. Otherwise, audio is resampled
            on-the-fly from the original files using librosa's kaiser_best method. If
            None, audio is returned at its original (variable) sample rate.
        data_root : str | AnyPathT, optional
            The root directory for the dataset, prepended to relative audio paths. If
            None, defaults to the GCS bucket path for this dataset.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars".
        streaming : bool, optional
            Whether to use streaming mode, by default False.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self._load()
        self.sample_rate = sample_rate

        if data_root is None:
            self.data_root = anypath(self._DATA_ROOT)
        else:
            self.data_root = anypath(data_root)

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset."""
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Return the available pre-resampled sample rates.

        Returns
        -------
        list[int]
            Sample rates (Hz) for which pre-resampled audio is available, based on
            which path columns exist in the loaded data.
        """
        available = []
        for sr, path_column in self._sample_rate_paths.items():
            if path_column in self._data.columns:
                available.append(sr)
        return available

    def _load(self) -> None:
        """Load the dataset.

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

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["ObservationOrg", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration containing dataset parameters.

        Returns
        -------
        tuple[Dataset, dict[str, Any]]
            The dataset instance and transformation metadata (empty if none applied).
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
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata

        return ds, {}

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of the dataset.

        Parameters
        ----------
        row : dict[str, Any]
            A dictionary representing a single row of the dataset.

        Returns
        -------
        dict[str, Any]
            The processed row with ``audio`` and ``sample_rate`` populated.
        """
        # Use a pre-resampled version if available for the requested sample rate;
        # otherwise resample on-the-fly from the original file.
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if path_column in row:
                val = row[path_column]
                s = str(val).strip() if val is not None else ""
                if s and s.lower() != "nan":
                    # Pre-resampled paths include the audio_16k/ or audio_32k/ prefix.
                    audio_path = anypath(self.data_root) / s
                    use_presampled = True

        if use_presampled:
            audio, sample_rate = read_audio(audio_path)
            audio = audio.astype(np.float32)
            audio = audio_stereo_to_mono(audio, mono_method="average")
        else:
            # Original variable-rate files; relative_path is relative to the audio/ dir.
            rel_path = row[self._originals_path_column]
            if not str(rel_path).startswith("audio/"):
                audio_path = anypath(self.data_root) / "audio" / rel_path
            else:
                audio_path = anypath(self.data_root) / rel_path
            audio, sample_rate = read_audio(audio_path)
            audio = audio.astype(np.float32)
            audio = audio_stereo_to_mono(audio, mono_method="average")

            if self.sample_rate is not None and sample_rate != self.sample_rate:
                audio = librosa.resample(
                    y=audio,
                    orig_sr=sample_rate,
                    target_sr=self.sample_rate,
                    scale=True,
                    res_type="kaiser_best",
                )
                sample_rate = self.sample_rate

        row["audio"] = audio
        row["sample_rate"] = sample_rate

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the processed data.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        Dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
        str
            Human-readable dataset summary.
        """
        base_info = f"{self.info.name} (v{self.info.version})"

        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
