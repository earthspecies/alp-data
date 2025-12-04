"""Xeno-canto dataset"""

from typing import Any, Dict, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class XenoCanto(Dataset):
    """Xeno-canto audio dataset.

    Description
    -----------
    Xeno-canto is a website dedicated to sharing wildlife sounds from around
    the world. This dataset includes audio recordings from Xeno-canto with
    associated metadata about species, locations, and other observation details.

    The dataset contains audio recordings with rich taxonomic information,
    including species scientific and common names, family, genus, order,
    and other metadata such as location, date, and recordist information.

    Available Metadata Fields
    -------------------------
    **Taxonomic Information:**
        - ``canonical_name``: Canonical species name (primary identifier)
        - ``species_common``: Common/vernacular species name
        - ``scientificName``: Scientific species name (legacy field)
        - ``vernacularName``: Vernacular name (if available)
        - ``genus``, ``family``, ``order``, ``class``, ``phylum``, ``kingdom``: Taxonomic hierarchy
        - ``gbifID``: GBIF (Global Biodiversity Information Facility) identifier

    **Audio File Paths:**
        - ``relative_path``: Path to original audio relative to data_root (variable sample rate)
        - ``gcs_path``: Full GCS path to original audio
        - ``32khz_path``: Path to pre-resampled 32kHz audio (if available)

    **Recording Metadata:**
        - ``eventDate``, ``eventTime``: When the recording was made
        - ``behavior``: Behavior being recorded (e.g., "calling song")
        - ``sex``: Sex of the recorded animal(s)
        - ``lifeStage``: Life stage (e.g., "adult")

    **Location:**
        - ``latitudeDecimal``, ``longitudeDecimal``: GPS coordinates
        - ``locality``, ``location``: Geographic location information

    **Rights & Attribution:**
        - ``rightsHolder``: Copyright holder
        - ``license``, ``license_text``: License information (e.g., CC BY-SA 4.0)
        - ``url``, ``associatedMedia``: Original Xeno-canto sound URL

    **Additional Fields:**
        - ``fieldNotes``, ``description``: Observer's notes about the recording
        - ``caption``, ``caption2``: Recording captions
        - ``xc_id``: Xeno-canto recording ID
        - ``dataset``, ``source_version``: Data source information

    References
    ----------
    Xeno-canto: https://www.xeno-canto.org/

    Examples
    --------
    >>> from esp_data.datasets import XenoCanto
    >>> dataset = XenoCanto(
    ...     split="train",
    ...     output_take_and_give={"canonical_name": "species"}
    ... )
    >>> print(dataset.info.name)
    xeno-canto
    >>> print(dataset.available_sample_rates)
    [32000]

    # Load with pre-resampled 32kHz audio (when available)
    >>> dataset_32k = XenoCanto(split="train", sample_rate=32000)

    # Load with on-the-fly resampling to 16kHz from original (variable rate) files
    >>> dataset_16k = XenoCanto(split="train", sample_rate=16000)
    """

    info = DatasetInfo(
        name="xeno-canto",
        owner="david; gagan",
        split_paths={
            "train": "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/xeno_curated_train.csv",
            "validation": "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/xeno_curated_val.csv",
            "all": "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/xeno_curated_all.csv",
        },
        version="0.1.0",
        description="Xeno-canto audio dataset with taxonomic metadata. "
        "Available at original (variable) sample rates and 32kHz (pre-resampled). "
        "Pre-resampled audio uses librosa's kaiser_best resampling method. "
        "Xeno-canto dump as of Oct 2025. "
        "Train/val split is 99%/1% with random seed 42.",
        sources=["Xeno-canto"],
        license="multiple (mostly CC BY-NC-SA 4.0, CC BY-NC 4.0, CC BY-SA, CC0)",
    )

    # Mapping of sample rates to their corresponding path columns
    _sample_rate_paths = {
        32000: "32khz_path",  # Pre-resampled to 32kHz
    }

    # Column name for original variable-rate audio files
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
        """Initialize the Xeno-canto dataset.

        Parameters
        ----------
        split : str, default="train"
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str], optional
            A dictionary mapping the original column names to the new column names.
        sample_rate : int, optional
            The sample rate to which audio files should be resampled. If the requested
            sample rate is available as pre-resampled audio (see `available_sample_rates`),
            the pre-resampled version will be loaded directly. Otherwise, audio will be
            resampled on-the-fly from the original files (at variable sample rates) using
            librosa's kaiser_best method. If None, audio is returned at its original
            (variable) sample rate.
        data_root : str | AnyPathT, optional
            The root directory for the dataset. This is prepended to the path
            column value to construct the full path to audio files. If None, defaults
            to the GCS bucket path for this dataset.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self._load()
        self.sample_rate = sample_rate

        if data_root is None:
            self.data_root = anypath("gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/")
            self._data_root_32k = anypath("gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/audio_32k/")
        else:
            self.data_root = anypath(data_root)
            self._data_root_32k = anypath(data_root)

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
            List of sample rates (in Hz) for which pre-resampled audio is available.
            Audio at these sample rates can be loaded directly without on-the-fly resampling.
            This checks which path columns actually exist in the loaded data.
        """
        available = []
        for sr, path_column in self._sample_rate_paths.items():
            # Check if the path column exists in the loaded data
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
                f"Invalid split: {self.split}."
                "Expected one of {list(self.info.split_paths.keys())}"
            )

        location = self.info.split_paths[self.split]
        # Read CSV directly from GCS path to avoid memory issues
        self._data = self._backend_class.from_csv(location, streaming=self._streaming)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["XenoCanto", dict[str, Any]]:
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
            The processed row.
        """
        # Determine which path column to use based on requested sample rate
        # If a pre-resampled version is available, use it; otherwise resample on-the-fly
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            # Check if the pre-resampled path column exists in the data
            if path_column in row and row[path_column] is not None and row[path_column] != "":
                # Use pre-resampled audio with appropriate data root
                audio_path = self._data_root_32k / row[path_column]
                use_presampled = True

        if use_presampled:
            audio, sr = read_audio(audio_path)
            audio = audio.astype(np.float32)
            audio = audio_stereo_to_mono(audio, mono_method="average")
            # Audio is already at the correct sample rate, no resampling needed
        else:
            # Use original variable-rate files and resample on-the-fly if needed
            # For original files, relative_path needs audio/ prefix if not already present
            rel_path = row[self._originals_path_column]
            if not rel_path.startswith("audio/"):
                audio_path = anypath(self.data_root) / "audio" / rel_path
            else:
                audio_path = anypath(self.data_root) / rel_path
            audio, sr = read_audio(audio_path)
            audio = audio.astype(np.float32)
            audio = audio_stereo_to_mono(audio, mono_method="average")

            if self.sample_rate is not None and sr != self.sample_rate:
                audio = librosa.resample(
                    y=audio,
                    orig_sr=sr,
                    target_sr=self.sample_rate,
                    scale=True,
                    res_type="kaiser_best",
                )

        row["audio"] = audio

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
            A string representation of the dataset including its name, version,
            and basic statistics if data is loaded.
        """
        base_info = f"{self.info.name} (v{self.info.version})"

        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
