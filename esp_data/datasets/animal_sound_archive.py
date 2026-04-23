"""Animal Sound Archive dataset"""

from typing import Any, Dict, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class AnimalSoundArchive(Dataset):
    """Animal Sound Archive (Tierstimmenarchiv) audio dataset.

    Description
    -----------
    The Tierstimmenarchiv (Animal Sound Archive) at the Museum für Naturkunde
    Berlin hosts ~46k downloadable recordings covering birds, mammals, insects,
    amphibians, and other taxa. Recordings are linked to GBIF backbone taxonomy.

    Audio is available as original MP3 files (variable sample rate) and
    pre-resampled WAV at 16kHz and 32kHz using librosa's kaiser_best method.

    Available Metadata Fields
    -------------------------
    **Taxonomic Information:**
        - ``canonical_name``: Canonical species name from GBIF (primary identifier)
        - ``species_scientific``: Scientific species name
        - ``species_common``: Common/vernacular species name (enriched via GBIF, ~99%)
        - ``genus``, ``family``, ``order``, ``class``, ``phylum``, ``kingdom``: Taxonomic hierarchy
        - ``gbifID``: GBIF (Global Biodiversity Information Facility) identifier

    **Audio File Paths:**
        - ``originals_path``: Path to original MP3 audio relative to data root
        - ``32khz_path``: Path to pre-resampled 32kHz WAV audio relative to data root
        - ``16khz_path``: Path to pre-resampled 16kHz WAV audio relative to data root

    **Recording Metadata:**
        - ``tsa_id``: Tierstimmenarchiv unique identifier
        - ``eventDate``: When the recording was made (~83%)
        - ``eventTime``: Time of recording (~45%)
        - ``soundType``: Type of sound, e.g. "song", "call" (~88%)
        - ``soundQuality``: Recording quality assessment (~45%)
        - ``duration_seconds``: Recording duration in seconds (~94%)
        - ``sex``: Sex of the recorded animal(s) (~39%)
        - ``lifeStage``: Life stage, e.g. "adult" (~40%)
        - ``backgroundSpecies``: Species audible in the background (~22%)

    **Location:**
        - ``latitudeDecimal``, ``longitudeDecimal``: GPS coordinates (~56%)
        - ``locality``: Geographic location (~89%)
        - ``country``: Country (~93%)
        - ``habitat``: Habitat description (~23%)

    **Rights & Attribution:**
        - ``recordist``: Person who made the recording (~100%)
        - ``license``, ``media_license``: License information (mostly CC-BY-NC-SA, some CC BY-NC-SA)
        - ``url``: Original archive download URL

    **Additional Fields:**
        - ``occurrenceRemarks``: Description of the recording in German (~95%)
        - ``occurrenceRemarks_en``: Description in English (~81%)
        - ``fieldNotes``: Observer's field notes (~70%)
        - ``weather``: Weather conditions during recording (~35%)
        - ``recordingEquipment``: Equipment used (~81%)

    Available Splits
    ----------------
    - ``train``: Training set (all minus 3000 held-out samples, random split)
    - ``validation``: Validation set (3000 samples, random split)
    - ``all``: Complete dataset (train + validation)
    - ``train_excl_beanszero``: Training set excluding taxa evaluated in BEANS-Zero benchmark
    - ``validation_excl_beanszero``: Validation set excluding taxa evaluated in BEANS-Zero benchmark
    - ``all_excl_beanszero``: Complete dataset excluding BEANS-Zero taxa

    References
    ----------
    Tierstimmenarchiv: https://www.tierstimmenarchiv.de/

    Examples
    --------
    >>> from esp_data.datasets import AnimalSoundArchive
    >>> dataset = AnimalSoundArchive(
    ...     split="train",
    ...     output_take_and_give={"canonical_name": "species"},
    ...     streaming=True
    ... )
    >>> print(dataset.info.name)
    animal-sound-archive
    >>> print(dataset.available_sample_rates)
    [32000, 16000]

    >>> dataset_32k = AnimalSoundArchive(split="train", sample_rate=32000, streaming=True)
    """

    info = DatasetInfo(
        name="animal-sound-archive",
        owner="david",
        split_paths={
            "train": "gs://esp-ml-datasets/tierstimmenarchiv/v0.1.0/raw/train_v2.csv",
            "validation": "gs://esp-ml-datasets/tierstimmenarchiv/v0.1.0/raw/val_v2.csv",
            "all": "gs://esp-ml-datasets/tierstimmenarchiv/v0.1.0/raw/all_v2.csv",
            "train_excl_beanszero": "gs://esp-ml-datasets/tierstimmenarchiv/v0.1.0/raw/train_unseen_v2.csv",
            "validation_excl_beanszero": "gs://esp-ml-datasets/tierstimmenarchiv/v0.1.0/raw/val_unseen_v2.csv",
            "all_excl_beanszero": "gs://esp-ml-datasets/tierstimmenarchiv/v0.1.0/raw/all_unseen_v2.csv",
        },
        version="0.1.0",
        description="Animal Sound Archive (Tierstimmenarchiv) audio dataset with "
        "taxonomic metadata. ~46k recordings of birds, mammals, insects, amphibians "
        "and other taxa from Museum für Naturkunde Berlin. "
        "Available at original (variable) sample rates, 16kHz, and 32kHz (pre-resampled). "
        "Pre-resampled audio uses librosa's kaiser_best resampling method. "
        "Train/val split: val_size=3000, random seed 42.",
        sources=["Tierstimmenarchiv (Museum für Naturkunde Berlin)"],
        license="mostly CC-BY-NC-SA (unversioned)",
    )

    _sample_rate_paths = {
        32000: "32khz_path",
        16000: "16khz_path",
    }

    _originals_path_column = "originals_path"

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the Animal Sound Archive dataset.

        Parameters
        ----------
        split : str, default="train"
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str], optional
            A dictionary mapping the original column names to the new column names.
        sample_rate : int, optional
            The sample rate to which audio files should be resampled. If the requested
            sample rate is available as pre-resampled audio (see ``available_sample_rates``),
            the pre-resampled version will be loaded directly. Otherwise, audio will be
            resampled on-the-fly from the original files (at variable sample rates) using
            librosa's kaiser_best method. If None, audio is returned at its original
            (variable) sample rate.
        data_root : str | AnyPathT, optional
            The root directory for the dataset. All path columns in the CSV are
            relative to this root. If None, defaults to the GCS bucket path.
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
            self.data_root = anypath("gs://esp-ml-datasets/tierstimmenarchiv/v0.1.0/raw/")
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
            List of sample rates (in Hz) for which pre-resampled audio is available.
            Audio at these sample rates can be loaded directly without on-the-fly resampling.
            This checks which path columns actually exist in the loaded data.
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
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["AnimalSoundArchive", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters.

        Returns
        -------
        tuple[AnimalSoundArchive, dict[str, Any]]
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
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if path_column in row and row[path_column] is not None and row[path_column] != "":
                audio_path = self.data_root / row[path_column]
                use_presampled = True

        if use_presampled:
            audio, sample_rate = read_audio(audio_path)
            audio = audio.astype(np.float32)
            audio = audio_stereo_to_mono(audio, mono_method="average")
        else:
            audio_path = self.data_root / row[self._originals_path_column]
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
