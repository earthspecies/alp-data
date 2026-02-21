"""AnimalSpeak dataset"""

from typing import Any, Dict, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio
from esp_data.schema import ColumnSchema, DatasetSchema


@register_dataset
class AnimalSpeak(Dataset):
    """AnimalSpeak dataset.

    Description
    -----------
    A part of NatureLM training and BioLingual, AnimalSpeak,
    as over a million audio-caption pairs holding information on
    species, vocalization context, and animal behavior.

    References
    ----------
    TRANSFERABLE MODELS FOR BIOACOUSTICS WITH HUMAN LANGUAGE SUPERVISION
    Robinson et al 2023
    https://arxiv.org/pdf/2308.04978

    Examples
    --------
    >>> from esp_data.datasets import AnimalSpeak
    >>> dataset = AnimalSpeak(
    ...     split="validation",
    ...     output_take_and_give={"species_common": "comm"}
    ... )
    >>> print(dataset.info.name)
    animalspeak
    """

    info = DatasetInfo(
        name="animalspeak",
        owner="david; marius; masato",
        split_paths={
            "train": "gs://esp-ml-datasets/animalspeak/v0.1.0/raw/16KHz/animalspeak2_train.csv",
            "validation": "gs://esp-ml-datasets/animalspeak/v0.1.0/raw/16KHz/animalspeak2_validation.csv",
        },
        version="0.1.0",
        description="AnimalSpeak dataset",
        sources=["Xeno-canto", "iNaturalist", "Watkins"],
        license="CC BY",
    )

    schema = DatasetSchema(
        columns=[
            ColumnSchema(
                name="Associated Taxa",
                dtype="str",
                required=False,
                description="Associated taxa from GBIF occurrence record",
            ),
            ColumnSchema(
                name="audiocap_id",
                dtype="float",
                required=False,
                description="AudioCaps dataset identifier",
            ),
            ColumnSchema(
                name="background_family",
                dtype="str",
                required=False,
                description="Taxonomic family of background species",
            ),
            ColumnSchema(
                name="background_genus",
                dtype="str",
                required=False,
                description="Taxonomic genus of background species",
            ),
            ColumnSchema(
                name="background_species_common",
                dtype="str",
                required=False,
                description="Common name of background species",
            ),
            ColumnSchema(
                name="background_species_sci",
                dtype="str",
                required=False,
                description="Scientific name of background species",
            ),
            ColumnSchema(
                name="background_taxonomic",
                dtype="str",
                required=False,
                description="Taxonomic classification of background species",
            ),
            ColumnSchema(
                name="behavior",
                dtype="str",
                required=False,
                description="Behavioral context of the vocalization",
            ),
            ColumnSchema(
                name="canonical_name",
                dtype="str",
                required=False,
                description="Canonical species name from GBIF",
            ),
            ColumnSchema(
                name="caption",
                dtype="str",
                required=True,
                description="Primary text caption describing the audio",
            ),
            ColumnSchema(
                name="caption2", dtype="str", required=False, description="Secondary text caption"
            ),
            ColumnSchema(
                name="caption3", dtype="str", required=False, description="Tertiary text caption"
            ),
            ColumnSchema(name="class", dtype="str", required=False, description="Taxonomic class"),
            ColumnSchema(
                name="eventDate",
                dtype="str",
                required=False,
                description="Date the recording was made",
            ),
            ColumnSchema(
                name="eventTime",
                dtype="str",
                required=False,
                description="Time the recording was made",
            ),
            ColumnSchema(
                name="family", dtype="str", required=False, description="Taxonomic family"
            ),
            ColumnSchema(
                name="fieldNotes",
                dtype="str",
                required=False,
                description="Field notes from the recording session",
            ),
            ColumnSchema(
                name="gbifID",
                dtype="float",
                required=False,
                description="GBIF occurrence record identifier",
            ),
            ColumnSchema(name="genus", dtype="str", required=False, description="Taxonomic genus"),
            ColumnSchema(
                name="identifier",
                dtype="str",
                required=False,
                description="Record identifier from source dataset",
            ),
            # TODO(#232): latitudeDecimal should be float, not str
            ColumnSchema(
                name="latitudeDecimal",
                dtype="str",
                required=False,
                description="Decimal latitude of recording location",
            ),
            ColumnSchema(
                name="license", dtype="str", required=False, description="License of the recording"
            ),
            ColumnSchema(
                name="lifeStage",
                dtype="str",
                required=False,
                description="Life stage of the animal (e.g. adult, juvenile)",
            ),
            ColumnSchema(
                name="local_path",
                dtype="str",
                required=True,
                description="Relative path to the audio file",
            ),
            # TODO(#232): longitudeDecimal should be float, not str
            ColumnSchema(
                name="longitudeDecimal",
                dtype="str",
                required=False,
                description="Decimal longitude of recording location",
            ),
            ColumnSchema(name="order", dtype="str", required=False, description="Taxonomic order"),
            ColumnSchema(
                name="path",
                dtype="str",
                required=False,
                description="Alternative path to the audio file",
            ),
            ColumnSchema(
                name="phylum", dtype="str", required=False, description="Taxonomic phylum"
            ),
            ColumnSchema(
                name="recordist",
                dtype="str",
                required=False,
                description="Name of the person who made the recording",
            ),
            ColumnSchema(
                name="rightsHolder",
                dtype="str",
                required=False,
                description="Rights holder of the recording",
            ),
            ColumnSchema(name="sex", dtype="str", required=False, description="Sex of the animal"),
            ColumnSchema(
                name="source",
                dtype="str",
                required=False,
                description="Source dataset (e.g. Xeno-canto, iNaturalist)",
            ),
            ColumnSchema(
                name="species_common",
                dtype="str",
                required=True,
                description="Common name of the species",
            ),
            ColumnSchema(
                name="species_scientific",
                dtype="str",
                required=True,
                description="Scientific name of the species",
            ),
            ColumnSchema(
                name="species_scientific_normalized",
                dtype="str",
                required=False,
                description="Normalized scientific name",
            ),
            ColumnSchema(
                name="start_time",
                dtype="float",
                required=False,
                description="Start time offset within the audio file in seconds",
            ),
            ColumnSchema(
                name="subspecies", dtype="str", required=False, description="Subspecies name"
            ),
            ColumnSchema(
                name="taxonomic_name",
                dtype="str",
                required=False,
                description="Full taxonomic name from GBIF",
            ),
            ColumnSchema(
                name="url", dtype="str", required=False, description="URL to the original recording"
            ),
            ColumnSchema(
                name="youtube_id",
                dtype="str",
                required=False,
                description="YouTube video identifier",
            ),
            ColumnSchema(
                name="country",
                dtype="str",
                required=False,
                description="Country where the recording was made",
            ),
            ColumnSchema(
                name="locality",
                dtype="str",
                required=False,
                description="Locality where the recording was made",
            ),
            ColumnSchema(
                name="verbatimElevation",
                dtype="str",
                required=False,
                description="Elevation as recorded in the original data",
            ),
            ColumnSchema(
                name="extracted_name",
                dtype="str",
                required=False,
                description="Species name extracted from metadata",
            ),
            ColumnSchema(
                name="cluster_path",
                dtype="str",
                required=False,
                description="Path to cluster assignment data",
            ),
            ColumnSchema(
                name="file_name", dtype="str", required=False, description="Original file name"
            ),
        ]
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the AnimalSpeak dataset.

        Parameters
        ----------
        split : str
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        sample_rate : int
            The sample rate to which audio files should be resampled.
        data_root : str | AnyPathT, optional
            The root directory for the dataset. This is optionally appended to the
            path item of a sample in the dataset.
            If None, the default is the parent directory of the split path.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self._load()  # Load the dataset (fills self._data)
        self.sample_rate = sample_rate

        if data_root is None:
            # TODO switch to gs://esp-ml-datasets/animalspeak once we're sure everything is there
            self.data_root = "gs://animalspeak2"
        else:
            self.data_root = data_root

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset."""
        return self._data.columns

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

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

        # TODO: Polars needs a lot of rows to figure out types correctly
        # which is why we set infer_schema_length here to 10,000
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
            infer_schema_length=10_000,
            keep_default_na=False,
            na_values=[""],
        )

        # Validate schema after load
        self._validate_schema()

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["AnimalSpeak", dict[str, Any]]:
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
                "Length is not available in streaming mode.Iterate over the dataset instead."
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
        # TODO (milad) this column shouldn't start with the bucket name because that is
        # essentially the root. We only need the relative paths there. Removing so that
        # audio_path assignment works with or without root
        # An example of the local_path: local_path = "animalspeak2/16khz/WavCaps/253918.flac"
        relative_path = row["local_path"].removeprefix("animalspeak2/")

        audio_path = anypath(self.data_root) / relative_path

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

        # AnimalSpeak likes to call this 'raw_wav'
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
            A dictionary containing the audio data, text label, label, and path.
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
        base_info = f"{self.info.name} (v{self.info.version}), split: {self.split}"

        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
