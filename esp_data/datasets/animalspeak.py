"""AnimalSpeak dataset"""

from io import StringIO
from typing import Any, Dict, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio, read_text


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
    -------
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

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
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
        """
        super().__init__(output_take_and_give)  # Initialize the parent Dataset class
        self.split = split
        self._data: pd.DataFrame = None
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
        return list(self._data.columns)

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
        # Read CSV content
        csv_text = read_text(location, encoding="utf-8")
        self._data = pd.read_csv(StringIO(csv_text))

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
        return len(self._data)

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

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self._data)}.")

        row = self._data.iloc[idx].to_dict()

        # TODO (milad) this column shouldn't start with the bucket name because that is
        # essentially the root. We only need the relative paths there. Removing so that
        # audio_path assignment works with or without root
        # An example of the local_path: local_path = "animalspeak2/16khz/WavCaps/253918.flac"
        relative_path = row["local_path"].removeprefix("animalspeak2/")

        audio_path = anypath(self.data_root) / relative_path

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

        # AnimalSpeak likes to call this 'raw_wav'
        row["audio"] = audio

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        return item

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        Dict[str, Any]
            Each sample in the dataset.
        """
        for idx in range(len(self)):
            yield self[idx]

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
