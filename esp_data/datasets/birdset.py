"""BEANS dataset"""

from typing import Any, Dict, Iterator, Optional

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class BirdSet(Dataset):
    """BirdSet dataset

    Description
    -----------
    BirdSet a large-scale benchmark dataset for audio classification focusing on avian
    bioacoustics. BirdSet surpasses AudioSet with over 6,800 recording hours from nearly
    10,000 classes for training and more than 400 hours across eight strongly labeled
    evaluation datasets. It serves as a versatile resource for use cases such as
    multi-label classification, covariate shift or self-supervised learning.

    References
    ----------
    Birdset: A multi-task benchmark for classification in avian bioacoustics
    Rauch, Lukas, et al. "Birdset: A multi-task benchmark for classification in avian
    bioacoustics."
    https://github.com/DBD-research-group/BirdSet
    https://arxiv.org/abs/2403.10380

    Examples
    -------
    >>> from esp_data.datasets import BirdSet
    >>> dataset = BirdSet(
    ...     split="HSN-test",
    ...     output_take_and_give={"label": "label"},
    ...     sample_rate=16000,
    ...     data_root="gs://foundation-model-data/"
    ... )
    """

    info = DatasetInfo(
        name="birdset",
        owner="gagan",
        split_paths={
            "HSN-train": "gs://foundation-model-data/data/birdset-train/HSN/HSN_taxonomic.jsonl",
            "HSN-validation": "gs://foundation-model-data/data/birdset-train/HSN/HSN_taxonomic.jsonl",
            "HSN-test": "gs://foundation-model-data/data/birdset-test/HSN/HSN_taxonomic.jsonl",
            "NBP-train": "gs://foundation-model-data/data/birdset-train/NBP/NBP_taxonomic.jsonl",
            "NBP-validation": "gs://foundation-model-data/data/birdset-train/NBP/NBP_taxonomic.jsonl",
            "NBP-test": "gs://foundation-model-data/data/birdset-test/NBP/NBP_taxonomic.jsonl",
            "NES-train": "gs://foundation-model-data/data/birdset-train/NES/NES_taxonomic.jsonl",
            "NES-validation": "gs://foundation-model-data/data/birdset-train/NES/NES_taxonomic.jsonl",
            "NES-test": "gs://foundation-model-data/data/birdset-test/NES/NES_taxonomic.jsonl",
            "PER-train": "gs://foundation-model-data/data/birdset-train/PER/PER_taxonomic.jsonl",
            "PER-validation": "gs://foundation-model-data/data/birdset-train/PER/PER_taxonomic.jsonl",
            "PER-test": "gs://foundation-model-data/data/birdset-test/PER/PER_taxonomic.jsonl",
            "POW-train": "gs://foundation-model-data/data/birdset-train/POW/POW_taxonomic.jsonl",
            "POW-validation": "gs://foundation-model-data/data/birdset-train/POW/POW_taxonomic.jsonl",
            "POW-test": "gs://foundation-model-data/data/birdset-test/POW/POW_taxonomic.jsonl",
        },
        version="0.1.0",
        description="BirdSet dataset",
        sources=["HSN", "NBP", "NES", "PER", "POW"],
        license="CC-BY-4.0, CC0",
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] = None,
        sample_rate: Optional[int] = None,
        data_root: Optional[str | AnyPathT] = None,
    ) -> None:
        """Initialize the BirdSet dataset.

        Parameters
        ----------
        split : str
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        sample_rate : int
            The sample rate to which audio files should be resampled.
        data_root : Optional[str | AnyPathT]
            The root directory where the dataset is stored.
            If None, it will use the default path from the DatasetInfo.
        """
        super().__init__(output_take_and_give)  # Initialize the parent Dataset class
        self.split = split
        self._data: pd.DataFrame = None
        self._load()  # Load the dataset (fills self._data)
        self.sample_rate = sample_rate
        self.data_root = data_root
        if self.data_root is None:
            # we assume that parent dir of the split path is the data root
            self.data_root = anypath(self.info.split_paths[self.split]).parent

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
                f"Invalid split: {self.split}.Expected one of {list(self.info.split_paths.keys())}"
            )

        location = self.info.split_paths[self.split]
        if anypath(location).suffix == ".jsonl":
            # For JSONL files, read them directly into a DataFrame
            self._data = pd.read_json(location, lines=True, orient="records")
        else:
            # Read CSV content
            self._data = pd.read_csv(
                location, keep_default_na=False, na_values=[""]
            )  # This setting avoids setting 'None' to a pd.NA type

    @classmethod
    def from_config(cls, cfg: DatasetConfig) -> "BirdSet":
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        cfg : DatasetConfig
            Configuration dictionary containing dataset parameters

        Returns
        -------
        Dataset
            An instance of the Dataset class.

        Raises
        -------
        LookupError
            If the specified split is not available in the dataset info.
        """
        cfg = cfg.model_dump(exclude=("dataset_name", "transformations"))

        split = cfg.get("split", None)
        if not split or split not in cls.info.split_paths:
            raise LookupError(
                f"Invalid split '{split}'."
                f"Available splits: {', '.join(cls.info.split_paths.keys())}"
            )

        return cls(
            split=split,
            output_take_and_give=cfg.get("output_take_and_give", None),
            data_root=cfg.get("data_root"),
            sample_rate=cfg["sample_rate"],
        )

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
            raise RuntimeError("No split has been loaded yet. Call load() first.")
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
            A dictionary containing the data.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self._data)}.")

        row = self._data.iloc[idx].to_dict()
        # Ensure audio path is valid
        if self.data_root:
            audio_path = anypath(self.data_root) / row["path"]
        else:
            audio_path = anypath(row["path"])

        # Read the audio clip
        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)
        # Stereo to mono if necessary.
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

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        Dict[str, Any]
            Each sample in the dataset.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call load() first.")

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
