"""AudioSet dataset"""

import json
from io import StringIO
from typing import Any, Dict, Iterator, Optional

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class AudioSet(Dataset):
    """AudioSet dataset.

    Description
    -----------
    AudioSet is largescale dataset of manually-annotated audio events that endeavors
    to bridge the gap in data availability between image and audio research.
    Using a carefully structured hierarchical ontology of 632 audio classes
    in 10 second segments of YouTube videos.

    References
    ----------
    AUDIO SET: AN ONTOLOGY AND HUMAN-LABELED DATASET FOR AUDIO EVENTS
    Gemmeke et al 2017
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45857.pdf

    The train and validation splits (balanced and unbalanced)
    correspond to the official ones in the paper (https://research.google.com/audioset/download.html).
    The train-animal, train-noise, validation-animal, and validation-noise splits
    are created for animal and non-animal (noise) classes in the ontology.

    The "caption" column contains the caption from AudioSetCaps when available.
    AudioSetCaps Paper: https://arxiv.org/abs/2411.18953
    AudioSetCaps Dataset: https://huggingface.co/datasets/baijs/AudioSetCaps
    Note these are empty with the exception of the unbalanced_train split.


    Examples
    -------
    >>> from esp_data.datasets import AudioSet
    >>> dataset = AudioSet(
    ...     split="train",
    ...     output_take_and_give={"label": "audio_label"}
    ... )
    >>> print(dataset.info.name)
    audioset
    """

    info = DatasetInfo(
        name="audioset",
        owner="david; marius; masato",
        split_paths={
            "train": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/unbalanced_train_segments_processed.csv",
            "train-balanced": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/balanced_train_segments_processed.csv",
            "validation": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/eval_segments_processed.csv",
            "train-animal": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/unbalanced_train_segments_processed_animal.csv",
            "validation-animal": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/eval_segments_processed_animal.csv",
            "train-noise": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/unbalanced_train_segments_processed_noise.csv",
            "validation-noise": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/eval_segments_processed_noise.csv",
        },
        version="0.1.0",
        description="AudioSet dataset",
        sources=["YouTube"],
        license="Mixed",
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] = None,
        sample_rate: Optional[int] = None,
        data_root: Optional[str | AnyPathT] = None,
    ) -> None:
        """Initialize the AudioSet dataset.

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
                f"Invalid split: {self.split}."
                "Expected one of {list(self.info.split_paths.keys())}"
            )

        location = self.info.split_paths[self.split]
        # Read CSV content
        csv_text = anypath(location).read_text(encoding="utf-8")

        # Converter to parse JSON-encoded labels into Python lists
        def parse_label(value: str) -> list:
            if pd.isna(value) or value == "":
                return []
            return json.loads(value)

        self._data = pd.read_csv(StringIO(csv_text), converters={"labels": parse_label})

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["AudioSet", dict[str, Any]]:
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

        Raises
        -------
        LookupError
            If the specified split is not available in the dataset info.
        """
        cfg = dataset_config.model_dump(exclude=("dataset_name", "transformations"))

        split = cfg.get("split", None)
        if not split or split not in cls.info.split_paths:
            raise LookupError(
                f"Invalid split '{split}'."
                f"Available splits: {', '.join(cls.info.split_paths.keys())}"
            )

        ds = cls(
            split=split,
            output_take_and_give=cfg.get("output_take_and_give", None),
            data_root=cfg.get("data_root"),
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

        # Ensure audio path is valid
        if self.data_root:
            audio_path = anypath(self.data_root) / row["local_path"]
        else:
            audio_path = anypath(row["local_path"])

        audio, sr = read_audio(audio_path, start_time=row["start"], end_time=row["end"])
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

        # AudioSet likes to call this 'raw_wav'
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
