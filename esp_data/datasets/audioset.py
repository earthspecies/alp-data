"""AudioSet dataset"""

import json
from typing import Any, Dict, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.dataset import register_config
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_config
class AudioSetConfig(DatasetConfig):
    """Configuration for the AudioSet dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset. Must be "audioset".
    split : str
        The split to load. One of AudioSet.info.split_paths keys.
    version : str | None
        The version of the dataset to use. If None, uses DEFAULT_VERSION.
        Available versions: "0.1.0", "0.2.0"
    output_take_and_give : dict[str, str] | None
        A dictionary mapping the original column names to the new column names.
        It acts as a filter as well.
    sample_rate : int | None
        The sample rate to which audio files should be resampled. For v0.2.0, if
        sample_rate=32000, pre-resampled audio will be loaded directly (faster).
    data_root : str | AnyPathT | None
        The root directory for the dataset. This is optionally appended to the
        path item of a sample in the dataset.
    backend : BackendType
        The backend to use ("pandas" or "polars"), by default "polars"
    streaming : bool
        Whether to use streaming mode, by default False
    """

    dataset_name: str = "audioset"
    split: str = "train"
    version: str | None = None
    output_take_and_give: dict[str, str] | None = None
    sample_rate: int | None = None
    data_root: str | AnyPathT | None = None
    backend: BackendType = "polars"
    streaming: bool = False


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
    Note these are empty with the exception of the unbalanced_train split of the V1 dataset.

    Note that AudioSet contains different files depending on YouTube video availability at
    time of download. Version 0.1.0 contains a dump of AudioSet pulled in 2021 and resampled
    to 16khz. Version 0.2.0 contains a larger set of audios pulled from this HuggingFace
    release https://huggingface.co/datasets/agkphysics/AudioSet and maintaining the sample
    rates of the original files.

    Pre-resampled Audio
    -------------------
    Version 0.2.0 includes pre-resampled 32kHz audio that can be loaded directly
    without on-the-fly resampling for faster data loading:

    Load with pre-resampled 32kHz audio (v0.2.0, no resampling needed)
    >>> dataset_32k = AudioSet(split="validation", version="0.2.0", sample_rate=32000,
    ... streaming=True)
    >>> print(dataset_32k.available_sample_rates)
    [32000]

    Load with on-the-fly resampling to 16kHz
    >>> dataset_16k = AudioSet(split="validation", version="0.2.0", sample_rate=16000,
    ... streaming=True)

    Examples
    --------
    >>> from esp_data.datasets import AudioSet
    >>> dataset = AudioSet(
    ...     split="train",
    ...     output_take_and_give={"label": "audio_label"},
    ...     version="0.1.0",
    ...     streaming=True
    ... )
    >>> print(dataset.info.name)
    audioset
    """

    # Version registry with version-specific configurations
    VERSIONS = {
        "0.1.0": {
            "split_paths": {
                "train": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/unbalanced_train_segments_processed.csv",
                "train-balanced": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/balanced_train_segments_processed.csv",
                "validation": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/eval_segments_processed.csv",
            },
            "data_root": "gs://esp-ml-datasets/audioset/v0.1.0/raw/",
        },
        "0.2.0": {
            "split_paths": {
                "train": "gs://esp-ml-datasets/audioset/v0.2.0/raw/csv-data/unbalanced_train_segments_processed.csv",
                "validation": "gs://esp-ml-datasets/audioset/v0.2.0/raw/csv-data/eval_segments_processed.csv",
                "train-environmental": "gs://esp-ml-datasets/audioset/v0.2.0/raw/csv-data/unbalanced_train_environmental_sounds.csv",
            },
            "data_root": "gs://esp-ml-datasets/audioset/v0.2.0/raw/",
        },
    }

    # Default version (keep as 0.1.0 if we want backward compatibility)
    DEFAULT_VERSION = "0.1.0"

    info = DatasetInfo(
        name="audioset",
        owner="david; marius; masato",
        split_paths={},  # Will be populated based on version
        version="0.1.0",  # Default version
        description="AudioSet dataset",
        sources=["YouTube"],
        license="CC BY 4.0",
    )

    # Mapping of sample rates to their corresponding path columns
    # Pre-resampled audio is available for v0.2.0 only
    _sample_rate_paths = {
        32000: "32khz_path",  # Pre-resampled to 32kHz (v0.2.0 only)
    }

    # Column name for original variable-rate audio files
    _originals_path_column = "local_path"

    def __init__(
        self,
        split: str = "train",
        version: str | None = None,
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the AudioSet dataset.

        Parameters
        ----------
        split : str
            The split to load. One of info.split_paths keys.
        version : str, optional
            The version of the dataset to use. If None, uses DEFAULT_VERSION.
            Available versions: "0.1.0", "0.2.0"
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        sample_rate : int, optional
            The sample rate to which audio files should be resampled. For v0.2.0, if
            sample_rate=32000, pre-resampled audio will be loaded directly (faster).
            Otherwise, audio will be resampled on-the-fly from the original files using
            librosa's kaiser_best method. If None, audio is returned at its original
            sample rate.
        data_root : str | AnyPathT, optional
            The root directory for the dataset. This is optionally appended to the
            path item of a sample in the dataset.
            If None, uses the default data_root for the specified version.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False

        Raises
        ------
        ValueError
            If the specified version is not available.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        # Copy class-level DatasetInfo to avoid cross-instance mutation (versions/splits)
        self.info = self.info.model_copy(deep=True)

        # Handle version selection
        if version is None:
            version = self.DEFAULT_VERSION

        if version not in self.VERSIONS:
            raise ValueError(
                f"Version '{version}' is not available. "
                f"Available versions: {list(self.VERSIONS.keys())}"
            )

        self.version = version
        self.version_config = self.VERSIONS[version]

        # Update info with version-specific split paths
        self.info.split_paths = self.version_config["split_paths"]
        self.info.version = version

        self.split = split
        self._data = None
        self._load()  # Load the dataset (fills self._data)
        self.sample_rate = sample_rate

        if data_root is None:
            self.data_root = anypath(self.version_config["data_root"])
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
            Note: Pre-resampled audio is only available for v0.2.0.
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

        self._data = self._backend_class.from_csv(location, streaming=self._streaming)

    @classmethod
    def from_config(cls, dataset_config: AudioSetConfig) -> tuple["AudioSet", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : AudioSetConfig
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
            version=cfg.get("version"),
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
        """
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of the dataset.

        Returns
        -------
        dict[str, Any]
            Processed row with audio loaded and labels parsed.
        """
        # Parse JSON-encoded labels if present
        if "labels" in row:
            v = row["labels"]
            if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
                row["labels"] = []
            elif isinstance(v, str):
                # Labels are stored as JSON arrays of strings
                row["labels"] = json.loads(v)

        # Determine which path column to use based on requested sample rate
        # If a pre-resampled version is available, use it; otherwise resample on-the-fly
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if (
                path_column in row
                and row[path_column] not in (None, "")
                and not (isinstance(row[path_column], float) and np.isnan(row[path_column]))
            ):
                audio_path = anypath(self.data_root) / str(row[path_column])
                use_presampled = True

        if use_presampled:
            audio, sample_rate = read_audio(audio_path)
            audio = audio.astype(np.float32)
            audio = audio_stereo_to_mono(audio, mono_method="average")
        else:
            # Resample on-the-fly from original variable-rate audio
            audio_path = anypath(self.data_root) / str(row[self._originals_path_column])
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
            item: dict[str, Any] = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
            return item

        return row

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
        base_info = f"{self.info.name} (v{self.version})"

        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available versions: {', '.join(self.VERSIONS.keys())}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
