"""BarkleyCanyon dataset"""

from typing import Any, Dict, Iterator

import librosa
import numpy as np
import soundfile as sf

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import (
    AnyPathT,
    anypath,
    audio_stereo_to_mono,
    filesystem_from_path,
    read_audio,
)


@register_dataset
class BarkleyCanyon(Dataset):
    """BarkleyCanyon dataset.

    Description
    -----------
    Excerpt from the original Abstract:
    "...This dataset, which is being made publicly available for further use,
    includes strong-label annotations of phonations from blue whales, fin whales,
    humpback whales, sperm whales, orcas, Pacific white-sided dolphins,
    Risso's dolphins, and other delphinids that could not be identified to species.
    All regional orca communities are represented within the dataset,
    and phonations are labelled to ecotype and pod level when possible..."


    References
    ----------
    K. S. Kanes,
    "Recycling data: An annotated marine acoustic data set that
    is publicly available for use  in classifier development
    and marine mammal research,"
    The Journal of the Acoustical Society of America, vol. 148,

    Examples
    --------
    >>> from esp_data.datasets import BarkleyCanyon
    >>> dataset = BarkleyCanyon(split="train")
    >>> print(dataset.info.name)
    barkley_canyon
    """

    info = DatasetInfo(
        name="barkley_canyon",
        owner="gagan",
        split_paths={
            "train": "gs://esp-ml-datasets/barkley_canyon/v0.1.0/raw/barkley_canyon_annotations_non_null_gbif_v2.csv",
        },
        version="0.1.0",
        description="BarkleyCanyon dataset",
        sources=["BarkleyCanyon"],
        license="unknown",
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: str = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the BarkleyCanyon dataset.

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
        backend : str
            Backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend, streaming)
        self.split = split
        self._data = None
        self._load()  # Load the dataset (fills self._data)
        self.sample_rate = sample_rate

        if data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent
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
                f"Invalid split: {self.split}.Expected one of {list(self.info.split_paths.keys())}"
            )

        location = self.info.split_paths[self.split]
        # Read CSV content using backend
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
        )

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["BarkleyCanyon", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters

        Returns
        -------
        tuple[Dataset, dict[str, Any]]
            A tuple containing the dataset instance and metadata.
            If the dataset_config contains transformations, they will be applied
            and the metadata will be returned as dict, otherwise None.
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
            transform_metadata = ds.apply_transformations(
                dataset_config.transformations
            )
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
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row from the dataset.
        Parameters
        ----------
        row : dict[str, Any]
            A dictionary representing a single row from the dataset.
        Returns
        -------
        dict[str, Any]
            A dictionary containing the processed data.

        Raises
        ------
        ValueError
            If the start time is beyond the audio duration.
        """
        # Ensure audio path is valid
        audio_path = anypath(self.data_root) / row["local_path"]

        with filesystem_from_path(audio_path).open(str(audio_path), "rb") as f:
            info = sf.info(f)
            sr = info.samplerate
            start_frame = (
                int(row["start_times(sec)"] * sr)
                if row["start_times(sec)"] is not None
                else 0
            )
            end_frame = (
                int(row["end_times(sec)"] * sr)
                if row["end_times(sec)"] is not None
                else info.frames
            )
            frames_to_read = end_frame - start_frame

            if frames_to_read <= 0:
                raise ValueError(
                    f"start_time ({row['start_times(sec)']}s) is beyond the audio duration"
                )

        # Read the audio clip
        audio, sr = read_audio(
            audio_path,
            frames=frames_to_read,
            start=start_frame,
        )
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


@register_dataset
class BarkleyCanyonDetection(Dataset):
    """BarkleyCanyonDetection dataset.

    Processed version of BarkleyCanyon dataset with audio files resampled and windowed.

    See:
    https://github.com/earthspecies/foundation-model-data/blob/main/scripts/convert_barkley_canyon.py

    Examples
    --------
    >>> from esp_data.datasets import BarkleyCanyon
    >>> dataset = BarkleyCanyonDetection(split="train")
    >>> print(dataset.info.name)
    barkley_canyon_detection
    """

    info = DatasetInfo(
        name="barkley_canyon_detection",
        owner="gagan",
        split_paths={
            "train": "gs://esp-ml-datasets/barkley_canyon_detection/v0.1.0/raw/barkley_canyon_detection_gbif.csv",
        },
        version="0.1.0",
        description="""
        BarkleyCanyon detection dataset,
        processed version of BarkleyCanyon dataset
        with audio files resampled and windowed.

        See:
        https://github.com/earthspecies/foundation-model-data/blob/main/scripts/convert_barkley_canyon.py
        """,
        sources=["BarkleyCanyon"],
        license="unknown",
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: str = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the BarkleyCanyonDetection dataset.

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
        backend : str
            Backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend, streaming)
        self.split = split
        self._data = None
        self._load()  # Load the dataset (fills self._data)
        self.sample_rate = sample_rate
        self.data_root = data_root

        if data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent
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
                f"Invalid split: {self.split}.Expected one of {list(self.info.split_paths.keys())}"
            )

        location = self.info.split_paths[self.split]
        # Read CSV content using backend
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
        )

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["BarkleyCanyonDetection", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters

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
            transform_metadata = ds.apply_transformations(
                dataset_config.transformations
            )
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
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        # Ensure audio path is valid
        audio_path = anypath(self.data_root) / row["local_path"]

        # Read the audio clip
        audio, sr = read_audio(
            audio_path,
            frames=-1,
        )

        # Stereo to mono if necessary.
        # Find the channel dimension (typically the smaller dimension)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        audio = audio.astype(np.float32)

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
            A dictionary containing the data.
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
