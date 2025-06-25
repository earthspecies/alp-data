"""Macaques Coo Calls dataset"""

from typing import Any, Dict, Iterator, Optional

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class MacaquesCooCalls(Dataset):
    """Macaques Coo Calls dataset

    Description
    -----------
    Coo calls from male and female macaques (Macaca mulatta)
    including macaque id,sex, weight.

    References
    ----------

    https://archive.org/details/macaque_coo_calls

    Examples
    -------
    >>> from esp_data.datasets import MacaquesCooCalls
    >>> dataset = MacaquesCooCalls(
    ...     split="test",
    ...     output_take_and_give={"id": "label"},
    ...     sample_rate=16000,
    ...     data_root="gs://esp-ml-datasets/macaques_coo_calls/v0.1.0/raw/"
    ... )
    """

    info = DatasetInfo(
        name="macaques_coo_calls",
        owner="marius",
        split_paths={
            "test": "gs://esp-ml-datasets/macaques_coo_calls/v0.1.0/raw/test.csv",
            "train": "gs://esp-ml-datasets/macaques_coo_calls/v0.1.0/raw/train.csv",
            "val": "gs://esp-ml-datasets/macaques_coo_calls/v0.1.0/raw/validation.csv",
        },
        version="0.1.0",
        description=(
            "Coo calls from male and female macaques (Macaca mulatta) including id, sex, weight_kg"
        ),
        sources=["archive.org"],
        license="CC0 1.0 Universal",
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] = None,
        sample_rate: Optional[int] = None,
        data_root: Optional[str | AnyPathT] = None,
    ) -> None:
        """Initialize the Macaques Coo Calls dataset.

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
        self.sample_rate = sample_rate
        self.data_root = data_root
        if self.data_root is None:
            # we assume that parent dir of the csv_data directory is the data root
            # The split path is: .../raw/csv_data/test.csv
            # We want the data root to be: .../raw/
            split_path = anypath(self.info.split_paths[self.split])
            self.data_root = split_path.parent.parent

        self._data: pd.DataFrame = None
        self._load()  # Load the dataset (fills self._data)

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
        self._data = pd.read_csv(location)

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["MacaquesCooCalls", dict[str, Any]]:
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
            and the metadata will be returned as dict, otherwise empty dict.

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
        if idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self._data)}.")

        row = self._data.iloc[idx].to_dict()
        # Ensure audio path is valid
        if self.data_root:
            # If data_root is provided, we need to construct the path properly
            # The local_path in CSV might be absolute, so we extract just the filename
            # and construct the path relative to data_root
            local_path = anypath(row["local_path"])
            filename = local_path.name

            # Check if the file exists in the train/valid subdirectories
            # Try train first, then valid
            if self.split == "train":
                audio_path = anypath(self.data_root) / "audio" / "train" / filename
            else:
                audio_path = anypath(self.data_root) / "audio" / "valid" / filename
        else:
            audio_path = anypath(row["local_path"])

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
