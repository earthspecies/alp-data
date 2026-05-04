"""BEANS dataset"""

from typing import Any, Dict, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio
from esp_data.utils import DATA_HOME


@register_dataset
class Beans(Dataset):
    """BEANS dataset

    Description
    -----------
    BEANS (the BEnchmark of ANimal Sounds) is a collection of bioacoustics tasks
    and public datasets, specifically designed to measure the performance of machine
    learning algorithms in the field of bioacoustics. The benchmark proposed here
    consists of two common tasks in bioacoustics: classification and detection.
    It includes 12 datasets covering various species, including birds, land and
    marine mammals, anurans, and insects.

    References
    ----------
    BEANS: The Benchmark of Animal Sounds
    Masato Hagiwara et al 2022
    https://arxiv.org/abs/2210.12300
    https://github.com/earthspecies/beans

    Examples
    --------
    >>> from esp_data.datasets import Beans
    >>> dataset = Beans(
    ...     split="validation",
    ...     output_take_and_give={"species_scientific": "species"},
    ...     sample_rate=16000,
    ...     streaming=True,
    ... )
    """

    info = DatasetInfo(
        name="beans",
        owner="gagan",
        split_paths={
            "train": f"{DATA_HOME}/beans/v0.1.0/raw/beans_train_v3.csv",
            "validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/beans_val_v3.csv",
            "test": "gs://esp-ml-datasets/beans/v0.1.0/raw/beans_test_v3.csv",
            "cbi_test": "gs://esp-ml-datasets/beans/v0.1.0/raw/cbi_test.jsonl",
            "cbi_validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/cbi_val.jsonl",
            "cbi_train": "gs://esp-ml-datasets/beans/v0.1.0/raw/cbi_train.jsonl",
            "watkins_test": "gs://esp-ml-datasets/beans/v0.1.0/raw/watkins_test.jsonl",
            "watkins_validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/watkins_val.jsonl",
            "watkins_train": "gs://esp-ml-datasets/beans/v0.1.0/raw/watkins_train.jsonl",
            "dogs_test": "gs://esp-ml-datasets/beans/v0.1.0/raw/dogs_test.jsonl",
            "dogs_validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/dogs_val.jsonl",
            "dogs_train": "gs://esp-ml-datasets/beans/v0.1.0/raw/dogs_train.jsonl",
            "egyptian_fruit_bats_test": "gs://esp-ml-datasets/beans/v0.1.0/raw/egyptian_fruit_bats_test.jsonl",
            "egyptian_fruit_bats_validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/egyptian_fruit_bats_val.jsonl",
            "egyptian_fruit_bats_train": "gs://esp-ml-datasets/beans/v0.1.0/raw/egyptian_fruit_bats_train.jsonl",
            "hiceas_test": "gs://esp-ml-datasets/beans/v0.1.0/raw/hiceas_test.jsonl",
            "hiceas_validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/hiceas_val.jsonl",
            "hiceas_train": "gs://esp-ml-datasets/beans/v0.1.0/raw/hiceas_train.jsonl",
            "dcase_test": "gs://esp-ml-datasets/beans/v0.1.0/raw/dcase_test.jsonl",
            "dcase_validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/dcase_val.jsonl",
            "dcase_train": "gs://esp-ml-datasets/beans/v0.1.0/raw/dcase_train.jsonl",
            "enabirds_test": "gs://esp-ml-datasets/beans/v0.1.0/raw/enabirds_test.jsonl",
            "enabirds_validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/enabirds_val.jsonl",
            "enabirds_train": "gs://esp-ml-datasets/beans/v0.1.0/raw/enabirds_train.jsonl",
            "esc50_test": "gs://esp-ml-datasets/beans/v0.1.0/raw/esc50_test.jsonl",
            "esc50_validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/esc50_val.jsonl",
            "esc50_train": "gs://esp-ml-datasets/beans/v0.1.0/raw/esc50_train.jsonl",
            "speech_commands_test": "gs://esp-ml-datasets/beans/v0.1.0/raw/speech_commands_test_v2.jsonl",
            "speech_commands_validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/speech_commands_val_v2.jsonl",
            "speech_commands_train": "gs://esp-ml-datasets/beans/v0.1.0/raw/speech_commands_train_v2.jsonl",
            "humbugdb_test": "gs://esp-ml-datasets/beans/v0.1.0/raw/humbugdb_test.jsonl",
            "humbugdb_validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/humbugdb_val.jsonl",
            "humbugdb_train": "gs://esp-ml-datasets/beans/v0.1.0/raw/humbugdb_train.jsonl",
            "rfcx_test": "gs://esp-ml-datasets/beans/v0.1.0/raw/rfcx_test.jsonl",
            "rfcx_validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/rfcx_val.jsonl",
            "rfcx_train": "gs://esp-ml-datasets/beans/v0.1.0/raw/rfcx_train.jsonl",
            "hainan_gibbons_test": "gs://esp-ml-datasets/beans/v0.1.0/raw/hainan_gibbons_test.jsonl",
            "hainan_gibbons_validation": "gs://esp-ml-datasets/beans/v0.1.0/raw/hainan_gibbons_val.jsonl",
            "hainan_gibbons_train": "gs://esp-ml-datasets/beans/v0.1.0/raw/hainan_gibbons_train.jsonl",
        },
        version="0.1.0",
        description="BEANS benchmark dataset",
        sources=[
            "cbi",
            "watkins",
            "dogs",
            "egyptian_fruit_bats",
            "hiceas",
            "dcase",
            "enabirds",
            "esc50",
            "speech_commands",
            "humbugdb",
            "rfcx",
            "hainan_gibbons",
        ],
        license="CC-BY-4.0, CC0",
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
        """Initialize the BEANS dataset.

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
        self._data: pd.DataFrame = None
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
            self._data = self._backend_class.from_json(location, lines=True, orient="records")
        else:
            # Read CSV content
            self._data = self._backend_class.from_csv(
                location, keep_default_na=False, na_values=[""], null_values=[]
            )  # This setting avoids setting 'None' to a pd.NA type

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["Beans", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parametesf

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

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        audio_path = anypath(self.data_root) / row["local_path"]

        # Read the audio clip
        audio, sample_rate = read_audio(audio_path)
        audio = audio.astype(np.float32)
        # Stereo to mono if necessary.
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
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
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
