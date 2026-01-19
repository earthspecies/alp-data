"""Tree Pipit ID dataset"""

from typing import Any, Dict, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class PipitId(Dataset):
    """Tree Pipit individual ID dataset.

    Description
    -----------
    Vocalisations released by Stowell et al. for individual Tree Pipits males
    (Anthus trivialis). Provides both *within-year* and *across-year*
    evaluation schemes.

    This dataset includes train and test splits within year
    (train_within_year, test_within_year)
    and across year (train_across_year, test_across_year). Test within year tests
    on recordings from the same year as the training data, though different days,
    while test across year tests on recordings from different years, giving harder
    test conditions, with potential differences in acoustic environment or
    vocalisation characteristics.

    References
    ----------
    https://royalsocietypublishing.org/doi/10.1098/rsif.2018.0940
    Zenodo: https://zenodo.org/records/1413495

    Examples
    --------
    >>> from esp_data.datasets import PipitId
    >>> dataset = PipitId(
    ...     split="test_within_year",
    ...     sample_rate=16000,
    ...     data_root="gs://esp-ml-datasets/pipit_id/v0.1.0/raw/"
    ... )
    """

    info = DatasetInfo(
        name="pipit_id",
        owner="david",
        split_paths={
            "train_within_year": "gs://esp-ml-datasets/pipit_id/v0.1.0/raw/withinyear_fg_train.csv",
            "test_within_year": "gs://esp-ml-datasets/pipit_id/v0.1.0/raw/withinyear_fg_test.csv",
            "train_across_year": "gs://esp-ml-datasets/pipit_id/v0.1.0/raw/acrossyear_fg_train.csv",
            "test_across_year": "gs://esp-ml-datasets/pipit_id/v0.1.0/raw/acrossyear_fg_test.csv",
        },
        version="0.1.0",
        description="Individual identification of tree pipits (Anthus trivialis)",
        sources=["https://royalsocietypublishing.org/doi/10.1098/rsif.2018.0940"],
        license="CC-BY-4.0",
    )

    def __init__(
        self,
        split: str = "train_within_year",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the PipitId dataset.

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
        self._load()
        self.sample_rate = sample_rate

        if data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent
        else:
            self.data_root = data_root

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )

        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
            infer_schema_length=10000,
            columns=["local_path", "individual_id"],  # for polars
            usecols=["local_path", "individual_id"],  # for pandas
        )

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["PipitId", dict[str, Any]]:
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
        """

        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})

        # Do not include split in kwargs if not defined and let __init__ use the default
        kwargs = {
            "output_take_and_give": cfg["output_take_and_give"],
            "data_root": cfg["data_root"],
            "sample_rate": cfg["sample_rate"],
            "backend": cfg["backend"],
            "streaming": cfg["streaming"],
            "split": cfg["split"],
        }

        ds = cls(**kwargs)

        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("Dataset not loaded.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode for this dataset.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        audio_path = anypath(self.data_root) / row["local_path"]

        audio, sr = read_audio(audio_path)
        sample_rate = sr
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")
        if self.sample_rate and sr != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sample_rate = self.sample_rate
        row["audio"] = audio
        row["sample_rate"] = sample_rate
        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
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
        base_info = f"{self.info.name} (v{self.info.version}), split='{self.split}'"

        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
