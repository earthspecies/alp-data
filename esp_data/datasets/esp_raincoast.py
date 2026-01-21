"""ESP Raincoast.org dataset"""

from typing import Any, Dict, Iterator, Literal

import librosa
import numpy as np
import pandas as pd

from esp_data import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    register_config,
    register_dataset,
)
from esp_data.io import (
    AnyPathT,
    anypath,
    audio_stereo_to_mono,
    read_audio,
    read_text,
)


@register_config
class ESPRaincoastConfig(DatasetConfig):
    """Configuration for the ESP Raincoast dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset. Default is "esp_raincoast".
    split : str
        The data split to use. Default is "full".
    output_take_and_give : dict[str, str] | None
        A mapping of original column names to new column names.
        If None, all columns are kept with their original names.
    sample_rate : int | None
        The sample rate to which audio files should be resampled.
        If None, audio files are loaded at their original sample rate.
    load_audio_segments : bool
        If True, audio files will be spliced between 'Begin Time(s)' and 'End Time(s)'.
        If False, entire audio files will be loaded. Default is True.
    mono_method : str | None
        Method to convert stereo audio to mono. If None, no conversion is done.
        Options are ["keep_first", "average"]. Default is None.
    data_root : str | AnyPathT | None
        The root directory for the dataset. This is optionally appended to the
        path item of a sample in the dataset.
        If None, the default is the parent directory of the split path.
        Default is "gs://esp-raincoast/2023-2024".
    backend : str, optional
        The backend to use ("pandas" or "polars"), by default "polars"
    streaming : bool, optional
        Whether to use streaming mode, by default False
    """

    dataset_name: str = "esp_raincoast"
    split: str = "full"
    output_take_and_give: dict[str, str] | None = None
    sample_rate: int | None = None
    load_audio_segments: bool = True
    mono_method: Literal["keep_first", "average"] | None = None
    data_root: str | AnyPathT | None = "gs://esp-raincoast/2023-2024"
    backend: str = "polars"
    streaming: bool = False


@register_dataset
class ESPRaincoast(Dataset):
    """ESP Raincoast.org dataset
    Recorded by Dylan Smyth, Valeria Vergara lab.
    """

    info = DatasetInfo(
        name="esp_raincoast",
        owner="emmanuel; gagan; dylansmyth; maddie",
        split_paths={
            "full": "gs://esp-raincoast/2023-2024/full_selection_table.csv",
        },
        version="0.1.0",
        description="Orca vocal repertoire dataset",
        sources=["esp-raincoast"],
        license="private",
    )

    def __init__(
        self,
        split: str = "full",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        load_audio_segments: bool = True,
        mono_method: Literal["keep_first", "average"] | None = None,
        data_root: str | AnyPathT | None = None,
        backend: str = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the ESPRaincoast dataset.

        Parameters
        ----------
        split : str
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        sample_rate : int
            The sample rate to which audio files should be resampled.
        load_audio_segments : bool
            If True, the audio files will be spliced between the 'Begin time(s)'
            and 'End time (s)' columns in the dataset.
            If False, the entire audio file will be loaded.
        mono_method : str | None
            Method to convert stereo audio to mono. If None, no conversion is done.
        data_root : str | AnyPathT, optional
            The root directory for the dataset. This is optionally appended to the
            path item of a sample in the dataset.
            If None, the default is the parent directory of the split path.
        backend : str, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend, streaming)
        self.split = split
        self.sample_rate = sample_rate
        self.data_root = data_root
        self.load_audio_segments = load_audio_segments
        self.mono_method = mono_method

        if data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent
        else:
            self.data_root = data_root

        self._data: pd.DataFrame = None
        self._load()

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
            # self._data = pd.read_json(location, lines=True, orient="records")
            self._data = self._backend_class.from_json(
                location, lines=True, streaming=self._streaming
            )
        else:
            from io import StringIO

            csv_text = read_text(location, encoding="utf-8")
            # TODO: Polars picked up some inconsistencies in the data!
            # Column "Call Quality" has a mix of f64 and string types
            self._data = self._backend_class.from_csv(
                StringIO(csv_text), streaming=self._streaming, infer_schema_length=10000
            )

    @classmethod
    def from_config(
        cls, dataset_config: ESPRaincoastConfig
    ) -> tuple["ESPRaincoast", dict[str, Any]]:
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

        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            data_root=cfg["data_root"],
            sample_rate=cfg["sample_rate"],
            load_audio_segments=cfg["load_audio_segments"],
            mono_method=cfg["mono_method"],
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
            raise RuntimeError("No split has been loaded yet. Call load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode.Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        # Ensure audio path is valid
        audio_path = anypath(self.data_root) / row["local_path"]

        # Read the audio clip
        if self.load_audio_segments:
            start_time = row.get("Begin Time (s)", 0.0)
            end_time = row.get("End Time (s)", None)
            audio, sample_rate = read_audio(audio_path, start_time=start_time, end_time=end_time)
        else:
            audio, sample_rate = read_audio(audio_path)
        audio = audio.astype(np.float32)

        # Stereo to mono if necessary.
        if self.mono_method is not None:
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
        base_info = f"{self.info.name} (v{self.info.version}), split='{self.split}'"

        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
