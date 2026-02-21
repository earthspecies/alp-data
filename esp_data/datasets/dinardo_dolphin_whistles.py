"""Dolphin whistles dataset, Di Nardo et al 2023"""

from typing import Any, Dict, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio
from esp_data.schema import ColumnSchema, DatasetSchema


@register_dataset
class DinardoDolphinWhistles(Dataset):
    """Dolphin whistles dataset, Di Nardo et al 2023

    Description
    -----------
    Authors: Francesco Di Nardo,  Rocco De Marco, Alessandro Lucchetti & DavidScaradozzi
    Globally, interactions between fishing activities and dolphins are cause for concern
    due to their negative effects on both mammals and fishermen.
    The recording of acoustic emissions could aid in detecting the
    presence of dolphins in close proximity to fishing gear,
    elucidating their behavior, and guiding potential
    management measures designed to limit this harmful phenomenon.
    This data descriptor presents a dataset of acoustic recordings (WAV files) collected
    during interactions between common bottlenose dolphins (Tursiops truncatus) and
    fishing activities in the Adriatic Sea. This dataset is distinguished by the high
    complexity of its repertoire, which includes various different typologies of dolphin emission.
    Specifically, a group of free-ranging dolphins was found to emit frequency-modulated whistles,
    echolocation clicks, and burst pulse signals, including feeding buzzes.
    An analysis of signal quality based on the signal-to-noise ratio was
    conducted to validate the dataset. The signal digital files and corresponding features
    make this dataset suitable for studying
    dolphin behavior in order to gain a deeper understanding of their communication and
    interaction with fishing gear (trawl).

    References
    ----------
    A WAV file dataset of bottlenose dolphin whistles, clicks, and pulse
    sounds during trawling interactions
    https://doi.org/10.1038/s41597-023-02547-8


    Examples
    --------
    >>> from esp_data.datasets import DinardoDolphinWhistles
    >>> dataset = DinardoDolphinWhistles(
    ...     split="all",
    ...     sample_rate=16000,
    ...     streaming=True)
    """

    info = DatasetInfo(
        name="dinardo_dolphin_whistles",
        owner="gagan",
        split_paths={
            "all": "gs://esp-ml-datasets/dinardo2023_dolphin_whistles/v0.1.0/raw/dinardo2023_annotations.csv",
        },
        version="0.1.0",
        description="Dolphin whistles dataset, Di Nardo et al 2023",
        sources=["Nature Scientific Data"],
        license="CC-BY-4.0",
    )

    schema = DatasetSchema(
        columns=[
            ColumnSchema(
                name="local_path",
                dtype="str",
                required=True,
                description="Relative path to the audio file",
            ),
            ColumnSchema(
                name="file_name", dtype="str", required=False, description="Original file name"
            ),
            ColumnSchema(
                name="gbifID",
                dtype="int",
                required=False,
                description="GBIF occurrence record identifier",
            ),
            ColumnSchema(
                name="species_common",
                dtype="str",
                required=False,
                description="Common name of the species",
            ),
            ColumnSchema(
                name="species_scientific",
                dtype="str",
                required=False,
                description="Scientific name of the species",
            ),
            ColumnSchema(
                name="kingdom", dtype="str", required=False, description="Taxonomic kingdom"
            ),
            ColumnSchema(
                name="phylum", dtype="str", required=False, description="Taxonomic phylum"
            ),
            ColumnSchema(name="order", dtype="str", required=False, description="Taxonomic order"),
            ColumnSchema(name="class", dtype="str", required=False, description="Taxonomic class"),
            ColumnSchema(
                name="family", dtype="str", required=False, description="Taxonomic family"
            ),
            ColumnSchema(name="genus", dtype="str", required=False, description="Taxonomic genus"),
            ColumnSchema(
                name="taxonomic_name",
                dtype="str",
                required=False,
                description="Full taxonomic name from GBIF",
            ),
            ColumnSchema(
                name="call_type",
                dtype="str",
                required=False,
                description="Type of dolphin emission (whistle, click, or burst pulse)",
            ),
        ]
    )

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the DinardoDolphinWhistles dataset.

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
        self.sample_rate = sample_rate

        if data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent
        else:
            self.data_root = data_root

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
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["DinardoDolphinWhistles", dict[str, Any]]:
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
