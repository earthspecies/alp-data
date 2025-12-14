"""Gelada vocal sequences dataset, Gustison et al 2016"""

from typing import Any, Dict, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class Geladas(Dataset):
    """Gelada vocal sequences follow Menzerath's linguistic law,
    Gustison et al 2016

    Description
    -----------
    Identifying universal principles underpinning diverse natural systems is a
    key goal of the life sciences. A powerful approach in
    addressing this goal has been to test whether patterns consistent
    with linguistic laws are found in nonhuman animals. Menzerath's
    law is a linguistic law that states that, the larger the construct, the
    smaller the size of its constituents. Here, to our knowledge, we
    present the first evidence that Menzerath's law holds in the vocal
    communication of a nonhuman species. We show that, in vocal
    sequences of wild male geladas (Theropithecus gelada), construct
    size (sequence size in number of calls) is negatively correlated with
    constituent size (duration of calls). Call duration does not vary
    significantly with position in the sequence, but call sequence composition does
    change with sequence size and most call types are
    abbreviated in larger sequences. We also find that intercall intervals follow the
    same relationship with sequence size as do calls.
    Finally, we provide formal mathematical support for the idea that
    Menzerath's law reflects compression—the principle of minimizing
    the expected length of a code. Our findings suggest that a common principle
    underpins human and gelada vocal communication,
    highlighting the value of exploring the applicability of linguistic
    laws in vocal systems outside the realm of language.

    References
    ----------
    Gelada vocal sequences follow Menzerath's linguistic law (PNAS, 2016)
    https://doi.org/10.1073/pnas.1522072113
    Also:
    Morgan L. Gustison, Thore J. Bergman, Divergent acoustic properties of gelada
    and baboon vocalizations and their implications for the evolution of human speech.
    Journal of Language Evolution.
    https://doi.org/10.1093/jole/lzx015

    Examples
    --------
    >>> from esp_data.datasets import Geladas
    >>> dataset = Geladas(
    ...     split="test",
    ...     output_take_and_give={"label": "label"},
    ...     sample_rate=16000,
    ...     streaming=True)
    """

    info = DatasetInfo(
        name="geladas",
        owner="gagan",
        split_paths={
            "all": "gs://esp-ml-datasets/geladas/v0.1.0/raw/geladas_annotations.csv",
        },
        version="0.1.0",
        description="Gelada vocal sequences dataset, Gustison et al 2016",
        sources=["PNAS"],
        license="CC-BY-4.0",
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
        """Initialize the Geladas dataset.

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["Geladas", dict[str, Any]]:
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
