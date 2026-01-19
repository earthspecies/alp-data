"""InfantMarmosetsVox dataset

Infant marmoset vocalizations dataset for call type and caller identification.
"""

from typing import Any, Dict, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

# Call type mapping (0-10, excluding 11=silence and 12=noise)
CALLTYPE_NAMES = {
    0: "Peep(Pre-Phee)",
    1: "Phee",
    2: "Twitter",
    3: "Trill",
    4: "Trillphee",
    5: "Tsik Tse",
    6: "Egg",
    7: "Pheecry(cry)",
    8: "TrllTwitter",
    9: "Pheetwitter",
    10: "Peep",
}


@register_dataset
class InfantMarmosetsVox(Dataset):
    """InfantMarmosetsVox dataset

    Description
    -----------
    InfantMarmosetsVox is a dataset for multi-class call-type and caller
    identification. It contains audio recordings of different individual
    marmosets and their call-types. The dataset contains a total of 350 files
    of precisely labelled 10-minute audio recordings across all caller classes.
    The audio was recorded from five pairs of infant marmoset twins, each
    recorded individually in two separate sound-proofed recording rooms at a
    sampling rate of 44.1 kHz. The start and end time, call-type, and marmoset
    identity of each vocalization are provided, labeled by an experienced
    researcher.

    Each entry in the dataset corresponds to a single vocalization segment,
    with the audio loaded from the corresponding time range in the source file.

    Labels
    ------
    - ``calltypeID``: Call type (0-10, 11 classes)
    - ``callerID``: Caller identity (0-9, 10 individuals from 5 twin pairs)

    Additional Fields
    -----------------
    - ``audio``: Vocalization segment waveform (numpy array).
    - ``path``: Relative path to audio file.
    - ``start``: Start time in seconds.
    - ``end``: End time in seconds.
    - ``duration``: Vocalization segment duration in seconds.
    - ``twinID``: Marmoset twin pair (1-5).
    - ``vocID``: Unique vocalization ID (row index).

    References
    ----------
    Sarkar, E., Magimai.-Doss, M. (2023)
    "Can Self-Supervised Neural Representations Pre-Trained on Human Speech
    distinguish Animal Callers?" Proc. Interspeech 2023, 1189-1193.
    doi: 10.21437/Interspeech.2023-1968
    https://www.isca-speech.org/archive/interspeech_2023/sarkar23_interspeech.html

    Examples
    --------
    >>> from esp_data.datasets import InfantMarmosetsVox
    >>> dataset = InfantMarmosetsVox(
    ...     split="all",
    ...     output_take_and_give={"calltypeID": "label", "audio": "audio"},
    ...     sample_rate=16000,
    ... )
    >>> sample = dataset[0]
    >>> print(dataset.info.name)
    InfantMarmosetsVox
    >>> print(dataset.available_sample_rates)
    [44100, 16000]
    """

    info = DatasetInfo(
        name="InfantMarmosetsVox",
        owner="eklavya",
        split_paths={
            "all": "gs://esp-ml-datasets/infant_marmosets_vox/labels.csv",
        },
        version="0.1.0",
        description=(
            "InfantMarmosetsVox is a dataset for multi-class call-type and caller identification. "
            "Available at original 44.1 kHz and pre-resampled 16kHz. "
            "Pre-resampled audio uses librosa's kaiser_best resampling method. "
            "Contains approx. 73k vocalization segments of infant marmoset vocalizations. "
            "Each vocalization segment sample has a calltypeID and callerID label. "
            "There are 11 calltype classes (0-10) and 10 caller identity classes (0-9). "
            "A calltypeID index can be associated with its calltype through `CALLTYPE_NAMES`. "
            "An unfiltered version (labels_raw.csv) with silence/noise is also available."
        ),
        sources=["Zenodo"],
        license="CC-BY-4.0",
    )

    # Mapping of sample rates to audio subdirectory names
    _sample_rate_paths = {
        44100: "audio_44k",  # Original 44.1kHz
        16000: "audio_16k",  # Pre-resampled to 16kHz
    }

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the InfantMarmosetsVox dataset.

        Parameters
        ----------
        split : str
            The split to load. Currently only "all" is available.
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        sample_rate : int
            The sample rate to which audio files should be resampled.
        data_root : str | AnyPathT, optional
            The root directory for the dataset audio files.
            If None, defaults to parent directory of the split CSV.
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

        self._data = None
        self._load()

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
            Audio at these sample rates can be loaded directly without on-the-fly
            resampling. Original audio is at 44100 Hz.
        """
        return list(self._sample_rate_paths.keys())

    def _load(self) -> None:
        """Load the dataset from preprocessed CSV.

        Raises
        ------
        LookupError
            If the requested split is not available.
        """
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
        )

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["InfantMarmosetsVox", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters

        Returns
        -------
        tuple[Dataset, dict[str, Any]]
            A tuple containing the dataset instance and metadata.
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
            The number of samples.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        NotImplementedError
            If streaming mode is enabled.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row to load audio segment.

        Parameters
        ----------
        row : dict
            A row from the dataset containing path, start, end times.

        Returns
        -------
        dict
            The processed sample with audio loaded.
        """
        # Determine which audio directory to use based on requested sample rate
        audio_rel_path = row["path"]

        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            # Use pre-resampled audio - replace "audio_44k" with appropriate subdirectory
            audio_subdir = self._sample_rate_paths[self.sample_rate]
            audio_rel_path = audio_rel_path.replace("audio_44k", audio_subdir, 1)

        audio_path = anypath(self.data_root) / audio_rel_path
        start = float(row["start"])
        end = float(row["end"])

        # Read the audio clip
        audio, sr = read_audio(audio_path, start_time=start, end_time=end)
        audio = audio.astype(np.float32)
        # Stereo to mono if necessary.
        audio = audio_stereo_to_mono(audio, mono_method="average")

        # Resample on-the-fly if requested sample rate doesn't have pre-resampled version
        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio, orig_sr=sr, target_sr=self.sample_rate, res_type="kaiser_best"
            )

        row["audio"] = audio
        row["path"] = audio_rel_path  # Update path to reflect actual file loaded

        if self.output_take_and_give:
            return {value: row[key] for key, value in self.output_take_and_give.items()}
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
            A dictionary containing audio and metadata.
        """
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        ------
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
            A formatted string with dataset information.
        """
        return (
            f"{self.info.name} (v{self.info.version}), split='{self.split}'\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )

    @property
    def calltype_names(self) -> dict[int, str]:
        """Return mapping from call type ID to name."""
        return CALLTYPE_NAMES

    @property
    def num_callers(self) -> int:
        """Return number of unique callers."""
        return len(self._data.get_unique("callerID"))
