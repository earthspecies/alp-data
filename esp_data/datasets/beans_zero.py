"""BEANS-Zero dataset"""

from typing import Any, Dict, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio
from esp_data.schema import ColumnSchema, DatasetSchema


@register_dataset
class BeansZero(Dataset):
    """BEANS-Zero dataset

    Description
    -----------
    BEANS-Zero is a bioacoustics benchmark designed to evaluate multimodal
    audio-language models in zero-shot settings. Introduced in the paper
    NatureLM-audio paper (Robinson et al., 2025), it brings together tasks
    from both existing datasets and newly curated resources.
    The benchmark focuses on models that take a bioacoustic audio input
    (e.g., bird or mammal vocalizations) and a text instruction
    (e.g., "What species is in this audio?"),
    and return a textual output (e.g., "Taeniopygia guttata").
    As a zero-shot benchmark, BEANS-Zero contains only a test
    split—no training or in-context examples are provided.

    References
    ----------
    NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics
    David Robinson, Marius Miron, Masato Hagiwara, Olivier Pietquin
    https://openreview.net/forum?id=hJVdwBpWjt

    Huggingface Dataset:
    https://huggingface.co/datasets/EarthSpeciesProject/BEANS-Zero


    Examples
    --------
    >>> from esp_data.datasets import Beans
    >>> dataset = BeansZero(
    ...     split="test",
    ...     output_take_and_give={"species_scientific": "species"},
    ...     sample_rate=16000,
    ...     data_root="gs://esp-ml-datasets/beans-zero/v0.1.0/raw/"
    ... )
    """

    info = DatasetInfo(
        name="beans_zero",
        owner="gagan, masato, david, marius",
        split_paths={
            # 'test' is the full test set combining all tasks
            "test": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/test.jsonl",
            "cbi": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/cbi_test.jsonl",
            "watkins": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/watkins_test.jsonl",
            "hiceas": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/hiceas_test.jsonl",
            "dcase": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/dcase_test.jsonl",
            "enabirds": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/enabirds_test.jsonl",
            "esc50": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/esc50_test.jsonl",
            "humbugdb": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/humbugdb_test.jsonl",
            "rfcx": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/rfcx_test.jsonl",
            "gibbons": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/gibbons_test.jsonl",
            "lifestage": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/lifestage_test.jsonl",
            "call-type": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/call-type_test.jsonl",
            "captioning": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/captioning_test.jsonl",
            "zf-indiv": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/zf-indiv_test.jsonl",
            "unseen-family-cmn": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/unseen-family-cmn_test.jsonl",
            "unseen-family-sci": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/unseen-family-sci_test.jsonl",
            "unseen-family-tax": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/unseen-family-tax_test.jsonl",
            "unseen-genus-cmn": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/unseen-genus-cmn_test.jsonl",
            "unseen-genus-sci": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/unseen-genus-sci_test.jsonl",
            "unseen-genus-tax": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/unseen-genus-tax_test.jsonl",
            "unseen-species-cmn": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/unseen-species-cmn_test.jsonl",
            "unseen-species-sci": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/unseen-species-sci_test.jsonl",
            "unseen-species-tax": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/unseen-species-tax_test.jsonl",
        },
        version="0.1.0",
        description="BEANS-Zero benchmark dataset",
        sources=[
            "Xeno-canto",
            "iNaturalist",
            "Animal Sound Archive",
            "Elie and Theunissen 2016",
            "Beans",
            "esc50",
            "rfcx",
            "CBI",
            "HumBugDB",
            "Enabirds",
            "HICEAS",
            "Watkins",
            "Gibbons",
            "DCASE-2021-Task-5",
        ],
        license="CC-BY-4.0, CC0",
    )

    # Mapping of sample rates to their corresponding path columns
    _sample_rate_paths = {
        32000: "audio_path_32KHz",  # Pre-resampled to 32kHz with librosa.resample
        16000: "audio_path_16KHz",  # Pre-resampled to 16kHz with librosa.resample
    }

    # Column name for original variable-rate audio files
    _originals_path_column = "audio_path_original_sample_rate"

    schema = DatasetSchema(
        columns=[
            ColumnSchema(name="source_dataset", dtype="str", required=False),
            ColumnSchema(name="dataset_name", dtype="str", required=False),
            ColumnSchema(name="output", dtype="str", required=False),
            ColumnSchema(name="instruction_text", dtype="str", required=False),
            ColumnSchema(name="instruction", dtype="str", required=False),
            ColumnSchema(name="task", dtype="str", required=False),
            ColumnSchema(name="file_name", dtype="str", required=False),
            ColumnSchema(name="license", dtype="str", required=False),
            ColumnSchema(name="id", dtype="str", required=False),
            ColumnSchema(name="metadata", dtype="str", required=False),
            ColumnSchema(name="audio_path_original_sample_rate", dtype="str", required=True),
            ColumnSchema(name="audio_path_16KHz", dtype="str", required=False),
            ColumnSchema(name="audio_path_32KHz", dtype="str", required=False),
        ]
    )

    def __init__(
        self,
        split: str = "test",
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

    @property
    def available_sample_rates(self) -> list[int]:
        """Return the available pre-resampled sample rates.

        Returns
        -------
        list[int]
            List of sample rates (in Hz) for which pre-resampled audio is available.
            Audio at these sample rates can be loaded directly without on-the-fly resampling.
            This checks which path columns actually exist in the loaded data.
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["BeansZero", dict[str, Any]]:
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
        # Determine which path column to use based on requested sample rate
        # If a pre-resampled version is available, use it; otherwise resample on-the-fly
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            # Check if the pre-resampled path column exists in the data
            if path_column in row and row[path_column] is not None and row[path_column] != "":
                # Use pre-resampled audio
                audio_path = anypath(self.data_root) / row[path_column]
                use_presampled = True

        if use_presampled:
            audio, sample_rate = read_audio(audio_path)
            audio = audio.astype(np.float32)
            audio = audio_stereo_to_mono(audio, mono_method="average")
            # Audio is already at the correct sample rate, no resampling needed
        else:
            # Use original variable-rate files and resample on-the-fly if needed
            audio_path = anypath(self.data_root) / row[self._originals_path_column]
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
