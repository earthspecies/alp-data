"""BEANS-Pro dataset: acoustic description matching benchmark."""

from typing import Any, Dict, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class BeansPro(Dataset):
    """BEANS-Pro acoustic description matching benchmark.

    Description
    -----------
    BEANS-Pro evaluates multimodal audio-language models on their ability
    to match animal vocalizations to expert acoustic descriptions. Each
    example presents an audio clip and four acoustic descriptions (one
    correct, three distractors from the same species), and the model must
    identify the correct description.

    Descriptions are sourced verbatim from published bioacoustics papers
    and verified against the original figures and tables.

    Available splits
    ----------------
    - ``crow-description``: 200 examples, 25 call types (merged from 40),
      carrion crow (*Corvus corone*). Source: ESP cooperative crows preprint.
    - ``zebra-description``: 40 examples, 4 call types, plains zebra
      (*Equus quagga*). Source: Xie et al. 2024, R. Soc. Open Sci.

    Schema
    ------
    Each row is a JSONL record with fields matching BEANS-Zero:
    - ``instruction``: Full prompt with ``<Audio><AudioHere></Audio>`` tag,
      question text, and four labelled choices (A-D).
    - ``output``: Correct answer letter (A, B, C, or D).
    - ``audio_path_original_sample_rate``: Relative path to audio file.
    - ``metadata``: JSON string with call_type, species, duration, etc.

    Examples
    --------
    >>> from esp_data.datasets import BeansPro
    >>> dataset = BeansPro(
    ...     split="crow-description",
    ...     sample_rate=16000,
    ...     data_root="gs://esp-data-ingestion/beans-pro/v0.1.0/raw/carrion_crow_descriptions/"
    ... )
    """

    info = DatasetInfo(
        name="beans_pro",
        owner="david",
        split_paths={
            "crow-description": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/carrion_crow_descriptions/test.jsonl",
            "zebra-description": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/zebra_descriptions/test.jsonl",
        },
        version="0.1.0",
        description=(
            "BEANS-Pro acoustic description matching benchmark. "
            "4-choice multiple choice: given audio, pick the correct acoustic "
            "description from paper-verified options. "
            "Crow split: 200 examples, 25 call types (Corvus corone). "
            "Zebra split: 40 examples, 4 call types (Equus quagga)."
        ),
        sources=[
            "ESP cooperative crows preprint",
            "Xie et al. 2024, R. Soc. Open Sci.",
        ],
        license="CC-BY-NC-4.0, CC0-1.0",
    )

    # Data roots per split (used when data_root is None)
    _default_data_roots = {
        "crow-description": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/carrion_crow_descriptions/",
        "zebra-description": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/zebra_descriptions/",
    }

    _originals_path_column = "audio_path_original_sample_rate"

    def __init__(
        self,
        split: str = "crow-description",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the BEANS-Pro dataset.

        Parameters
        ----------
        split : str
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
        sample_rate : int
            The sample rate to which audio files should be resampled.
        data_root : str | AnyPathT, optional
            The root directory for the dataset. If None, uses the default
            GCS path for the selected split.
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
            self.data_root = self._default_data_roots.get(
                split, anypath(self.info.split_paths[split]).parent
            )
        else:
            self.data_root = data_root

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        return []  # Only original sample rate available

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. "
                f"Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_json(
            location, lines=True, orient="records"
        )

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["BeansPro", dict[str, Any]]:
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
        audio_path = anypath(self.data_root) / row[self._originals_path_column]
        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)
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

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No split has been loaded yet.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    def __str__(self) -> str:
        base_info = f"{self.info.name} (v{self.info.version}), split: {self.split}"
        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
