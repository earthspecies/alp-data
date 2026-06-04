"""Birdeep dataset"""

from __future__ import annotations

import warnings
from io import StringIO
from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import pandas as pd

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import DATA_HOME, AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class Birdeep(Dataset):
    """Birdeep Dataset

    Description
    -----------
    Dataset of bird vocalizations with bounding boxes, originally released in:
    "A Bird Song Detector for improving bird identification through Deep Learning:
    a case study from Doñana" by Alba Márquez-Rodríguez et al. (2025)

    Description from the github:

    "Data was collected using automatic audio recording devices (AudioMoths) in
    three different habitats in Doñana National Park. Approximately 500 minutes
    of audio data were recorded. There are 9 recorders in 3 different habitats
    (marshland, scrubland, and ecotone), which are constantly running, recording
    1 minute and leaving 9 minutes between recordings. That is, 1 minute is
    recorded for every 10 minutes, with a sampling rate of 32 kHz. The
    recordings were made prioritising those times when the birds are most active
    in order to try to have as many audio recordings of songs as possible,
    specifically a few hours before dawn until midday.

    Expert annotators labeled 461 minutes of audio data, identifying bird
    vocalizations and other relevant sounds. Annotations are provided in a
    standard format with start time, end time, and frequency range for each
    bird vocalization."

    Each entry consists of:
    - an audio recording
    - a selection table (Raven format), with Species labels

    Note that some birds were not identifiable to species, and are annotated as "Unknown".

    The dataset splits are the same as in the original publication.

    Pre-resampled Audio
    -------------------
    Pre-resampled audio is available at 16 kHz. When ``sample_rate=16000`` is
    passed, the pre-resampled files are loaded directly (no on-the-fly
    resampling). For any other target rate, audio is resampled on-the-fly from
    the native 32 kHz files using librosa's ``kaiser_best`` method.

    References
    ----------
    https://huggingface.co/datasets/GrunCrow/BIRDeep_AudioAnnotations
    https://www.sciencedirect.com/science/article/pii/S1574954125002638?via%3Dihub

    """

    info = DatasetInfo(
        name="birdeep",
        owner="benjamin",
        split_paths={
            "train": f"{DATA_HOME}/birdeep/train_formatted_v3.csv",
            "val": f"{DATA_HOME}/birdeep/val_formatted_v3.csv",
            "test": f"{DATA_HOME}/birdeep/test_formatted_v3.csv",
            "all": f"{DATA_HOME}/birdeep/all_formatted_v3.csv",
        },
        version="0.1.0",
        description="Dataset of bird vocalizations with bounding boxes, originally released in: "
        "A Bird Song Detector for improving bird identification "
        "through Deep Learning: a case study from Doñana by Alba Márquez-Rodríguez et al. (2025)",
        sources="HuggingFace",
        license="MIT",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path"}
    _originals_path_column = "audio_path"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: Dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load (key in info.split_paths).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns as well).
        sample_rate : int | None
            If set, audio is resampled to this rate.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row['audio_path'].
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.annotation_columns = ["Species"]
        self.unknown_label = "Unknown"

        self.sample_rate = sample_rate

        self._load()

        if data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent
        else:
            self.data_root = anypath(data_root)

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Return pre-resampled sample rates whose path columns exist in the data."""
        return [sr for sr, col in self._sample_rate_paths.items() if col in self._data.columns]

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            location, streaming=self._streaming, keep_default_na=False, na_values=[""]
        )

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of the dataset.

        Parameters
        ----------
        row : dict[str, Any]
            A dictionary representing a single row of the dataset.

        Returns
        -------
        dict[str, Any]
            The processed row.
        """
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if path_column in row and row[path_column] is not None and row[path_column] != "":
                audio_path = anypath(self.data_root) / row[path_column]
                use_presampled = True

        if not use_presampled:
            audio_path = anypath(self.data_root) / row[self._originals_path_column]

        audio, sr = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        if not use_presampled and self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sr = self.sample_rate

        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
        audio_dur = len(audio) / float(sr)
        st = st[st["Begin Time (s)"] < audio_dur].copy()

        row["audio"] = audio
        row["sample_rate"] = sr
        row["selection_table"] = st

        if self.output_take_and_give:
            item = {}
            for old_key, new_key in self.output_take_and_give.items():
                item[new_key] = row[old_key]
            return item

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
            A dictionary containing the processed data.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["Birdeep", dict[str, Any]]:
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
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def get_available_labels(self, anno_column: str = "Species") -> List[str]:
        """
        Return all possible labels for a given annotation column
        anno_column is included as an optional argument for consistency
        with other detection datasets.

        Returns
        ---------
        A list of all the available labels for anno_column
        """
        available_labels = set()
        for row in self._data:
            st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
            available_labels.update(st[anno_column].astype(str).tolist())
        if self.unknown_label in available_labels:
            available_labels.remove(self.unknown_label)

        warnings.warn(
            f"Events with unknown label={self.unknown_label} exist in dataset"
            f"but {self.unknown_label} suppressed from get_available_labels output",
            stacklevel=2,
        )

        return sorted(available_labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
