"""AudioSet Strong dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class AudioSetStrong(Dataset):
    """AudioSet Strong Dataset

    Description
    -----------
    AudioSet Strong is a strongly-labeled subset of AudioSet with temporal annotations
    (start and end times) for sound events. This dataset provides precise timing
    information for when each sound event occurs within the 10-second audio clips.

    AudioSet is a large-scale dataset of manually-annotated audio events that endeavors
    to bridge the gap in data availability between image and audio research, using a
    carefully structured hierarchical ontology of 632 audio classes in 10-second
    segments of YouTube videos.

    This class makes the AudioSet Strong subset available in the esp-data strongly-labeled
    format, where each entry consists of:
    - An audio recording (10 seconds)
    - A selection table with temporal annotations (begin time, end time, label)

    The strong labels provide temporal boundaries for sound events, making this dataset
    suitable for sound event detection and temporal localization tasks.

    References
    ----------
    AUDIO SET: AN ONTOLOGY AND HUMAN-LABELED DATASET FOR AUDIO EVENTS
    Gemmeke et al. 2017
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45857.pdf

    AudioSet Homepage:
    https://research.google.com/audioset/

    Examples
    --------
    >>> from esp_data.datasets import AudioSetStrong
    >>> dataset = AudioSetStrong(split="train", sample_rate=16000)
    >>> print(len(dataset))
    8841
    >>> item = dataset[0]
    >>> sorted(item.keys())
    ['audio', 'audio_path', 'segment_id', 'segment_start', 'selection_table', 'youtube_id']
    >>> print(item['selection_table'].columns)
    Index(['Selection', 'Begin Time (s)', 'End Time (s)', 'Label'], dtype='object')
    """

    info = DatasetInfo(
        name="audioset_strong",
        owner="david; marius; masato",
        split_paths={
            "train": "gs://esp-ml-datasets/audioset/v0.2.0/raw/csv-data/audioset_train_strong_selection_tables_filtered.csv",
        },
        version="0.1.0",
        description="AudioSet Strong: Strongly-labeled subset with temporal annotations",
        sources=["YouTube"],
        license="Mixed",
    )

    def __init__(
        self,
        split: str = "train",
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
            Split to load (key in info.split_paths). Currently only "train" is available.
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
        self.annotation_columns = ["Label"]

        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None

        # Load split CSV
        self._load()

        # If no explicit data_root, set to the raw directory (go up two levels from csv file)
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent.parent

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

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
            location, streaming=self._streaming, keep_default_na=False, na_values=[""]
        )

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No split loaded.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    @staticmethod
    def _empty_selection_table() -> pd.DataFrame:
        # Default Raven-style selection table columns we expect for strong labels.
        return pd.DataFrame(columns=["Selection", "Begin Time (s)", "End Time (s)", "Label"])

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        # Resolve audio path
        audio_path = (
            (self.data_root / row["audio_path"]) if self.data_root else anypath(row["audio_path"])
        )

        # Read audio
        audio, sr = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        # Resample if necessary
        target_sr = self.sample_rate
        if target_sr is not None and sr != target_sr:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=target_sr,
                scale=True,
                res_type="kaiser_best",
            )
            sr = target_sr

        # Selection table
        selection_table_blob = row.get("selection_table", "")
        if selection_table_blob is None or selection_table_blob == "":
            st = self._empty_selection_table()
        else:
            st = pd.read_csv(StringIO(selection_table_blob), sep="\t")

        # Clip events outside audio (keep only events that begin before audio end)
        audio_dur = len(audio) / float(sr)
        if "Begin Time (s)" in st.columns:
            st = st[st["Begin Time (s)"] < audio_dur].copy()

        # Build output
        row["audio"] = audio
        row["selection_table"] = st

        if self.output_take_and_give:
            item: dict[str, Any] = {}
            for old_key, new_key in self.output_take_and_give.items():
                item[new_key] = row[old_key]
            return item

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._data is None:
            raise RuntimeError("No split loaded.")
        if self._streaming:
            raise NotImplementedError(
                "Random access (__getitem__) is not available in streaming mode. "
                "Iterate over the dataset instead."
            )
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self._data)}")

        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._data is None:
            raise RuntimeError("No split loaded.")
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["AudioSetStrong", dict[str, Any]]:
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

    def get_available_labels(self) -> List[str]:
        """
        Return all possible labels found in the dataset.

        Returns
        -------
        List[str]
            A sorted list of all unique labels in the dataset.
        """
        if self._data is None:
            return []

        labels: set[str] = set()
        for row in self._data:
            selection_table_blob = row.get("selection_table", "")
            if selection_table_blob is None or selection_table_blob == "":
                continue
            st = pd.read_csv(StringIO(selection_table_blob), sep="\t")
            if "Label" in st.columns:
                labels.update(st["Label"].astype(str).tolist())

        return sorted(labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
