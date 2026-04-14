"""Birdeep dataset"""

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

    References
    ----------
    https://huggingface.co/datasets/GrunCrow/BIRDeep_AudioAnnotations
    https://www.sciencedirect.com/science/article/pii/S1574954125002638?via%3Dihub

    """

    info = DatasetInfo(
        name="birdeep",
        owner="benjamin",
        split_paths={
            "train": "gs://esp-ml-datasets/birdeep/train_formatted.csv",
            "val": "gs://esp-ml-datasets/birdeep/val_formatted.csv",
            "test": "gs://esp-ml-datasets/birdeep/test_formatted.csv",
            "all": "gs://esp-ml-datasets/birdeep/all_formatted.csv",
        },
        version="0.1.0",
        description="[MISSING]",
        sources="HuggingFace",
        license="MIT",
    )

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
        self.data_root = anypath(data_root) if data_root is not None else None

        # Load split CSV
        self._load()

        # If no explicit data_root, assume parent dir of the split path
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

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
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of the dataset.

        When the row contains ``window_start_sec`` / ``window_end_sec``
        (added by the ``window_annotations`` transform), only the matching
        audio segment is loaded. If ``selection_table`` has been removed by
        upstream transforms such as ``select_columns``, audio loading still
        proceeds and the missing table is ignored.

        Parameters
        ----------
        row : dict[str, Any]
            A dictionary representing a single row of the dataset.

        Returns
        -------
        dict[str, Any]
            The processed row.
        """

        # Resolve audio path
        audio_path = (
            (self.data_root / row["audio_path"]) if self.data_root else anypath(row["audio_path"])
        )

        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")

        # Read audio
        if window_start is not None and window_end is not None:
            audio, sample_rate = read_audio(
                audio_path,
                start_time=float(window_start),
                end_time=float(window_end),
            )
        else:
            audio, sample_rate = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        # Resample if necessary
        if self.sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sample_rate = self.sample_rate

        raw_st = row.get("selection_table")
        if raw_st is not None:
            if isinstance(raw_st, str):
                st = pd.read_csv(StringIO(raw_st), sep="\t")
            elif isinstance(raw_st, pd.DataFrame):
                st = raw_st
            else:
                st = pd.DataFrame()

            # Clip events outside audio (keep only events that begin before audio end)
            audio_dur = len(audio) / float(sample_rate)
            if "Begin Time (s)" in st.columns:
                st = st[st["Begin Time (s)"] < audio_dur].copy()

            row["selection_table"] = st

        # Build output
        row["audio"] = audio
        row["sample_rate"] = sample_rate

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
        return sorted(available_labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
