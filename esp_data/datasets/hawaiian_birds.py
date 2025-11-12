"""Hawaiian Birds dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class HawaiianBirds(Dataset):
    """HawaiianBirds Dataset

    Description
    -----------
    Annotated soundscapes from Hawaii, provided by Cornell Lab of Ornithology

    Description from the Zenodo:

    "This collection contains 635 soundscape recordings with a total duration
    of almost 51 hours, which have been annotated by expert ornithologists
    who provided 59,583 bounding box labels for 27 different bird species
    from the Hawaiian Islands, including 6 threatened or endangered native
    birds. The data were recorded between 2016 and 2022 at four sites across
    Hawaii Island. This collection has partially been featured as test data
    in the 2022 BirdCLEF competition and can primarily be used for training
    and evaluation of machine learning algorithms.

    Data collection

    Soundscapes for this collection were recorded for various research projects
    by the Listening Observatory for Hawaiian Ecosystems (LOHE) at the
    University of Hawaii at Hilo. The recordings were collected using Wildlife
    Acoustics Inc. Song Meters (models 2, 4, or Mini), as 16-bit wav files at a
    sampling rate of 44.1 kHz, using the default gain settings of each model.
    Further specifics for each recording, such as recording location and habitat
    type, can be found in the metadata provided. Soundscapes in this collection
    vary in length, ranging from just under a minute to 9 minutes in duration.
    All audio was unified, converted to FLAC, and resampled to 32 kHz for this
    collection. Parts of this dataset have previously been used in the 2022
    BirdCLEF competition.

    Sampling and annotation protocol

    This collection is a subset of the files recorded over the course of the LOHE
    lab’s respective studies. The data were subsampled for annotation by aurally
    scanning the recordings and visually scanning spectrograms generated using
    Raven Pro software for target species of interest to the individual research
    project for which each recording was collected. Recordings that did not
    contain vocalizations of the species of interest were excluded from full
    annotation and thus this collection.

    Using Raven Pro, annotators were asked to create a selection box around every
    bird call they could recognize, ignoring those that were too faint or
    unidentifiable at a spectrogram window size of 700 points. Provided labels
    contain full bird calls that are boxed in time and frequency. Annotators were
    allowed to combine multiple consecutive calls of the same species into one
    bounding box label if pauses between calls were shorter than 0.5 seconds. We
    converted labels to eBird species codes, following the 2021 eBird taxonomy
    (Clements list)."


    Each entry consists of:
    - an audio recording
    - a selection table (Raven format), with Species labels

    References
    ----------
    https://zenodo.org/records/7078499

    """

    info = DatasetInfo(
        name="hawaiian_birds",
        owner="benjamin",
        split_paths={
            "all": "gs://esp-ml-datasets/hawaiian_birds/all.csv",
        },
        version="0.1.0",
        description="[MISSING]",
        sources="Zenodo",
        license="CC-BY-4.0",
    )

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: Dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
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
        """
        super().__init__(output_take_and_give)
        self.split = split
        self.annotation_columns = ["Species"]

        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None

        # Load split CSV
        self._load()

        # If no explicit data_root, assume parent dir of the split path
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

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
        self._data = pd.read_csv(location, keep_default_na=False, na_values=[""])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self._data)}")

        row = self._data.iloc[idx].to_dict()

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
        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")

        # Clip events outside audio (keep only events that begin before audio end)
        audio_dur = len(audio) / float(sr)
        st = st[st["Begin Time (s)"] < audio_dur].copy()

        # Build output
        row["audio"] = audio
        row["selection_table"] = st

        if self.output_take_and_give:
            item = {}
            for old_key, new_key in self.output_take_and_give.items():
                item[new_key] = row[old_key]
            return item

        return row

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["HawaiianBirds", dict[str, Any]]:
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            data_root=cfg["data_root"],
            sample_rate=cfg["sample_rate"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def get_available_labels(self, anno_column: str = "Species") -> List[str]:
        """
        Return all possible labels for a given annotation column

        Returns
        ---------
        A list of all the available labels for anno_column
        """
        available_labels = set()
        for _, row in self._data.iterrows():
            st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
            available_labels.update(st[anno_column].astype(str).tolist())
        return sorted(available_labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
