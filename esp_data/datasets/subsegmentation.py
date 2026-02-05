"""Subsegmentation dataset"""

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
class Subsegmentation(Dataset):
    """Bird Song Subsegmentation Dataset

    Description
    -----------
    Bird Song subsegmentation dataset from Logan James' paper "Pervasive patterns in
    the songs of passerine birds resemble human music universals and are linked with
    production and cognitive mechanisms"

    Currently, this dataset is for internal use but we hope to release it publicly.
    The recordings come from xeno-canto and the annotations come from the paper.

    Each entry consists of:
    - an audio recording
    - a selection table with start- and stop-times of song syllables
    - a boolean indicating if it passed quality control (i.e. if it was sub-segmentable)
    - annotations of Species, Genus, Order, and Family.

    Each selection table has, for each syllable, column for the Species, Genus, Order,
    and Family, as well as an Annotation:

    'a' indicates a syllable that is the beginning of a song (we define as at least 500 ms
        silence before)
    'z' indicates a syllable that is the end of a song (we define as at least 500 ms silence
        after)
    's' indicates all other syllables


    References
    ----------
    https://www.biorxiv.org/content/biorxiv/early/2024/07/17/2024.07.15.603339.full.pdf


    """

    info = DatasetInfo(
        name="subsegmentation",
        owner="benjamin",
        split_paths={
            "all": "gs://subsegmentation/xeno_canto_annotations/all.csv",
            "train": "gs://subsegmentation/xeno_canto_annotations/train.csv",
            "val": "gs://subsegmentation/xeno_canto_annotations/val.csv",
            "test": "gs://subsegmentation/xeno_canto_annotations/test.csv",
        },
        version="0.1.0",
        description="[MISSING]",
        sources="Logan James",
        license="private",
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
        self.annotation_columns = ["Species", "Genus", "Family", "Order", "Annotation"]

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

        # Resolve audio path
        audio_path = (
            (self.data_root / row["audio_path"]) if self.data_root else anypath(row["audio_path"])
        )

        # Read audio
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

        # Selection table
        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")

        # Clip events outside audio (keep only events that begin before audio end)
        audio_dur = len(audio) / float(sample_rate)
        st = st[st["Begin Time (s)"] < audio_dur].copy()

        # Build output
        row["audio"] = audio
        row["sample_rate"] = sample_rate
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["Subsegmentation", dict[str, Any]]:
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

        Returns
        ---------
        A list of all the available labels for anno_column
        """
        available_labels = set()
        for row in self._data:
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
