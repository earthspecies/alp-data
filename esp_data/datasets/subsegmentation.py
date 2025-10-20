"""Subsegmentation dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Iterator, List

import numpy as np
import pandas as pd
import torch
import torchaudio

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
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
        license="Internal (currently)",
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
        self._data: pd.DataFrame | None = None
        self.annotation_columns = ["Species", "Genus", "Family", "Order", "Annotation"]

        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None

        # Load split CSV
        self._load()

        # If no explicit data_root, assume parent dir of the split path
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

    # -----------------------------
    # Properties
    # -----------------------------
    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    # -----------------------------
    # Internals
    # -----------------------------
    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = pd.read_csv(location, keep_default_na=False, na_values=[""])

    # -----------------------------
    # Dataset API
    # -----------------------------
    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No split loaded.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._data is None:
            raise RuntimeError("No split loaded.")
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

        # Resample if needed
        target_sr = self.sample_rate
        if target_sr is not None and sr != target_sr:
            audio = torchaudio.functional.resample(
                torch.tensor(audio),
                sr,
                target_sr,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            ).numpy()
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

    # -----------------------------
    # Factory
    # -----------------------------
    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["Subsegmentation", dict[str, Any]]:
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

    # -----------------------------
    # Convenience
    # -----------------------------
    def get_available_labels(self, anno_column: str) -> List[str]:
        """
        Return all possible labels for a given annotation column

        Returns
        ---------
        A list of all the available labels for anno_column
        """
        if self._data is None:
            return []
        available_labels = set()
        for _, row in self._data.iterrows():
            st = pd.read_csv(StringIO(row["selection_table_str"]), sep="\t")
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


if __name__ == "__main__":
    ds = Subsegmentation()

    sample = ds[100]

    assert "audio" in sample
    print(f"Audio shape: {sample['audio'].shape}")
    st1 = sample["selection_table"]
    print(st1.head(10))
    print(sample["pass_qc"])
