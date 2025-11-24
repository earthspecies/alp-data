"""Superb Starling dataset"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import pandas as pd
import os

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class SuperbStarling(Dataset):
    """Superb Starling Dataset

    Description
    -----------
    Dataset of superb starling (Lamprotornis superbus) flight calls with precise time bounds, 
    individual ID, and social group ID 

    Each entry includes:
    - An audio clip containing one flight call
    - Annotations for exact start/stop of the call in audio clip 
    - Metadata (bird ID, group, sex, ring, timestamp)

    The metadata file is a tab-separated text file that is formatted as a Raven selection table. 
    This lets you open all sound files in Raven and see the annotations aligned for every selection, which each correspond to a single flight call

    

    References
    ----------
    Keen, S. C., Meliza, C. D., & Rubenstein, D. R. (2013). Flight calls signal group and individual identity but not kinship in a cooperatively breeding bird. Behavioral Ecology, 24(6), 1279-1285.

    """

    info = DatasetInfo(
        name="superb_starling",
        owner="Sara",  
        split_paths={
            "all": "gs://esp-ml-datasets/superb-starlings-keen/v0.1.0/organized_data/superb_starlings_flightcalls.txt",   
        },
        version="0.1.0",
        description="superb starling flight calls with individual ID and group ID annotations",
        sources="Kenya field recordings",  
        license="CC0 1.0",  #https://datadryad.org/dataset/doi:10.5061/dryad.p1n88
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
            Optional root directory to prepend to each row's audio path.
            If None, will use the parent directory of the split file.
        """
        super().__init__(output_take_and_give)
        self.split = split
        self._data: pd.DataFrame | None = None
        self.annotation_columns = ["Species", "bird", "group", "sex", "ring"]

        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None

        # Load split file
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
        # Read tab-separated file
        self._data = pd.read_csv(location, sep="\t", keep_default_na=False, na_values=[""])

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

        # Resolve audio path - use "Begin File" column for the filename
        audio_filename = row["Begin Path"]
        audio_path = (
            (self.data_root / audio_filename) if self.data_root else anypath(audio_filename)
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

        # Add audio and sample rate to output
        row["audio"] = audio
        row["sample_rate"] = sr

        # Calculate duration from audio
        row["duration_secs"] = len(audio) / float(sr)

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["SuperbStarling", dict[str, Any]]:
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
        Return all possible labels for a given annotation column.

        Parameters
        ----------
        anno_column : str
            The annotation column to get labels from. Options include:
            'Species', 'bird', 'group', 'sex', 'ring'

        Returns
        -------
        list[str]
            A sorted list of all unique values in the specified column.
        """
        if self._data is None:
            return []
        if anno_column not in self._data.columns:
            raise ValueError(
                f"Column '{anno_column}' not found. Available columns: {self.columns}"
            )
        return sorted(self._data[anno_column].astype(str).unique().tolist())

    def get_individual_stats(self) -> pd.DataFrame:
        """
        Get statistics per individual bird.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: bird, group, sex, ring, num_vocalizations
        """
        if self._data is None:
            return pd.DataFrame()
        
        stats = (
            self._data.groupby(["bird", "group", "sex", "ring"])
            .size()
            .reset_index(name="num_vocalizations")
        )
        return stats.sort_values("num_vocalizations", ascending=False)

    def get_group_stats(self) -> pd.DataFrame:
        """
        Get statistics per social group.

        Returns
        -------
        pd.DataFrame
            DataFrame with group-level statistics
        """
        if self._data is None:
            return pd.DataFrame()
        
        stats = (
            self._data.groupby("group")
            .agg({
                "bird": "nunique",
                "Selection": "count",
            })
            .rename(columns={"bird": "num_individuals", "Selection": "num_vocalizations"})
            .reset_index()
        )
        return stats.sort_values("num_vocalizations", ascending=False)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}\n"
            f"Total vocalizations: {len(self) if self._data is not None else 'N/A'}"
        )