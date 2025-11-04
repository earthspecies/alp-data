"""Powdermill dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class Powdermill(Dataset):
    """Powdermill Dataset

    Description
    -----------
    Dataset of bird vocalizations with bounding boxes, originally released in:
    "An annotated set of audio recordings of Eastern North American birds containing
    frequency, time, and species information" by Lauren Chronister et al. (2021).

    Description from the original:

    "Acoustic recordings of soundscapes are an important category of audio data that
    can be useful for answering a variety of questions, and an entire discipline
    within ecology, dubbed “soundscape ecology,” has risen to study them. Bird sound
    is often the focus of studies of soundscapes due to the ubiquitousness of birds
    in most terrestrial environments and their high vocal activity. Autonomous
    acoustic recorders have increased the quantity and availability of recordings
    of natural soundscapes while mitigating the impact of human observers on
    community behavior. However, such recordings are of little use without analysis
    of the sounds they contain. Manual analysis currently stands as the best means
    of processing this form of data for use in certain applications within
    soundscape ecology, but it is a laborious task, sometimes requiring many hours
    of human review to process comparatively few hours of recording. For this reason,
    few annotated data sets of soundscape recordings are publicly available. Further
    still, there are no publicly available strongly labeled soundscape recordings of
    bird sounds that contain information on timing, frequency, and species. Therefore,
    we present the first data set of strongly labeled bird sound soundscape recordings
    under free use license. These data were collected in the Northeastern United States
    at Powdermill Nature Reserve, Rector, Pennsylvania, USA. Recordings encompass 385
    minutes of dawn chorus recordings collected by autonomous acoustic recorders between
    the months of April through July 2018. Recordings were collected in continuous bouts
    on four days during the study period and contain 48 species and 16,052 annotations.
    Applications of this data set may be numerous and include the training, validation,
    and testing of certain advanced machine-learning models that detect or classify bird
    sounds. There are no copyright or propriety restrictions; please cite this paper when
    using materials within."

    Note that this data was included in the BEANS "detection", i.e. multi-label
    classification, benchmark, under the name ENABirds.

    Each entry consists of:
    - an audio recording
    - a selection table (Raven format), with Species labels

    References
    ----------
    https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/ecy.3329

    """

    info = DatasetInfo(
        name="powdermill",
        owner="benjamin",
        split_paths={
            "all": "gs://esp-ml-datasets/powdermill/all_gbif.csv",
        },
        version="0.1.0",
        description="[MISSING]",
        sources="Dryad",
        license="Public Domain",
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
        self._data = pd.read_csv(location, keep_default_na=False, na_values=[""])

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["Powdermill", dict[str, Any]]:
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
        anno_column is included as an optional argument for consistency
        with other detection datasets.

        Returns
        ---------
        A list of all the available labels for anno_column
        """
        if self._data is None:
            return []
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
