"""NBM dataset"""

from __future__ import annotations

import warnings
from io import StringIO
from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class NocturnalBirdMigration(Dataset):
    """NocturnalBirdMigration Dataset

    Description
    -----------
    Dataset of nocturnal vocalizations from migratory birds in Europe.
    Vocalizations are annotated with start- and end- times, as well as high- and
    low-frequencies.

    The dataset consists of a train split and a test split. The test split consists
    entirely of xeno-canto recordings which the dataset authors annotated. The train
    split consists of recordings submitted by French citizen-scientists, as well as
    xeno-canto recordings annotated by the dataset authors.

    Note that the license is mixed (due to origins on xeno-canto); most restrictive
    is CC-BY-NC-SA 4.0. See zenodo page for full license details.

    Description from the paper:

    The persisting threats on migratory bird populations
    highlight the urgent need for effective monitoring tech-
    niques that could assist in their conservation. Among
    these, passive acoustic monitoring is an essential tool,
    particularly for nocturnal migratory species that are
    difficult to track otherwise. This work presents the Noc-
    turnal Bird Migration (NBM) dataset, a collection of
    13,359 annotated vocalizations from 117 species of the
    Western Palearctic. The dataset includes precise time
    and frequency annotations, gathered by dozens of bird
    enthusiasts across France, enabling novel downstream
    acoustic analysis. In particular, we prove the utility of
    this database by training an original two-stage deep ob-
    ject detection model tailored for the processing of audio
    data. While allowing the precise localization of bird calls
    in spectrograms, this model shows competitive accuracy
    on the 45 main species of the dataset with state-of-the-
    art systems trained on much larger audio collections.
    These results highlight the interest of fostering similar
    open-science initiatives to acquire costly but valuable
    fine-grained annotations of audio files. All data and
    code are made openly available.

    Each entry consists of:
    - an audio recording
    - a selection table (Raven format), with Species labels
    - xeno-canto id, if applicable (else, empty string)

    Pre-resampled Audio
    -------------------
    Pre-resampled audio is available at 16 kHz and 32 kHz. When
    ``sample_rate`` matches one of these rates, the pre-resampled files are
    loaded directly (no on-the-fly resampling). For any other target rate,
    audio is resampled on-the-fly using librosa's ``kaiser_best`` method.

    References
    ----------
    https://zenodo.org/records/17573913
    https://arxiv.org/pdf/2412.03633

    """

    info = DatasetInfo(
        name="nocturnal_bird_migration",
        owner="benjamin",
        split_paths={
            # Full training set
            "train": "gs://esp-ml-datasets/nocturnal_bird_migration/train_v2.csv",
            # Training subset: no xeno-canto
            "train_nonxc": "gs://esp-ml-datasets/nocturnal_bird_migration/train_nonxc_v2.csv",
            # Training subset: xeno-canto recordings only
            "train_xc": "gs://esp-ml-datasets/nocturnal_bird_migration/train_xc_v2.csv",
            # Held-out test set
            "test": "gs://esp-ml-datasets/nocturnal_bird_migration/test_v2.csv",
        },
        version="0.1.0",
        description="[MISSING]",
        sources="Zenodo, xeno-canto",
        license="CC-BY-NC-SA 4.0",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_path"

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
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["NocturnalBirdMigration", dict[str, Any]]:
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
