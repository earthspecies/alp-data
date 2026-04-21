"""BirdVox-DCASE-20k dataset (DCASE 2018 Task 3, audio v2).

20 000 ten-second clips labelled with binary bird-presence (``hasbird``).
Labels are expert-annotated with estimated accuracy of 99.5–99.95 %.

Requires local data:
- Audio: https://zenodo.org/records/1208080 (BirdVox-DCASE-20k.zip)
- Labels: https://figshare.com/articles/dataset/6026471
  (BirdVox-DCASE-20k_csv-public.csv)
"""

from __future__ import annotations

import pathlib
from typing import Any, Iterator

import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import audio_stereo_to_mono, read_audio


@register_dataset
class BirdVoxDCASE20k(Dataset):
    """BirdVox-DCASE-20k binary bird-presence dataset.

    Description
    -----------
    20 000 ten-second audio clips (44.1 kHz mono) labelled with binary
    bird-presence (``hasbird`` 0/1). Expert-annotated with estimated accuracy
    of 99.5–99.95 %, making this the highest-quality negative-biophony
    benchmark in the BAD suite.

    Requires local data — audio and labels CSV must be downloaded separately.

    Each entry contains:

    - an audio clip (10 s, resampled to 16 kHz mono by default)
    - ``itemid``: UUID filename stem
    - ``hasbird``: 0 (no bird) or 1 (bird present)

    References
    ----------
    https://zenodo.org/records/1208080
    https://figshare.com/articles/dataset/6026471
    Lostanlen et al., "BirdVox-DCASE-20k: a dataset for bird audio detection",
    DCASE 2018.
    """

    info = DatasetInfo(
        name="birdvox_dcase_20k",
        owner="christos",
        split_paths={},
        version="1.0.0",
        description="20k 10-second clips with binary bird-presence labels (local data required).",
        sources="zenodo.org/records/1208080, figshare.com/articles/dataset/6026471",
        license="CC-BY-4.0",
    )

    def __init__(
        self,
        data_root: str,
        labels_csv: str,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        data_root : str
            Directory containing the extracted ``<uuid>.wav`` files.
        labels_csv : str
            Path to ``BirdVox-DCASE-20k_csv-public.csv``.
        split : {"all", "train", "test"}
            Subset to expose. ``"train"`` returns the first 80 % sorted by
            ``itemid``; ``"test"`` the remaining 20 %; ``"all"`` the full set.
            Default ``"all"``.
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys. Omit ``"audio"``
            to suppress audio loading.
        sample_rate : int | None
            Target audio sample rate. Default 16000.
        backend : BackendType
            Unused; present for API consistency.
        streaming : bool
            Unused; present for API consistency.

        Raises
        ------
        ValueError
            If ``split`` is not ``"all"``, ``"train"``, or ``"test"``.
        FileNotFoundError
            If ``data_root`` or ``labels_csv`` do not exist.
        """
        if split not in ("all", "train", "test"):
            raise ValueError(f"split must be 'all', 'train', or 'test', got {split!r}")
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.data_root = pathlib.Path(data_root)
        self.labels_csv = pathlib.Path(labels_csv)
        self.split = split
        self.sample_rate = sample_rate

        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root does not exist: {self.data_root}")
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"labels_csv not found: {self.labels_csv}")

        self._records = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        import pandas as pd
        df = pd.read_csv(self.labels_csv, dtype={"itemid": str, "hasbird": int})
        df = df.sort_values("itemid").reset_index(drop=True)
        if self.split != "all":
            cutoff = int(len(df) * 0.8)
            df = df.iloc[:cutoff] if self.split == "train" else df.iloc[cutoff:]
        return [
            {"itemid": str(row["itemid"]), "hasbird": int(row["hasbird"]),
             "audio_path": str(self.data_root / f"{row['itemid']}.wav")}
            for _, row in df.iterrows()
        ]

    @property
    def columns(self) -> list[str]:
        return ["itemid", "hasbird", "audio_path", "audio", "sample_rate"]

    @property
    def available_splits(self) -> list[str]:
        return ["all", "train", "test"]

    def __len__(self) -> int:
        return len(self._records)

    def _process(self, record: dict[str, Any]) -> dict[str, Any]:
        """Build a sample dict for a single record.

        Parameters
        ----------
        record : dict[str, Any]
            Record dict with ``itemid``, ``hasbird``, ``audio_path``.

        Returns
        -------
        dict[str, Any]
            Sample with ``itemid``, ``hasbird``, ``audio_path``, and
            optionally ``audio`` and ``sample_rate``.
        """
        need_audio = self.output_take_and_give is None or "audio" in self.output_take_and_give
        item = dict(record)
        if need_audio:
            audio, sr = read_audio(record["audio_path"])
            audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
            if self.sample_rate is not None and sr != self.sample_rate:
                import librosa
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.sample_rate,
                                         scale=True, res_type="kaiser_best")
                sr = self.sample_rate
            item["audio"] = audio
            item["sample_rate"] = sr
        if self.output_take_and_give is not None:
            return {new: item[old] for old, new in self.output_take_and_give.items() if old in item}
        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the sample at position ``idx``.

        Parameters
        ----------
        idx : int
            Index into the dataset.

        Returns
        -------
        dict[str, Any]
            Processed sample dict.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for BirdVoxDCASE20k with {len(self)} samples")
        return self._process(self._records[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["BirdVoxDCASE20k", dict[str, Any]]:
        """Create an instance from a DatasetConfig.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration object.

        Returns
        -------
        tuple[BirdVoxDCASE20k, dict[str, Any]]
            Dataset instance and transformation metadata.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            data_root=cfg["data_root"],
            labels_csv=cfg["labels_csv"],
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version})\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Split: {self.split} ({len(self)} samples)"
        )
