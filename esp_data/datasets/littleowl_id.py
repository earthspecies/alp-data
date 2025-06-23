"""Little Owl ID dataset"""

from typing import Any, Dict, Iterator, Optional

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class LittleOwlId(Dataset):
    """Little Owl individual ID dataset.

    Description
    -----------
    Vocalisations released by Stowell et al. for individual Little Owls
    (Athene noctua). Provides both *within-year* and *across-year* evaluation schemes.
    https://royalsocietypublishing.org/doi/10.1098/rsif.2018.0940

    References
    ----------
    https://royalsocietypublishing.org/doi/10.1098/rsif.2018.0940
    Zenodo: https://zenodo.org/records/1413495
    Examples
    -------
    >>> from esp_data.datasets import LittleOwlId
    >>> dataset = LittleOwlId(
    ...     split="test_across_year",
    ...     sample_rate=16000,
    ...     data_root="gs://esp-ml-datasets/littleowl_id/v0.1.0/raw/"
    ... )
    """

    info = DatasetInfo(
        name="littleowl_id",
        owner="david",
        split_paths={
            "train_across_year": "gs://esp-ml-datasets/littleowl_id/v0.1.0/raw/acrossyear_fg_train.csv",
            "test_across_year": "gs://esp-ml-datasets/littleowl_id/v0.1.0/raw/acrossyear_fg_test.csv",
        },
        version="0.1.0",
        description="Individual identification of little owls (Athene noctua)",
        sources=["https://royalsocietypublishing.org/doi/10.1098/rsif.2018.0940"],
        license="CC-BY-4.0",
    )

    def __init__(
        self,
        split: str = "train_across_year",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: Optional[int] = None,
        data_root: Optional[str | AnyPathT] = None,
    ) -> None:
        super().__init__(output_take_and_give)
        self.split = split
        self._data: pd.DataFrame = None
        self._load()
        self.sample_rate = sample_rate
        self.data_root = data_root or anypath(self.info.split_paths[self.split]).parent

    # Common helper methods are identical to ChiffchaffId --------------------------------

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
        df = pd.read_csv(location, keep_default_na=False, na_values=[""])

        required_cols = {"local_path", "individual_id"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"CSV at {location} must contain columns {required_cols}. Found {set(df.columns)}."
            )

        self._data = df[list(required_cols)].copy()

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["LittleOwlId", dict[str, Any]]:
        cfg = dataset_config.model_dump(exclude=("dataset_name", "transformations"))
        split = cfg.get("split", "train_across_year")
        if split not in cls.info.split_paths:
            available_splits = ", ".join(cls.info.split_paths.keys())
            raise LookupError(f"Invalid split '{split}'. Available splits: {available_splits}")
        ds = cls(
            split=split,
            output_take_and_give=cfg.get("output_take_and_give"),
            data_root=cfg.get("data_root"),
            sample_rate=cfg["sample_rate"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("Dataset not loaded.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        row = self._data.iloc[idx].to_dict()
        audio_path = (
            anypath(self.data_root) / row["local_path"]
            if self.data_root
            else anypath(row["local_path"])
        )
        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")
        if self.sample_rate and sr != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
        row["audio"] = audio
        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        splits_str = ", ".join(self.available_splits)
        return f"{self.info.name} (v{self.info.version}) | splits: {splits_str}"
