"""Synthetic SED (Sound Event Detection) dataset.

Synthetically generated soundscape recordings with expert-quality annotations,
stored on GCS at 16 kHz. Each scene is a multi-species audio recording with a
paired Raven-format selection table.

Audio and selection tables are stored as separate files; the scene manifest
(parquet) lists all scenes with their GCS paths and durations.
"""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import audio_stereo_to_mono, read_audio

_GCS_BASE = "gs://foundation-model-data/synthetic/synthetic_sed_scenes_16k"


@register_dataset
class SyntheticSED(Dataset):
    """Synthetic SED soundscape recordings with selection tables.

    Description
    -----------
    Synthetically generated multi-species soundscape recordings at 16 kHz,
    paired with Raven-format selection tables. Each scene contains multiple
    annotated vocalizations with ``Begin Time (s)``, ``End Time (s)``,
    ``Low Freq (Hz)``, ``High Freq (Hz)``, and ``Annotation`` (species) columns.

    Audio and selection tables are stored as separate GCS files. The manifest
    parquet lists all scenes with ``fn``, ``audio_fp``, ``selection_table_fp``,
    and ``audio_duration``.

    Each entry contains:

    - an audio recording (16 kHz mono)
    - a Raven-format selection table (DataFrame)
    - ``fn``, ``audio_duration``

    References
    ----------
    gs://foundation-model-data/synthetic/synthetic_sed_scenes_16k
    """

    info = DatasetInfo(
        name="synthetic_sed",
        owner="christos",
        split_paths={
            "all": f"{_GCS_BASE}/manifest.parquet",
        },
        version="1.0.0",
        description="Synthetically generated SED soundscapes at 16 kHz with selection tables.",
        sources="gs://foundation-model-data/synthetic/synthetic_sed_scenes_16k",
        license="CC-BY-4.0",
    )

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load (key in info.split_paths). Default ``"all"``.
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys. Omit ``"audio"``
            to suppress audio loading.
        sample_rate : int | None
            Target audio sample rate. Default 16000.
        backend : BackendType
            Unused; present for API consistency.
        streaming : bool
            Unused; present for API consistency.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data: pd.DataFrame | None = None
        self._load()

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split!r}. Expected one of {list(self.info.split_paths.keys())}"
            )
        import fsspec
        path = self.info.split_paths[self.split]
        with fsspec.open(path, "rb") as f:
            self._data = pd.read_parquet(f)

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def __len__(self) -> int:
        """Return number of scenes.

        Returns
        -------
        int
            Number of scenes in the loaded split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split loaded yet.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single manifest row into a sample dict.

        Parameters
        ----------
        row : dict[str, Any]
            Raw row from the manifest DataFrame.

        Returns
        -------
        dict[str, Any]
            Processed sample with ``fn``, ``audio_duration``,
            ``selection_table`` (DataFrame), and optionally ``audio`` and
            ``sample_rate``.
        """
        import fsspec

        need_audio = self.output_take_and_give is None or "audio" in self.output_take_and_give

        if need_audio:
            audio, sr = read_audio(row["audio_fp"])
            audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
            if self.sample_rate is not None and sr != self.sample_rate:
                import librosa
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.sample_rate,
                                         scale=True, res_type="kaiser_best")
                sr = self.sample_rate
            row["audio"] = audio
            row["sample_rate"] = sr

        need_st = self.output_take_and_give is None or "selection_table" in self.output_take_and_give
        if need_st:
            with fsspec.open(row["selection_table_fp"], "r") as f:
                row["selection_table"] = pd.read_csv(f, sep="\t")

        if self.output_take_and_give is not None:
            return {new: row[old] for old, new in self.output_take_and_give.items() if old in row}
        return row

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
            raise IndexError(f"index {idx} out of range for SyntheticSED with {len(self)} samples")
        return self._process(self._data.iloc[idx].to_dict())

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["SyntheticSED", dict[str, Any]]:
        """Create an instance from a DatasetConfig.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration object.

        Returns
        -------
        tuple[SyntheticSED, dict[str, Any]]
            Dataset instance and transformation metadata.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
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
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
