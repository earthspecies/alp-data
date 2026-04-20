"""Pre-cropped Powdermill dataset.

Each row corresponds to a single deterministic crop (1–30 s, seed 42, tile mode)
of a Powdermill recording. Audio is stored as 16 kHz mono WAV on GCS; selection
tables have already been filtered and time-shifted to the crop window.
"""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class PowdermillCropped(Dataset):
    """Pre-cropped Powdermill dataset.

    Description
    -----------
    Deterministic 1–30 second crops of Powdermill dawn chorus recordings
    (~300 s per recording), stored as 16 kHz mono WAV files on GCS. Each
    crop's selection table is pre-filtered and time-shifted relative to the
    crop start. Cropping was performed with seed=42, tile mode,
    min_duration=1.0 s, max_duration=30.0 s — identical to the
    ``PowdermillCropped`` class in the NatureLM-audio-data-synth package.

    Each entry contains:

    - an audio clip (1–30 s, 16 kHz mono)
    - a Raven-style selection table for the clip window (times relative to
      the crop start; includes ``partial`` boolean column; Species column)
    - provenance fields: ``original_fn``, ``crop_start``, ``crop_end``

    References
    ----------
    https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/ecy.3329
    """

    info = DatasetInfo(
        name="powdermill_cropped",
        owner="christos",
        split_paths={
            "all": "gs://foundation-model-data/synthetic/cropped/powdermill/all.csv",
        },
        version="1.0.0",
        description="Pre-cropped Powdermill clips (1–30 s, seed 42, tile, 16 kHz mono).",
        sources="Dryad / Chronister et al. 2021",
        license="Public Domain",
    )

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
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
            Optional mapping of original → new output keys. Omit ``"audio"``
            to suppress audio loading.
        sample_rate : int | None
            If set, audio is resampled to this rate. Default ``16000``.
        data_root : str | AnyPathT | None
            If given, relative ``audio_path`` values are resolved against this
            root. Leave ``None`` when using the default GCS URIs.
        backend : BackendType
            CSV backend, by default ``"polars"``.
        streaming : bool
            Whether to use streaming mode, by default ``False``.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None
        self._load()

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split!r}. Expected one of {list(self.info.split_paths.keys())}"
            )
        self._data = self._backend_class.from_csv(
            self.info.split_paths[self.split],
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

    def __len__(self) -> int:
        """Return the number of crop samples.

        Returns
        -------
        int
            Number of samples in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        NotImplementedError
            In streaming mode.
        """
        if self._data is None:
            raise RuntimeError("No split loaded yet.")
        if self._streaming:
            raise NotImplementedError("Length unavailable in streaming mode.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single CSV row into a sample dict.

        Parameters
        ----------
        row : dict[str, Any]
            Raw row from the metadata CSV.

        Returns
        -------
        dict[str, Any]
            Processed sample with at minimum ``fn``, ``audio_path``,
            ``selection_table`` (DataFrame), ``audio_duration``,
            ``original_fn``, ``crop_start``, ``crop_end``.  Also includes
            ``audio`` (float32 ndarray) and ``sample_rate`` when not suppressed
            via ``output_take_and_give``.
        """
        need_audio = self.output_take_and_give is None or "audio" in self.output_take_and_give

        if need_audio:
            if self.data_root is not None:
                audio_path = self.data_root / row["audio_path"]
            else:
                audio_path = anypath(row["audio_path"])

            audio, sr = read_audio(audio_path)
            audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

            if self.sample_rate is not None and sr != self.sample_rate:
                audio = librosa.resample(
                    y=audio,
                    orig_sr=sr,
                    target_sr=self.sample_rate,
                    scale=True,
                    res_type="kaiser_best",
                )
                sr = self.sample_rate

            row["audio"] = audio
            row["sample_rate"] = sr

        raw_st = row.get("selection_table")
        if isinstance(raw_st, str):
            row["selection_table"] = pd.read_csv(StringIO(raw_st), sep="\t")
        elif not isinstance(raw_st, pd.DataFrame):
            row["selection_table"] = pd.DataFrame()

        if self.output_take_and_give is not None:
            return {
                new_key: row[old_key]
                for old_key, new_key in self.output_take_and_give.items()
                if old_key in row
            }
        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        dict[str, Any]
            Processed sample dict.
        """
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all samples.

        Yields
        ------
        dict[str, Any]
            Each processed sample.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["PowdermillCropped", dict[str, Any]]:
        """Create an instance from a DatasetConfig.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration object.

        Returns
        -------
        tuple[PowdermillCropped, dict[str, Any]]
            Dataset instance and transformation metadata.
        """
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

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
