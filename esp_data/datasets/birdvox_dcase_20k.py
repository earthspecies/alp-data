"""BirdVox-DCASE-20k dataset (DCASE 2018 Task 3, audio v2).

20 000 ten-second clips labelled with binary bird-presence (``hasbird``),
stored at 16 kHz mono on GCS.

References
----------
https://zenodo.org/records/1208080
https://figshare.com/articles/dataset/6026471
Lostanlen et al., "BirdVox-DCASE-20k: a dataset for bird audio detection",
DCASE 2018.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_BASE = "gs://foundation-model-data/synthetic/cropped/birdvox_dcase_20k"


@register_dataset
class BirdVoxDCASE20k(Dataset):
    """BirdVox-DCASE-20k binary bird-presence dataset.

    Description
    -----------
    20 000 ten-second audio clips (16 kHz mono) labelled with binary
    bird-presence (``hasbird`` 0/1). Expert-annotated with estimated accuracy
    of 99.5–99.95 %, making this the highest-quality negative-biophony
    benchmark in the BAD suite.

    Each entry contains:

    - an audio clip (10 s, 16 kHz mono)
    - ``fn``: clip UUID
    - ``hasbird``: 0 (no bird) or 1 (bird present)
    - ``audio_duration``: clip duration in seconds

    References
    ----------
    https://zenodo.org/records/1208080
    https://figshare.com/articles/dataset/6026471
    """

    info = DatasetInfo(
        name="birdvox_dcase_20k",
        owner="christos",
        split_paths={
            "all": f"{_GCS_BASE}/all.csv",
        },
        version="1.0.0",
        description="20k 10-second clips with binary bird-presence labels (16 kHz mono).",
        sources="zenodo.org/records/1208080, figshare.com/articles/dataset/6026471",
        license="CC-BY-4.0",
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
            Split to load (key in info.split_paths). Default ``"all"``.
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys. Omit ``"audio"``
            to suppress audio loading.
        sample_rate : int | None
            Target audio sample rate. Default 16000.
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

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def __len__(self) -> int:
        """Return number of clips.

        Returns
        -------
        int
            Number of clips in the loaded split.

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
            Processed sample with ``fn``, ``audio_path``, ``hasbird``,
            ``audio_duration``, and optionally ``audio`` and ``sample_rate``.
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
                import librosa
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.sample_rate,
                                         scale=True, res_type="kaiser_best")
                sr = self.sample_rate

            row["audio"] = audio
            row["sample_rate"] = sr

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
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

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
        return (
            f"{self.info.name} (v{self.info.version})\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
