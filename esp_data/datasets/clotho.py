"""Clotho v2.1 dataset — Freesound audio with human-written captions."""

from __future__ import annotations

from typing import Any, Iterator

import librosa
import numpy as np
import polars as pl

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/clotho/v0.1.0"


@register_dataset
class Clotho(Dataset):
    """Clotho v2.1 Dataset.

    Description
    -----------
    Clotho (Drossos et al., ICASSP 2020) is a dataset of **15-30 s sound
    snippets from Freesound** paired with **five independently-written
    English captions per clip** (~5,929 clips × 5 captions = ~29,645
    captions). Caption length is 8-20 words.

    Audio was originally distributed by Freesound under per-clip Creative
    Commons licenses (CC-BY, CC0, etc., see ``freesound_license`` column).
    Captions are released under Tampere University's non-commercial
    attribution licence. Combined-use should follow the more restrictive
    of the two for any given clip.

    Splits
    ------
        - ``all`` : every clip across train+val+test.
        - ``train`` : 3,839 clips (Clotho's "development" split).
        - ``val``   : 1,045 clips (Clotho's "validation" split).
        - ``test``  : 1,045 clips (Clotho's "evaluation" split).

    Pre-resampled Audio
    -------------------
    Originals are 44.1 kHz mono WAV (Freesound source rate normalised by
    the Clotho authors). Pre-resampled 16 kHz and 32 kHz mono WAV are
    available under ``audio_16k/{development,validation,evaluation}/`` and
    ``audio_32k/...`` respectively. When ``sample_rate`` matches one of
    these rates the pre-resampled file is loaded directly; otherwise
    librosa ``kaiser_best`` is used.

    References
    ----------
    Drossos, Lipping, Virtanen (2020). "Clotho: An Audio Captioning Dataset",
    ICASSP 2020. DOI: 10.1109/ICASSP40776.2020.9052990
    Zenodo v2.1: https://zenodo.org/records/4783391
    """

    info = DatasetInfo(
        name="clotho_v2_1",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/all.csv",
            "train": f"{_GCS_ROOT}/train.csv",
            "val": f"{_GCS_ROOT}/val.csv",
            "test": f"{_GCS_ROOT}/test.csv",
        },
        version="0.1.0",
        description=(
            "Clotho v2.1 — 15-30 s Freesound snippets paired with 5 "
            "human-written captions each (Drossos et al., ICASSP 2020). "
            "5,929 clips total (train: 3,839; val: 1,045; test: 1,045). "
            "44.1 kHz mono WAV originals; pre-resampled 16/32 kHz mono "
            "available."
        ),
        sources=["https://zenodo.org/records/4783391"],
        license="CC-BY (Freesound audio) + Tampere non-commercial caption licence",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_path"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load: ``all`` / ``train`` / ``val`` / ``test``.
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns).
        sample_rate : int | None
            If set, audio is loaded at this rate.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row's audio path.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "pandas".
        streaming : bool, optional
            Whether to use streaming mode, by default False.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
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
        """Return pre-resampled sample rates whose path columns exist.

        Returns
        -------
        list[int]
            Sample rates (Hz) for which pre-resampled audio is available.
        """
        return [sr for sr, col in self._sample_rate_paths.items() if col in self._data.columns]

    def _load(self) -> None:
        """Load the manifest CSV for ``self.split``.

        Raises
        ------
        LookupError
            If ``self.split`` is not a key in ``info.split_paths``.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
            schema_overrides={"sound_id": pl.Utf8},
            dtype={"sound_id": "string"},
            null_values=["", "Not found"],
            keep_default_na=False,
            na_values=["", "Not found"],
        )

    def __len__(self) -> int:
        """Return the number of clips in the dataset.

        Returns
        -------
        int
            Number of audio clips in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Load audio + return the row with 5 captions.

        Returns
        -------
        dict[str, Any]
            Row dict with ``audio`` (numpy float32) and ``sample_rate`` (int)
            populated alongside the original caption / metadata columns.
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

        row["audio"] = audio
        row["sample_rate"] = sr

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific sample from the dataset.

        Returns
        -------
        dict[str, Any]
            Dict with audio, sample_rate, file_name, caption_1..5, metadata.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        ------
        dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["Clotho", dict[str, Any]]:
        """Create a Clotho instance from a DatasetConfig.

        Returns
        -------
        tuple[Clotho, dict[str, Any]]
            The instantiated dataset + apply_transformations metadata (empty
            when no transforms are configured).
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
