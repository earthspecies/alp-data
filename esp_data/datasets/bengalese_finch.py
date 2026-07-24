"""Bengalese Finch Song Repository (Nicholson, Queen & Sober 2017).

Held-out subsegmentation evaluation dataset: 32 kHz song bouts from 4 birds,
each with per-syllable onset/offset + single-character labels in an inline
``selection_table`` TSV (WABAD/CEB shape). Built by
``scripts/data_preprocessing_scripts/bengalese_finch/build_bengalese_finch.py``.
Source: figshare 10.6084/m9.figshare.4805749 (CC BY 4.0).
"""
from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
from typing import Any

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_ROOT = "gs://esp-data-ingestion/bengalese_finch/v0.1.0"


@register_dataset
class BengaleseFinch(Dataset):
    """Bengalese finch song bouts with per-syllable onset/offset + labels.

    Columns
    -------
    filepath / audio_fp : str
        Wav path relative to :attr:`data_root` (``<bird>/<day>/<name>.wav``).
    bird_id, day, source_cbin : str
    sample_rate, duration_s, n_syllables : numeric
    selection_table : str
        TSV with ``Begin Time (s)``, ``End Time (s)``, ``Annotation`` (the
        single-character syllable label).

    Splits
    ------
    ``all`` (and ``<bird_id>`` per-bird splits, if built).
    """

    info = DatasetInfo(
        name="bengalese_finch",
        owner="david",
        split_paths={
            "all": f"{_ROOT}/bf_manifest.csv",
        },
        version="0.1.0",
        description=(
            "Bengalese Finch Song Repository (Nicholson, Queen & Sober 2017): "
            "32 kHz song bouts from 4 birds with per-syllable onset/offset + "
            "labels; held-out subsegmentation boundary evaluation."
        ),
        sources="figshare 10.6084/m9.figshare.4805749",
        license="CC BY 4.0",
    )

    _sample_rate_paths: dict[int, str] = {}
    _originals_path_column = "audio_fp"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """Initialise the dataset (see class docstring for columns/splits)."""
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        self.annotation_columns = ["Annotation"]
        self._load()
        self.data_root = anypath(data_root) if data_root is not None else anypath(f"{_ROOT}/audio")

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        """Load the split manifest CSV.

        Raises
        ------
        LookupError
            If the split is not valid.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        self._data = self._backend_class.from_csv(
            self.info.split_paths[self.split],
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

    def __len__(self) -> int:
        """Return the number of recordings in the split.

        Raises
        ------
        RuntimeError
            If no split has been loaded.
        NotImplementedError
            In streaming mode.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Read audio (optionally windowed) and parse the selection table.

        Returns
        -------
        dict[str, Any]
            The row with ``audio``, ``sample_rate`` and a parsed
            ``selection_table`` DataFrame.
        """
        audio_path = anypath(self.data_root) / str(row[self._originals_path_column])
        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")
        if window_start is not None and window_end is not None:
            audio, sr = read_audio(audio_path, start_time=float(window_start), end_time=float(window_end))
        else:
            audio, sr = read_audio(audio_path)

        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.sample_rate,
                                     scale=True, res_type="kaiser_best")
            sr = self.sample_rate
        row["audio"] = audio
        row["sample_rate"] = sr

        raw_st = row.get("selection_table")
        if raw_st is not None:
            if isinstance(raw_st, str):
                st = pd.read_csv(StringIO(raw_st), sep="\t", keep_default_na=False)
            elif isinstance(raw_st, pd.DataFrame):
                st = raw_st
            else:
                st = pd.DataFrame()
            audio_dur = len(audio) / float(sr)
            if "Begin Time (s)" in st.columns:
                st = st[st["Begin Time (s)"] < audio_dur].copy()
            row["selection_table"] = st

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single processed row."""
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over processed rows."""
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["BengaleseFinch", dict[str, Any]]:
        """Create an instance from a configuration."""
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            data_root=cfg["data_root"],
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __str__(self) -> str:
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{self.info.name} (v{self.info.version})\n"
            f"Recordings: {n} (split={self.split})\n"
            f"License: {self.info.license}"
        )
