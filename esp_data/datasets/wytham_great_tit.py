"""Wytham Great Tit Song Dataset (Merino Recalde et al. 2024).

Held-out subsegmentation evaluation dataset: short great-tit song bouts (one
22.05 kHz mono wav per song), each with per-note onset/offset in an inline
``selection_table`` TSV (WABAD/CEB shape; ``Annotation`` = per-song song-type,
notes are unlabelled). Built (on a stratified subset by default) by
``scripts/data_preprocessing_scripts/wytham_great_tit/build_wytham_great_tit.py``.
Source: OSF 10.17605/OSF.IO/N8AC9 (CC BY 4.0). Audio is resampled to 32 kHz on
load to match the NatureLM stack.
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

_ROOT = "gs://esp-data-ingestion/wytham_great_tit/v0.1.0"


@register_dataset
class WythamGreatTit(Dataset):
    """Great tit song bouts with per-note onset/offset (note boundaries).

    Columns
    -------
    filepath / audio_fp : str
        Wav path relative to :attr:`data_root`.
    bird_id, song_type : str
    n_notes : int
    selection_table : str
        TSV with ``Begin Time (s)``, ``End Time (s)``, ``Annotation`` (the
        song-type label; note class is not annotated).

    Splits
    ------
    ``eval_subset`` (stratified subset built for evaluation).
    """

    info = DatasetInfo(
        name="wytham_great_tit",
        owner="david",
        split_paths={
            "eval_subset": f"{_ROOT}/gt_manifest_subset.csv",
        },
        version="0.1.0",
        description=(
            "Wytham Great Tit Song Dataset (Merino Recalde et al. 2024): great "
            "tit song bouts with per-note onset/offset boundaries; held-out "
            "subsegmentation boundary evaluation. Audio 22.05 kHz -> 32 kHz on load."
        ),
        sources="OSF 10.17605/OSF.IO/N8AC9",
        license="CC BY 4.0",
    )

    _sample_rate_paths: dict[int, str] = {}
    _originals_path_column = "audio_fp"

    def __init__(
        self,
        split: str = "eval_subset",
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
        """Return the number of songs in the split.

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
        """Read audio (optionally windowed, resampled to 32 kHz) + parse table.

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["WythamGreatTit", dict[str, Any]]:
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
            f"Songs: {n} (split={self.split})\n"
            f"License: {self.info.license}"
        )
