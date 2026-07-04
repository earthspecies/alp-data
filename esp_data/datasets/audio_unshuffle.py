"""Audio unshuffling: multi-audio temporal-order prediction.

Each example presents ``N`` (up to 10) short audio clips taken from a *single*
recording, in a shuffled order, and the model must recover their original
temporal order.  The clips are referenced via ``N``
``<Audio><AudioHere></Audio>`` placeholders in the prompt and the ``audios``
field is a list of numpy arrays ordered to match those placeholders.

The task is self-supervised: labels come purely from the (known) temporal
position of each clip within its source recording.  Two regimes are produced
by the builder (``scripts/build_audio_unshuffle.py``) so we can probe
ordering at different timescales:

- ``local``: short call-length clips packed close together (small gaps) —
  tests local ordering.
- ``longrange``: longer clips spread across the recording with large gaps —
  tests long-range ordering.

Storage is lazy: a row stores a single ``recording_path`` plus a list of
``windows`` (``[start_sec, end_sec]`` pairs in presentation order).  At load
time the covering span is read once and sliced into per-clip waveforms, so
each example triggers at most one audio read regardless of clip count.

Available splits
----------------
- ``train`` / ``val``: built by ``scripts/build_audio_unshuffle.py`` over a
  fixed set of structure-diverse Xeno-canto + iNaturalist species, split by
  recording so val recordings are unseen during training.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

import librosa
import numpy as np
import polars as pl

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.backends.polars_backend import PolarsBackend
from esp_data.io import (
    AnyPathT,
    anypath,
    audio_stereo_to_mono,
    filesystem_from_path,
    read_audio,
)

logger = logging.getLogger(__name__)

_ROOT = "gs://foundation-model-data/synthetic/audio-unshuffle/v0"


@register_dataset
class AudioUnshuffle(Dataset):
    """Multi-audio temporal-order ("unshuffling") dataset.

    Each row returns ``audios`` (a list of numpy arrays, one per clip, in the
    shuffled presentation order) alongside a pre-baked ``messages``
    conversation whose ``<AudioHere>`` placeholder count equals
    ``len(audios)``.  The assistant target lists the clip numbers in original
    temporal order (earliest first).

    Examples
    --------
    >>> from esp_data.datasets.audio_unshuffle import AudioUnshuffle
    >>> ds = AudioUnshuffle(split="val", sample_rate=32000)
    >>> row = ds[0]
    >>> len(row["audios"]) == row["n_clips"]
    True
    >>> row["messages"][0]["content"].count("<AudioHere>") == row["n_clips"]
    True
    """

    info = DatasetInfo(
        name="audio-unshuffle",
        owner="david",
        split_paths={
            "train": f"{_ROOT}/train.jsonl",
            "val": f"{_ROOT}/val.jsonl",
        },
        version="0.1.0",
        description=(
            "Multi-audio unshuffling: given N shuffled clips from one "
            "recording, predict their original temporal order. Built over "
            "structure-diverse Xeno-canto + iNaturalist species, split by "
            "recording. Clips stored lazily as windows into the source "
            "recording (read once, sliced per clip at load time)."
        ),
        sources=[_ROOT],
        license="CC BY-NC-SA 4.0, CC BY-NC 4.0, CC BY-SA, CC0",
    )

    _ALL_SPLITS = ("train", "val")

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the AudioUnshuffle dataset.

        Parameters
        ----------
        split : str
            One of ``train`` or ``val``.
        output_take_and_give : dict[str, str] | None
            Optional column rename mapping.
        sample_rate : int | None
            Target sample rate; clips are resampled to this rate. The source
            recordings are pre-resampled to 32 kHz, so 32000 avoids any
            on-the-fly resampling.
        data_root : str | AnyPathT | None
            Optional override for the directory holding ``{split}.jsonl``. When
            given, the manifest is read from ``{data_root}/{split}.jsonl``
            instead of ``info.split_paths``; this lets one registered dataset
            serve multiple builds (e.g. the synthetic-window v0 build and the
            BirdCODE-detection build) selected purely via config. The audio
            ``recording_path`` URIs inside the JSONL are always absolute, so
            ``data_root`` never affects audio resolution.
        backend : BackendType
            Tabular backend (``"polars"`` or ``"pandas"``).
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        if split not in self._ALL_SPLITS:
            raise LookupError(f"Invalid split: {split!r}. Expected one of {self._ALL_SPLITS}")
        self.split = split
        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None
        self._data = None
        self._load()

    def _load(self) -> None:
        if self.data_root is not None:
            jsonl_path = f"{str(self.data_root).rstrip('/')}/{self.split}.jsonl"
        else:
            jsonl_path = self.info.split_paths[self.split]
        fs = filesystem_from_path(jsonl_path)
        records: list[dict[str, Any]] = []
        skipped = 0
        with fs.open(str(jsonl_path), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
        if skipped:
            logger.warning("Skipped %d malformed lines in %s", skipped, jsonl_path)
        self._data = PolarsBackend(pl.DataFrame(records))

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self._ALL_SPLITS)

    def _read_clips(self, recording_path: str, windows: list[list[float]]) -> list[np.ndarray]:
        """Read the covering span once and slice it into per-clip waveforms.

        Parameters
        ----------
        recording_path : str
            Absolute path to the source recording.
        windows : list[list[float]]
            ``[start_sec, end_sec]`` pairs in presentation order.

        Returns
        -------
        list[np.ndarray]
            Mono float32 clips, resampled to ``self.sample_rate``, in the same
            order as ``windows``.
        """
        path = anypath(recording_path)
        span_start = min(float(w[0]) for w in windows)
        span_end = max(float(w[1]) for w in windows)

        audio, sr = read_audio(path, start_time=span_start, end_time=span_end)
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

        clips: list[np.ndarray] = []
        for start_sec, end_sec in windows:
            offset = int(round((float(start_sec) - span_start) * sr))
            length = int(round((float(end_sec) - float(start_sec)) * sr))
            offset = max(0, min(offset, audio.shape[-1]))
            end = max(offset, min(offset + length, audio.shape[-1]))
            clip = audio[offset:end]
            if clip.shape[-1] == 0:
                # Degenerate window (rounding at the recording edge); emit a
                # single silent sample so the placeholder count stays valid.
                clip = np.zeros(1, dtype=np.float32)
            clips.append(np.ascontiguousarray(clip, dtype=np.float32))
        return clips

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        windows = row.get("windows")
        recording_path = row.get("recording_path")
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"Expected non-empty 'windows' in row {row.get('id', '<unknown>')!r}")
        if not recording_path:
            raise ValueError(f"Expected 'recording_path' in row {row.get('id', '<unknown>')!r}")

        row["audios"] = self._read_clips(str(recording_path), windows)
        # Emit a distinct task label per variant so each shows up separately in
        # per-task train/val logging (train/task/<task>/...). BirdCODE builds
        # carry a ``mode`` (birdcode_order / birdcode_order_gaps); synthetic
        # builds carry a ``regime`` (local / longrange).
        mode = row.get("mode")
        regime = row.get("regime")
        if mode:
            row["task"] = str(mode)
        elif regime:
            row["task"] = f"unshuffle_{regime}"
        else:
            row["task"] = "audio_unshuffle"

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No data loaded.")
        if self._streaming:
            raise NotImplementedError("Length not available in streaming mode.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["AudioUnshuffle", dict[str, Any]]:
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
            f"{self.info.name} (v{self.info.version}), split: {self.split}, {n} examples\n"
            f"Description: {self.info.description}\n"
            f"Available splits: {', '.join(self._ALL_SPLITS)}"
        )
