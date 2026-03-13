"""Synthetic audio conversations dataset."""

from __future__ import annotations

from typing import Any, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_SPLIT_ROOTS: dict[str, str] = {
    "train_v2": "gs://foundation-model-data/synthetic/synth_v2_32k_noisy_v2",
    "train_v4_wbio": "gs://foundation-model-data/synthetic/synth_v2_32k_noisy_v4_wbio",
}
_SOURCE_SAMPLE_RATE = 32000

_TASK_BY_TEMPLATE_PATH: dict[str, str] = {
    "audio_synth/caption": "synthetic_captioning",
    "audio_synth/summary": "synthetic_summary",
    "audio_synth/perceptual": "synthetic_perceptual_description",
    "audio_synth/qa": "synthetic_audio_qa",
    "audio_synth/json": "synthetic_structured_annotation",
    "audio_synth/tags": "synthetic_audio_tags",
    "audio_synth/f0_contour": "synthetic_f0_contour",
}


@register_dataset
class SyntheticAudioConversations(Dataset):
    """Synthetic multi-task audio conversations for NatureLM stage 2."""

    info = DatasetInfo(
        name="synthetic_audio_conversations",
        owner="david",
        split_paths={split: f"{root}/conversations.jsonl" for split, root in _SPLIT_ROOTS.items()},
        version="0.2.0",
        description=(
            "Synthetic audio conversations with authored chat turns for captioning, "
            "QA, structured annotations, tags, perceptual summaries, and F0 tracing "
            "across multiple synthetic corpora."
        ),
        sources=["foundation-model-data synthetic audio"],
        license="internal",
    )

    def __init__(
        self,
        split: str = "train_v2",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        default_root = _SPLIT_ROOTS[self.split]
        self.data_root = anypath(data_root) if data_root else anypath(default_root)
        self._load()

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        self._data = self._backend_class.from_json(
            self.info.split_paths[self.split],
            lines=True,
            streaming=self._streaming,
        )

    def _resolve_audio_path(self, row: dict[str, Any]) -> AnyPathT:
        audio_ids = row.get("audio_ids")
        if not isinstance(audio_ids, list) or not audio_ids or not isinstance(audio_ids[0], str):
            raise ValueError(f"Expected non-empty 'audio_ids' list in row '{row.get('id', '<unknown>')}'")
        return self.data_root / "audio" / f"{audio_ids[0]}.wav"

    def _resolve_task(self, row: dict[str, Any]) -> str:
        template_path = row.get("template_path")
        if not isinstance(template_path, str):
            raise ValueError(f"Expected string 'template_path' in row '{row.get('id', '<unknown>')}'")
        try:
            return _TASK_BY_TEMPLATE_PATH[template_path]
        except KeyError as exc:
            raise ValueError(f"Unknown synthetic template_path: {template_path}") from exc

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        audio_path = self._resolve_audio_path(row)
        audio, sample_rate = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        if self.sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sample_rate = self.sample_rate

        row["audio"] = audio
        row["audio_path"] = str(audio_path)
        row["task"] = self._resolve_task(row)
        row["sample_rate"] = sample_rate

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["SyntheticAudioConversations", dict[str, Any]]:
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
        base = f"{self.info.name} (v{self.info.version}), split: {self.split}"
        return (
            f"{base}\n"
            f"Description: {self.info.description}\n"
            f"Source sample rate: {_SOURCE_SAMPLE_RATE} Hz\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
