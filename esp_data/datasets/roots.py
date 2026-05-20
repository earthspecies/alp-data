"""ROOTS synthetic audio conversation dataset."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_T1_V2_ROOT = "gs://foundation-model-data/synthetic/train_final/T1_with_captions_train_v2"
_F0_AUDIO_ROOT = "gs://esp-data-ingestion/f0-prediction/audio_32k"
_INAT_AUDIO_ROOT = "gs://esp-ml-datasets/inaturalist/v0.1.0/raw"
_XC_AUDIO_ROOT = "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/audio"

_SPLIT_SPECS: dict[str, dict[str, str | None]] = {
    "tier1_v2_acoustic_caption_f0bioacoustic": {
        "jsonl_path": f"{_T1_V2_ROOT}/acoustic_caption_f0bioacoustic_train_unseen_v1_clean.jsonl",
        "audio_root": _F0_AUDIO_ROOT,
        "task": "roots_tier1_acoustic_caption",
    },
    "tier1_v2_acoustic_caption_field_notes_inat": {
        "jsonl_path": (
            f"{_T1_V2_ROOT}/acoustic_caption_field_notes_inat_train_unseen_v2_clean.jsonl"
        ),
        "audio_root": _INAT_AUDIO_ROOT,
        "task": "roots_tier1_acoustic_caption",
    },
    "tier1_v2_acoustic_caption_field_notes_xc": {
        "jsonl_path": f"{_T1_V2_ROOT}/acoustic_caption_field_notes_xc_train_unseen_v2_clean.jsonl",
        "audio_root": _XC_AUDIO_ROOT,
        "task": "roots_tier1_acoustic_caption",
    },
    "tier1_v2_acoustic_caption_pseudovox": {
        "jsonl_path": f"{_T1_V2_ROOT}/acoustic_caption_pseudovox_v1_clean.jsonl",
        "audio_root": None,
        "task": "roots_tier1_acoustic_caption",
    },
    "tier1_v2_cat_snr_mcq_custom_bins_xc": {
        "jsonl_path": f"{_T1_V2_ROOT}/cat_snr_mcq_custom_bins_xc_train_unseen_v1_clean.jsonl",
        "audio_root": _XC_AUDIO_ROOT,
        "task": "roots_tier1_snr_mcq",
    },
    "tier1_v2_cat_snr_mcq_inat": {
        "jsonl_path": f"{_T1_V2_ROOT}/cat_snr_mcq_inat_train_unseen_v2_clean.jsonl",
        "audio_root": _INAT_AUDIO_ROOT,
        "task": "roots_tier1_snr_mcq",
    },
    "tier1_v2_cat_snr_mcq_xc": {
        "jsonl_path": f"{_T1_V2_ROOT}/cat_snr_mcq_xc_train_unseen_v2_clean.jsonl",
        "audio_root": _XC_AUDIO_ROOT,
        "task": "roots_tier1_snr_mcq",
    },
    "tier1_v2_f0_bioacoustic_summary": {
        "jsonl_path": f"{_T1_V2_ROOT}/f0_bioacoustic_16khz_f0_summary.jsonl",
        "audio_root": _F0_AUDIO_ROOT,
        "task": "roots_tier1_f0_summary",
    },
    "tier1_v2_snr_binary_xc": {
        "jsonl_path": f"{_T1_V2_ROOT}/snr_binary_xc_train_unseen_v1_clean.jsonl",
        "audio_root": _XC_AUDIO_ROOT,
        "task": "roots_tier1_snr_binary",
    },
    "tier1_v2_snr_oe_xc": {
        "jsonl_path": f"{_T1_V2_ROOT}/snr_oe_xc_train_unseen_v1_clean.jsonl",
        "audio_root": _XC_AUDIO_ROOT,
        "task": "roots_tier1_snr_open",
    },
    "tier1_v2_voc_desc_f0_mcq_f0bioacoustic": {
        "jsonl_path": f"{_T1_V2_ROOT}/voc_desc_f0_mcq_f0bioacoustic_train_unseen_v2_clean.jsonl",
        "audio_root": _F0_AUDIO_ROOT,
        "task": "roots_tier1_vocal_description_mcq",
    },
    "tier1_v2_voc_desc_mcq_field_notes_inat": {
        "jsonl_path": f"{_T1_V2_ROOT}/voc_desc_mcq_field_notes_inat_train_unseen_v1_clean.jsonl",
        "audio_root": _INAT_AUDIO_ROOT,
        "task": "roots_tier1_vocal_description_mcq",
    },
    "tier1_v2_voc_desc_mcq_field_notes_xc": {
        "jsonl_path": f"{_T1_V2_ROOT}/voc_desc_mcq_field_notes_xc_train_unseen_v1_clean.jsonl",
        "audio_root": _XC_AUDIO_ROOT,
        "task": "roots_tier1_vocal_description_mcq",
    },
    "tier1_v2_voc_desc_mcq_pseudovox": {
        "jsonl_path": f"{_T1_V2_ROOT}/voc_desc_mcq_pseudovox_train_unseen_v1_clean.jsonl",
        "audio_root": None,
        "task": "roots_tier1_vocal_description_mcq",
    },
}


@register_dataset
class ROOTS(Dataset):
    """ROOTS synthetic audio conversations for NatureLM training."""

    info = DatasetInfo(
        name="ROOTS",
        owner="david",
        split_paths={split: str(spec["jsonl_path"]) for split, spec in _SPLIT_SPECS.items()},
        version="0.1.0",
        description=(
            "Chat-native ROOTS synthetic audio tasks. Splits are organized by "
            "tier and task so later Tier 2 and Tier 3 final-format JSONL files "
            "can be added without introducing a new dataset adapter."
        ),
        sources=["foundation-model-data synthetic train_final"],
        license="internal",
    )

    def __init__(
        self,
        split: str,
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        task: str | None = None,
        jsonl_path: str | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize a ROOTS split.

        Parameters
        ----------
        split
            Named ROOTS split. Built-in splits follow ``tier{n}_...`` naming.
        output_take_and_give
            Optional mapping of original to output field names.
        sample_rate
            Target audio sample rate. If ``None``, source sample rates are kept.
        data_root
            Optional audio root override for relative ``audio_paths`` values.
        task
            Optional task override. When omitted, the built-in split task is used.
        jsonl_path
            Optional JSONL path override for ad hoc ROOTS-compatible splits.
        backend
            Backend used to load the JSONL metadata.
        streaming
            Whether to load the backend in streaming mode.

        Raises
        ------
        LookupError
            If ``split`` is unknown and no explicit ``jsonl_path`` is provided.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None

        spec = _SPLIT_SPECS.get(split, {})
        self.jsonl_path = jsonl_path or spec.get("jsonl_path")
        if not isinstance(self.jsonl_path, str):
            raise LookupError(
                f"Invalid ROOTS split: {split}. Expected one of {list(_SPLIT_SPECS)} "
                "or provide jsonl_path explicitly."
            )

        audio_root = data_root if data_root is not None else spec.get("audio_root")
        self.data_root = anypath(audio_root) if audio_root is not None else None
        self.task = task or spec.get("task") or "roots"
        self._load()

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        self._data = self._backend_class.from_json(
            self.jsonl_path,
            lines=True,
            streaming=self._streaming,
        )

    def _resolve_audio_path(self, row: dict[str, Any]) -> AnyPathT:
        audio_paths = row.get("audio_paths")
        if not isinstance(audio_paths, list) or not audio_paths:
            raise ValueError(
                f"Expected non-empty 'audio_paths' list in row '{row.get('id', '<unknown>')}'"
            )

        audio_path = str(audio_paths[0])
        if audio_path.startswith(("gs://", "s3://", "r2://", "/")):
            return anypath(audio_path)
        if self.data_root is None:
            raise ValueError(
                f"ROOTS row '{row.get('id', '<unknown>')}' has relative audio path "
                f"{audio_path!r}, but no data_root is configured."
            )
        return self.data_root / audio_path

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
        row["sample_rate"] = sample_rate
        row["task"] = str(self.task)

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["ROOTS", dict[str, Any]]:
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            data_root=cfg["data_root"],
            task=cfg.get("task"),
            jsonl_path=cfg.get("jsonl_path"),
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version}), split: {self.split}\n"
            f"JSONL: {self.jsonl_path}\n"
            f"Task: {self.task}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
