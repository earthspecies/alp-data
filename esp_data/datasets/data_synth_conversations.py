"""Data-synth merged conversations dataset.

Loads pre-merged multi-turn conversation parquets produced by the
NatureLM-audio-data-synth pipeline (scripts/merge_conversations.py).

Each row covers one source audio file and may contain multiple QA/caption
turns. The first user message retains the <Audio><AudioHere></Audio> tag;
subsequent turns reference the audio in context.

Audio path resolution
---------------------
Paths stored in the parquet are relative for XenoCanto and iNaturalist
(as they come from the synthesis pipeline), and absolute GCS paths for
SED / BirdVox / WABAD datasets.  Relative paths are resolved against
known dataset roots defined in ``_AUDIO_ROOTS``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator
from urllib.parse import unquote
import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://foundation-model-data/synthetic/data-synth/merged"

# Relative audio_path → prepend this root to get a resolvable GCS URI.
# Only needed for datasets whose synthesis pipeline stores relative paths.
_AUDIO_ROOTS: dict[str, str] = {
    "XenoCanto": "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/audio_16k/",
    "INaturalist": "gs://esp-ml-datasets/inaturalist/v0.1.0/raw/",
    "F0Bioacoustic": "gs://esp-data-ingestion/f0-prediction/audio/",
    "NocturnalBirdMigration": "gs://esp-ml-datasets/nocturnal_bird_migration/",
    "BirdVoxDCASE20k": "gs://foundation-model-data/synthetic/cropped/birdvox_dcase_20k/audio/",
}


@register_dataset
class DataSynthConversations(Dataset):
    """Pre-merged bioacoustic QA/caption conversations from NatureLM data-synth.

    Each sample is a multi-turn conversation covering one source audio file,
    with turns drawn from one or more synthesis runs (SNR, call type, behavior,
    vocalization tasks, etc.).  The ``messages`` field is compatible with the
    NatureLM-audio training pipeline.

    Examples
    --------
    >>> ds = DataSynthConversations(split="tier1_tier2_xc_inat_v1")
    >>> sample = ds[0]
    >>> sample.keys()
    dict_keys(['audio', 'sample_rate', 'messages', 'audio_paths', 'dataset',
               'audio_id', 'n_turns', 'source_keys'])
    """

    info = DatasetInfo(
        name="data_synth_conversations",
        owner="chrispla",
        split_paths={
            "tier1_tier2_xc_inat_v1": f"{_GCS_ROOT}/tier1_tier2_xc_inat_v1.parquet",
            "t1": "gs://foundation-model-data/synthetic/merged/t1.parquet",
            "t1_val_existing_5k": "gs://foundation-model-data/synthetic/merged/t1_val_existing_5k.parquet",
            "t1_val_f0_mean_seen": "gs://foundation-model-data/synthetic/merged/t1_val_f0_mean_seen.parquet",
            "t1_val_f0_mean_heldout": "gs://foundation-model-data/synthetic/merged/t1_val_f0_mean_heldout.parquet",
            "t1_val_voc_desc_xc": "gs://foundation-model-data/synthetic/merged/t1_val_voc_desc_xc.parquet",
            "t3": "gs://foundation-model-data/synthetic/merged/t3.parquet",
            "t3_val": "gs://foundation-model-data/synthetic/merged/t3_val.parquet",
        },
        version="0.1.0",
        description=(
            "Pre-merged bioacoustic QA/caption conversations produced by the "
            "NatureLM-audio-data-synth pipeline. Each row is one source audio "
            "file with multiple task turns merged into a single conversation."
        ),
        sources=["XenoCanto", "iNaturalist", "SyntheticSED", "WABAD", "Birdeep"],
        license="internal",
    )

    def __init__(
        self,
        split: str = "tier1_tier2_xc_inat_v1",
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
        if split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {split!r}. Expected one of {list(self.info.split_paths.keys())}"
            )
        self._load()

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        self._data = self._backend_class.from_parquet(
            self.info.split_paths[self.split],
            streaming=self._streaming,
        )

    def _resolve_audio_path(self, row: dict[str, Any]) -> str:
        audio_paths = json.loads(row["audio_paths"])
        if not audio_paths:
            raise ValueError(
                f"Empty audio_paths for audio_id={row.get('audio_id')!r}"
            )
        path = audio_paths[0]
        # Some paths were double-encoded by the synthesis pipeline (%25XX).
        # Decode once so gcsfs sees the single-encoded form that matches GCS object names.
        if re.search(r'%25[0-9A-Fa-f]{2}', path):
            path = unquote(path)
        if path.startswith("gs://") or path.startswith("/"):
            return path
        dataset = row.get("dataset", "")
        root = _AUDIO_ROOTS.get(dataset)
        if root is None:
            raise ValueError(
                f"No audio root configured for dataset {dataset!r}. "
                f"Add it to _AUDIO_ROOTS in data_synth_conversations.py."
            )
        if dataset == "XenoCanto":
            stem = path.rsplit(".", 1)[0] if "." in path.split("/")[-1] else path
            path = stem + ".wav"
        return root + path

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        audio_path = self._resolve_audio_path(row)
        try:
            audio, sr = read_audio(anypath(audio_path))
        except (FileNotFoundError, OSError):
            # XenoCanto bucket has a mix of .wav and .WAV extensions
            if audio_path.endswith(".wav"):
                audio, sr = read_audio(anypath(audio_path[:-4] + ".WAV"))
            else:
                raise
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

        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        row["audio"] = audio
        row["sample_rate"] = sr
        row["messages"] = messages

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
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["DataSynthConversations", dict[str, Any]]:
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            data_root=cfg.get("data_root"),
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
            f"{self.info.name} (v{self.info.version}), split: {self.split}\n"
            f"Samples: {n}\n"
            f"Description: {self.info.description}"
        )
