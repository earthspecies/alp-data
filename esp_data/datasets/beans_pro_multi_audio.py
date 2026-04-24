"""BEANS-Pro multi-audio evaluation benchmark.

Pre-computed multi-audio evaluation tasks where each example contains
2+ audio files and a conversation with multiple ``<AudioHere>``
placeholders. Includes few-shot gibbon detection and other multi-audio
evaluation tasks.

Available splits
----------------

- ``gibbon-fewshot-detection``: 18,554 examples, 3-way gibbon call
  detection with fixed A/B/C support exemplars and optional background
  environment audio.
- ``gibbon-fewshot-detection-balanced``: 868 examples, balanced
  present-vs-none subset of the same task.
- ``giant-otter-same-different``: 1000 examples, same/different call-type
  pairs from the giant otter vocal repertoire (22 call types).
- ``giant-otter-4way``: 500 examples, 4-way multiple-choice call-type
  matching from the giant otter vocal repertoire.
- ``dcase-4way``: 1378 examples, 4-way multiple-choice species/sound
  detection from DCASE 2021 Task 5 (17 sound types).
- ``dcase-fewshot-detection``: 13,688 examples, 4-way few-shot multi-label
  sound detection from DCASE 2021 Task 5 with fixed support exemplars and
  optional background environment audio.
- ``dcase-fewshot-detection-balanced``: 3,158 examples, balanced
  present-vs-none subset of the same task.
- ``crow-4way``: 200 examples, 4-way multiple-choice call-type matching
  for carrion crow (*Corvus corone*, 25 call types). Aligned 1:1 with
  the ``crow-description`` split in `BeansPro`.
- ``zebra-4way``: 40 examples, 4-way multiple-choice call-type matching
  for plains zebra (*Equus quagga*, 4 call types). Aligned 1:1 with
  the ``zebra-description`` split in `BeansPro`.
- ``unseen-species-4way``: 1227 examples, 4-way species classification
  for 172 held-out species (genus seen), random confusers.
- ``unseen-species-4way-hard``: 1227 examples, same targets as above
  with same-genus confusers where available (~50%).
- ``unseen-genus-4way``: 927 examples, 4-way species classification
  for 77 species from held-out genera (family seen), random confusers.
- ``unseen-genus-4way-hard``: 927 examples, same targets as above
  with same-family confusers where available (~61%).
- ``unseen-family-4way``: 440 examples, 4-way species classification
  for 25 species from held-out families, random confusers.
- ``unseen-family-4way-hard``: 440 examples, same targets as above
  with same-genus confusers where available (~47%).
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
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, filesystem_from_path, read_audio

logger = logging.getLogger(__name__)

# ── Split configuration ──────────────────────────────────────────────────

_GCS_BASE = "gs://esp-data-ingestion/beans-pro/v0.1.0/raw"

_SPLITS: dict[str, str] = {
    "gibbon-fewshot-multipulse": f"{_GCS_BASE}/gibbon_fewshot_multipulse/test.jsonl",
    "gibbon-fewshot-singlepulse": f"{_GCS_BASE}/gibbon_fewshot_singlepulse/test.jsonl",
    "gibbon-fewshot-duet": f"{_GCS_BASE}/gibbon_fewshot_duet/test.jsonl",
    "gibbon-fewshot-tiny": f"{_GCS_BASE}/gibbon_fewshot_tiny/test.jsonl",
    "gibbon-fewshot-detection": f"{_GCS_BASE}/gibbon_fewshot_detection/test.jsonl",
    "gibbon-fewshot-detection-balanced": (
        f"{_GCS_BASE}/gibbon_fewshot_detection_balanced/test.jsonl"
    ),
    "same-species": f"{_GCS_BASE}/same_species/test.jsonl",
    "giant-otter-same-different": f"{_GCS_BASE}/giant_otter_same_different/test.jsonl",
    "giant-otter-4way": f"{_GCS_BASE}/giant_otter_4way/test.jsonl",
    "dcase-4way": f"{_GCS_BASE}/dcase_4way/test.jsonl",
    "dcase-fewshot-detection": f"{_GCS_BASE}/dcase_fewshot_detection/test.jsonl",
    "dcase-fewshot-detection-balanced": (
        f"{_GCS_BASE}/dcase_fewshot_detection_balanced/test.jsonl"
    ),
    "crow-4way": f"{_GCS_BASE}/crow_4way/test.jsonl",
    "zebra-4way": f"{_GCS_BASE}/zebra_4way/test.jsonl",
    "unseen-species-4way": f"{_GCS_BASE}/unseen_species_4way/test.jsonl",
    "unseen-species-4way-hard": f"{_GCS_BASE}/unseen_species_4way_hard/test.jsonl",
    "unseen-genus-4way": f"{_GCS_BASE}/unseen_genus_4way/test.jsonl",
    "unseen-genus-4way-hard": f"{_GCS_BASE}/unseen_genus_4way_hard/test.jsonl",
    "unseen-family-4way": f"{_GCS_BASE}/unseen_family_4way/test.jsonl",
    "unseen-family-4way-hard": f"{_GCS_BASE}/unseen_family_4way_hard/test.jsonl",
}

# Default audio root — gibbon audio is copied into the beans-pro folder,
# so audio paths like "audio/gibbons/32KHz/file.wav" resolve correctly.
_DEFAULT_AUDIO_ROOT = f"{_GCS_BASE}/"

# Per-split overrides when audio paths use a different root.
# same-species audio lives in XC/iNat under gs://esp-ml-datasets/.
_AUDIO_ROOT_OVERRIDES: dict[str, str] = {
    "gibbon-fewshot-detection": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/",
    "gibbon-fewshot-detection-balanced": ("gs://esp-ml-datasets/beans-zero/v0.1.0/raw/"),
    "same-species": "gs://esp-ml-datasets/",
    "dcase-fewshot-detection": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/",
    "dcase-fewshot-detection-balanced": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/",
    "crow-4way": f"{_GCS_BASE}/carrion_crow_descriptions/",
    "zebra-4way": f"{_GCS_BASE}/zebra_descriptions/",
    "unseen-species-4way": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/",
    "unseen-species-4way-hard": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/",
    "unseen-genus-4way": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/",
    "unseen-genus-4way-hard": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/",
    "unseen-family-4way": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/",
    "unseen-family-4way-hard": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/",
}


@register_dataset
class BeansProMultiAudio(Dataset):
    """BEANS-Pro multi-audio evaluation benchmark.

    Description
    -----------
    Pre-computed multi-audio evaluation tasks. Each example returns a
    list of audio arrays via the ``audios`` field, ordered to match
    ``<AudioHere>`` placeholder positions in the prompt.

    Includes both binary gibbon call-type detection (2 support clips for
    one target type, answer Yes/No), fixed-label gibbon detection
    (A/B/C exemplars plus optional background environment, answer
    A/B/C/None), and DCASE few-shot multi-label detection with answer
    sets such as ``A, C`` or ``None``.

    Examples
    --------
    >>> from esp_data.datasets.beans_pro_multi_audio import BeansProMultiAudio
    >>> ds = BeansProMultiAudio(split="gibbon-fewshot-tiny", sample_rate=32000)
    >>> row = ds[0]
    >>> len(row["audios"])
    3
    """

    info = DatasetInfo(
        name="beans_pro_multi_audio",
        owner="david",
        split_paths=_SPLITS,
        version="0.1.0",
        description=(
            "BEANS-Pro multi-audio evaluation benchmark. "
            "Includes few-shot gibbon detection and other multi-audio "
            "evaluation tasks."
        ),
        sources=["Hainan Gibbons (BEANS-Zero)"],
        license="CC-BY-NC-SA",
    )

    def __init__(
        self,
        split: str = "gibbon-fewshot-tiny",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        split : str
            Split to load. One of the keys in ``info.split_paths``.
        output_take_and_give : dict[str, str] | None
            Optional column rename mapping.
        sample_rate : int | None
            Target sample rate for audio resampling.
        data_root : str | AnyPathT | None
            Override for the audio root directory. If ``None``, uses
            the BEANS-Zero raw GCS path.
        backend : BackendType
            Backend for tabular loading.
        streaming : bool
            Whether to use streaming mode.

        Raises
        ------
        LookupError
            If ``split`` is not a valid split name.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        if split not in _SPLITS:
            raise LookupError(f"Invalid split: {split!r}. Expected one of {list(_SPLITS)}")
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        default_root = _AUDIO_ROOT_OVERRIDES.get(split, _DEFAULT_AUDIO_ROOT)
        self.data_root = anypath(data_root) if data_root else anypath(default_root)
        self._load()

    def _load(self) -> None:
        jsonl_path = _SPLITS[self.split]
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
        """Return column names of the loaded data."""
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        """Return all valid split names."""
        return list(_SPLITS)

    def _load_audio(self, rel_path: str) -> np.ndarray:
        """Load and optionally resample a single audio file.

        Parameters
        ----------
        rel_path : str
            Path relative to ``data_root``.

        Returns
        -------
        np.ndarray
            Mono float32 audio waveform.
        """
        full_path = self.data_root / rel_path
        audio, sr = read_audio(full_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
        return audio

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        audio_paths = row.get("audio_paths")
        if not isinstance(audio_paths, list) or not audio_paths:
            raise ValueError(
                f"Expected non-empty 'audio_paths' list in row {row.get('id', '<unknown>')!r}"
            )
        audios = [self._load_audio(p) for p in audio_paths]
        row["audios"] = audios
        row["task"] = row.get("task", self.split)

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
        cls,
        dataset_config: DatasetConfig,
    ) -> tuple["BeansProMultiAudio", dict[str, Any]]:
        """Create instance from a dataset config.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration with ``split``, ``sample_rate``, etc.

        Returns
        -------
        tuple[BeansProMultiAudio, dict[str, Any]]
            The dataset and any transformation metadata.
        """
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
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{base}, {n} examples\n"
            f"Description: {self.info.description}\n"
            f"Available splits: {', '.join(_SPLITS)}"
        )
