"""DRASDIC: multi-audio synthetic evaluation and training datasets.

Provides access to synthetic multi-audio tasks where each example contains
2-7 audio files paired with a conversation that references them via multiple
``<Audio><AudioHere></Audio>`` placeholders.

Available splits
----------------
- ``sed_fewshot``: ~50k few-shot sound event detection examples (2-4
  support clips + 1 query clip, predict target timestamps).
- ``call_type_all``: ~100k call-type tasks across 5 sub-tasks.
- ``call_type_multiple_choice``: ~25k 4-choice call-type matching.
- ``call_type_binary``: ~25k binary call-type detection (Yes/No).
- ``call_type_binary_timestamps``: ~25k binary detection with timestamps.
- ``call_type_same_different``: ~12.5k same/different call-type comparison.
- ``call_type_counting``: ~12.4k target-sound counting.
- ``fewshot_detection``: ~100k v2 few-shot multi-label detection
  examples (examples + query, predict which call types are present).
- ``feature_conditioned_detection``: ~100k v2 feature-conditioned
  detection examples (acoustic feature descriptions + examples + query,
  predict label).
- ``call_type_all_v2_16k`` / ``call_type_all_v2_32k`` (and matching
  ``call_type_*_v2_16k`` / ``call_type_*_v2_32k`` sub-splits): ~200k v2
  call-type tasks each. Same task layout (multiple-choice ~75k, binary
  ~50k, binary timestamps ~25k, same/different ~25k, counting ~25k);
  the suffix indicates the audio's native sample rate.
- ``species_mcq``: ~50k 4-way species multiple-choice (XC + iNat,
  natively 32 kHz audio).
- ``same_species``: ~200k binary same-species detection — given 2-5
  reference clips, predict whether a query clip is the same species.
  XC + iNat, natively 32 kHz.

Most splits ship 16 kHz WAV; ``*_v2_32k`` splits ship 32 kHz WAV. Each
row returns a list of numpy arrays in ``audios``, ordered to match the
``<AudioHere>`` positions in the prompt.
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

_BASE_ROOT = "gs://foundation-model-data/synthetic/multi-audio"
_V1_ROOT = "gs://foundation-model-data/synthetic"

_SPLIT_DIRS: dict[str, str] = {
    "sed_fewshot": f"{_BASE_ROOT}/synthetic_sed_fewshot_16k",
    "call_type_all": f"{_BASE_ROOT}/synthetic_call_type_tasks_16k",
    "call_type_all_v1": f"{_V1_ROOT}/synthetic_call_type_tasks_16k_v1",
    "call_type_all_v2_16k": f"{_BASE_ROOT}/synthetic_call_type_tasks_16k_v2",
    "call_type_all_v2_32k": f"{_BASE_ROOT}/synthetic_call_type_tasks_32k_v2",
    "fewshot_detection": f"{_BASE_ROOT}/synthetic_fewshot_detection_16k_v2",
    "feature_conditioned_detection": (f"{_BASE_ROOT}/synthetic_feature_detection_gcs_16k_v2"),
}

# Default audio root for v0 splits (audio paths include the dataset subdir).
_DEFAULT_AUDIO_ROOT = _BASE_ROOT

# Per-split audio root overrides (v1 audio paths are relative to a different root).
_AUDIO_ROOT_OVERRIDES: dict[str, str] = {
    "call_type_all_v1": _V1_ROOT,
    "call_type_multiple_choice_v1": _V1_ROOT,
    "call_type_binary_v1": _V1_ROOT,
    "call_type_same_different_v1": _V1_ROOT,
    "call_type_counting_v1": _V1_ROOT,
    "species_mcq": "gs://esp-ml-datasets/",
    "same_species": "gs://esp-ml-datasets/",
}

# Sub-splits that filter call_type_all by template_path
_CALL_TYPE_SUBSPLITS: dict[str, str] = {
    "call_type_multiple_choice": "audio_synth/multiple_choice",
    "call_type_binary": "audio_synth/binary_audio",
    "call_type_binary_timestamps": "audio_synth/binary_audio_timestamps",
    "call_type_same_different": "audio_synth/same_different",
    "call_type_counting": "audio_synth/counting",
}

# v1 sub-splits (filter call_type_all_v1 — excludes timestamps)
_CALL_TYPE_V1_SUBSPLITS: dict[str, str] = {
    "call_type_multiple_choice_v1": "audio_synth/multiple_choice",
    "call_type_binary_v1": "audio_synth/binary_audio",
    "call_type_same_different_v1": "audio_synth/same_different",
    "call_type_counting_v1": "audio_synth/counting",
}

# v2 sub-splits: maps split name → (parent split, template_path). The 16k and
# 32k flavors share the task layout but ship audio at different native rates;
# both filter their parent ``call_type_all_v2_*`` conversations.jsonl.
_CALL_TYPE_V2_TEMPLATES: dict[str, str] = {
    "multiple_choice": "audio_synth/multiple_choice",
    "binary": "audio_synth/binary_audio",
    "binary_timestamps": "audio_synth/binary_audio_timestamps",
    "same_different": "audio_synth/same_different",
    "counting": "audio_synth/counting",
}

_CALL_TYPE_V2_SUBSPLITS: dict[str, tuple[str, str]] = {
    f"call_type_{task}_v2_{rate}": (f"call_type_all_v2_{rate}", template)
    for rate in ("16k", "32k")
    for task, template in _CALL_TYPE_V2_TEMPLATES.items()
}

_TASK_BY_TEMPLATE: dict[str, str] = {
    "audio_synth/fewshot_sed": "few_shot_sed",
    "audio_synth/multiple_choice": "call_type_multiple_choice",
    "audio_synth/binary_audio": "call_type_binary",
    "audio_synth/binary_audio_timestamps": "call_type_binary_timestamps",
    "audio_synth/same_different": "call_type_same_different",
    "audio_synth/counting": "call_type_counting",
    "audio_synth/fewshot_detection": "fewshot_detection",
    "species_mcq": "species_mcq",
    "same_species": "same_species",
    "audio_synth/feature_conditioned_detection_gcs": "feature_conditioned_detection",
}

_ALL_SPLITS = (
    list(_SPLIT_DIRS)
    + list(_CALL_TYPE_SUBSPLITS)
    + list(_CALL_TYPE_V1_SUBSPLITS)
    + list(_CALL_TYPE_V2_SUBSPLITS)
    + ["species_mcq", "same_species"]
)


# ── Dataset class ────────────────────────────────────────────────────────


@register_dataset
class DRASDIC(Dataset):
    """DRASDIC multi-audio synthetic dataset.

    Description
    -----------
    Multi-audio tasks where each example presents 2-7 audio clips
    alongside a conversation containing multiple ``<AudioHere>``
    placeholders.  The ``audios`` field is a list of numpy arrays
    ordered to match the placeholder positions in the prompt.

    Two source corpora are included:

    - **SED fewshot**: given support clips with annotated timestamps,
      detect the target sound in a query clip.
    - **Call-type tasks**: given example call-type clips, answer
      questions about a target recording (multiple choice, binary
      detection, same/different, counting).

    Examples
    --------
    >>> from esp_data.datasets.drasdic import DRASDIC
    >>> ds = DRASDIC(split="sed_fewshot", sample_rate=16000)
    >>> row = ds[0]
    >>> len(row["audios"])  # 2-5 audio arrays
    3
    >>> row["messages"][0]["content"].count("<AudioHere>")
    3
    """

    info = DatasetInfo(
        name="drasdic",
        owner="david",
        split_paths={
            "sed_fewshot": f"{_SPLIT_DIRS['sed_fewshot']}/conversations.jsonl",
            "call_type_all": f"{_SPLIT_DIRS['call_type_all']}/conversations.jsonl",
            "fewshot_detection": f"{_SPLIT_DIRS['fewshot_detection']}/conversations.jsonl",
            "feature_conditioned_detection": (
                f"{_SPLIT_DIRS['feature_conditioned_detection']}/conversations.jsonl"
            ),
            **{
                sub: f"{_SPLIT_DIRS['call_type_all']}/conversations.jsonl"
                for sub in _CALL_TYPE_SUBSPLITS
            },
            "call_type_all_v1": f"{_SPLIT_DIRS['call_type_all_v1']}/conversations.jsonl",
            **{
                sub: f"{_SPLIT_DIRS['call_type_all_v1']}/{sub}.jsonl"
                for sub in _CALL_TYPE_V1_SUBSPLITS
            },
            "call_type_all_v2_16k": (f"{_SPLIT_DIRS['call_type_all_v2_16k']}/conversations.jsonl"),
            "call_type_all_v2_32k": (f"{_SPLIT_DIRS['call_type_all_v2_32k']}/conversations.jsonl"),
            **{
                sub: f"{_SPLIT_DIRS[parent]}/conversations.jsonl"
                for sub, (parent, _) in _CALL_TYPE_V2_SUBSPLITS.items()
            },
            "species_mcq": "gs://esp-data-ingestion/drasdic/v0.1.0/species_mcq.jsonl",
            "same_species": "gs://esp-data-ingestion/drasdic/v0.1.0/same_species.jsonl",
        },
        version="0.1.0",
        description=(
            "DRASDIC multi-audio synthetic tasks: few-shot SED, "
            "call-type classification/detection, few-shot detection, and "
            "feature-conditioned detection with 2-7 audio clips per example."
        ),
        sources=[
            "gs://foundation-model-data/synthetic/multi-audio/synthetic_sed_fewshot_16k",
            "gs://foundation-model-data/synthetic/multi-audio/synthetic_call_type_tasks_16k",
            "gs://foundation-model-data/synthetic/multi-audio/synthetic_fewshot_detection_16k_v2",
            "gs://foundation-model-data/synthetic/multi-audio/synthetic_feature_detection_gcs_16k_v2",
            "gs://foundation-model-data/synthetic/multi-audio/synthetic_call_type_tasks_16k_v2",
            "gs://foundation-model-data/synthetic/multi-audio/synthetic_call_type_tasks_32k_v2",
            "gs://esp-data-ingestion/drasdic/v0.1.0/same_species.jsonl",
        ],
        license="internal",
    )

    def __init__(
        self,
        split: str = "sed_fewshot",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the DRASDIC dataset.

        Parameters
        ----------
        split : str
            Split to load. One of: ``sed_fewshot``, ``call_type_all``,
            ``call_type_multiple_choice``, ``call_type_binary``,
            ``call_type_binary_timestamps``, ``call_type_same_different``,
            ``call_type_counting``, ``fewshot_detection``,
            ``feature_conditioned_detection``, ``call_type_all_v1`` (with
            ``_v1`` sub-splits), ``call_type_all_v2_16k`` (with
            ``_v2_16k`` sub-splits), or ``call_type_all_v2_32k`` (with
            ``_v2_32k`` sub-splits, natively 32 kHz).
        output_take_and_give : dict[str, str] | None
            Optional column rename mapping.
        sample_rate : int | None
            Target sample rate for resampling. Source audio is 16 kHz for
            most splits; ``*_v2`` splits are natively 32 kHz.
        data_root : str | AnyPathT | None
            Override for the audio root directory. If ``None``, uses
            the default GCS path for the split.
        backend : BackendType
            Backend for tabular loading (``"polars"`` or ``"pandas"``).
        streaming : bool
            Whether to use streaming mode.

        Raises
        ------
        LookupError
            If ``split`` is not a valid split name.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        if split not in _ALL_SPLITS:
            raise LookupError(f"Invalid split: {split!r}. Expected one of {_ALL_SPLITS}")
        self.split = split
        self.sample_rate = sample_rate
        self._data = None

        # Audio paths in the JSONL include the dataset subdirectory name
        # (e.g. "synthetic_sed_fewshot_16k/audio/file.wav"), so the audio
        # root is the shared parent directory.  v1 splits use a different root.
        default_root = _AUDIO_ROOT_OVERRIDES.get(split, _DEFAULT_AUDIO_ROOT)
        self.data_root = anypath(data_root) if data_root else anypath(default_root)

        self._load()

    def _load(self) -> None:
        jsonl_path = self.info.split_paths[self.split]

        # Read line-by-line to skip malformed rows (some JSONL files
        # contain lines broken by unescaped newlines inside strings).
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

        # Filter by template_path for v0 and v2 call-type sub-splits (single
        # combined conversations.jsonl). v1 sub-splits have pre-split JSONL
        # files and don't need filtering.
        if self.split in _CALL_TYPE_SUBSPLITS:
            template = _CALL_TYPE_SUBSPLITS[self.split]
            self._data = self._data.filter_isin("template_path", [template])
        elif self.split in _CALL_TYPE_V2_SUBSPLITS:
            _, template = _CALL_TYPE_V2_SUBSPLITS[self.split]
            self._data = self._data.filter_isin("template_path", [template])

    @property
    def columns(self) -> list[str]:
        """Return column names of the loaded data."""
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        """Return all valid split names."""
        return list(_ALL_SPLITS)

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
        row["task"] = _TASK_BY_TEMPLATE.get(row.get("template_path", ""), self.split)

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["DRASDIC", dict[str, Any]]:
        """Create a DRASDIC instance from a dataset config.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration with ``split``, ``sample_rate``, etc.

        Returns
        -------
        tuple[DRASDIC, dict[str, Any]]
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
            f"Available splits: {', '.join(_ALL_SPLITS)}"
        )
