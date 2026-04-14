"""AudioSkillsXL dataset.

Multi-task audio understanding dataset combining conversation-style QA from
nvidia/AudioSkills with AudioSet audio.  Each source is a split; the ``all``
split is the union of every source.  Audio for AudioSet-backed sources
(wavcaps, audioset, audioset_sl) is resolved against AudioSet v0.2.0 on GCS.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import librosa
import numpy as np
import polars as pl

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType, DataBackend, PandasBackend, PolarsBackend
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCES: list[str] = [
    "counting_qa",
    "wavcaps",
    "fsd50k",
    "clotho_v2",
    "audioset",
    "audioset_sl",
]

_LOCAL_SOURCES: set[str] = {"counting_qa", "fsd50k", "clotho_v2"}
_AUDIOSET_BACKED_SOURCES: set[str] = {"wavcaps", "audioset", "audioset_sl"}

_GCS_ROOT = "gs://esp-ml-datasets/audio_skills_xl/v0.1.0/raw/"
_AUDIOSET_GCS_ROOT = "gs://esp-ml-datasets/audioset/v0.2.0/raw/"
_MESSAGES_DTYPE = pl.List(
    pl.Struct(
        [
            pl.Field("role", pl.String),
            pl.Field("content", pl.String),
        ]
    )
)


def _parse_messages_json(value: Any) -> Any:
    """Parse JSON-encoded chat messages into structured turns."""
    if isinstance(value, str):
        return json.loads(value)
    return value


@register_dataset
class AudioSkillsXL(Dataset):
    """AudioSkillsXL – multi-source audio understanding dataset.

    Description
    -----------
    Combines conversation-annotated audio splits from
    `nvidia/AudioSkills <https://huggingface.co/datasets/nvidia/AudioSkills>`_
    with AudioSet audio resolved against AudioSet v0.2.0 on GCS.

    **Local-audio sources** (audio hosted under ``audio_skills_xl``):

    - ``counting_qa`` – Sound event counting and temporal QA.
    - ``fsd50k`` – Sound event classification and captioning QA.
    - ``clotho_v2`` – Audio captioning QA.

    **AudioSet-backed sources** (audio from AudioSet v0.2.0 on GCS):

    - ``wavcaps`` – WavCaps / AudioSet Strongly Labeled captioning QA.
    - ``audioset`` – Full AudioSet conversation QA (2.8 M → ~1.6 M with audio).
    - ``audioset_sl`` – AudioSet Strongly Labeled subset QA.

    All entries carry a ``messages`` column (JSON-encoded list of
    ``{"role": "user"|"assistant", "content": "..."}`` turns).

    Pre-resampled Audio
    -------------------
    AudioSet-backed entries include ``16khz_path`` and ``32khz_path``
    columns pointing to pre-resampled audio on GCS.  Local-audio sources
    gain these columns after running the resample pipeline step.

    Examples
    --------
    >>> from esp_data.datasets import AudioSkillsXL
    >>> ds = AudioSkillsXL(split="fsd50k")
    >>> len(ds)
    2004

    >>> ds_all = AudioSkillsXL(split="all")
    >>> ds_32k = AudioSkillsXL(split="wavcaps", sample_rate=32000)
    """

    info = DatasetInfo(
        name="audio_skills_xl",
        owner="david",
        split_paths={
            "counting_qa": f"{_GCS_ROOT}counting_qa.csv",
            "wavcaps": f"{_GCS_ROOT}wavcaps.csv",
            "fsd50k": f"{_GCS_ROOT}fsd50k.csv",
            "clotho_v2": f"{_GCS_ROOT}clotho_v2.csv",
            "audioset": f"{_GCS_ROOT}audioset.csv",
            "audioset_sl": f"{_GCS_ROOT}audioset_sl.csv",
            "all": "all",
        },
        version="0.1.0",
        description=(
            "Multi-task audio understanding: conversation-annotated splits "
            "from nvidia/AudioSkills (CountingQA, WavCaps, FSD50k, Clotho-v2, "
            "AudioSet, AudioSet-SL) with audio resolved against AudioSet v0.2.0."
        ),
        sources=[
            "nvidia/AudioSkills",
            "AudioSet (Gemmeke et al. 2017)",
        ],
        license="multiple (CC BY 4.0 for AudioSet, see individual sources)",
    )

    _sample_rate_paths: dict[int, str] = {
        32000: "32khz_path",
        16000: "16khz_path",
    }

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        sources: list[str] | None = None,
        data_root: str | AnyPathT | None = None,
        audioset_data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load — one of the source names or ``"all"``.
        output_take_and_give : dict[str, str] | None
            Optional column rename/filter mapping.
        sample_rate : int | None
            Target sample rate.  Pre-resampled audio is used when the
            path column is populated; otherwise resampled on-the-fly.
        sources : list[str] | None
            Restrict ``"all"`` to a subset of sources.
        data_root : str | AnyPathT | None
            Root for local-audio sources.  Defaults to GCS.
        audioset_data_root : str | AnyPathT | None
            Root for AudioSet-backed audio.  Defaults to GCS.
        backend : BackendType
            ``"polars"`` or ``"pandas"``.
        streaming : bool
            Lazy / streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None

        self.data_root = anypath(data_root) if data_root else anypath(_GCS_ROOT)
        self.audioset_data_root = (
            anypath(audioset_data_root) if audioset_data_root else anypath(_AUDIOSET_GCS_ROOT)
        )

        self._load()

        if sources is not None:
            unknown = set(sources) - set(SOURCES)
            if unknown:
                raise ValueError(f"Unknown sources: {sorted(unknown)}. Valid: {SOURCES}")
            self._data = self._data.filter_isin("source", sources)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    @property
    def available_sources(self) -> list[str]:
        if self._data is None:
            return []
        return sorted(self._data.get_unique("source"))

    @property
    def available_sample_rates(self) -> list[int]:
        available = []
        if self._data is not None:
            for sr, col in self._sample_rate_paths.items():
                if col in self._data.columns:
                    available.append(sr)
        return available

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        if self.split == "all":
            backends = [self._load_source(s) for s in SOURCES]
            self._data = self._backend_class.concat(backends)
        else:
            self._data = self._load_source(self.split)

    def _resolve_split_path(self, source: str) -> str:
        gcs_path = self.info.split_paths[source]
        csv_name = gcs_path.rsplit("/", 1)[-1]

        for root in (self.data_root, self.audioset_data_root):
            root_str = str(root)
            if not root_str.startswith("gs://"):
                from pathlib import Path

                local = Path(root_str) / csv_name
                if local.exists():
                    return str(local)
        return gcs_path

    def _load_source(self, source: str):
        path = self._resolve_split_path(source)
        if self._backend_class is PandasBackend:
            return PandasBackend.from_csv(
                path,
                streaming=self._streaming,
                converters={"messages": _parse_messages_json},
            )

        backend = self._backend_class.from_csv(path, streaming=self._streaming)
        return self._normalize_messages_backend(backend)

    def _normalize_messages_backend(self, backend: DataBackend) -> DataBackend:
        """Ensure `messages` has the same structured schema as chat-native datasets."""
        if "messages" not in backend.columns or not isinstance(backend, PolarsBackend):
            return backend

        schema = backend._df.collect_schema() if backend._streaming else backend._df.schema
        if schema["messages"] != pl.String:
            return backend

        normalized_df = backend._df.with_columns(
            pl.col("messages").str.json_decode(_MESSAGES_DTYPE).alias("messages")
        )
        return PolarsBackend(normalized_df, streaming=backend._streaming)

    # ------------------------------------------------------------------
    # Audio path resolution
    # ------------------------------------------------------------------

    _local_sr_prefix: dict[int, str] = {
        16000: "audio_16k",
        32000: "audio_32k",
    }

    def _resolve_audio_path(self, row: dict[str, Any]) -> tuple[AnyPathT, bool]:
        """Return ``(audio_path, is_presampled)``."""
        source = row.get("source", "")
        is_audioset_backed = source in _AUDIOSET_BACKED_SOURCES
        root = self.audioset_data_root if is_audioset_backed else self.data_root

        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            col = self._sample_rate_paths[self.sample_rate]
            val = row.get(col)
            if val is not None and str(val).strip() and str(val).lower() != "nan":
                if is_audioset_backed:
                    return root / str(val), True
                # Local sources: 16khz_path is relative (no dir prefix),
                # prepend audio_16k/ or audio_32k/
                prefix = self._local_sr_prefix.get(self.sample_rate, "")
                return root / prefix / str(val), True

        return root / str(row["audio_path"]), False

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        audio_path, is_presampled = self._resolve_audio_path(row)

        audio, sr = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        if not is_presampled and self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )

        row["audio"] = audio

        if "messages" in row and isinstance(row["messages"], str):
            try:
                row["messages"] = json.loads(row["messages"])
            except (json.JSONDecodeError, TypeError):
                pass

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}

        return row

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["AudioSkillsXL", dict[str, Any]]:
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            sources=cfg.get("sources"),
            data_root=cfg.get("data_root"),
            audioset_data_root=cfg.get("audioset_data_root"),
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        n = len(self) if self._data is not None and not self._streaming else "?"
        srcs = ", ".join(self.available_sources) if self._data is not None else "?"
        return (
            f"{base}\n"
            f"Samples: {n}\n"
            f"Sources: {srcs}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
