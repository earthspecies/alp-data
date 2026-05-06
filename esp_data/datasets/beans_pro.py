"""BEANS-Pro dataset: acoustic description matching benchmark."""

import json
import re
from typing import Any, Dict, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_AUDIO_TAG_RE = re.compile(r"\s*<Audio><AudioHere></Audio>\s*")
_CROPPED_AUDIO_ROOTS = {
    "birdeep_cropped": "gs://foundation-model-data/synthetic/cropped/birdeep/audio",
    "BirdeepCropped": "gs://foundation-model-data/synthetic/cropped/birdeep/audio",
    "powdermill_cropped": "gs://foundation-model-data/synthetic/cropped/powdermill/audio",
    "PowdermillCropped": "gs://foundation-model-data/synthetic/cropped/powdermill/audio",
    "birdvox_full_night_cropped": "gs://foundation-model-data/synthetic/cropped/birdvox_full_night/audio",
    "BirdVoxFullNightCropped": "gs://foundation-model-data/synthetic/cropped/birdvox_full_night/audio",
    "wabad_cropped": "gs://foundation-model-data/synthetic/cropped/wabad/audio",
    "WABADCropped": "gs://foundation-model-data/synthetic/cropped/wabad/audio",
}
_BIRDVOX_DCASE_AUDIO_ROOT = (
    "gs://foundation-model-data/synthetic/cropped/birdvox_dcase_20k/audio"
)

_T3_SPLIT_PATHS = {
    "t3-bird-presence-biophony-binary": "gs://foundation-model-data/synthetic/beanspro_draft/t3/bird_presence_biophony_binary.jsonl",
    "t3-frequency-range-description": "gs://foundation-model-data/synthetic/beanspro_draft/t3/frequency_range_description_oe.jsonl",
    "t3-ordered-species-summary": "gs://foundation-model-data/synthetic/beanspro_draft/t3/ordered_species_summary_oe.jsonl",
    "t3-species-by-highest-pitch-mcq": "gs://foundation-model-data/synthetic/beanspro_draft/t3/species_by_highest_pitch_mcq.jsonl",
    "t3-species-by-highest-pitch-oe": "gs://foundation-model-data/synthetic/beanspro_draft/t3/species_by_highest_pitch_oe.jsonl",
    "t3-species-by-longest-vocalization-mcq": "gs://foundation-model-data/synthetic/beanspro_draft/t3/species_by_longest_vocalization_mcq.jsonl",
    "t3-species-by-longest-vocalization-oe": "gs://foundation-model-data/synthetic/beanspro_draft/t3/species_by_longest_vocalization_oe.jsonl",
    "t3-species-by-lowest-pitch-mcq": "gs://foundation-model-data/synthetic/beanspro_draft/t3/species_by_lowest_pitch_mcq.jsonl",
    "t3-species-by-lowest-pitch-oe": "gs://foundation-model-data/synthetic/beanspro_draft/t3/species_by_lowest_pitch_oe.jsonl",
    "t3-species-by-vocalization-frequency-mcq": "gs://foundation-model-data/synthetic/beanspro_draft/t3/species_by_vocalization_frequency_mcq.jsonl",
    "t3-species-by-vocalization-frequency-oe": "gs://foundation-model-data/synthetic/beanspro_draft/t3/species_by_vocalization_frequency_oe.jsonl",
    "t3-species-by-vocalization-order-mcq": "gs://foundation-model-data/synthetic/beanspro_draft/t3/species_by_vocalization_order_mcq.jsonl",
    "t3-species-by-vocalization-order-oe": "gs://foundation-model-data/synthetic/beanspro_draft/t3/species_by_vocalization_order_oe.jsonl",
    "t3-species-count-oe": "gs://foundation-model-data/synthetic/beanspro_draft/t3/species_count_oe.jsonl",
    "t3-species-listing-open-list": "gs://foundation-model-data/synthetic/beanspro_draft/t3/species_listing_open_list.jsonl",
    "t3-structural-captioning": "gs://foundation-model-data/synthetic/beanspro_draft/t3/structural_captioning_caption.jsonl",
    "t3-vocalization-cooccurrence-binary": "gs://foundation-model-data/synthetic/beanspro_draft/t3/vocalization_cooccurrence_binary.jsonl",
    "t3-vocalization-count-per-species-oe": "gs://foundation-model-data/synthetic/beanspro_draft/t3/vocalization_count_per_species_oe.jsonl",
    "t3-vocalization-count-total-mcq": "gs://foundation-model-data/synthetic/beanspro_draft/t3/vocalization_count_total_mcq.jsonl",
    "t3-vocalization-count-total-oe": "gs://foundation-model-data/synthetic/beanspro_draft/t3/vocalization_count_total_oe.jsonl",
    "t3-vocalization-presence-binary": "gs://foundation-model-data/synthetic/beanspro_draft/t3/vocalization_presence_binary.jsonl",
    "t3-vocalization-referring-mcq": "gs://foundation-model-data/synthetic/beanspro_draft/t3/vocalization_referring_mcq.jsonl",
}


def _message_content(row: dict[str, Any], role: str) -> str | None:
    """Return the first conversation message content for a role.

    Parameters
    ----------
    row : dict[str, Any]
        Raw JSONL row.
    role : str
        Message role to extract.

    Returns
    -------
    str or None
        Message content when present.
    """
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != role:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _metadata_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Parse the JSON metadata column when available.

    Parameters
    ----------
    row : dict[str, Any]
        Raw JSONL row.

    Returns
    -------
    dict[str, Any]
        Parsed metadata mapping, or an empty mapping when unavailable.
    """
    meta = row.get("metadata")
    if isinstance(meta, dict):
        return dict(meta)
    if isinstance(meta, str) and meta.strip():
        try:
            parsed = json.loads(meta)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _first_string(value: object) -> str | None:
    """Return the first non-empty string from a scalar or list-like value.

    Parameters
    ----------
    value : object
        Scalar string or list-like value.

    Returns
    -------
    str or None
        First non-empty string when present.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _basename_without_suffix(path: str) -> str:
    """Return the final path component without a trailing ``.wav`` suffix.

    Parameters
    ----------
    path : str
        Audio path or URI.

    Returns
    -------
    str
        Basename with a trailing ``.wav`` removed when present.
    """
    name = path.rstrip("/").rsplit("/", 1)[-1]
    if name.lower().endswith(".wav"):
        return name[:-4]
    return name


def _cropped_audio_path(row: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    """Return the canonical cropped audio URI for a BeansPro row.

    Parameters
    ----------
    row : dict[str, Any]
        Raw or partially-normalized BeansPro row.
    metadata : dict[str, Any]
        Parsed metadata for ``row``.

    Returns
    -------
    str or None
        Canonical cropped audio URI when the row contains enough information.
    """
    audio_path = _first_string(row.get("audio_path"))
    if audio_path and "/synthetic/cropped/" in audio_path:
        return audio_path

    audio_id = _first_string(row.get("audio_ids"))
    audio_paths_value = _first_string(row.get("audio_paths")) or ""

    if "birdvox_dcase_20k" in audio_paths_value:
        dcase_id = audio_id or _basename_without_suffix(audio_paths_value)
        return f"{_BIRDVOX_DCASE_AUDIO_ROOT}/{dcase_id}.wav"

    source_dataset = metadata.get("source_dataset") or row.get("dataset")
    if not isinstance(source_dataset, str):
        return None

    root = _CROPPED_AUDIO_ROOTS.get(source_dataset)
    if root is None or audio_id is None:
        return None

    if source_dataset in {"BirdVoxFullNightCropped", "birdvox_full_night_cropped"}:
        if not audio_id.startswith("BirdVox-full-night_"):
            return None
    elif "__crop_" not in audio_id:
        return None

    return f"{root}/{audio_id}.wav"


def _normalize_synthetic_conversation_row(row: dict[str, Any]) -> dict[str, Any]:
    """Expose conversation-style synthetic rows through the flat BeansPro schema.

    Parameters
    ----------
    row : dict[str, Any]
        Raw JSONL row from a synthetic BeansPro split.

    Returns
    -------
    dict[str, Any]
        Row with flat ``instruction``, ``output``, and
        ``audio_path_original_sample_rate`` fields populated when possible.
    """
    out = dict(row)
    metadata = _metadata_dict(out)

    cropped_path = _cropped_audio_path(out, metadata)
    if cropped_path is not None:
        out["audio_path_original_sample_rate"] = cropped_path

    if "audio_path_original_sample_rate" not in out:
        audio_path = _first_string(out.get("audio_path"))
        if audio_path is not None:
            out["audio_path_original_sample_rate"] = audio_path
    if "audio_path_original_sample_rate" not in out:
        audio_path = _first_string(out.get("audio_paths"))
        if audio_path is not None:
            out["audio_path_original_sample_rate"] = audio_path

    instruction = out.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        user_content = _message_content(out, "user")
        if user_content is not None:
            out["instruction"] = user_content
            out.setdefault("instruction_text", _AUDIO_TAG_RE.sub("", user_content).strip())

    output = out.get("output")
    if not isinstance(output, str) or not output.strip():
        assistant_content = _message_content(out, "assistant")
        if assistant_content is not None:
            out["output"] = assistant_content

    for source_key, target_key in (
        ("source_dataset", "source_dataset"),
        ("source_file", "file_name"),
        ("beanspro_category", "category"),
        ("beanspro_task", "task"),
        ("beanspro_format", "format"),
    ):
        value = metadata.get(source_key)
        if isinstance(value, str | int | float | bool):
            out.setdefault(target_key, value)
    return out


@register_dataset
class BeansPro(Dataset):
    """BEANS-Pro acoustic description matching benchmark.

    Description
    -----------
    BEANS-Pro evaluates multimodal audio-language models on their ability
    to match animal vocalizations to expert acoustic descriptions. Each
    example presents an audio clip and four acoustic descriptions (one
    correct, three distractors from the same species), and the model must
    identify the correct description.

    Descriptions are sourced verbatim from published bioacoustics papers
    and verified against the original figures and tables.

    Available splits
    ----------------
    - ``crow-description``: 200 examples, 25 call types (merged from 40),
      carrion crow (*Corvus corone*). Source: ESP cooperative crows preprint.
    - ``zebra-description``: 40 examples, 4 call types, plains zebra
      (*Equus quagga*). Source: Xie et al. 2024, R. Soc. Open Sci.
    - ``f0-mean-seen-taxa``: 2086 examples, mean F0 prediction across
      9 seen taxa. Source: Musikhin et al. 2025, F0 Bioacoustic Benchmark.
    - ``f0-mean-heldout-taxa``: 571 examples, mean F0 prediction for
      spotted hyenas (held-out taxon). Source: Musikhin et al. 2025.
    - ``bird-presence``: 3478 balanced examples, bird vocalization
      detection (Yes/No). Source: XC + iNat val_unseen.
    - ``mammal-presence``: 468 balanced examples, mammal vocalization
      detection. Source: XC + iNat val_unseen.
    - ``insect-presence``: 1176 balanced examples, insect sound
      detection. Source: XC + iNat val_unseen.
    - ``amphibian-presence``: 1818 balanced examples, amphibian
      vocalization detection. Source: XC + iNat val_unseen.
    - ``alarm-call-presence``: 36 balanced examples, alarm call binary
      detection. Source: BEANS-Zero call variants.
    - ``flight-call-presence``: 192 balanced examples, flight call
      binary detection. Source: BEANS-Zero call variants.
    - ``call-type-fixed-vocab``: 999 examples, 5-label multilabel
      call-type classification. Source: BEANS-Zero call variants.

    Tier-1 synthetic splits (XC + iNat beanszero and new splits)
    -------------------------------------------------------------
    - ``t1-snr-mcq``: 2068 examples, SNR prediction MCQ (fixed-bin and
      custom-bin variants merged; ``task`` field distinguishes them).
      Source: XC new split.
    - ``t1-snr-binary``: 1881 examples, SNR binary threshold (Yes/No).
      Source: XC new split.
    - ``t1-snr-regression``: 2014 examples, SNR open-ended regression.
      Source: XC new split.
    - ``t1-description-mcq``: 1368 examples, vocalization description MCQ
      from field notes / occurrence remarks.
      Source: XC + iNat beanszero and new splits.
    - ``t1-caption``: 984 examples, acoustic captioning from field notes.
      Source: XC + iNat beanszero and new splits.
    - ``t2-captioning``: 10783 examples, semantic captioning from field notes,
      filtered by LLM-judge audibility/interest/difficulty scores.
      Source: XC + iNat beanszero and new splits.
    - ``t2-behavior``: 3824 examples, behavior multiple-choice questions
      filtered by LLM-judge audibility and difficulty scores.
      Source: XC + iNat beanszero and new splits.

    Audio paths in the synthetic tier splits are absolute GCS URIs (``gs://``);
    ``data_root`` is ignored for these splits.

    Schema
    ------
    Each row is a JSONL record with fields matching BEANS-Zero:
    - ``instruction``: Full prompt with ``<Audio><AudioHere></Audio>`` tag,
      question text, and four labelled choices (A-D).
    - ``output``: Correct answer letter (A, B, C, or D).
    - ``audio_path_original_sample_rate``: Relative path to audio file.
    - ``metadata``: JSON string with call_type, species, duration, etc.

    Examples
    --------
    >>> from esp_data.datasets import BeansPro
    >>> dataset = BeansPro(
    ...     split="crow-description",
    ...     sample_rate=16000,
    ...     data_root="gs://esp-data-ingestion/beans-pro/v0.1.0/raw/carrion_crow_descriptions/"
    ... )
    """

    info = DatasetInfo(
        name="beans_pro",
        owner="david",
        split_paths={
            "crow-description": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/carrion_crow_descriptions/test.jsonl",
            "zebra-description": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/zebra_descriptions/test.jsonl",
            "f0-mean-seen-taxa": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/f0_mean_seen_taxa/test.jsonl",
            "f0-mean-heldout-taxa": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/f0_mean_heldout_taxa/test.jsonl",
            "bird-presence": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/bird_presence/test.jsonl",
            "mammal-presence": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/mammal_presence/test.jsonl",
            "insect-presence": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/insect_presence/test.jsonl",
            "amphibian-presence": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/amphibian_presence/test.jsonl",
            "alarm-call-presence": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/alarm_call_presence/test.jsonl",
            "flight-call-presence": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/flight_call_presence/test.jsonl",
            "call-type-fixed-vocab": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/call_type_fixed_vocab/test.jsonl",
            "t1-snr-mcq": "gs://foundation-model-data/synthetic/beanspro/t1-snr-mcq.jsonl",
            "t1-snr-binary": "gs://foundation-model-data/synthetic/beanspro/t1-snr-binary.jsonl",
            "t1-snr-regression": "gs://foundation-model-data/synthetic/beanspro/t1-snr-regression.jsonl",
            "t1-description-mcq": "gs://foundation-model-data/synthetic/beanspro/t1-description-mcq.jsonl",
            "t1-caption": "gs://foundation-model-data/synthetic/beanspro/t1-caption.jsonl",
            "t2-captioning": "gs://foundation-model-data/synthetic/beanspro/t2-captioning.jsonl",
            "t2-behavior": "gs://foundation-model-data/synthetic/beanspro/t2-behavior.jsonl",
            **_T3_SPLIT_PATHS,
        },
        version="0.1.0",
        description=(
            "BEANS-Pro evaluation benchmark. "
            "Includes acoustic description matching, mean F0 prediction, "
            "binary taxonomic presence, and call-type tasks."
        ),
        sources=[
            "ESP cooperative crows preprint",
            "Xie et al. 2024, R. Soc. Open Sci.",
            "Musikhin et al. 2025, F0 Bioacoustic Benchmark",
            "Xeno-canto / iNaturalist (val_unseen splits)",
            "BEANS-Zero call variants",
            "Xeno-canto / iNaturalist (beanszero + new splits, LLM-synthesized)",
        ],
        license="CC-BY-NC-4.0, CC0-1.0",
    )

    # Data roots per split (used when data_root is None)
    _default_data_roots = {
        "crow-description": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/carrion_crow_descriptions/",
        "zebra-description": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/zebra_descriptions/",
        "f0-mean-seen-taxa": "gs://esp-data-ingestion/f0-prediction/audio/",
        "f0-mean-heldout-taxa": "gs://esp-data-ingestion/f0-prediction/audio/",
        "bird-presence": "gs://esp-ml-datasets/",
        "mammal-presence": "gs://esp-ml-datasets/",
        "insect-presence": "gs://esp-ml-datasets/",
        "amphibian-presence": "gs://esp-ml-datasets/",
        "alarm-call-presence": "gs://esp-ml-datasets/",
        "flight-call-presence": "gs://esp-ml-datasets/",
        "call-type-fixed-vocab": "gs://esp-ml-datasets/",
        # Tier-1 splits use absolute GCS paths; data_root is unused.
        "t1-snr-mcq": "",
        "t1-snr-binary": "",
        "t1-snr-regression": "",
        "t1-description-mcq": "",
        "t1-caption": "",
        "t2-captioning": "",
        "t2-behavior": "",
        **{split: "" for split in _T3_SPLIT_PATHS},
    }

    _originals_path_column = "audio_path_original_sample_rate"
    _synthetic_splits = {
        "t1-snr-mcq",
        "t1-snr-binary",
        "t1-snr-regression",
        "t1-description-mcq",
        "t1-caption",
        "t2-captioning",
        "t2-behavior",
        *_T3_SPLIT_PATHS,
    }

    def __init__(
        self,
        split: str = "crow-description",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the BEANS-Pro dataset.

        Parameters
        ----------
        split : str
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
        sample_rate : int
            The sample rate to which audio files should be resampled.
        data_root : str | AnyPathT, optional
            The root directory for the dataset. If None, uses the default
            GCS path for the selected split.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data: pd.DataFrame = None
        self._load()
        self.sample_rate = sample_rate

        if data_root is None:
            self.data_root = self._default_data_roots.get(
                split, anypath(self.info.split_paths[split]).parent
            )
        else:
            self.data_root = data_root

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        return []  # Only original sample rate available

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_json(location, lines=True, orient="records")
        if self.split in self._synthetic_splits:
            self._make_synthetic_ids_unique()

    def _make_synthetic_ids_unique(self) -> None:
        """Rewrite synthetic row ids to be unique within a BEANS-Pro split.

        Some source synthetic JSONLs reuse local/source ids across rows and
        across task variants. Preserve those ids in `source_id` and expose a
        deterministic split-qualified row id in `id` so prediction/evaluation
        systems can safely use it as a primary key.
        """
        if len(self._data) == 0:
            return
        source_ids = [
            str(row.get("source_id", row.get("id", i))) for i, row in enumerate(self._data)
        ]
        row_ids = [f"{self.split}:{source_id}:{i}" for i, source_id in enumerate(source_ids)]
        self._data = self._data.add_column("source_id", source_ids)
        self._data = self._data.add_column("id", row_ids)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["BeansPro", dict[str, Any]]:
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
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata
        return ds, {}

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        row = _normalize_synthetic_conversation_row(row)
        path_val = row[self._originals_path_column]
        if "://" in str(path_val):
            audio_path = anypath(path_val)
        else:
            audio_path = anypath(self.data_root) / path_val
        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )

        row["audio"] = audio

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        return item

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No split has been loaded yet.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    def __str__(self) -> str:
        base_info = f"{self.info.name} (v{self.info.version}), split: {self.split}"
        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
