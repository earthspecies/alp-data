"""BEANS-Pro dataset: acoustic description matching benchmark."""

from typing import Any, Dict, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_SYNTH = "gs://foundation-model-data/synthetic/data-synth"
_NBM_AUDIO_ROOT = "gs://esp-ml-datasets/nocturnal_bird_migration/"

# Mapping from BeansPro split name → data-synth run key (used to build GCS path).
_SYNTH_SPLIT_KEYS: dict[str, str] = {
    # Powdermill
    "species-count-oe-powdermill": "species_count_oe_powdermill_v1_clean",
    "voc-loc-mcq-powdermill": "voc_loc_mcq_powdermill_v1_clean",
    "voc-cooccurrence-binary-powdermill": "voc_cooccurrence_binary_powdermill_v1_clean",
    "vocal-dominance-oe-powdermill": "vocal_dominance_oe_powdermill_v1_clean",
    "vocal-dominance-mcq-powdermill": "vocal_dominance_mcq_powdermill_v1_clean",
    "species-voc-order-oe-powdermill": "species_voc_order_oe_powdermill_v1_clean",
    "species-voc-order-mcq-powdermill": "species_voc_order_mcq_powdermill_v1_clean",
    "highest-pitch-species-oe-powdermill": "highest_pitch_species_oe_powdermill_v1_clean",
    "highest-pitch-species-mcq-powdermill": "highest_pitch_species_mcq_powdermill_v1_clean",
    "lowest-pitch-species-oe-powdermill": "lowest_pitch_species_oe_powdermill_v1_clean",
    "lowest-pitch-species-mcq-powdermill": "lowest_pitch_species_mcq_powdermill_v1_clean",
    "longest-voc-species-oe-powdermill": "longest_voc_species_oe_powdermill_v1_clean",
    "longest-voc-species-mcq-powdermill": "longest_voc_species_mcq_powdermill_v1_clean",
    "structural-caption-powdermill": "tier1_structural_caption_powdermill_v1_clean",
    # NBM (Nocturnal Bird Migration)
    "species-count-oe-nbm": "species_count_oe_nbm_v1_clean",
    "voc-loc-mcq-nbm": "voc_loc_mcq_nbm_v1_clean",
    "voc-cooccurrence-binary-nbm": "voc_cooccurrence_binary_nbm_v1_clean",
    "vocal-dominance-oe-nbm": "vocal_dominance_oe_nbm_v1_clean",
    "vocal-dominance-mcq-nbm": "vocal_dominance_mcq_nbm_v1_clean",
    "species-voc-order-oe-nbm": "species_voc_order_oe_nbm_v1_clean",
    "species-voc-order-mcq-nbm": "species_voc_order_mcq_nbm_v1_clean",
    "highest-pitch-species-oe-nbm": "highest_pitch_species_oe_nbm_v1_clean",
    "highest-pitch-species-mcq-nbm": "highest_pitch_species_mcq_nbm_v1_clean",
    "lowest-pitch-species-oe-nbm": "lowest_pitch_species_oe_nbm_v1_clean",
    "lowest-pitch-species-mcq-nbm": "lowest_pitch_species_mcq_nbm_v1_clean",
    "longest-voc-species-oe-nbm": "longest_voc_species_oe_nbm_v1_clean",
    "longest-voc-species-mcq-nbm": "longest_voc_species_mcq_nbm_v1_clean",
    "structural-caption-nbm": "tier1_structural_caption_nbm_v1_clean",
}

_SYNTH_SPLIT_PATHS: dict[str, str] = {
    split: f"{_GCS_SYNTH}/{key}.jsonl"
    for split, key in _SYNTH_SPLIT_KEYS.items()
}

# NBM audio paths in conversations are relative to the NBM audio root.
# Powdermill audio paths are absolute GCS paths (no root needed → None).
_SYNTH_DATA_ROOTS: dict[str, str | None] = {
    **{s: None for s in _SYNTH_SPLIT_KEYS if s.endswith("-powdermill")},
    **{s: _NBM_AUDIO_ROOT for s in _SYNTH_SPLIT_KEYS if s.endswith("-nbm")},
}


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

    Original BEANS-Pro splits (BEANS-Zero schema)
    ---------------------------------------------
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

    Powdermill data-synth splits (Conversation schema)
    ---------------------------------------------------
    Generated from PowdermillCropped dawn-chorus recordings; stored in
    data-synth Conversation format (``audio_paths``, ``messages``).
    ``_process`` normalises these to ``instruction`` / ``output`` / ``audio``.

    - ``species-count-oe-powdermill``: 1,632 OE species-counting questions.
    - ``voc-loc-mcq-powdermill``: 892 MCQ vocalization-location questions.
    - ``voc-cooccurrence-binary-powdermill``: 902 binary co-occurrence questions.
    - ``vocal-dominance-oe-powdermill``: 1,177 OE vocal-dominance questions.
    - ``vocal-dominance-mcq-powdermill``: 901 MCQ vocal-dominance questions.
    - ``species-voc-order-oe-powdermill``: 1,308 OE species-by-vocalization-order questions.
    - ``species-voc-order-mcq-powdermill``: 868 MCQ species-by-vocalization-order questions.
    - ``highest-pitch-species-oe-powdermill``: 963 OE highest-pitch-species questions.
    - ``highest-pitch-species-mcq-powdermill``: 566 MCQ highest-pitch-species questions.
    - ``lowest-pitch-species-oe-powdermill``: 684 OE lowest-pitch-species questions.
    - ``lowest-pitch-species-mcq-powdermill``: 401 MCQ lowest-pitch-species questions.
    - ``longest-voc-species-oe-powdermill``: 1,113 OE longest-vocalization-species questions.
    - ``longest-voc-species-mcq-powdermill``: 752 MCQ longest-vocalization-species questions.
    - ``structural-caption-powdermill``: 1,586 structural captions.

    NBM data-synth splits (Conversation schema)
    --------------------------------------------
    Generated from NocturnalBirdMigration train recordings; same schema as
    Powdermill synth splits above. Audio paths are relative to
    ``gs://esp-ml-datasets/nocturnal_bird_migration/``.

    - ``species-count-oe-nbm``: 883 OE species-counting questions.
    - ``voc-loc-mcq-nbm``: 115 MCQ vocalization-location questions.
    - ``voc-cooccurrence-binary-nbm``: 88 binary co-occurrence questions.
    - ``vocal-dominance-oe-nbm``: 113 OE vocal-dominance questions.
    - ``vocal-dominance-mcq-nbm``: 117 MCQ vocal-dominance questions.
    - ``species-voc-order-oe-nbm``: 438 OE species-by-vocalization-order questions.
    - ``species-voc-order-mcq-nbm``: 185 MCQ species-by-vocalization-order questions.
    - ``highest-pitch-species-oe-nbm``: 136 OE highest-pitch-species questions.
    - ``highest-pitch-species-mcq-nbm``: 108 MCQ highest-pitch-species questions.
    - ``lowest-pitch-species-oe-nbm``: 128 OE lowest-pitch-species questions.
    - ``lowest-pitch-species-mcq-nbm``: 99 MCQ lowest-pitch-species questions.
    - ``longest-voc-species-oe-nbm``: 110 OE longest-vocalization-species questions.
    - ``longest-voc-species-mcq-nbm``: 98 MCQ longest-vocalization-species questions.
    - ``structural-caption-nbm``: 883 structural captions.

    Schema
    ------
    Original splits — each row is a JSONL record with fields:
    - ``instruction``: Full prompt with ``<Audio><AudioHere></Audio>`` tag,
      question text, and four labelled choices (A-D).
    - ``output``: Correct answer letter (A, B, C, or D).
    - ``audio_path_original_sample_rate``: Relative path to audio file.
    - ``metadata``: JSON string with call_type, species, duration, etc.

    Powdermill / NBM synth splits — each row is a data-synth Conversation:
    - ``audio_paths``: List of audio file paths (one element per clip).
    - ``messages``: List of ``{role, content}`` dicts (user + assistant).
    ``_process`` adds ``instruction`` and ``output`` keys derived from messages.

    Examples
    --------
    >>> from esp_data.datasets import BeansPro
    >>> dataset = BeansPro(
    ...     split="crow-description",
    ...     sample_rate=16000,
    ...     data_root="gs://esp-data-ingestion/beans-pro/v0.1.0/raw/carrion_crow_descriptions/"
    ... )
    >>> synth = BeansPro(split="species-count-oe-powdermill", sample_rate=16000)
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
            **_SYNTH_SPLIT_PATHS,
        },
        version="0.1.0",
        description=(
            "BEANS-Pro evaluation benchmark. "
            "Includes acoustic description matching, mean F0 prediction, "
            "binary taxonomic presence, call-type tasks, and data-synth "
            "multi-species reasoning splits from Powdermill and NBM."
        ),
        sources=[
            "ESP cooperative crows preprint",
            "Xie et al. 2024, R. Soc. Open Sci.",
            "Musikhin et al. 2025, F0 Bioacoustic Benchmark",
            "Xeno-canto / iNaturalist (val_unseen splits)",
            "BEANS-Zero call variants",
            "Powdermill / Chronister et al. 2021",
            "NocturnalBirdMigration / Zenodo 14039937",
        ],
        license="CC-BY-NC-4.0, CC0-1.0, Public Domain, CC BY-ND 3.0",
    )

    # Data roots per split (used when data_root is None).
    _default_data_roots: dict[str, str | None] = {
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
        **_SYNTH_DATA_ROOTS,
    }

    _originals_path_column = "audio_path_original_sample_rate"

    # Splits stored in data-synth Conversation format (audio_paths + messages).
    _synth_format_splits: frozenset[str] = frozenset(_SYNTH_SPLIT_KEYS)

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
        if self.split in self._synth_format_splits:
            # data-synth Conversation format: audio_paths is a list; messages holds the QA.
            audio_path_raw = row["audio_paths"][0]
            if self.data_root is not None:
                audio_path = anypath(self.data_root) / audio_path_raw
            else:
                audio_path = anypath(audio_path_raw)
            messages = row["messages"]
            row["instruction"] = messages[0]["content"]
            row["output"] = messages[-1]["content"]
        else:
            audio_path = anypath(self.data_root) / row[self._originals_path_column]

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
