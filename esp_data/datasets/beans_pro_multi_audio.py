"""BEANS-Pro multi-audio evaluation benchmark.

Pre-computed multi-audio evaluation tasks where each example contains
1+ audio files and a conversation with one or more ``<AudioHere>``
placeholders. Includes few-shot detection and 4-way audio MCQ tasks.

Available splits
----------------

- ``gibbon-fewshot-detection``: 18,554 examples, 3-way gibbon call
  detection with fixed A/B/C support exemplars and optional background
  environment audio.
- ``gibbon-fewshot-detection-balanced``: 868 examples, balanced
  present-vs-none subset of the same task.
- ``giant-otter-4way``: 500 examples, 4-way multiple-choice call-type
  matching from the giant otter vocal repertoire.
- ``dcase-fewshot-detection-balanced``: 3,158 examples, balanced
  present-vs-none 4-way few-shot multi-label sound detection from DCASE
  2021 Task 5.
- ``crow-4way``: 200 examples, 4-way multiple-choice call-type matching
  for carrion crow (*Corvus corone*, 25 call types). Aligned 1:1 with
  the ``crow-description`` split in `BeansPro`.
- ``zebra-4way``: 40 examples, 4-way multiple-choice call-type matching
  for plains zebra (*Equus quagga*, 4 call types). Aligned 1:1 with the
  ``zebra-description`` split in `BeansPro`.
- ``unseen-species-4way``: 1227 examples, 4-way species classification
  for 172 held-out species (genus seen), random confusers.
- ``weldy-multi-call-type-fewshot``: Within-species call-type
  discrimination from the Weldy NW Dawn Chorus dataset, presented as a
  variable-N-way (2-4) few-shot multi-audio MCQ. One exemplar clip per
  call variant of the species + 1 query. Aligned 1:1 with the text-only
  ``weldy-multi-call-type`` split in `BeansPro`.
- ``raincoast-2025-pulsed-whistle-fewshot``: Raincoast 2025 killer whale
  pulsed-call vs whistle classification with fixed labeled support clips.
- ``ford-catalogue-pulsed-discrete-4way``: Ford catalogue Northern Resident
  pulsed-discrete 4-way call-type classification with audio support options.
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
    "gibbon-fewshot-detection": f"{_GCS_BASE}/gibbon_fewshot_detection/test.jsonl",
    "gibbon-fewshot-detection-balanced": (
        f"{_GCS_BASE}/gibbon_fewshot_detection_balanced/test.jsonl"
    ),
    "giant-otter-4way": f"{_GCS_BASE}/giant_otter_4way/test.jsonl",
    "dcase-fewshot-detection-balanced": (
        f"{_GCS_BASE}/dcase_fewshot_detection_balanced/test.jsonl"
    ),
    "crow-4way": f"{_GCS_BASE}/crow_4way/test.jsonl",
    # Holt et al. 2017 noisy-miner repertoire, crow-4way prompt, all 13
    # supplement exemplars as options. Local episodes, absolute audio paths.
    "noisy-miner-repertoire-13way": (
        "/home/david_earthspecies_org/esp-data-dev/esp-research/projects/"
        "NatureLM-audio-v1.5/results/noisy_miner_repertoire_icl/n226_13way_n30/episodes.jsonl"
    ),
    # Same 13 Holt exemplars as options; queries are A2O windows. See
    # scripts/build_noisy_miner_a2o_icl.py (NatureLM-audio-v1.5).
    "noisy-miner-repertoire-13way-a2o-magpie": (
        "/home/david_earthspecies_org/esp-data-dev/esp-research/projects/"
        "NatureLM-audio-v1.5/results/noisy_miner_repertoire_icl/"
        "n226_13way_a2o_magpie_n30/episodes.jsonl"
    ),
    "noisy-miner-repertoire-13way-a2o-miner-only": (
        "/home/david_earthspecies_org/esp-data-dev/esp-research/projects/"
        "NatureLM-audio-v1.5/results/noisy_miner_repertoire_icl/"
        "n226_13way_a2o_miner_only_n30/episodes.jsonl"
    ),
    "noisy-miner-repertoire-13way-a2o-miner-sed-crop": (
        "/home/david_earthspecies_org/esp-data-dev/esp-research/projects/"
        "NatureLM-audio-v1.5/results/noisy_miner_repertoire_icl/"
        "n226_13way_a2o_miner_sed_crop_n30/episodes.jsonl"
    ),
    # crow-4way with each option's acoustic description inline after its audio
    # slot. See scripts/build_crow_4way_audio_desc.py (NatureLM-audio-v1.5).
    "crow-4way-audio-desc": f"{_GCS_BASE}/crow_4way_audio_desc/test.jsonl",
    # crow-4way-audio-desc in the TRAINED calltype_desc_mcq / calltype_desc_textonly prompt
    # format ("Here are 4 call types."), each row carrying its trained 1/2/4 s audio_max_sec.
    # See scripts/build_crow_4way_calltype_desc_trained.py (NatureLM-audio-v1.5).
    "crow-4way-calltype-desc": f"{_GCS_BASE}/crow_4way_calltype_desc/test.jsonl",
    "crow-4way-calltype-desc-textonly": f"{_GCS_BASE}/crow_4way_calltype_desc_textonly/test.jsonl",
    "zebra-4way": f"{_GCS_BASE}/zebra_4way/test.jsonl",
    # zebra-4way with each option's acoustic description inline after its audio
    "zebra-4way-audio-desc": f"{_GCS_BASE}/zebra_4way_audio_desc/test.jsonl",
    # Zebra-finch vocal repertoire (Elie & Theunissen 2016): 11 call types with
    # canonical paper descriptions; bird-disjoint supports. audio / text /
    # audio_desc on identical episodes so the text channel can be isolated.
    "zf-repertoire-4way-1shot-audio": f"{_GCS_BASE}/zf_repertoire_4way_1shot_audio/test.jsonl",
    "zf-repertoire-4way-1shot-audio-desc": f"{_GCS_BASE}/zf_repertoire_4way_1shot_audio_desc/test.jsonl",
    "zf-repertoire-4way-2shot-audio": f"{_GCS_BASE}/zf_repertoire_4way_2shot_audio/test.jsonl",
    "zf-repertoire-4way-2shot-audio-desc": f"{_GCS_BASE}/zf_repertoire_4way_2shot_audio_desc/test.jsonl",
    "zf-repertoire-4way-3shot-audio": f"{_GCS_BASE}/zf_repertoire_4way_3shot_audio/test.jsonl",
    "zf-repertoire-4way-3shot-audio-desc": f"{_GCS_BASE}/zf_repertoire_4way_3shot_audio_desc/test.jsonl",
    "zf-repertoire-4way-text": f"{_GCS_BASE}/zf_repertoire_4way_text/test.jsonl",
    "zf-repertoire-8way-1shot-audio": f"{_GCS_BASE}/zf_repertoire_8way_1shot_audio/test.jsonl",
    "zf-repertoire-8way-1shot-audio-desc": f"{_GCS_BASE}/zf_repertoire_8way_1shot_audio_desc/test.jsonl",
    "zf-repertoire-8way-2shot-audio": f"{_GCS_BASE}/zf_repertoire_8way_2shot_audio/test.jsonl",
    "zf-repertoire-8way-2shot-audio-desc": f"{_GCS_BASE}/zf_repertoire_8way_2shot_audio_desc/test.jsonl",
    "zf-repertoire-8way-3shot-audio": f"{_GCS_BASE}/zf_repertoire_8way_3shot_audio/test.jsonl",
    "zf-repertoire-8way-3shot-audio-desc": f"{_GCS_BASE}/zf_repertoire_8way_3shot_audio_desc/test.jsonl",
    "zf-repertoire-8way-text": f"{_GCS_BASE}/zf_repertoire_8way_text/test.jsonl",
    "zf-repertoire-4way-1shot-audio-desc-acoustic": f"{_GCS_BASE}/zf_repertoire_4way_1shot_audio_desc_acoustic/test.jsonl",
    "zf-repertoire-4way-2shot-audio-desc-acoustic": f"{_GCS_BASE}/zf_repertoire_4way_2shot_audio_desc_acoustic/test.jsonl",
    "zf-repertoire-4way-3shot-audio-desc-acoustic": f"{_GCS_BASE}/zf_repertoire_4way_3shot_audio_desc_acoustic/test.jsonl",
    "zf-repertoire-4way-text-acoustic": f"{_GCS_BASE}/zf_repertoire_4way_text_acoustic/test.jsonl",
    "zf-repertoire-8way-1shot-audio-desc-acoustic": f"{_GCS_BASE}/zf_repertoire_8way_1shot_audio_desc_acoustic/test.jsonl",
    "zf-repertoire-8way-2shot-audio-desc-acoustic": f"{_GCS_BASE}/zf_repertoire_8way_2shot_audio_desc_acoustic/test.jsonl",
    "zf-repertoire-8way-3shot-audio-desc-acoustic": f"{_GCS_BASE}/zf_repertoire_8way_3shot_audio_desc_acoustic/test.jsonl",
    "zf-repertoire-8way-text-acoustic": f"{_GCS_BASE}/zf_repertoire_8way_text_acoustic/test.jsonl",
    "unseen-species-4way": f"{_GCS_BASE}/unseen_species_4way/test.jsonl",
    # Fair species-exemplar MCQ from BirdSet SNE+UHH soundscapes (singleton
    # windows, 5 distinct recordings per row). See
    # scripts/build_birdset_species_mcq.py (NatureLM-audio-v1.5).
    "birdset-species-mcq-4way": f"{_GCS_BASE}/birdset_species_mcq_4way/test.jsonl",
    "birdset-species-mcq-4way-full": f"{_GCS_BASE}/birdset_species_mcq_4way_full/test.jsonl",
    # label-inclusion variant: each option line names its species
    "birdset-species-mcq-4way-full-labeled": (
        f"{_GCS_BASE}/birdset_species_mcq_4way_full_labeled/test.jsonl"
    ),
    # Weldy NW Dawn Chorus within-species call-type discrimination as a
    # few-shot multi-audio MCQ (K exemplar clips per call variant + 1 query).
    # Aligned 1:1 with the text-only ``weldy-multi-call-type`` split in
    # `BeansPro`, but presents audio exemplars instead of text descriptions.
    "weldy-multi-call-type-fewshot": (f"{_GCS_BASE}/weldy_multi_call_type_fewshot/test.jsonl"),
    "weldy-multi-call-type-fewshot-1shot": (
        f"{_GCS_BASE}/weldy_multi_call_type_fewshot_1shot/test.jsonl"
    ),
    # 1-shot MCQ with Weldy's OWN sonotype descriptions (annotation_metadata.tsv)
    # appended to each option; episodes/answer key identical to the audio-only split.
    "weldy-multi-call-type-fewshot-1shot-audio-desc": (
        f"{_GCS_BASE}/weldy_multi_call_type_fewshot_1shot_audio_desc/test.jsonl"
    ),
    # text-only (zero-shot) arm: descriptions alone, query audio only
    "weldy-multi-call-type-fewshot-1shot-text": (
        f"{_GCS_BASE}/weldy_multi_call_type_fewshot_1shot_text/test.jsonl"
    ),
    "weldy-multi-call-type-fewshot-2shot": (
        f"{_GCS_BASE}/weldy_multi_call_type_fewshot_2shot/test.jsonl"
    ),
    "weldy-multi-call-type-fewshot-3shot": (
        f"{_GCS_BASE}/weldy_multi_call_type_fewshot_3shot/test.jsonl"
    ),
    # V2 splits — drop 2-s query slots with multi-species contamination.
    # v2 = query + all option slots singleton (n=124, 10.7% of v1).
    # v2_qonly = only query singleton (n=511, 44.3% of v1). See
    # scripts/build_weldy_fewshot_v2_singleton.py.
    "weldy-multi-call-type-fewshot-v2": (
        f"{_GCS_BASE}/weldy_multi_call_type_fewshot_v2/test.jsonl"
    ),
    "weldy-multi-call-type-fewshot-v2-qonly": (
        f"{_GCS_BASE}/weldy_multi_call_type_fewshot_v2_qonly/test.jsonl"
    ),
    # v3: singleton-clean query AND supports (interval-overlap test) with
    # recording-disjoint support resampling. See
    # scripts/build_weldy_fewshot_v3.py.
    "weldy-multi-call-type-fewshot-v3-1shot": (
        f"{_GCS_BASE}/weldy_multi_call_type_fewshot_v3_1shot/test.jsonl"
    ),
    "weldy-multi-call-type-fewshot-v3-2shot": (
        f"{_GCS_BASE}/weldy_multi_call_type_fewshot_v3_2shot/test.jsonl"
    ),
    "weldy-multi-call-type-fewshot-v3-3shot": (
        f"{_GCS_BASE}/weldy_multi_call_type_fewshot_v3_3shot/test.jsonl"
    ),
    # Weldy within-species same/different call-type verification (audio-only).
    # 2422 balanced pairs (Same vs Different) across 8 species with >=2
    # ``call_N`` variants of the same sonotype; recording-disjoint. Mirrors the
    # DRASDIC ``call_type_same_different`` task shape: 2 audios + "Same" or
    # "Different". See scripts/build_weldy_same_different.py.
    "weldy-same-different": f"{_GCS_BASE}/weldy_same_different/test.jsonl",
    # Two-image (montage) variant of the same corpus: each record adds a
    # ``montage_path`` = full-res render of clip_0, and the collater renders
    # clip_1 as its standard image (routes as the DRASDIC same_different_montage
    # training task). Requires scripts/build_weldy_sd_montages.py to have
    # rendered + uploaded the PNGs to weldy_same_different/montage/.
    "weldy-same-different-montage": f"{_GCS_BASE}/weldy_same_different/test_montage.jsonl",
    "raincoast-2025-pulsed-whistle-fewshot": (
        f"{_GCS_BASE}/raincoast_2025_pulsed_whistle_fewshot/test.jsonl"
    ),
    "ford-catalogue-pulsed-discrete-4way": (
        f"{_GCS_BASE}/ford_catalogue_pulsed_discrete_4way/test.jsonl"
    ),
    # Ford catalogue NRKW pulsed-discrete few-shot MCQ variants.
    "ford-catalogue-pulsed-discrete-4way-1shot": (
        f"{_GCS_BASE}/ford_catalogue_pulsed_discrete_4way_1shot/test.jsonl"
    ),
    "ford-catalogue-pulsed-discrete-4way-2shot": (
        f"{_GCS_BASE}/ford_catalogue_pulsed_discrete_4way_2shot/test.jsonl"
    ),
    "ford-catalogue-pulsed-discrete-4way-3shot": (
        f"{_GCS_BASE}/ford_catalogue_pulsed_discrete_4way_3shot/test.jsonl"
    ),
    "ford-catalogue-pulsed-discrete-8way-1shot": (
        f"{_GCS_BASE}/ford_catalogue_pulsed_discrete_8way_1shot/test.jsonl"
    ),
    "ford-catalogue-pulsed-discrete-8way-2shot": (
        f"{_GCS_BASE}/ford_catalogue_pulsed_discrete_8way_2shot/test.jsonl"
    ),
    "ford-catalogue-pulsed-discrete-8way-3shot": (
        f"{_GCS_BASE}/ford_catalogue_pulsed_discrete_8way_3shot/test.jsonl"
    ),
    # Raincoast 2025 NRKW MA few-shot MCQ with Ford-catalogue support clips.
    "raincoast-2025-nrkw-ma-ford-support-4way-1shot": (
        f"{_GCS_BASE}/raincoast_2025_nrkw_calltype_fewshot/"
        "raincoast_2025_nrkw_ma_ford_support_4way_1shot/test.jsonl"
    ),
    "raincoast-2025-nrkw-ma-ford-support-4way-2shot": (
        f"{_GCS_BASE}/raincoast_2025_nrkw_calltype_fewshot/"
        "raincoast_2025_nrkw_ma_ford_support_4way_2shot/test.jsonl"
    ),
    "raincoast-2025-nrkw-ma-ford-support-4way-3shot": (
        f"{_GCS_BASE}/raincoast_2025_nrkw_calltype_fewshot/"
        "raincoast_2025_nrkw_ma_ford_support_4way_3shot/test.jsonl"
    ),
    # Raincoast 2025 NRKW MA few-shot MCQ with Raincoast support clips.
    "raincoast-2025-nrkw-ma-raincoast-support-4way-1shot": (
        f"{_GCS_BASE}/raincoast_2025_nrkw_calltype_fewshot/"
        "raincoast_2025_nrkw_ma_raincoast_support_4way_1shot/test.jsonl"
    ),
    "raincoast-2025-nrkw-ma-raincoast-support-4way-2shot": (
        f"{_GCS_BASE}/raincoast_2025_nrkw_calltype_fewshot/"
        "raincoast_2025_nrkw_ma_raincoast_support_4way_2shot/test.jsonl"
    ),
    "raincoast-2025-nrkw-ma-raincoast-support-4way-3shot": (
        f"{_GCS_BASE}/raincoast_2025_nrkw_calltype_fewshot/"
        "raincoast_2025_nrkw_ma_raincoast_support_4way_3shot/test.jsonl"
    ),
    # Generic in-context-learning (ICL) episode family (TRAINING split, not eval): W-way x K-shot
    # episodes over Xeno-Canto / iNaturalist (group_key, label_key) pairs. Every row carries its
    # own per-episode ``task`` (``icl_<axis>_<arm>``) and absolute ``gs://`` ``audio_paths`` +
    # parallel ``audio_windows`` (long source recordings decoded as ~10 s spans). Built offline by
    # scripts/gen_icl_episodes.py + jobs/build_icl_episodes.sh. See configs/icl_axes.yml.
    "icl-episodes-v1": "gs://foundation-model-data/synthetic/icl-episodes/v1/train.jsonl",
    # SYNTHETIC ICL episodes: worked-example / class-match / multi-axis episodes over structural
    # acoustic properties, with PRE-CUT 32 kHz clips (no windowing needed) and exact synthetic
    # labels. 199,975 rows / 2-26 clips each / 53 tasks ``icl_synth_<family>_<axis>``. This points
    # at ``train.jsonl``, NOT the raw ``conversations.jsonl`` upload: the raw file has no ``task``
    # column and still carries the generation node's local paths. See
    # scripts/build_synth_icl_manifest.py.
    "synthetic-icl-v1": (
        "gs://foundation-model-data/synthetic/multi-audio/"
        "synthetic_icl_episodes_32k_v1/train.jsonl"
    ),
    # Duck (Anatidae) detection+sub-segmentation REVIEW tasks. Each row is one detection window:
    # single audio clip + a `montage_path` overlay PNG (detection box red, syllable units yellow)
    # + the prediction echoed as bbox_2d. Three task families keyed by the `task` column
    # (duck_review_pass / _individuals / _reason). Human labels, 620 train detections.
    "duck-review-v1": (
        "gs://foundation-model-data/synthetic/multi-audio/duck_review_v1/train.jsonl"
    ),
    "duck-review-v1-test": (
        "gs://foundation-model-data/synthetic/multi-audio/duck_review_v1/test.jsonl"
    ),
    # v2 re-render: spectrograms are now padded to exactly 10 s before rendering and pixel
    # coordinates mapped against 10 s, matching the collater (dynamic_padding=false,
    # audio_max_sec=10). v1's bbox_2d were scaled by 10/window -- up to 2x wrong. Detections are
    # also capped at 9 s so window+context never exceeds 10 s and is never randomly cropped.
    "duck-review-v2": (
        "gs://foundation-model-data/synthetic/multi-audio/duck_review_v2/train.jsonl"
    ),
    "duck-review-v2-test": (
        "gs://foundation-model-data/synthetic/multi-audio/duck_review_v2/test.jsonl"
    ),
    # Subsegmentation SELF-REVIEW over GPT-6-sol-scored cascade detections. Same row shape as
    # duck-review: one detection window of audio + a `montage_path` overlay PNG (detection red,
    # predicted syllable units yellow) + the prediction echoed as bbox_2d in the 0-1000 frame.
    # Labels are GPT-6-sol verdicts (the `lenient` framing, validated at 0.711 balanced accuracy
    # against human review) rather than human ones, so this is far larger than duck-review but
    # noisier. Two task families keyed by `task`: subseg_review_pass (keep/throw_away) and
    # subseg_review_count (visible syllable count).
    "cascade-review-v1": (
        "gs://foundation-model-data/synthetic/multi-audio/cascade_review_v1/train.jsonl"
    ),
    # FIT-render twin of cascade-review-v1: the same detections, but each spectrogram spans the
    # CLIP's own duration instead of the 10 s canvas, and every bbox_2d is recomputed in that
    # frame. On the 10 s build the median clip used 25% of the canvas and 69% of the predicted
    # unit boxes were thinner than one ViT token column (10.7 units of 1000) -- the model could
    # not resolve the boxes it was asked to judge. Rows carry `render_duration_s`, which is
    # RENDER-ONLY: the collater crops a local copy for the image and leaves the waveform BEATs
    # sees padded to audio_max_sec. Tasks are suffixed `_fit` so the router and per-task metrics
    # can separate the two regimes.
    "cascade-review-fit-v1": (
        "gs://foundation-model-data/synthetic/multi-audio/cascade_review_fit_v1/train.jsonl"
    ),
    # GPT call-type acoustic descriptions over the synthetic MCQ v5 corpus, in four variants
    # (option-audio / text-only x audio / vision) keyed by the `task` column. Rows carry a
    # PER-ROW audio_max_sec of 1/2/4 s: these are ~0.4 s calls and a 17-clip episode would
    # otherwise push 170 s of padded audio through BEATs for ~7 s of content.
    "calltype-desc-v1": (
        "gs://foundation-model-data/synthetic/multi-audio/calltype_desc_v1/train.jsonl"
    ),
    # F0 contour self-review: is this pitch trace correct, and if not why / what should it be.
    # Corruptions are real tracker failure modes (flat-mean, octave errors, octave jump,
    # truncation, time shift). Rows carry a PER-ROW `audio_max_sec` bucket (1/2/4/6 s) rather
    # than the global 10 s: the median F0 contour is 0.28 s, which at 10 s spans ~2 ViT token
    # columns, and a trace occupying two tokens cannot be reviewed.
    "f0-review-v1": (
        "gs://foundation-model-data/synthetic/multi-audio/f0_review_v1/train.jsonl"
    ),
    # XC-strong detection self-review: corrupt the human selection table in one nameable way and
    # ask whether the shown detections are right, optionally with the correction.
    "xcstrong-review-v1": (
        "gs://foundation-model-data/synthetic/multi-audio/xcstrong_review_v1/train.jsonl"
    ),
}

# Default audio root for splits whose audio was copied into the beans-pro folder.
_DEFAULT_AUDIO_ROOT = f"{_GCS_BASE}/"

# Per-split overrides when audio paths use a different root.
_AUDIO_ROOT_OVERRIDES: dict[str, str] = {
    "gibbon-fewshot-detection": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/",
    "gibbon-fewshot-detection-balanced": ("gs://esp-ml-datasets/beans-zero/v0.1.0/raw/"),
    "dcase-fewshot-detection-balanced": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/",
    "crow-4way": f"{_GCS_BASE}/carrion_crow_descriptions/",
    "crow-4way-audio-desc": f"{_GCS_BASE}/carrion_crow_descriptions/",
    "crow-4way-calltype-desc": f"{_GCS_BASE}/carrion_crow_descriptions/",
    "crow-4way-calltype-desc-textonly": f"{_GCS_BASE}/carrion_crow_descriptions/",
    "zebra-4way": f"{_GCS_BASE}/zebra_descriptions/",
    "zebra-4way-audio-desc": f"{_GCS_BASE}/zebra_descriptions/",
    "zf-repertoire-4way-1shot-audio": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-4way-1shot-audio-desc": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-4way-2shot-audio": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-4way-2shot-audio-desc": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-4way-3shot-audio": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-4way-3shot-audio-desc": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-4way-text": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-8way-1shot-audio": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-8way-1shot-audio-desc": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-8way-2shot-audio": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-8way-2shot-audio-desc": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-8way-3shot-audio": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-8way-3shot-audio-desc": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-8way-text": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-4way-1shot-audio-desc-acoustic": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-4way-2shot-audio-desc-acoustic": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-4way-3shot-audio-desc-acoustic": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-4way-text-acoustic": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-8way-1shot-audio-desc-acoustic": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-8way-2shot-audio-desc-acoustic": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-8way-3shot-audio-desc-acoustic": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "zf-repertoire-8way-text-acoustic": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    "unseen-species-4way": "gs://esp-data-ingestion/",
    "birdset-species-mcq-4way": "gs://esp-ml-datasets/birdset/v0.1.0/raw/",
    "birdset-species-mcq-4way-full": "gs://esp-ml-datasets/birdset/v0.1.0/raw/",
    "birdset-species-mcq-4way-full-labeled": "gs://esp-ml-datasets/birdset/v0.1.0/raw/",
    "weldy-multi-call-type-fewshot": (f"{_GCS_BASE}/weldy_multi_call_type_fewshot/"),
    "weldy-multi-call-type-fewshot-1shot": (f"{_GCS_BASE}/weldy_multi_call_type_fewshot_1shot/"),
    "weldy-multi-call-type-fewshot-1shot-audio-desc": (
        f"{_GCS_BASE}/weldy_multi_call_type_fewshot_1shot/"
    ),
    "weldy-multi-call-type-fewshot-1shot-text": (
        f"{_GCS_BASE}/weldy_multi_call_type_fewshot_1shot/"
    ),
    "weldy-multi-call-type-fewshot-2shot": (f"{_GCS_BASE}/weldy_multi_call_type_fewshot_2shot/"),
    "weldy-multi-call-type-fewshot-3shot": (f"{_GCS_BASE}/weldy_multi_call_type_fewshot_3shot/"),
    # v2 variants reuse the v1 audio dir — audio_paths in the row are relative
    # to the original fewshot audio/ folder.
    "weldy-multi-call-type-fewshot-v2": (f"{_GCS_BASE}/weldy_multi_call_type_fewshot/"),
    "weldy-multi-call-type-fewshot-v2-qonly": (f"{_GCS_BASE}/weldy_multi_call_type_fewshot/"),
    # v3 resamples supports from the full v1 clip pool, so audio resolves
    # against the base fewshot dir.
    "weldy-multi-call-type-fewshot-v3-1shot": (f"{_GCS_BASE}/weldy_multi_call_type_fewshot/"),
    "weldy-multi-call-type-fewshot-v3-2shot": (f"{_GCS_BASE}/weldy_multi_call_type_fewshot/"),
    "weldy-multi-call-type-fewshot-v3-3shot": (f"{_GCS_BASE}/weldy_multi_call_type_fewshot/"),
    # Same-different verification: reuses the v1 audio dir (audio_paths are
    # ``audio/<window_key>.wav``, resolved against the fewshot clip corpus).
    "weldy-same-different": (f"{_GCS_BASE}/weldy_multi_call_type_fewshot/"),
    "weldy-same-different-montage": (f"{_GCS_BASE}/weldy_multi_call_type_fewshot/"),
    "raincoast-2025-pulsed-whistle-fewshot": (
        f"{_GCS_BASE}/raincoast_2025_pulsed_whistle_fewshot/"
    ),
    "ford-catalogue-pulsed-discrete-4way": "gs://esp-data-ingestion/ford-catalogue/",
    "ford-catalogue-pulsed-discrete-4way-1shot": "gs://esp-data-ingestion/ford-catalogue/",
    "ford-catalogue-pulsed-discrete-4way-2shot": "gs://esp-data-ingestion/ford-catalogue/",
    "ford-catalogue-pulsed-discrete-4way-3shot": "gs://esp-data-ingestion/ford-catalogue/",
    "ford-catalogue-pulsed-discrete-8way-1shot": "gs://esp-data-ingestion/ford-catalogue/",
    "ford-catalogue-pulsed-discrete-8way-2shot": "gs://esp-data-ingestion/ford-catalogue/",
    "ford-catalogue-pulsed-discrete-8way-3shot": "gs://esp-data-ingestion/ford-catalogue/",
    "raincoast-2025-nrkw-ma-ford-support-4way-1shot": (
        f"{_GCS_BASE}/raincoast_2025_nrkw_calltype_fewshot/"
    ),
    "raincoast-2025-nrkw-ma-ford-support-4way-2shot": (
        f"{_GCS_BASE}/raincoast_2025_nrkw_calltype_fewshot/"
    ),
    "raincoast-2025-nrkw-ma-ford-support-4way-3shot": (
        f"{_GCS_BASE}/raincoast_2025_nrkw_calltype_fewshot/"
    ),
    "raincoast-2025-nrkw-ma-raincoast-support-4way-1shot": (
        f"{_GCS_BASE}/raincoast_2025_nrkw_calltype_fewshot/"
    ),
    "raincoast-2025-nrkw-ma-raincoast-support-4way-2shot": (
        f"{_GCS_BASE}/raincoast_2025_nrkw_calltype_fewshot/"
    ),
    "raincoast-2025-nrkw-ma-raincoast-support-4way-3shot": (
        f"{_GCS_BASE}/raincoast_2025_nrkw_calltype_fewshot/"
    ),
    # ICL episode audio_paths are already absolute gs:// URIs (see _load_audio's passthrough),
    # so this root is only a nominal default and is never joined onto a path.
    "icl-episodes-v1": "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/audio_32k/",
    # Same as above: synthetic-ICL audio_paths are absolute gs:// URIs, so this root is nominal
    # and never joined onto a path.
    "synthetic-icl-v1": (
        "gs://foundation-model-data/synthetic/multi-audio/synthetic_icl_episodes_32k_v1/"
    ),
    # duck-review audio_paths are absolute gs:// URIs; root is nominal.
    "duck-review-v1": "gs://foundation-model-data/synthetic/multi-audio/duck_review_v1/",
    "duck-review-v1-test": "gs://foundation-model-data/synthetic/multi-audio/duck_review_v1/",
    "duck-review-v2": "gs://foundation-model-data/synthetic/multi-audio/duck_review_v2/",
    "duck-review-v2-test": "gs://foundation-model-data/synthetic/multi-audio/duck_review_v2/",
    "xcstrong-review-v1": "gs://foundation-model-data/synthetic/multi-audio/xcstrong_review_v1/",
    "f0-review-v1": "gs://foundation-model-data/synthetic/multi-audio/f0_review_v1/",
    # cascade-review audio_paths are absolute gs:// URIs; root is nominal.
    "cascade-review-v1": "gs://foundation-model-data/synthetic/multi-audio/cascade_review_v1/",
    "cascade-review-fit-v1": "gs://foundation-model-data/synthetic/multi-audio/cascade_review_fit_v1/",
    "calltype-desc-v1": "gs://foundation-model-data/synthetic/multi-audio/",
}


@register_dataset
class BeansProMultiAudio(Dataset):
    """BEANS-Pro multi-audio evaluation benchmark.

    Description
    -----------
    Pre-computed multi-audio evaluation tasks. Each example returns a
    list of audio arrays via the ``audios`` field, ordered to match
    ``<AudioHere>`` placeholder positions in the prompt.

    Includes fixed-label gibbon detection (A/B/C exemplars plus optional
    background environment, answer A/B/C/None), DCASE few-shot multi-label
    detection with answer sets such as ``A, C`` or ``None``, and 4-way audio
    MCQ tasks.

    Examples
    --------
    >>> from esp_data.datasets.beans_pro_multi_audio import BeansProMultiAudio
    >>> ds = BeansProMultiAudio(split="gibbon-fewshot-detection-balanced", sample_rate=32000)
    >>> row = ds[0]
    >>> len(row["audios"]) >= 4
    True
    """

    info = DatasetInfo(
        name="beans_pro_multi_audio",
        owner="david",
        split_paths=_SPLITS,
        version="0.1.0",
        description=(
            "BEANS-Pro multi-audio evaluation benchmark. "
            "Includes few-shot detection and 4-way audio MCQ tasks."
        ),
        sources=[
            "Hainan Gibbons (BEANS-Zero)",
            "DCASE 2021 Task 5",
            "Giant Otter Vocal Repertoire",
            "Carrion Crow and Plains Zebra call descriptions",
            "BEANS-Zero unseen species holdout",
            "Raincoast 2025 killer whale hydrophone annotations",
            "Ford catalogue Northern Resident killer whale calls",
        ],
        license="Mixed source licenses; evaluation-only manifests.",
    )

    def __init__(
        self,
        split: str = "gibbon-fewshot-detection-balanced",
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

    def _load_audio(
        self, rel_path: str, window: tuple[float, float] | list[float] | None = None
    ) -> np.ndarray:
        """Load and optionally resample a single audio file.

        Parameters
        ----------
        rel_path : str
            Path relative to ``data_root``.
        window : tuple[float, float] or list[float] or None, optional
            Optional ``(start_sec, end_sec)`` crop. When given, only that span is
            decoded. ICL episodes reference long source recordings (Xeno-Canto
            recordings run to several minutes) but use ~10 s per clip, and an
            episode can hold up to 32 clips -- decoding each in full would cost
            minutes of audio per row. ``None`` reads the whole file, preserving
            the behaviour every pre-existing split relies on.

        Returns
        -------
        np.ndarray
            Mono float32 audio waveform.
        """
        # ICL-episode rows (and any prebuilt split whose audio spans several buckets) carry
        # already-absolute ``gs://`` / POSIX paths; use them directly and ignore ``data_root``.
        src = anypath(rel_path) if rel_path.startswith(("gs://", "/")) else self.data_root / rel_path
        if window is not None:
            audio, sr = read_audio(src, start_time=float(window[0]), end_time=float(window[1]))
        else:
            audio, sr = read_audio(src)
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
        windows = row.get("audio_windows")
        if windows is not None:
            if not isinstance(windows, (list, tuple)) or len(windows) != len(audio_paths):
                raise ValueError(
                    f"'audio_windows' must be None or parallel to 'audio_paths' "
                    f"({len(audio_paths)} paths, got {windows!r}) in row {row.get('id', '<unknown>')!r}"
                )
            audios = [
                self._load_audio(str(path), windows[i]) for i, path in enumerate(audio_paths)
            ]
        else:
            audios = [self._load_audio(str(path)) for path in audio_paths]
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
