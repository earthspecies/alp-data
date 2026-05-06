"""Per-dataset configuration for the audio + spectrogram precompute.

Each entry tells `build_assets.py build-samples` how to:

- locate audio (`audio_columns` — first present wins)
- assemble human-readable metadata (`label_column`, `license_column`,
  `url_column`)
- size the log-mel spectrogram (`mel_*` keys)
- pick a target sample rate that the dataset class supports as a
  pre-resampled column

Only the curated subset of datasets used in step-4b (weak-label) is
included; strong-label datasets (`geladas`, `gibbon_solos`) are added in
step 4c.
"""

from __future__ import annotations

from typing import Any

# Fixed seed for reproducible sample selection across precompute runs.
SAMPLE_SEED: int = 42
# How many valid samples we want per dataset (10 per the design plan).
SAMPLES_PER_DATASET: int = 10
# Skip clips shorter than this; center-crop anything longer.
MIN_DURATION_S: float = 2.0
MAX_DURATION_S: float = 30.0
# How many random indices we draw before giving up; needs to exceed
# `SAMPLES_PER_DATASET` because some clips may be too short or fail to
# read.
SAMPLE_POOL_FACTOR: int = 6


SAMPLE_CONFIG: dict[str, dict[str, Any]] = {
    "inaturalist": {
        "class_name": "INaturalist",
        "split": "train",
        "extractor": "weak",
        "audio_columns": ("16khz_path", "32khz_path", "originals_path"),
        "label_column": "species_common",
        "license_column": "media_license",
        "url_column": "url",
        "target_sr": 16000,
        "mel_n_fft": 1024,
        "mel_hop": 256,
        "mel_n_mels": 128,
        "mel_fmin": 50,
        "mel_fmax": 8000,
    },
    "xeno-canto": {
        "class_name": "XenoCanto",
        "split": "train",
        "extractor": "weak",
        "audio_columns": ("16khz_path", "32khz_path", "gcs_path", "path"),
        "label_column": "species_common",
        "license_column": "license",
        "url_column": "url",
        "target_sr": 16000,
        "mel_n_fft": 1024,
        "mel_hop": 256,
        "mel_n_mels": 128,
        "mel_fmin": 50,
        "mel_fmax": 8000,
    },
    "insectset_459": {
        "class_name": "InsectSet459",
        "split": "train",
        "extractor": "weak",
        "audio_columns": ("local_path",),
        "label_column": "species_common",
        "license_column": "license",
        "url_column": None,
        # Insect calls have a lot of energy above 8 kHz; keep a wider
        # spectrum than the bird configs.
        "target_sr": 22050,
        "mel_n_fft": 2048,
        "mel_hop": 512,
        "mel_n_mels": 128,
        "mel_fmin": 200,
        "mel_fmax": 11025,
    },
    "watkins": {
        "class_name": "Watkins",
        "split": "train",
        "extractor": "weak",
        "audio_columns": ("32khz_path", "16khz_path", "audio_path"),
        "label_column": "species_common",
        "license_column": "license",
        "url_column": None,
        # Marine mammals span a huge range; 32 kHz with high fmax keeps
        # dolphin clicks visible while avoiding huge files.
        "target_sr": 32000,
        "mel_n_fft": 2048,
        "mel_hop": 512,
        "mel_n_mels": 128,
        "mel_fmin": 30,
        "mel_fmax": 16000,
    },
    "geladas": {
        "class_name": "Geladas",
        "split": "all",
        # Each row is one annotated event; use the row's onset/offset.
        "extractor": "event_columns",
        "audio_path_column": "local_path",
        "onset_column": "vocal_onset",
        "offset_column": "vocal_offset",
        "label_column": "vocal_type",
        "license_column": None,
        "url_column": None,
        "target_sr": 16000,
        "mel_n_fft": 1024,
        "mel_hop": 256,
        "mel_n_mels": 128,
        "mel_fmin": 80,
        "mel_fmax": 8000,
    },
    "gibbon_solos": {
        "class_name": "GibbonSolos",
        "split": "all",
        # Each row points to a long file; pick an event from the inline
        # Raven selection table.
        "extractor": "selection_table",
        "audio_path_column": "local_path",
        "selection_table_column": "selection_table",
        "label_column": "species_common",
        "license_column": None,
        "url_column": None,
        "target_sr": 16000,
        "mel_n_fft": 2048,
        "mel_hop": 512,
        "mel_n_mels": 128,
        "mel_fmin": 200,
        "mel_fmax": 8000,
    },
}


__all__ = [
    "MAX_DURATION_S",
    "MIN_DURATION_S",
    "SAMPLES_PER_DATASET",
    "SAMPLE_CONFIG",
    "SAMPLE_POOL_FACTOR",
    "SAMPLE_SEED",
]
