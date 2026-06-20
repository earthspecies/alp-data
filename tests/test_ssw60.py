"""Unit tests for the SSW60 multimodal dataset.

These tests load the manifests from GCS, so they require network access
and the ``gs://esp-data-ingestion/ssw60/v0.1.0/`` upload to be present.
Video-decode tests additionally require the optional ``video`` extra
(PyAV); they are skipped when ``av`` is unavailable.

Run with:
    pytest -q tests/test_ssw60.py
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from esp_data.datasets import SSW60

# --- snapshot (source counts from visipedia/ssw60) ---
EXPECTED_AUDIO_ALL = 3861
EXPECTED_VIDEO_ALL = 5400
# 21,600 iNat2021 + 10,221 NABirds = 31,821, minus 1 NABirds asset
# (13e5d907…) whose JPG is missing from the upstream tarball.
EXPECTED_IMAGE_ALL = 31820
EXPECTED_NUM_SPECIES = 60
# ------------------------------------------------------

_HAS_AV = importlib.util.find_spec("av") is not None


@pytest.fixture(scope="module")
def audio_ds() -> SSW60:
    """Load the audio_all split at 16 kHz (pandas backend).

    Returns
    -------
    SSW60
        The loaded audio_all dataset.
    """
    return SSW60(split="audio_all", sample_rate=16000, backend="pandas")


def test_audio_split_length(audio_ds: SSW60) -> None:
    """The audio_all split length should match the source count."""
    assert len(audio_ds) == EXPECTED_AUDIO_ALL


def test_available_splits(audio_ds: SSW60) -> None:
    """available_splits should expose all modality splits."""
    for split in [
        "all",
        "audio_all", "audio_train", "audio_test",
        "video_all", "video_train", "video_test",
        "image_all", "image_train", "image_test", "image_val",
    ]:
        assert split in audio_ds.available_splits


def test_modality_labels(audio_ds: SSW60) -> None:
    """get_available_labels exposes the modality enum."""
    assert set(audio_ds.get_available_labels("modality")) == {"audio", "video", "image"}


def test_required_columns(audio_ds: SSW60) -> None:
    """Unified manifest columns should be present."""
    expected = [
        "asset_id", "modality", "label", "species_code", "canonical_name",
        "species_common", "family", "order", "split",
        "audio_path", "16khz_path", "32khz_path", "image_path", "video_path",
        "kingdom", "phylum", "class", "genus",
    ]
    for col in expected:
        assert col in audio_ds.columns, f"missing column {col}"


def test_gbif_linked(audio_ds: SSW60) -> None:
    """Every audio row should carry a non-empty GBIF canonical name."""
    rows = list(audio_ds._data)
    assert all(str(r.get("canonical_name", "")).strip() for r in rows)


def test_species_count(audio_ds: SSW60) -> None:
    """There should be exactly 60 distinct species labels."""
    rows = list(audio_ds._data)
    assert len({r["label"] for r in rows}) == EXPECTED_NUM_SPECIES


def test_audio_item(audio_ds: SSW60) -> None:
    """An audio item should decode to a mono 16 kHz waveform."""
    item = audio_ds[0]
    assert item["modality"] == "audio"
    assert isinstance(item["audio"], np.ndarray)
    assert item["audio"].ndim == 1
    assert item["sample_rate"] == 16000


def test_image_item() -> None:
    """An image item should decode to an HWC uint8 array."""
    ds = SSW60(split="image_test", backend="pandas")
    assert len(ds) > 0
    item = ds[0]
    assert item["modality"] == "image"
    assert isinstance(item["image"], np.ndarray)
    assert item["image"].dtype == np.uint8
    assert item["image"].ndim == 3


@pytest.mark.skipif(not _HAS_AV, reason="PyAV (video extra) not installed")
def test_video_item() -> None:
    """A video item should decode frames and the aligned audio track."""
    ds = SSW60(split="video_test", sample_rate=16000, backend="pandas", max_frames=8)
    assert len(ds) > 0
    item = ds[0]
    assert item["modality"] == "video"
    assert isinstance(item["video_frames"], np.ndarray)
    assert item["video_frames"].ndim == 4  # (T, H, W, C)
    assert item["video_frames"].shape[0] <= 8
    # Aligned audio (best-effort): present for most clips.
    if item["audio"] is not None:
        assert item["audio"].ndim == 1
        assert item["sample_rate"] == 16000
