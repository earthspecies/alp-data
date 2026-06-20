"""Unit tests for the PIFSC PIPAN dataset (Phase 1).

Run with:
    pytest -q tests/test_pifsc_pipan.py
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from esp_data.datasets import PIFSCPipan

# --- snapshot (fill after pifsc_pipan_*.csv are uploaded) ---
EXPECTED_LEN_ALL = 38857
EXPECTED_LABEL_COUNTS = {
    "Mn": 32189,
    "Background": 5928,
    "Other": 306,
    "Vessel": 242,
    "Fish": 98,
    "Device": 94,
}
EXPECTED_DEPLOYMENTS = {
    "crosssm", "equator", "hawaii", "howland", "kauai", "kingman",
    "laddsm_d", "laddsm_s", "pagan", "palmyra_ns", "palmyra_wt",
    "phr_a", "phr_b", "saipan", "tinian", "wake",
}
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> PIFSCPipan:
    """Load the 'all' split at 16 kHz (pandas backend, no streaming)."""
    return PIFSCPipan(split="all", sample_rate=16000, backend="pandas")


def test_ds_not_empty(ds: PIFSCPipan) -> None:
    """The 'all' split should have annotation events."""
    assert len(ds) > 0


def test_expected_length(ds: PIFSCPipan) -> None:
    """The 'all' split length should be stable across builds."""
    assert len(ds) == EXPECTED_LEN_ALL


def test_available_splits(ds: PIFSCPipan) -> None:
    """available_splits should expose all split names."""
    for split in ["all", "train", "val"]:
        assert split in ds.available_splits


def test_label_columns(ds: PIFSCPipan) -> None:
    """Event-level columns should be present."""
    expected = [
        "audio_path", "16khz_path", "32khz_path", "deployment",
        "xwav_subchunk_index", "begin_in_subchunk_s", "end_in_subchunk_s",
        "begin_in_file_s", "end_in_file_s",
        "begin_utc", "end_utc",
        "label", "label_is_strong", "implicit_negatives", "audit_name",
        "coarse_call_type", "species",
    ]
    for col in expected:
        assert col in ds.columns, f"missing column {col}"


def test_label_distribution(ds: PIFSCPipan) -> None:
    """Per-label counts should match the source CSV."""
    rows = list(ds._data)
    actual = {}
    for r in rows:
        actual[r["label"]] = actual.get(r["label"], 0) + 1
    for label, count in EXPECTED_LABEL_COUNTS.items():
        assert actual.get(label) == count, (
            f"Label '{label}' expected {count}, got {actual.get(label, 0)}"
        )


def test_deployments(ds: PIFSCPipan) -> None:
    """Every deployment should be represented and known."""
    deploys = {r["deployment"] for r in ds._data}
    assert deploys == EXPECTED_DEPLOYMENTS, deploys ^ EXPECTED_DEPLOYMENTS


def test_gbif_link_humpback(ds: PIFSCPipan) -> None:
    """Mn rows carry the humpback GBIF canonical name; others do not."""
    def _empty(v):
        s = str(v).strip()
        return s == "" or s.lower() == "nan"
    for r in ds._data:
        if r["label"] == "Mn":
            assert r["canonical_name"] == "Megaptera novaeangliae", r
            assert not _empty(r["gbifID"]), r
        else:
            assert _empty(r["species"]), r


def test_available_labels(ds: PIFSCPipan) -> None:
    """get_available_labels returns the documented vocabulary."""
    labels = ds.get_available_labels("label")
    assert set(labels) == set(EXPECTED_LABEL_COUNTS.keys())
    deploys = ds.get_available_labels("deployment")
    assert set(deploys) == EXPECTED_DEPLOYMENTS


def test_split_disjoint() -> None:
    """train / val splits split at the file level: no overlap of audio_path."""
    train = PIFSCPipan(split="train", sample_rate=16000, backend="pandas")
    val = PIFSCPipan(split="val", sample_rate=16000, backend="pandas")
    train_files = {r["audio_path"] for r in train._data}
    val_files = {r["audio_path"] for r in val._data}
    assert not (train_files & val_files), (
        f"{len(train_files & val_files)} files shared between train and val"
    )


@pytest.mark.slow
def test_check_audio(ds: PIFSCPipan) -> None:
    """Audio integrity on a few deterministic indices (requires GCS read)."""
    rng = random.Random(42)
    indices = [rng.randrange(len(ds)) for _ in range(3)]
    for idx in indices:
        item = ds[idx]
        audio = item["audio"]
        assert isinstance(audio, np.ndarray), f"[{idx}] audio not ndarray"
        assert audio.dtype == np.float32, f"[{idx}] dtype {audio.dtype}"
        assert audio.size >= 10, f"[{idx}] too short"
        assert not np.any(np.isnan(audio)), f"[{idx}] NaN"
        assert item["sample_rate"] == 16000, f"[{idx}] sr {item['sample_rate']}"
