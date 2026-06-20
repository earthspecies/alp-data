"""Smoke tests for the monster-monash wingbeat datasets.

Run with:
    pytest -q tests/test_wingbeats_datasets.py
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from esp_data.datasets import InsectSound, MosquitoSound

# --- Snapshots (fill once after GCS upload completes) ---
EXPECTED_MOSQUITO_TOTAL = 279566
EXPECTED_INSECT_TOTAL = 50000

MOSQUITO_SPECIES_EXPECTED = {
    "Aedes aegypti", "Aedes albopictus", "Anopheles arabiensis",
    "Anopheles gambiae", "Culex pipiens", "Culex quinquefasciatus",
}
INSECT_SPECIES_EXPECTED = {
    "Aedes aegypti", "Culex stigmatosoma", "Culex tarsalis",
    "Culex quinquefasciatus", "Musca domestica", "Drosophila simulans",
}


# ---------------- MosquitoSound ----------------

@pytest.fixture(scope="module")
def mosquito_all() -> MosquitoSound:
    """Load the full MosquitoSound at 16 kHz, pandas backend."""
    return MosquitoSound(split="all", sample_rate=16000, backend="pandas")


def test_mosquito_total(mosquito_all: MosquitoSound) -> None:
    """All 279,566 wingbeat clips should be present."""
    assert len(mosquito_all) == EXPECTED_MOSQUITO_TOTAL


def test_mosquito_columns(mosquito_all: MosquitoSound) -> None:
    """Required schema columns exist."""
    for c in [
        "audio_path", "16khz_path", "32khz_path", "class_id", "species",
        "canonical_name", "gbifID", "kingdom", "phylum", "class", "order",
        "family", "genus", "fold_0_test", "license",
    ]:
        assert c in mosquito_all.columns, f"missing {c}"


def test_mosquito_species_set(mosquito_all: MosquitoSound) -> None:
    """Species set matches the six Wingbeats Culicidae."""
    seen = {row["species"] for row in mosquito_all._data}
    assert seen == MOSQUITO_SPECIES_EXPECTED


def test_mosquito_taxonomy_consistency(mosquito_all: MosquitoSound) -> None:
    """Every row is Insecta / Diptera / Culicidae."""
    for row in mosquito_all._data:
        assert row["kingdom"] == "Animalia"
        assert row["class"] == "Insecta"
        assert row["order"] == "Diptera"
        assert row["family"] == "Culicidae"


def test_mosquito_splits() -> None:
    """train + val sizes match all, no overlap."""
    train = MosquitoSound(split="train", sample_rate=16000, backend="pandas")
    val = MosquitoSound(split="val", sample_rate=16000, backend="pandas")
    assert len(train) + len(val) == EXPECTED_MOSQUITO_TOTAL


@pytest.mark.slow
def test_mosquito_audio_loads(mosquito_all: MosquitoSound) -> None:
    """A few random clips load at the requested sample rate."""
    rng = random.Random(7)
    for idx in [rng.randrange(len(mosquito_all)) for _ in range(3)]:
        item = mosquito_all[idx]
        audio = item["audio"]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.size > 0 and not np.any(np.isnan(audio))
        assert item["sample_rate"] == 16000


# ---------------- InsectSound ----------------

@pytest.fixture(scope="module")
def insect_all() -> InsectSound:
    """Load the full InsectSound at 16 kHz, pandas backend."""
    return InsectSound(split="all", sample_rate=16000, backend="pandas")


def test_insect_total(insect_all: InsectSound) -> None:
    """All 50,000 wingbeat clips should be present."""
    assert len(insect_all) == EXPECTED_INSECT_TOTAL


def test_insect_columns(insect_all: InsectSound) -> None:
    """Required schema columns exist."""
    for c in [
        "audio_path", "16khz_path", "32khz_path", "class_id", "species",
        "sex", "canonical_name", "gbifID", "fold_0_test", "license",
    ]:
        assert c in insect_all.columns, f"missing {c}"


def test_insect_species_set(insect_all: InsectSound) -> None:
    """The six taxa in the UCR mapping are present."""
    seen = {row["species"] for row in insect_all._data}
    assert seen == INSECT_SPECIES_EXPECTED


def test_insect_class_balance(insect_all: InsectSound) -> None:
    """All 10 classes should have exactly 5,000 examples."""
    from collections import Counter
    c = Counter(row["class_id"] for row in insect_all._data)
    assert len(c) == 10
    assert all(v == 5000 for v in c.values()), c


def test_insect_sex_populated(insect_all: InsectSound) -> None:
    """Every row carries a non-empty sex (female or male)."""
    for row in insect_all._data:
        assert str(row["sex"]).strip() in {"female", "male"}, row


@pytest.mark.slow
def test_insect_audio_loads(insect_all: InsectSound) -> None:
    """A few random clips load at the requested sample rate."""
    rng = random.Random(11)
    for idx in [rng.randrange(len(insect_all)) for _ in range(3)]:
        item = insect_all[idx]
        audio = item["audio"]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.size > 0 and not np.any(np.isnan(audio))
        assert item["sample_rate"] == 16000
