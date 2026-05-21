"""
Unit tests for SpanishCarrionCrowsVox dataset.

Run with:
    pytest -q test_spanish_carrion_crows_vox.py
"""

from __future__ import annotations

import hashlib
import random
from typing import List

import numpy as np
import pytest

from esp_data.datasets import SpanishCarrionCrowsVox


# --- Dataset snapshot ---
#
# Code to generate snapshot
#
# import hashlib
# from esp_data.datasets import SpanishCarrionCrowsVox

# ds = SpanishCarrionCrowsVox(split="all", sample_rate=16000, backend="pandas")

# print("len(ds) =", len(ds))
# # len(ds) = 54288

# item0 = ds[0]
# audio0 = item0["audio"]
# print("dtype:", audio0.dtype, "shape:", audio0.shape)
# h = hashlib.sha256(audio0.tobytes()).hexdigest()
# print("audio sha256:", h)
# # audio sha256: c5a94be195ea9c2d89a927a94e09c054c28a10886e04a972565240abb63aff2d

# csv_bytes = (
#     ds._data.unwrap.sort_index(axis=0)
#     .sort_index(axis=1)
#     .to_csv(index=True)
#     .encode("utf-8")
# )
# h = hashlib.sha256(csv_bytes).hexdigest()
# print("csv sha256:", h)
# # csv sha256: 9ba2ccd223dc541f6cc4183b5181dbbcf68db09b03fe563c5765591a932e6dcb

# quit()
# ---------------------------------------------------------------------------

EXPECTED_LEN_ALL = 54288
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = "c5a94be195ea9c2d89a927a94e09c054c28a10886e04a972565240abb63aff2d"
EXPECTED_CSV_SHA256 = "9ba2ccd223dc541f6cc4183b5181dbbcf68db09b03fe563c5765591a932e6dcb"
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> SpanishCarrionCrowsVox:
    return SpanishCarrionCrowsVox(split="all", sample_rate=16000)


@pytest.fixture(scope="module")
def ds_pandas() -> SpanishCarrionCrowsVox:
    return SpanishCarrionCrowsVox(split="all", sample_rate=16000, backend="pandas")


@pytest.fixture(scope="module")
def ds_denoised() -> SpanishCarrionCrowsVox:
    return SpanishCarrionCrowsVox(split="all", sample_rate=16000, denoised=True)


@pytest.fixture(scope="module")
def ds_window(ds: SpanishCarrionCrowsVox) -> SpanishCarrionCrowsVox:
    window_id = next(s for s in ds.available_splits if s != "all")
    return SpanishCarrionCrowsVox(split=window_id, sample_rate=16000)


@pytest.fixture(scope="module")
def sample_indices(ds: SpanishCarrionCrowsVox) -> List[int]:
    n = len(ds)
    rng = random.Random(42)
    return [rng.randrange(n) for _ in range(min(5, n))]


def test_ds_not_empty(ds: SpanishCarrionCrowsVox):
    assert len(ds) > 0, "Dataset appears empty"


def test_window_split(ds: SpanishCarrionCrowsVox, ds_window: SpanishCarrionCrowsVox):
    """Filtering by overlap_window_id should return a strict subset."""
    assert 0 < len(ds_window) <= len(ds)
    for item in ds_window:
        assert item["overlap_window_id"] == ds_window.split


def test_invalid_window_split_raises():
    with pytest.raises(LookupError, match="overlap_window_id"):
        SpanishCarrionCrowsVox(split=999999999)


def test_check_audio(ds: SpanishCarrionCrowsVox, sample_indices: List[int]):
    for idx in sample_indices:
        item = ds[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert audio.dtype == np.float32, f"[{idx}] audio dtype is {audio.dtype}, expected float32"
        assert audio.size >= 10, f"[{idx}] audio too short (size={audio.size})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


def test_metadata_fields(ds: SpanishCarrionCrowsVox, sample_indices: List[int]):
    required = {"audio", "sample_rate", "call_type", "focal_individual", "timestamp_start", "timestamp_end", "overlap_window_id"}
    for idx in sample_indices:
        item = ds[idx]
        missing = required - set(item.keys())
        assert not missing, f"[{idx}] item missing keys: {sorted(missing)}"

        assert isinstance(item["sample_rate"], int), f"[{idx}] sample_rate should be int"
        assert item["sample_rate"] == 16000, f"[{idx}] expected sample_rate=16000"
        assert isinstance(item["call_type"], str), f"[{idx}] call_type should be str"
        assert isinstance(item["focal_individual"], str), f"[{idx}] focal_individual should be str"


def test_padding_extends_audio(ds: SpanishCarrionCrowsVox, sample_indices: List[int]):
    ds_padded = SpanishCarrionCrowsVox(split="all", sample_rate=16000, padding_sec=0.5)
    idx = sample_indices[0]
    unpadded_len = ds[idx]["audio"].shape[0]
    padded_len = ds_padded[idx]["audio"].shape[0]
    assert padded_len >= unpadded_len, (
        f"[{idx}] padded audio ({padded_len}) should be >= unpadded ({unpadded_len})"
    )


def test_denoised_audio_present(ds: SpanishCarrionCrowsVox, ds_denoised: SpanishCarrionCrowsVox):
    success_idx = next(
        (i for i, row in enumerate(ds._data) if row.get("derived.denoised_focal.success") is True),
        None,
    )
    if success_idx is None:
        pytest.skip("No successfully denoised rows found")
    item = ds_denoised[success_idx]
    assert "denoised_success" in item, "denoised_success key missing in denoised mode"
    assert item["denoised_success"] is True
    audio = item["audio"]
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.size >= 10
    assert not np.any(np.isnan(audio))


def test_denoised_fallback_to_noisy(ds: SpanishCarrionCrowsVox):
    """With fallback_to_noisy=True, failed rows should return audio with denoised_success=False."""
    ds_fallback = SpanishCarrionCrowsVox(split="all", sample_rate=16000, denoised=True, fallback_to_noisy=True)
    for i, row in enumerate(ds_fallback._data):
        success = row.get("derived.denoised_focal.success", "True")
        if isinstance(success, str):
            success = success == "True"
        if not success:
            item = ds_fallback[i]
            assert item["denoised_success"] is False
            assert isinstance(item["audio"], np.ndarray)
            assert item["audio"].dtype == np.float32
            return
    pytest.skip("No failed-denoising rows found in split")


def test_denoised_failed_row_raises(ds: SpanishCarrionCrowsVox):
    """Accessing a row with no denoised file in denoised mode should raise ValueError."""
    ds_den = SpanishCarrionCrowsVox(split="all", sample_rate=16000, denoised=True)
    # Find first row where denoised_focal.success is False by inspecting the raw data
    for i, row in enumerate(ds_den._data):
        success = row.get("derived.denoised_focal.success", "True")
        if isinstance(success, str):
            success = success == "True"
        if not success:
            with pytest.raises(ValueError, match="Denoised audio not available"):
                ds_den[i]
            return
    pytest.skip("No failed-denoising rows found in split")


@pytest.mark.skipif(
    EXPECTED_FIRST_ITEM_AUDIO_SHA256 == "",
    reason="Reference hash not yet populated — run snapshot code above first",
)
def test_reference_item_stability(ds_pandas: SpanishCarrionCrowsVox):
    """Check that index-0 audio and the CSV are bitwise-stable."""
    item = ds_pandas[0]

    assert "audio" in item
    audio = item["audio"]
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32

    h = hashlib.sha256(audio.tobytes()).hexdigest()
    assert h == EXPECTED_FIRST_ITEM_AUDIO_SHA256, (
        "First item audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_FIRST_ITEM_AUDIO_SHA256}\n\n"
        "If this is intentional, update EXPECTED_FIRST_ITEM_AUDIO_SHA256."
    )

    csv_bytes = (
        ds_pandas._data.unwrap.sort_index(axis=0)
        .sort_index(axis=1)
        .to_csv(index=True)
        .encode("utf-8")
    )
    h = hashlib.sha256(csv_bytes).hexdigest()
    assert h == EXPECTED_CSV_SHA256, (
        "CSV hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_CSV_SHA256}\n\n"
        "If this is intentional, update EXPECTED_CSV_SHA256."
    )
