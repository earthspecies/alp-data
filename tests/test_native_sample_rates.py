"""Offline regression tests for native sample-rate metadata and diagnostics."""

import logging
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from alp_data.dataset import DatasetInfo
from alp_data.datasets.infant_marmosets_vox import InfantMarmosetsVox


def make_info(**kwargs):
    return DatasetInfo(
        name="test",
        owner="test",
        split_paths={},
        version="0.1.0",
        description="Test metadata",
        sources="local",
        **kwargs,
    )


def test_existing_metadata_does_not_require_native_rates():
    assert make_info().native_sample_rates is None


def test_marmoset_native_rate_excludes_pre_resampled_copy():
    assert InfantMarmosetsVox.info.native_sample_rates == [44100]


@pytest.mark.parametrize("rates", [None, [44100], [8000, 44100, 48000]])
def test_native_rates_round_trip(rates):
    info = make_info(native_sample_rates=rates)
    assert DatasetInfo.model_validate_json(info.model_dump_json()).native_sample_rates == rates


@pytest.mark.parametrize("rates", [[], [0], [-16000], [44100, 0], [44100.5]])
def test_invalid_native_rates_rejected(rates):
    with pytest.raises(ValidationError):
        make_info(native_sample_rates=rates)
    info = make_info()
    with pytest.raises(ValidationError):
        info.native_sample_rates = rates


@pytest.fixture
def local_marmosets(tmp_path, monkeypatch):
    """Create a local manifest and distinct native/pre-resampled audio fixtures."""
    for rate, directory, amplitude in [(44100, "audio_44k", 0.2), (16000, "audio_16k", 0.1)]:
        audio_dir = tmp_path / directory
        audio_dir.mkdir()
        time = np.arange(rate, dtype=np.float32) / rate
        audio = amplitude * np.sin(2 * np.pi * 440 * time)
        sf.write(audio_dir / "sample.wav", audio, rate, subtype="FLOAT")
    manifest = tmp_path / "labels.csv"
    manifest.write_text("path,start,end\naudio_44k/sample.wav,0.1,0.3\n", encoding="utf-8")
    info = InfantMarmosetsVox.info.model_copy(update={"split_paths": {"all": str(manifest)}})
    monkeypatch.setattr(InfantMarmosetsVox, "info", info)
    return tmp_path


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("requested_rate", [None, 44100, 16000, 22050])
def test_resampling_diagnostics(local_marmosets, backend, streaming, requested_rate, caplog):
    ds = InfantMarmosetsVox(
        data_root=local_marmosets,
        backend=backend,
        streaming=streaming,
        sample_rate=requested_rate,
    )
    assert ds.available_sample_rates == [44100, 16000]

    log_name = "alp_data.datasets.infant_marmosets_vox"
    with caplog.at_level(logging.DEBUG, logger=log_name):
        sample = next(iter(ds)) if streaming else ds[0]

    output_rate = requested_rate or 44100
    assert ds.info.native_sample_rates == [44100]
    assert sample["sample_rate"] == output_rate
    assert sample["audio"].dtype == np.float32
    assert len(sample["audio"]) == round(0.2 * output_rate)
    expected_directory = "audio_16k" if requested_rate == 16000 else "audio_44k"
    assert Path(sample["path"]).parts[0] == expected_directory

    records = [
        record
        for record in caplog.records
        if record.name == log_name and "resampl" in record.getMessage().lower()
    ]
    if requested_rate == 22050:
        assert len(records) == 1
        assert records[0].levelno == logging.DEBUG
        assert "on the fly" in records[0].message
        assert "44100 Hz" in records[0].message
        assert "22050 Hz" in records[0].message
        assert "sample.wav" in records[0].message
        # Compare to the known signal, independently of the resampler implementation.
        # Omit filter boundary transients at either end of the segment.
        time = 0.1 + np.arange(len(sample["audio"])) / output_rate
        expected = 0.2 * np.sin(2 * np.pi * 440 * time)
        np.testing.assert_allclose(sample["audio"][64:-64], expected[64:-64], rtol=0, atol=1e-4)
    else:
        assert not records
        original, _ = sf.read(
            local_marmosets / expected_directory / "sample.wav",
            start=round(0.1 * output_rate),
            stop=round(0.3 * output_rate),
            dtype="float32",
        )
        np.testing.assert_array_equal(sample["audio"], original)
