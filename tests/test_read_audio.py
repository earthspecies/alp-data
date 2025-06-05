"""Unitary tests of audio file reading functions."""

import io

import numpy as np
import soundfile as sf

from esp_data.io.read_utils import (
    _read_audio_from_bytes,
    read_audio,
    audio_stereo_to_mono,
)


def test_read_from_file() -> None:
    path = "tests/samples/noise.wav"
    data, sr = read_audio(path)
    assert sr == 16000
    assert data.shape == (524288,)


def test_read_no_offset() -> None:
    # Create some dummy audio data (pretend it's a short, single-channel sound)
    sample_rate = 16000
    test_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, test_data, sample_rate, format="WAV")
    audio_bytes = buffer.getvalue()

    data, sr = _read_audio_from_bytes(audio_bytes)

    assert sr == sample_rate
    np.testing.assert_allclose(data.flatten(), test_data, atol=1e-04)


def test_read_with_offset() -> None:
    # Create some dummy audio data
    sample_rate = 8000
    all_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    offset = 2  # We want to skip the first 2 frames
    expected_data = np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, all_data, sample_rate, format="WAV")
    audio_bytes = buffer.getvalue()

    # Call your function with an offset
    data, sr = _read_audio_from_bytes(audio_bytes, start=offset)

    assert sr == sample_rate
    np.testing.assert_allclose(data.flatten(), expected_data, atol=1e-04)


def test_read_with_offset_and_frames() -> None:
    # Create some dummy audio data
    sample_rate = 8000
    all_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    offset = 2  # We want to skip the first 2 frames
    frames = 3  # Number of frames we want to get.
    expected_data = np.array([0.3, 0.4, 0.5], dtype=np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, all_data, sample_rate, format="WAV")
    audio_bytes = buffer.getvalue()

    # Call your function with an offset
    data, sr = _read_audio_from_bytes(audio_bytes, start=offset, frames=frames)

    assert sr == sample_rate
    np.testing.assert_allclose(data.flatten(), expected_data, atol=1e-04)


def test_audio_stereo_to_mono() -> None:
    """Test if audio_stereo_to_mono converts stereo audio to mono correctly."""
    # Create dummy stereo audio data
    stereo_audio = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    # Convert to mono using average method
    mono_audio = audio_stereo_to_mono(stereo_audio, mono_method="average")

    # Expected result is the average of the two channels
    expected_mono = np.array([0.25, 0.35, 0.45], dtype=np.float32)

    np.testing.assert_allclose(mono_audio.flatten(), expected_mono, atol=1e-04)

    mono_audio2 = audio_stereo_to_mono(stereo_audio, mono_method="keep_first")

    # Expected result is the first channel
    expected_mono2 = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    np.testing.assert_allclose(mono_audio2.flatten(), expected_mono2, atol=1e-04)
