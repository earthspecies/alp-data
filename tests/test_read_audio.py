"""Unitary tests of audio file reading functions."""

import io

import numpy as np
import soundfile as sf

from esp_data.io.read_utils import _read_audio_from_bytes, read_audio

def test_read_from_file():
    path = "tests/samples/noise.wav"
    data, sr = read_audio(path)
    assert sr == 16000
    assert data.shape == (524288,)

def test_read_no_offset():
    # Create some dummy audio data (pretend it's a short, single-channel sound)
    sample_rate = 16000
    test_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, test_data, sample_rate, format="WAV")
    audio_bytes = buffer.getvalue()

    data, sr = _read_audio_from_bytes(audio_bytes)

    assert sr == sample_rate
    np.testing.assert_allclose(data.flatten(), test_data, atol=1e-04)


def test_read_with_offset():
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


def test_read_with_offset_and_frames():
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
