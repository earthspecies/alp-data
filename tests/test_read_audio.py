"""Unitary tests of audio file reading functions."""

import io

import numpy as np
import soundfile as sf

from esp_data.io.read_utils import (
    _read_audio_from_bytes,
    read_audio,
    audio_stereo_to_mono,
    get_audio_info,
    read_audio_by_time
)


def test_read_from_file() -> None:
    path = "tests/samples/noise.wav"
    data, sr = read_audio(path)
    assert sr == 16000
    assert data.shape == (524288,)

def test_read_from_file_from_time() -> None:
    path = "tests/samples/noise.wav"
    data, sr = read_audio(path, start_time=0.0)
    assert sr == 16000
    assert data.shape == (524288,)

    data, sr = read_audio(path, start_time=1.0)
    assert sr == 16000
    assert data.shape == (508288,)

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


def test_get_audio_info() -> None:
    """Test if get_audio_info retrieves correct sample rate and frame count."""
    path = "tests/samples/noise.wav"
    info = get_audio_info(path)

    assert info["sr"] == 16000
    assert info["num_frames"] == 524288

    remote_path = "gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3"
    info = get_audio_info(remote_path)

    assert info["sr"] == 44100
    assert info["num_frames"] == 241727


def test_read_audio_by_time() -> None:
    """Test if read_audio_by_time reads correct audio segment."""
    path = "tests/samples/noise.wav"
    start_time = 1.0  # Start at 1 second
    end_time = 2.0  # End at 2 seconds

    audio, sr = read_audio_by_time(path, start_time=start_time, end_time=end_time)

    assert sr == 16000
    expected_length = int((end_time - start_time) * sr)
    assert audio.shape[0] == expected_length
