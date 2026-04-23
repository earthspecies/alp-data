"""Unitary tests of audio file reading functions."""

import numpy as np
import soundfile as sf

from esp_data.io.read_utils import (
    _audio_open_uri_and_suffix,
    _read_audio_from_file,
    read_audio,
    audio_stereo_to_mono,
    get_audio_info,
    read_audio_by_time,
    _read_audio_from_tmpfile,
)


def test_audio_open_uri_preserves_gs_string() -> None:
    """GCS URIs from strings must be opened verbatim (no %2F-style re-encoding)."""
    raw = "gs://animalspeak2/animalspeak_pseudovox/sub%2Fdir/file.wav"
    open_uri, suffix = _audio_open_uri_and_suffix(raw)
    assert open_uri == raw
    assert suffix == ".wav"


def test_audio_open_uri_r2_to_s3() -> None:
    open_uri, suffix = _audio_open_uri_and_suffix("r2://mybucket/folder/a.ogg")
    assert open_uri == "s3://mybucket/folder/a.ogg"
    assert suffix == ".ogg"


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


def test_read_no_offset(tmp_path) -> None:
    # Create some dummy audio data (pretend it's a short, single-channel sound)
    sample_rate = 16000
    test_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    sf.write(tmp_path / "test.wav", test_data, sample_rate, format="WAV")

    f = open(tmp_path / "test.wav", "rb")
    data, sr = _read_audio_from_file(f, format="WAV")

    assert sr == sample_rate
    np.testing.assert_allclose(data.flatten(), test_data, atol=1e-04)


def test_read_with_offset(tmp_path) -> None:
    # Create some dummy audio data
    sample_rate = 8000
    all_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    offset = 2  # We want to skip the first 2 frames
    expected_data = np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32)

    sf.write(tmp_path / "test.wav", all_data, sample_rate, format="WAV")

    # Call your function with an offset
    f = open(tmp_path / "test.wav", "rb")
    data, sr = _read_audio_from_file(f, start=offset, format="WAV")

    assert sr == sample_rate
    np.testing.assert_allclose(data.flatten(), expected_data, atol=1e-04)


def test_read_with_offset_and_frames(tmp_path) -> None:
    # Create some dummy audio data
    sample_rate = 8000
    all_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    offset = 2  # We want to skip the first 2 frames
    frames = 3  # Number of frames we want to get.
    expected_data = np.array([0.3, 0.4, 0.5], dtype=np.float32)

    sf.write(tmp_path / "test.wav", all_data, sample_rate, format="WAV")
    f = open(tmp_path / "test.wav", "rb")

    # Call your function with an offset
    data, sr = _read_audio_from_file(f, start=offset, frames=frames, format="WAV")

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


def test_read_mp3_from_gcs() -> None:
    """Test reading MP3 file from GCS."""
    remote_path = "gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3"
    data, sr = read_audio(remote_path)

    assert sr == 44100
    # MP3 files may have encoder padding, so decoded frames differ from metadata
    # The actual decoded frames is 235008, not the metadata value of 241727
    assert data.shape[0] == 235008
    # MP3 files are typically stereo or mono
    assert data.ndim in (1, 2)


def test_read_mp3_by_time_from_gcs() -> None:
    """Test reading MP3 file segment by time from GCS."""
    remote_path = "gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3"
    start_time = 1.0  # Start at 1 second
    end_time = 2.0  # End at 2 seconds

    audio, sr = read_audio_by_time(remote_path, start_time=start_time, end_time=end_time)

    assert sr == 44100
    expected_length = int((end_time - start_time) * sr)
    assert audio.shape[0] == expected_length


def test_read_mp3_from_bytes() -> None:
    """Test reading MP3 from bytes using _read_audio_from_bytes."""
    # First read the remote MP3 to get the bytes
    from esp_data.io.filesystem import filesystem_from_path

    remote_path = "gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3"
    fs = filesystem_from_path(remote_path)

    with fs.open(remote_path, "rb") as f:
        # Now test reading from bytes with MP3 format
        data, sr = _read_audio_from_file(f, format="MP3")

    assert sr == 44100
    # Actual decoded frames (not metadata frames due to encoder padding)
    assert data.shape[0] == 235008


def test_read_mp3_from_bytes_with_offset() -> None:
    """Test reading MP3 from bytes with offset."""
    from esp_data.io.filesystem import filesystem_from_path

    remote_path = "gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3"
    fs = filesystem_from_path(remote_path)

    with fs.open(remote_path, "rb") as f:
        # Read with offset (skip first 44100 frames = 1 second)
        offset = 44100
        data, sr = _read_audio_from_file(f, start=offset, format="MP3")

    assert sr == 44100
    # Should have 1 second less of data (using actual decoded frames)
    assert data.shape[0] == 235008 - offset


def test_read_mp3_from_bytes_with_frames() -> None:
    """Test reading specific number of frames from MP3 bytes."""
    from esp_data.io.filesystem import filesystem_from_path

    remote_path = "gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3"
    fs = filesystem_from_path(remote_path)
    with fs.open(remote_path, "rb") as f:
        # Read only first 2 seconds (88200 frames at 44100 Hz)
        frames = 88200
        data, sr = _read_audio_from_file(f, frames=frames, format="MP3")

    assert sr == 44100
    assert data.shape[0] == frames

    # read from tmpfile and compare
    with fs.open(remote_path, "rb") as f:
        data2, sr2 = _read_audio_from_tmpfile(f.read(), frames=frames, format="MP3")
    assert sr2 == sr
    np.testing.assert_allclose(data, data2, atol=1e-04)



def test_read_troublesome_xc_file() -> None:
    """Test reading a known troublesome audio file from Xenocanto."""
    remote_path = "gs://esp-ci-cd-tests/esp-data-tests/XC_corcorax.mp3"
    data, sr = read_audio(remote_path)

    assert sr == 48000
    # Actual decoded frames (not metadata frames due to encoder padding)
    assert data.shape[0] == 2794752

    # test reading by time
    start_time = 5.0
    end_time = 10.0
    segment, sr3 = read_audio_by_time(remote_path, start_time=start_time, end_time=end_time)
    expected_length = int((end_time - start_time) * sr)
    assert sr3 == sr
    assert segment.shape[0] == expected_length


def test_get_audio_info_troublesome_xc_file() -> None:
    """Test getting audio info for a known troublesome audio file from Xenocanto."""
    remote_path = "gs://esp-ci-cd-tests/esp-data-tests/XC_corcorax.mp3"
    info = get_audio_info(remote_path)

    assert info["sr"] == 48000
    assert info["num_frames"] == 2807470  # FIXME: Info gives more frames


#Issue 215: Glob pattern fix
def test_read_audio_with_glob() -> None:
    """Test reading audio file using glob pattern."""
    path = "gs://esp-ci-cd-tests/esp-data-tests/test_glob_pattern_audio/test_audio[1].wav"
    data, sr = read_audio(path)
    assert sr == 16000
    assert data.shape == (16000,)
