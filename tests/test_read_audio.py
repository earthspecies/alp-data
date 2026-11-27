"""Unitary tests of audio file reading functions."""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import alp_data.io.read_utils as read_utils
from alp_data.io.read_utils import (
    FFmpegSegmentError,
    _gcs_path_to_url,
    _read_audio_ffmpeg,
    _read_audio_from_file,
    _read_audio_from_tmpfile,
    _warn_ffmpeg_fallback_once,
    audio_stereo_to_mono,
    get_audio_info,
    read_audio,
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
    """Test if read_audio reads the correct audio segment by time (local WAV)."""
    path = "tests/samples/noise.wav"
    start_time = 1.0  # Start at 1 second
    end_time = 2.0  # End at 2 seconds

    audio, sr = read_audio(path, start_time=start_time, end_time=end_time)

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
    """Test reading an MP3 segment by time from GCS (ffmpeg streaming path)."""
    remote_path = "gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3"
    start_time = 1.0  # Start at 1 second
    end_time = 2.0  # End at 2 seconds

    audio, sr = read_audio(remote_path, start_time=start_time, end_time=end_time)

    assert sr == 44100
    expected_length = int((end_time - start_time) * sr)
    # MP3 frame alignment means ffmpeg segment length is approximate.
    assert abs(audio.shape[0] - expected_length) < sr * 0.05


def test_read_mp3_from_bytes() -> None:
    """Test reading MP3 from bytes using _read_audio_from_bytes."""
    # First read the remote MP3 to get the bytes
    from alp_data.io.filesystem import filesystem_from_path

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
    from alp_data.io.filesystem import filesystem_from_path

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
    from alp_data.io.filesystem import filesystem_from_path

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

    # test reading by time (ffmpeg streaming path for GCS)
    start_time = 5.0
    end_time = 10.0
    segment, sr3 = read_audio(remote_path, start_time=start_time, end_time=end_time)
    expected_length = int((end_time - start_time) * sr)
    assert sr3 == sr
    # MP3 frame alignment means ffmpeg segment length is approximate.
    assert abs(segment.shape[0] - expected_length) < sr * 0.05


def test_read_audio_ffmpeg_segment_from_gcs() -> None:
    """ffmpeg streams a sample-accurate segment from a long GCS WAV file.

    The source is a deterministic 10-minute 220 Hz tone (FLOAT WAV, so decoding
    is lossless). Reading a mid-file segment via ffmpeg HTTP range requests must
    return exactly the corresponding source frames without downloading the whole
    file.
    """
    remote_path = "gs://esp-ci-cd-tests/esp-data-tests/long_tone_600s_16k_float.wav"
    sr = 16000
    start_time, end_time = 123.0, 128.0

    segment, file_sr = _read_audio_ffmpeg(remote_path, start_time, end_time)

    assert file_sr == sr
    assert segment.shape == (int((end_time - start_time) * sr),)

    # Reconstruct the exact source samples for the requested window.
    frame_idx = np.arange(int(start_time * sr), int(end_time * sr))
    expected = (0.5 * np.sin(2 * np.pi * 220.0 * frame_idx / sr)).astype(np.float32)
    np.testing.assert_allclose(segment, expected, atol=1e-06)


def test_get_audio_info_troublesome_xc_file() -> None:
    """Test getting audio info for a known troublesome audio file from Xenocanto."""
    remote_path = "gs://esp-ci-cd-tests/esp-data-tests/XC_corcorax.mp3"
    info = get_audio_info(remote_path)

    assert info["sr"] == 48000
    assert info["num_frames"] == 2807470  # FIXME: Info gives more frames


# Issue 215: Glob pattern fix
def test_read_audio_with_glob() -> None:
    """Test reading audio file using glob pattern."""
    path = "gs://esp-ci-cd-tests/esp-data-tests/test_glob_pattern_audio/test_audio[1].wav"
    data, sr = read_audio(path)
    assert sr == 16000
    assert data.shape == (16000,)


def test_read_audio_ffmpeg_segment_glob_char_object() -> None:
    """ffmpeg time-range read handles GCS object names with glob characters.

    The object name contains ``[`` and ``]``; ``_gcs_path_to_url`` must
    percent-encode these so ffmpeg requests the correct object rather than a
    malformed URL that 404s and falls back to a full download.
    """
    path = "gs://esp-ci-cd-tests/esp-data-tests/test_glob_pattern_audio/test_audio[1].wav"
    data, sr = _read_audio_ffmpeg(path, start_time=0.0, end_time=0.5)
    assert sr == 16000
    assert data.shape == (8000,)


class TestGcsPathToUrl:
    """Tests for _gcs_path_to_url."""

    def test_with_gs_prefix(self) -> None:
        result = _gcs_path_to_url("gs://my-bucket/path/to/file.wav")
        assert result == "https://storage.googleapis.com/my-bucket/path/to/file.wav"

    def test_without_gs_prefix(self) -> None:
        result = _gcs_path_to_url("my-bucket/path/to/file.wav")
        assert result == "https://storage.googleapis.com/my-bucket/path/to/file.wav"

    def test_strips_leading_slashes(self) -> None:
        result = _gcs_path_to_url("///my-bucket/path/to/file.wav")
        assert result == "https://storage.googleapis.com/my-bucket/path/to/file.wav"

    def test_rejects_s3_path(self) -> None:
        with pytest.raises(ValueError, match="Unsupported storage scheme"):
            _gcs_path_to_url("s3://my-bucket/path/to/file.wav")

    def test_rejects_r2_path(self) -> None:
        with pytest.raises(ValueError, match="Unsupported storage scheme"):
            _gcs_path_to_url("r2://my-bucket/path/to/file.wav")

    def test_encodes_spaces(self) -> None:
        result = _gcs_path_to_url("gs://my-bucket/path/my file.wav")
        assert result == "https://storage.googleapis.com/my-bucket/path/my%20file.wav"

    def test_encodes_hash(self) -> None:
        result = _gcs_path_to_url("gs://my-bucket/path/file#1.wav")
        assert result == "https://storage.googleapis.com/my-bucket/path/file%231.wav"

    def test_encodes_question_mark(self) -> None:
        result = _gcs_path_to_url("gs://my-bucket/path/file?.wav")
        assert result == "https://storage.googleapis.com/my-bucket/path/file%3F.wav"

    def test_encodes_percent(self) -> None:
        result = _gcs_path_to_url("gs://my-bucket/path/50%off.wav")
        assert result == "https://storage.googleapis.com/my-bucket/path/50%25off.wav"

    def test_encodes_glob_characters(self) -> None:
        result = _gcs_path_to_url("gs://my-bucket/path/file[0].wav")
        assert result == "https://storage.googleapis.com/my-bucket/path/file%5B0%5D.wav"

    def test_preserves_path_separators(self) -> None:
        result = _gcs_path_to_url("gs://my-bucket/a b/c d/e.wav")
        assert result == "https://storage.googleapis.com/my-bucket/a%20b/c%20d/e.wav"


def test_read_audio_ffmpeg_fallback(monkeypatch, caplog) -> None:
    """When the ffmpeg path fails, read_audio falls back to a download read.

    The fallback must produce the same segment as the download path and warn
    once for the failure cause.
    """
    remote_path = "gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3"
    start_time, end_time = 1.0, 2.0

    # Reference segment via the download path (force ffmpeg off by simulating
    # a missing binary).
    def _boom(*args: object, **kwargs: object) -> tuple[np.ndarray, int]:
        raise FFmpegSegmentError("ffmpeg not installed", "simulated missing binary")

    _warn_ffmpeg_fallback_once.cache_clear()
    monkeypatch.setattr("alp_data.io.read_utils._read_audio_ffmpeg", _boom)

    with caplog.at_level("WARNING", logger="alp_data"):
        audio, sr = read_audio(remote_path, start_time=start_time, end_time=end_time)

    assert sr == 44100
    expected_length = int((end_time - start_time) * sr)
    assert audio.shape[0] == expected_length
    assert any("ffmpeg not installed" in r.message for r in caplog.records)


def test_read_audio_ffmpeg_unparseable_probe_raises(monkeypatch) -> None:
    """Malformed ffprobe output surfaces as FFmpegSegmentError, not a raw ValueError.

    This lets `read_audio` catch it and fall back to the download path instead
    of crashing the caller.
    """

    # ffprobe exited 0 but produced no stream line.
    monkeypatch.setattr(
        read_utils.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="")
    )

    with pytest.raises(FFmpegSegmentError) as excinfo:
        _read_audio_ffmpeg("gs://bucket/file.wav", 0.0, 1.0, anonymous=True)
    assert excinfo.value.cause == "ffprobe output unparsable"


def test_read_audio_ffmpeg_input_sr_mismatch_warns(monkeypatch, caplog) -> None:
    """The ffmpeg path warns when `input_sr` disagrees with the native rate."""

    def _fake_run(cmd: list[str], *args: object, **kwargs: object) -> SimpleNamespace:
        if cmd[0] == "ffprobe":
            return SimpleNamespace(stdout="16000,1")
        return SimpleNamespace(stdout=np.zeros(16000, dtype=np.float32).tobytes())

    monkeypatch.setattr(read_utils.subprocess, "run", _fake_run)

    with caplog.at_level("WARNING", logger="alp_data"):
        _read_audio_ffmpeg("gs://bucket/file.wav", 0.0, 1.0, input_sr=44100, anonymous=True)
    assert any("doesn't match" in r.message for r in caplog.records)


class _LocalFS:
    """Stand-in for a GCS filesystem that opens a fixed local file."""

    def __init__(self, local_path: str) -> None:
        self.local_path = local_path
        self.open_kwargs: dict[str, object] = {}

    def open(self, path: str, mode: str = "rb", **kwargs: object) -> object:
        self.open_kwargs = kwargs
        return open(self.local_path, mode)  # noqa: SIM115 - closed by the caller's with


@pytest.mark.parametrize(
    ("start_time", "end_time", "expected"),
    [
        (0.0, 1.0, (0, 1000)),
        (0.25, 0.5, (250, 500)),
        (1.5, None, (1500, 2000)),
        (1.5, 5.0, (1500, 2000)),  # end past EOF is clipped
        (3.0, 4.0, (2000, 2000)),  # start past EOF gives an empty read
    ],
)
def test_read_audio_sfseek_frame_math(
    tmp_path, monkeypatch, start_time, end_time, expected
) -> None:
    """`_read_audio_sfseek` returns the same frames as slicing the decoded file."""
    sr = 1000
    data = np.linspace(-1, 1, 2 * sr, dtype=np.float32)
    local = tmp_path / "ramp.wav"
    sf.write(local, data, sr, subtype="FLOAT")
    fake_fs = _LocalFS(str(local))
    monkeypatch.setattr(read_utils, "filesystem_from_path", lambda _p: fake_fs)

    audio, out_sr = read_utils._read_audio_sfseek(
        read_utils.anypath("gs://bucket/ramp.wav"), start_time, end_time
    )

    assert out_sr == sr
    assert audio.dtype == np.float32
    np.testing.assert_array_equal(audio, data[expected[0] : expected[1]])
    assert fake_fs.open_kwargs == {"block_size": 2**18, "cache_type": "readahead"}


def test_read_audio_sfseek_stereo(tmp_path, monkeypatch) -> None:
    """Multi-channel segments come back as `(frames, channels)`."""
    sr = 1000
    data = np.random.default_rng(0).uniform(-1, 1, (sr, 2)).astype(np.float32)
    local = tmp_path / "stereo.wav"
    sf.write(local, data, sr, subtype="FLOAT")
    monkeypatch.setattr(read_utils, "filesystem_from_path", lambda _p: _LocalFS(str(local)))

    audio, _ = read_utils._read_audio_sfseek(read_utils.anypath("gs://b/s.wav"), 0.1, 0.2)

    np.testing.assert_array_equal(audio, data[100:200])


def _route_spies(monkeypatch, sfseek_error: Exception | None = None) -> dict[str, int]:
    """Replace the sfseek and ffmpeg readers with counters; return the counts."""
    calls = {"sfseek": 0, "ffmpeg": 0}

    def _sfseek(*args: object, **kwargs: object) -> tuple[np.ndarray, int]:
        calls["sfseek"] += 1
        if sfseek_error is not None:
            raise sfseek_error
        return np.zeros(10, dtype=np.float32), 1

    def _ffmpeg(*args: object, **kwargs: object) -> tuple[np.ndarray, int]:
        calls["ffmpeg"] += 1
        return np.ones(10, dtype=np.float32), 2

    monkeypatch.setattr(read_utils, "_read_audio_sfseek", _sfseek)
    monkeypatch.setattr(read_utils, "_read_audio_ffmpeg", _ffmpeg)
    return calls


def test_read_audio_segment_reader_default_is_ffmpeg(monkeypatch) -> None:
    """Without the env flag, gs:// segment reads go to ffmpeg only."""
    monkeypatch.delenv("ALP_DATA_SEGMENT_READER", raising=False)
    calls = _route_spies(monkeypatch)

    _, sr = read_audio("gs://bucket/a.wav", start_time=0.0, end_time=1.0)

    assert sr == 2
    assert calls == {"sfseek": 0, "ffmpeg": 1}


def test_read_audio_segment_reader_sfseek(monkeypatch) -> None:
    """With `ALP_DATA_SEGMENT_READER=sfseek`, gs:// WAV segments use sfseek."""
    monkeypatch.setenv("ALP_DATA_SEGMENT_READER", "sfseek")
    calls = _route_spies(monkeypatch)

    _, sr = read_audio("gs://bucket/a.WAV", start_time=0.0, end_time=1.0)

    assert sr == 1
    assert calls == {"sfseek": 1, "ffmpeg": 0}


def test_read_audio_segment_reader_sfseek_skips_other_formats(monkeypatch) -> None:
    """Formats outside WAV/FLAC/OGG stay on ffmpeg even with the flag set."""
    monkeypatch.setenv("ALP_DATA_SEGMENT_READER", "sfseek")
    calls = _route_spies(monkeypatch)

    read_audio("gs://bucket/a.mp3", start_time=0.0, end_time=1.0)

    assert calls == {"sfseek": 0, "ffmpeg": 1}


def test_read_audio_segment_reader_sfseek_falls_back(monkeypatch) -> None:
    """Any sfseek failure falls back to the ffmpeg path."""
    monkeypatch.setenv("ALP_DATA_SEGMENT_READER", "sfseek")
    calls = _route_spies(monkeypatch, sfseek_error=RuntimeError("boom"))

    _, sr = read_audio("gs://bucket/a.flac", start_time=0.0, end_time=1.0)

    assert sr == 2
    assert calls == {"sfseek": 1, "ffmpeg": 1}
