"""Tests for read_audio_ffmpeg and helper functions in read_gcs_audio_range."""

import numpy as np
import pytest

from esp_data.io.read_audio_ffmpeg import (
    _gcs_path_to_url,
    get_gcs_token,
    read_audio_ffmpeg,
)


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

    def test_gs_prefix_no_double_slash_strip(self) -> None:
        result = _gcs_path_to_url("gs://bucket/blob")
        assert result == "https://storage.googleapis.com/bucket/blob"

    def test_rejects_s3_path(self) -> None:
        with pytest.raises(ValueError, match="Unsupported storage scheme"):
            _gcs_path_to_url("s3://my-bucket/path/to/file.wav")

    def test_rejects_r2_path(self) -> None:
        with pytest.raises(ValueError, match="Unsupported storage scheme"):
            _gcs_path_to_url("r2://my-bucket/path/to/file.wav")


class TestGetGcsToken:
    """Tests for get_gcs_token."""

    def test_returns_string(self) -> None:
        token = get_gcs_token()
        assert isinstance(token, str)
        assert len(token) > 0


class TestReadAudioFfmpeg:
    """Tests for read_audio_ffmpeg using a known GCS test file."""

    GCS_PATH = "gs://esp-ci-cd-tests/esp-data-tests/XC_corcorax.mp3"

    def test_read_full_file(self) -> None:
        audio, sr = read_audio_ffmpeg(self.GCS_PATH)
        assert sr == 48000
        assert audio.dtype == np.float32
        assert audio.ndim >= 1
        assert audio.shape[-1] > 0

    def test_read_segment(self) -> None:
        start, end = 5.0, 10.0
        audio, sr = read_audio_ffmpeg(self.GCS_PATH, start_time=start, end_time=end)
        assert sr == 48000
        expected_frames = int((end - start) * sr)
        # Allow small tolerance for MP3 frame alignment
        assert abs(audio.shape[-1] - expected_frames) < sr * 0.05

    def test_read_from_start(self) -> None:
        audio, sr = read_audio_ffmpeg(self.GCS_PATH, start_time=0.0, end_time=1.0)
        assert sr == 48000
        expected_frames = sr  # 1 second
        assert abs(audio.shape[-1] - expected_frames) < sr * 0.05

    def test_sample_rate_matches_known_value(self) -> None:
        _, sr = read_audio_ffmpeg(self.GCS_PATH, start_time=0.0, end_time=0.5)
        assert sr == 48000
