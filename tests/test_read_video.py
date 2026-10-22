"""Unitary tests of video file reading functions."""

import numpy as np
import pytest
import torch

from alp_data.io.decode_utils import (
    decode_frame_at,
    decode_frames_at,
    decode_frames_in_range,
    decode_video,
)
from alp_data.io.read_utils import get_video_info, read_video

# 2 s, 30 fps, 64x48 h264 clip of ffmpeg's `testsrc` pattern
VIDEO_PATH = "tests/samples/video.mp4"


def test_read_from_file() -> None:
    path = VIDEO_PATH
    data, fps = read_video(path)
    assert fps == 30
    assert data.shape == (60, 3, 48, 64)
    assert data.dtype == np.uint8


def test_read_from_file_from_time() -> None:
    path = VIDEO_PATH
    data, fps = read_video(path, start_time=0.0)
    assert fps == 30
    assert data.shape == (60, 3, 48, 64)

    data, fps = read_video(path, start_time=1.0)
    assert fps == 30
    assert data.shape == (30, 3, 48, 64)


def test_read_from_file_to_time() -> None:
    """Test if read_video honours `end_time` when no `start_time` is given."""
    path = VIDEO_PATH
    data, fps = read_video(path, end_time=1.0)
    assert fps == 30
    assert data.shape == (30, 3, 48, 64)


def test_get_video_info() -> None:
    """Test if get_video_info retrieves correct stream metadata."""
    path = VIDEO_PATH
    info = get_video_info(path)

    assert info["duration"] == 2
    assert info["height"] == 48
    assert info["width"] == 64
    assert info["num_frames"] == 60
    assert info["average_fps"] == 30
    assert info["codec"] == "h264"
    assert info["pixel_format"] == "yuv420p"


def test_read_video_by_time() -> None:
    """Test if read_video reads the correct video segment by time (local MP4)."""
    path = VIDEO_PATH
    start_time = 0.5  # Start at 0.5 seconds
    end_time = 1.0  # End at 1 second

    video, fps = read_video(path, start_time=start_time, end_time=end_time)

    assert fps == 30
    expected_length = int((end_time - start_time) * fps)
    assert video.shape[0] == expected_length


def test_read_video_invalid_time_range() -> None:
    """Test if read_video rejects an `end_time` that is not after `start_time`."""
    with pytest.raises(ValueError, match="end_time must be greater than start_time"):
        read_video(VIDEO_PATH, start_time=1.0, end_time=0.5)

    with pytest.raises(ValueError, match="end_time must be greater than start_time"):
        read_video(VIDEO_PATH, start_time=1.0, end_time=1.0)


def test_read_video_resample_fps() -> None:
    """Test if read_video resamples to a lower frame rate."""
    data, fps = read_video(VIDEO_PATH, fps=10)
    assert fps == 10
    assert data.shape == (20, 3, 48, 64)

    data, fps = read_video(VIDEO_PATH, start_time=0.5, end_time=1.0, fps=10)
    assert fps == 10
    assert data.shape == (5, 3, 48, 64)


def test_read_video_fps_above_native_duplicates_frames() -> None:
    """Test if a frame rate above the native one duplicates frames instead of raising."""
    data, fps = read_video(VIDEO_PATH, fps=60)
    assert fps == 60
    assert data.shape == (120, 3, 48, 64)
    np.testing.assert_array_equal(data[0], data[1])


def test_decode_video() -> None:
    batch = decode_video(VIDEO_PATH)
    assert batch.data.shape == (60, 3, 48, 64)


def test_decode_frames_in_range_open_ended() -> None:
    """Test if decode_frames_in_range defaults to the stream bounds."""
    assert decode_frames_in_range(VIDEO_PATH).data.shape[0] == 60
    assert decode_frames_in_range(VIDEO_PATH, start_time=1.0).data.shape[0] == 30
    assert decode_frames_in_range(VIDEO_PATH, end_time=1.0).data.shape[0] == 30


def test_decode_frames_at_with_stride() -> None:
    batch = decode_frames_at(VIDEO_PATH, start=0, end=10, stride=3)
    assert batch.data.shape == (4, 3, 48, 64)


def test_decode_frames_at_custom_indices() -> None:
    """Test if custom indices are read in the requested order, with repeats."""
    indices = [5, 2, 5]
    batch = decode_frames_at(VIDEO_PATH, start=0, end=0, custom_frame_indices=indices)

    expected = torch.stack([decode_frame_at(VIDEO_PATH, i) for i in indices])
    torch.testing.assert_close(batch.data, expected)


def test_decode_frame_at() -> None:
    frame = decode_frame_at(VIDEO_PATH, 5)
    assert frame.shape == (3, 48, 64)


def test_decode_frame_at_out_of_range() -> None:
    with pytest.raises(IndexError):
        decode_frame_at(VIDEO_PATH, 60)
