"""Unitary tests of video file reading functions."""

from typing import Any

import pytest
import torch

from alp_data.io.decode_utils import (
    decode_frame_at,
    decode_frames_at,
    decode_frames_in_range,
    decode_video,
)
from alp_data.io.read_utils import get_video_info, read_video, read_video_at_frames

# 2 s, 30 fps, 64x48 h264 clip of ffmpeg's `testsrc` pattern
VIDEO_PATH = "tests/samples/video.mp4"


def test_read_from_file() -> None:
    path = VIDEO_PATH
    data = read_video(path)
    assert isinstance(data, torch.Tensor)
    assert data.shape == (60, 3, 48, 64)
    assert data.dtype == torch.uint8


def test_read_from_file_from_time() -> None:
    path = VIDEO_PATH
    data = read_video(path, start_time=0.0)
    assert data.shape == (60, 3, 48, 64)

    data = read_video(path, start_time=1.0)
    assert data.shape == (30, 3, 48, 64)


def test_read_from_file_to_time() -> None:
    """Test if read_video honours `end_time` when no `start_time` is given."""
    path = VIDEO_PATH
    data = read_video(path, end_time=1.0)
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

    video = read_video(path, start_time=start_time, end_time=end_time)

    expected_length = int((end_time - start_time) * 30)  # native fps
    assert video.shape[0] == expected_length


def test_read_video_invalid_time_range() -> None:
    """Test if read_video rejects an `end_time` that is not after `start_time`."""
    with pytest.raises(ValueError, match="end_time must be greater than start_time"):
        read_video(VIDEO_PATH, start_time=1.0, end_time=0.5)

    with pytest.raises(ValueError, match="end_time must be greater than start_time"):
        read_video(VIDEO_PATH, start_time=1.0, end_time=1.0)


def test_read_video_resample_fps() -> None:
    """Test if read_video resamples to a lower frame rate."""
    data = read_video(VIDEO_PATH, fps=10)
    assert data.shape == (20, 3, 48, 64)

    data = read_video(VIDEO_PATH, start_time=0.5, end_time=1.0, fps=10)
    assert data.shape == (5, 3, 48, 64)


def test_read_video_fps_above_native_duplicates_frames() -> None:
    """Test if a frame rate above the native one duplicates frames instead of raising."""
    data = read_video(VIDEO_PATH, fps=60)
    assert data.shape == (120, 3, 48, 64)
    torch.testing.assert_close(data[0], data[1])


def test_read_at_frames_from_file() -> None:
    path = VIDEO_PATH
    data = read_video_at_frames(path)
    assert isinstance(data, torch.Tensor)
    assert data.shape == (60, 3, 48, 64)
    assert data.dtype == torch.uint8


def test_read_at_frames_by_range() -> None:
    """Test if read_video_at_frames reads the frames in `[start_frame, end_frame)`."""
    path = VIDEO_PATH
    full = read_video(path)

    data = read_video_at_frames(path, start_frame=10, end_frame=20)
    torch.testing.assert_close(data, full[10:20])

    data = read_video_at_frames(path, start_frame=50)
    torch.testing.assert_close(data, full[50:])

    data = read_video_at_frames(path, end_frame=10)
    torch.testing.assert_close(data, full[:10])


def test_read_at_frames_with_stride() -> None:
    """Test if read_video_at_frames steps by `stride`."""
    path = VIDEO_PATH
    full = read_video(path)

    data = read_video_at_frames(path, start_frame=0, end_frame=10, stride=3)
    torch.testing.assert_close(data, full[0:10:3])

    data = read_video_at_frames(path, stride=2)
    torch.testing.assert_close(data, full[::2])


def test_read_at_frames_custom_indices() -> None:
    """Test if custom indices are read in order, with repeats and negative indices."""
    path = VIDEO_PATH
    full = read_video(path)

    data = read_video_at_frames(path, custom_frame_indices=[-1, 0, 0, 30])
    torch.testing.assert_close(data, full[[59, 0, 0, 30]])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"start_frame": 5, "end_frame": 5}, "end must be greater than start"),
        ({"start_frame": 10, "end_frame": 5}, "end must be greater than start"),
        ({"start_frame": -1}, "start and end must be non-negative"),
        ({"end_frame": -1}, "start and end must be non-negative"),
        ({"stride": 0}, "stride must be at least 1"),
        ({"custom_frame_indices": []}, "custom_frame_indices must not be empty"),
        ({"custom_frame_indices": [1], "start_frame": 0}, "cannot be combined"),
        ({"custom_frame_indices": [1], "stride": 2}, "cannot be combined"),
    ],
)
def test_read_at_frames_invalid_arguments(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        read_video_at_frames(VIDEO_PATH, **kwargs)


def test_read_at_frames_out_of_range() -> None:
    with pytest.raises(IndexError):
        read_video_at_frames(VIDEO_PATH, start_frame=60)

    with pytest.raises(IndexError):
        read_video_at_frames(VIDEO_PATH, start_frame=0, end_frame=61)

    with pytest.raises(IndexError):
        read_video_at_frames(VIDEO_PATH, custom_frame_indices=[60])


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


def test_decode_frames_at_open_ended() -> None:
    """Test if decode_frames_at defaults to the first and last frames."""
    assert decode_frames_at(VIDEO_PATH).data.shape[0] == 60
    assert decode_frames_at(VIDEO_PATH, start=50).data.shape[0] == 10
    assert decode_frames_at(VIDEO_PATH, end=10).data.shape[0] == 10


def test_decode_frames_at_custom_indices() -> None:
    """Test if custom indices are read in the requested order, with repeats."""
    indices = [5, 2, 5]
    batch = decode_frames_at(VIDEO_PATH, custom_frame_indices=indices)

    expected = torch.stack([decode_frame_at(VIDEO_PATH, i) for i in indices])
    torch.testing.assert_close(batch.data, expected)


def test_decode_frame_at() -> None:
    frame = decode_frame_at(VIDEO_PATH, 5)
    assert frame.shape == (3, 48, 64)


def test_decode_frame_at_out_of_range() -> None:
    with pytest.raises(IndexError):
        decode_frame_at(VIDEO_PATH, 60)
