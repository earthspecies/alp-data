"""Unitary tests of video file reading functions."""

from typing import Any

import numpy as np
import pytest
import torch
from torchcodec.decoders import VideoDecoder

from alp_data.io.read_utils import read_video, read_video_at_frames
from alp_data.io.video_utils import (
    decode_frame_at,
    decode_frames_at,
    decode_frames_in_range,
    decode_video,
    get_video_info,
)

# 2 s, 30 fps, 64x48 h264 clip of ffmpeg's `testsrc` pattern
VIDEO_PATH = "tests/samples/video.mp4"
# Byte-identical copy of `VIDEO_PATH` on GCS
REMOTE_VIDEO_PATH = "gs://esp-ci-cd-tests/esp-data-tests/video-samples/video.mp4"
# Byte-identical copy of `VIDEO_PATH` on a public HTTPS endpoint (R2 dev URL)
HTTPS_VIDEO_PATH = "https://pub-ad974658c4664976b92e83cd9e627e83.r2.dev/test/video.mp4"
# Same clip, with PTS starting at 0.5 s (`ffmpeg ... -output_ts_offset 0.5`)
OFFSET_VIDEO_PATH = "tests/samples/video_offset_start.mp4"
# Variable frame rate: 60 frames, the first 30 at 1/30 s and the rest at 1/15 s.
#   ffmpeg -f lavfi -i testsrc=size=64x48:rate=30:duration=2 \
#     -vf "setpts='if(lt(N,30),N/30,N/15)/TB'" -fps_mode vfr <codec args> <out>
# MKV (libx264): no frame count in the header, approximate mode estimates 118.
VFR_MKV_PATH = "tests/samples/video_vfr.mkv"
# MP4 (libx264): header count is right, but approximate seeking returns wrong frames.
VFR_MP4_PATH = "tests/samples/video_vfr.mp4"
# AVI (`-c:v mjpeg -q:v 31`): header claims 119 frames, consistent with duration * fps.
VFR_AVI_PATH = "tests/samples/video_vfr.avi"

# Largest allowed difference between the approximate and exact frame counts.
MAX_FRAME_GAP = 0  # tighten/loosen as we benchmark more datasets

_VFR_XFAIL = pytest.mark.xfail(
    strict=True, reason="torchcodec's approximate mode trusts header metadata, wrong for VFR"
)
GAP_SAMPLES = [
    VIDEO_PATH,
    OFFSET_VIDEO_PATH,
    pytest.param(VFR_MKV_PATH, marks=_VFR_XFAIL),
    pytest.param(VFR_MP4_PATH, marks=_VFR_XFAIL),
    pytest.param(VFR_AVI_PATH, marks=_VFR_XFAIL),
]


def test_read_from_file() -> None:
    path = VIDEO_PATH
    data = read_video(path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (60, 3, 48, 64)
    assert data.dtype == np.uint8


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
    np.testing.assert_array_equal(data[0], data[1])


def test_read_at_frames_from_file() -> None:
    path = VIDEO_PATH
    data = read_video_at_frames(path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (60, 3, 48, 64)
    assert data.dtype == np.uint8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_read_video_on_gpu_returns_numpy() -> None:
    """Test if GPU-decoded frames are copied back to a host NumPy array."""
    data = read_video(VIDEO_PATH, device="cuda")
    assert isinstance(data, np.ndarray)
    assert data.shape == (60, 3, 48, 64)
    assert data.dtype == np.uint8

    data = read_video_at_frames(VIDEO_PATH, start_frame=0, end_frame=10, device="cuda")
    assert isinstance(data, np.ndarray)
    assert data.shape == (10, 3, 48, 64)


def test_read_at_frames_by_range() -> None:
    """Test if read_video_at_frames reads the frames in `[start_frame, end_frame)`."""
    path = VIDEO_PATH
    full = read_video(path)

    data = read_video_at_frames(path, start_frame=10, end_frame=20)
    np.testing.assert_array_equal(data, full[10:20])

    data = read_video_at_frames(path, start_frame=50)
    np.testing.assert_array_equal(data, full[50:])

    data = read_video_at_frames(path, end_frame=10)
    np.testing.assert_array_equal(data, full[:10])


def test_read_at_frames_with_stride() -> None:
    """Test if read_video_at_frames steps by `stride`."""
    path = VIDEO_PATH
    full = read_video(path)

    data = read_video_at_frames(path, start_frame=0, end_frame=10, stride=3)
    np.testing.assert_array_equal(data, full[0:10:3])

    data = read_video_at_frames(path, stride=2)
    np.testing.assert_array_equal(data, full[::2])


def test_read_at_frames_custom_indices() -> None:
    """Test if custom indices are read in order, with repeats and negative indices."""
    path = VIDEO_PATH
    full = read_video(path)

    data = read_video_at_frames(path, custom_frame_indices=[-1, 0, 0, 30])
    np.testing.assert_array_equal(data, full[[59, 0, 0, 30]])


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


# -- GCS -------------------------------------------------------------------------


def test_get_video_info_from_gcs() -> None:
    """Test if get_video_info reads the same metadata from GCS as locally."""
    assert get_video_info(REMOTE_VIDEO_PATH) == get_video_info(VIDEO_PATH)


def test_read_video_from_gcs() -> None:
    """Test reading a whole video from GCS."""
    data = read_video(REMOTE_VIDEO_PATH)
    assert data.shape == (60, 3, 48, 64)
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.uint8
    np.testing.assert_array_equal(data, read_video(VIDEO_PATH))


def test_read_video_by_time_from_gcs() -> None:
    """Test reading a video segment by time from GCS, with and without resampling."""
    data = read_video(REMOTE_VIDEO_PATH, start_time=0.5, end_time=1.0)
    assert data.shape[0] == 15
    np.testing.assert_array_equal(data, read_video(VIDEO_PATH, start_time=0.5, end_time=1.0))

    data = read_video(REMOTE_VIDEO_PATH, start_time=1.0, fps=10)
    assert data.shape == (10, 3, 48, 64)


def test_read_at_frames_from_gcs() -> None:
    """Test reading frames by index from GCS."""
    full = read_video(VIDEO_PATH)

    data = read_video_at_frames(REMOTE_VIDEO_PATH, start_frame=10, end_frame=40, stride=3)
    np.testing.assert_array_equal(data, full[10:40:3])

    data = read_video_at_frames(REMOTE_VIDEO_PATH, custom_frame_indices=[-1, 0, 30])
    np.testing.assert_array_equal(data, full[[59, 0, 30]])


def test_decode_frame_at_from_gcs() -> None:
    torch.testing.assert_close(
        decode_frame_at(REMOTE_VIDEO_PATH, 5), decode_frame_at(VIDEO_PATH, 5)
    )


def test_read_video_missing_from_gcs() -> None:
    with pytest.raises(FileNotFoundError):
        read_video("gs://esp-ci-cd-tests/esp-data-tests/tests/samples/does_not_exist.mp4")


# -- HTTPS -----------------------------------------------------------------------


def test_get_video_info_from_https() -> None:
    """Test if get_video_info reads the same metadata over HTTPS as locally."""
    assert get_video_info(HTTPS_VIDEO_PATH) == get_video_info(VIDEO_PATH)


def test_read_video_from_https() -> None:
    """Test reading a whole video over HTTPS."""
    data = read_video(HTTPS_VIDEO_PATH)
    assert data.shape == (60, 3, 48, 64)
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.uint8
    np.testing.assert_array_equal(data, read_video(VIDEO_PATH))


def test_read_video_by_time_from_https() -> None:
    """Test reading a video segment by time over HTTPS, with and without resampling."""
    data = read_video(HTTPS_VIDEO_PATH, start_time=0.5, end_time=1.0)
    assert data.shape[0] == 15
    np.testing.assert_array_equal(data, read_video(VIDEO_PATH, start_time=0.5, end_time=1.0))

    data = read_video(HTTPS_VIDEO_PATH, start_time=1.0, fps=10)
    assert data.shape == (10, 3, 48, 64)


def test_read_at_frames_from_https() -> None:
    """Test reading frames by index over HTTPS."""
    full = read_video(VIDEO_PATH)

    data = read_video_at_frames(HTTPS_VIDEO_PATH, start_frame=10, end_frame=40, stride=3)
    np.testing.assert_array_equal(data, full[10:40:3])

    data = read_video_at_frames(HTTPS_VIDEO_PATH, custom_frame_indices=[-1, 0, 30])
    np.testing.assert_array_equal(data, full[[59, 0, 30]])


def test_decode_frame_at_from_https() -> None:
    torch.testing.assert_close(decode_frame_at(HTTPS_VIDEO_PATH, 5), decode_frame_at(VIDEO_PATH, 5))


def test_read_video_missing_from_https() -> None:
    with pytest.raises(FileNotFoundError):
        read_video("https://pub-ad974658c4664976b92e83cd9e627e83.r2.dev/test/does_not_exist.mp4")


def test_read_video_offset_start_from_zero() -> None:
    """Test if `start_time` is an offset from the first PTS, not absolute."""
    assert read_video(OFFSET_VIDEO_PATH, start_time=0.0).shape == (60, 3, 48, 64)


def test_read_video_offset_first_second() -> None:
    assert read_video(OFFSET_VIDEO_PATH, start_time=0.0, end_time=1.0).shape == (
        30,
        3,
        48,
        64,
    )


def test_read_video_offset_end_at_duration() -> None:
    duration = get_video_info(OFFSET_VIDEO_PATH)["duration"]
    assert read_video(OFFSET_VIDEO_PATH, end_time=duration).shape == (60, 3, 48, 64)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"start_time": -0.1}, "start_time must be non-negative"),
        ({"start_time": -0.1, "end_time": 1.0}, "start_time must be non-negative"),
        ({"end_time": 0.0}, "end_time must be greater than start_time"),
        ({"end_time": -1.0}, "end_time must be greater than start_time"),
        ({"start_time": 2.0}, "beyond video duration"),
        ({"start_time": 5.0, "end_time": 6.0}, "beyond video duration"),
    ],
)
def test_read_video_offset_invalid_time_range(kwargs: dict[str, Any], match: str) -> None:
    """Test if out-of-range times raise our `ValueError`, not torchcodec's."""
    with pytest.raises(ValueError, match=match):
        read_video(OFFSET_VIDEO_PATH, **kwargs)


# -- Approximate vs exact seek mode ------------------------------------------------


@pytest.mark.parametrize("path", GAP_SAMPLES)
def test_raw_approximate_matches_exact(path: str) -> None:
    """Measure torchcodec's own approximate-vs-exact gap, bypassing our fallback.

    The VFR samples are expected to fail; `test_approximate_matches_exact`
    checks that our readers hide the gap.
    """
    approx_decoder = VideoDecoder(path, seek_mode="approximate")
    exact_decoder = VideoDecoder(path, seek_mode="exact")
    exact = exact_decoder.metadata.num_frames
    assert abs(approx_decoder.metadata.num_frames - exact) <= MAX_FRAME_GAP

    idx = [0, exact // 2, exact - 1]
    assert torch.equal(
        approx_decoder.get_frames_at(idx).data, exact_decoder.get_frames_at(idx).data
    )


@pytest.mark.parametrize(
    "path", [VIDEO_PATH, OFFSET_VIDEO_PATH, VFR_MKV_PATH, VFR_MP4_PATH, VFR_AVI_PATH]
)
def test_approximate_matches_exact(path: str) -> None:
    """Test if the default approximate mode gives the same frames as exact mode."""
    approx = get_video_info(path, seek_mode="approximate")["num_frames"]
    exact = get_video_info(path, seek_mode="exact")["num_frames"]
    assert abs(approx - exact) <= MAX_FRAME_GAP

    # Reading "all frames" by index must not run past the decodable frames.
    frames = read_video_at_frames(path)
    assert frames.shape[0] == exact
    np.testing.assert_array_equal(frames, read_video_at_frames(path, seek_mode="exact"))

    # Same index -> same frame in both modes.
    idx = [0, exact // 2, exact - 1]
    a = decode_frames_at(path, custom_frame_indices=idx, seek_mode="approximate").data
    e = decode_frames_at(path, custom_frame_indices=idx, seek_mode="exact").data
    assert torch.equal(a, e)
