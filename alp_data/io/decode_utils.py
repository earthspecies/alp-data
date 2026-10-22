"""Low-level video decoding helpers built on `torchcodec`."""

from typing import Literal

import torch
from torchcodec import FrameBatch
from torchcodec.decoders import VideoDecoder

from alp_data.io.filesystem import filesystem_from_path
from alp_data.io.paths import AnyPathT

SeekMode = Literal["exact", "approximate"]

__all__ = [
    "SeekMode",
    "decode_frame_at",
    "decode_frames_at",
    "decode_frames_in_range",
    "decode_video",
]


def decode_video(
    path: str | AnyPathT, fps: float | None = None, seek_mode: SeekMode = "approximate"
) -> FrameBatch:
    """Read every frame of a video, optionally resampled to a target frame rate.

    Parameters
    ----------
    path : str or AnyPathT
        Path or URI of the video.
    fps : float | None, optional
        Target frame rate. If None, the video's native frame rate is used.
    seek_mode : SeekMode, optional
        Decoder seek mode, by default "approximate".

    Returns
    -------
    FrameBatch
        All frames, with `.data` shaped `(N, C, H, W)`.

    Notes
    -----
    Opening a missing `path` raises `FileNotFoundError` and an undecodable
    file raises `ValueError` or `RuntimeError`.
    """
    with filesystem_from_path(path).open(str(path), "rb") as f:
        decoder = VideoDecoder(f, seek_mode=seek_mode)
        return decoder.get_all_frames(fps)


def decode_frames_in_range(
    path: str | AnyPathT,
    start_time: float | None = None,
    end_time: float | None = None,
    fps: float | None = None,
    seek_mode: SeekMode = "approximate",
) -> FrameBatch:
    """Read the frames played within a time range of a video.

    Parameters
    ----------
    path : str or AnyPathT
        Path or URI of the video.
    start_time : float | None, optional
        Start time in seconds. If None, starts at the beginning of the stream.
    end_time : float | None, optional
        End time in seconds. If None, reads to the end of the stream.
    fps : float | None, optional
        Target frame rate. If None, the video's native frame rate is used.
    seek_mode : SeekMode, optional
        Decoder seek mode, by default "approximate".

    Returns
    -------
    FrameBatch
        The frames played in `[start_time, end_time)`, with `.data` shaped
        `(N, C, H, W)`.

    Notes
    -----
    Opening a missing `path` raises `FileNotFoundError` and an undecodable
    file raises `ValueError` or `RuntimeError`. A range lying outside the
    video's stream raises `ValueError`.
    """
    with filesystem_from_path(path).open(str(path), "rb") as f:
        decoder = VideoDecoder(f, seek_mode=seek_mode)
        if start_time is None:
            start_time = decoder.metadata.begin_stream_seconds
        if end_time is None:
            end_time = decoder.metadata.end_stream_seconds
        return decoder.get_frames_played_in_range(start_time, end_time, fps)


def decode_frames_at(
    path: str | AnyPathT,
    start: int,
    end: int,
    stride: int = 1,
    seek_mode: SeekMode = "approximate",
    custom_frame_indices: list[int] | None = None,
) -> FrameBatch:
    """Read a strided range of frames, or an explicit list of frames, from a video.

    By default frames are taken at `range(start, end, stride)`, so `end` is exclusive
    and the number of frames returned is `ceil((end - start) / stride)`. If
    `custom_frame_indices` is given, exactly those frames are read instead and `start`,
    `end` and `stride` are ignored.

    Parameters
    ----------
    path : str or AnyPathT
        Path or URI of the video.
    start : int
        First frame index, inclusive.
    end : int
        Last frame index, exclusive.
    stride : int, optional
        Step between frame indices, by default 1.
    seek_mode : SeekMode, optional
        Decoder seek mode, by default "approximate".
    custom_frame_indices : list[int] | None, optional
        Explicit frame indices to read, in order (repeats allowed). By default None.

    Returns
    -------
    FrameBatch
        The selected frames, with `.data` shaped `(N, C, H, W)`, in the requested
        order.

    Notes
    -----
    Opening a missing `path` raises `FileNotFoundError` and an undecodable
    file raises `ValueError` or `RuntimeError`; the decoder raises `IndexError`
    if any requested index is negative or greater than or equal to the video's
    frame count.
    """
    if custom_frame_indices is None:
        indices = list(range(start, end, stride))
    else:
        indices = list(custom_frame_indices)

    with filesystem_from_path(path).open(str(path), "rb") as f:
        decoder = VideoDecoder(f, seek_mode=seek_mode)
        return decoder.get_frames_at(indices)


def decode_frame_at(
    path: str | AnyPathT, index: int, seek_mode: SeekMode = "approximate"
) -> torch.Tensor:
    """Read a single frame from a video.

    Parameters
    ----------
    path : str or AnyPathT
        Path or URI of the video.
    index : int
        Frame index to read.
    seek_mode : SeekMode, optional
        Decoder seek mode, by default "approximate".

    Returns
    -------
    torch.Tensor
        The frame, shaped `(C, H, W)`.

    Notes
    -----
    Opening a missing `path` raises `FileNotFoundError` and an undecodable
    file raises `ValueError` or `RuntimeError`; the decoder raises `IndexError`
    if `index` is out of range for the video.
    """
    with filesystem_from_path(path).open(str(path), "rb") as f:
        decoder = VideoDecoder(f, seek_mode=seek_mode)
        return decoder[index]
