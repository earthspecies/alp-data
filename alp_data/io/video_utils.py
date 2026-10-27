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
    path: str | AnyPathT,
    fps: float | None = None,
    seek_mode: SeekMode = "approximate",
    device: str | torch.device | None = None,
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
        "approximate" seeks to the frames to decode by relying on video metadata,
        and returns the same result as "exact" for most videos. "exact" performs
        a full scan of the video first, so instantiating the VideoDecoder is slower.
    device : str | torch.device | None, optional
        Device to decode on, e.g. "cpu" or "cuda". If None (default), uses
        PyTorch's default device, `torch.get_default_device()`, which is the CPU
        unless changed with `torch.set_default_device`.

    Returns
    -------
    FrameBatch
        All frames, with `.data` shaped `(N, C, H, W)` and on `device`.

    Notes
    -----
    Opening a missing `path` raises `FileNotFoundError` and an undecodable
    file raises `ValueError` or `RuntimeError`.
    """
    with filesystem_from_path(path).open(str(path), "rb") as f:
        decoder = VideoDecoder(f, seek_mode=seek_mode, device=device)
        return decoder.get_all_frames(fps)


def decode_frames_in_range(
    path: str | AnyPathT,
    start_time: float | None = None,
    end_time: float | None = None,
    fps: float | None = None,
    seek_mode: SeekMode = "approximate",
    device: str | torch.device | None = None,
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
        "approximate" seeks to the frames to decode by relying on video metadata,
        and returns the same result as "exact" for most videos. "exact" performs
        a full scan of the video first, so instantiating the VideoDecoder is slower.
    device : str | torch.device | None, optional
        Device to decode on, e.g. "cpu" or "cuda". If None (default), uses
        PyTorch's default device, `torch.get_default_device()`, which is the CPU
        unless changed with `torch.set_default_device`.

    Returns
    -------
    FrameBatch
        The frames played in `[start_time, end_time)`, with `.data` shaped
        `(N, C, H, W)` and on `device`.

    Notes
    -----
    Opening a missing `path` raises `FileNotFoundError` and an undecodable
    file raises `ValueError` or `RuntimeError`. A range lying outside the
    video's stream raises `ValueError`.
    """
    with filesystem_from_path(path).open(str(path), "rb") as f:
        decoder = VideoDecoder(f, seek_mode=seek_mode, device=device)
        if start_time is None:
            start_time = decoder.metadata.begin_stream_seconds
        if end_time is None:
            end_time = decoder.metadata.end_stream_seconds
        return decoder.get_frames_played_in_range(start_time, end_time, fps)


def decode_frames_at(
    path: str | AnyPathT,
    start: int | None = None,
    end: int | None = None,
    stride: int = 1,
    seek_mode: SeekMode = "approximate",
    custom_frame_indices: list[int] | None = None,
    device: str | torch.device | None = None,
) -> FrameBatch:
    """Read a strided range of frames, or an explicit list of frames, from a video.

    By default frames are taken at `range(start, end, stride)`, so `end` is exclusive
    and the number of frames returned is `ceil((end - start) / stride)`. If
    `custom_frame_indices` is given, exactly those frames are read instead.

    Parameters
    ----------
    path : str or AnyPathT
        Path or URI of the video.
    start : int | None, optional
        First frame index, inclusive. If None, starts at the first frame.
    end : int | None, optional
        Last frame index, exclusive. If None, reads to the last frame.
    stride : int, optional
        Step between frame indices, by default 1.
    seek_mode : SeekMode, optional
        Decoder seek mode, by default "approximate".
        "approximate" seeks to the frames to decode by relying on video metadata,
        and returns the same result as "exact" for most videos. "exact" performs
        a full scan of the video first, so instantiating the VideoDecoder is slower.
    custom_frame_indices : list[int] | None, optional
        Explicit frame indices to read, in order (repeats allowed). Negative
        indices count from the end of the video. By default None.
    device : str | torch.device | None, optional
        Device to decode on, e.g. "cpu" or "cuda". If None (default), uses
        PyTorch's default device, `torch.get_default_device()`, which is the CPU
        unless changed with `torch.set_default_device`.

    Returns
    -------
    FrameBatch
        The selected frames, with `.data` shaped `(N, C, H, W)` and on `device`,
        in the requested order.

    Raises
    ------
    ValueError
        If `custom_frame_indices` is combined with `start`, `end` or `stride`.
        If `custom_frame_indices` is empty.
        If `start` or `end` is negative, `end` is not greater than `start`, or
        `stride` is less than 1.
        If `end` is None and the video's frame count is unknown.
    IndexError
        If `end` is None and `start` is not less than the video's frame count.

    Notes
    -----
    Opening a missing `path` raises `FileNotFoundError` and an undecodable
    file raises `ValueError` or `RuntimeError`. The decoder raises `IndexError`
    if any requested index is out of range for the video.
    """
    if custom_frame_indices is not None:
        if start is not None or end is not None or stride != 1:
            raise ValueError("custom_frame_indices cannot be combined with start, end or stride")
        if not custom_frame_indices:
            raise ValueError("custom_frame_indices must not be empty")
    else:
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")
        start = 0 if start is None else start
        if start < 0 or (end is not None and end < 0):
            raise ValueError("start and end must be non-negative")
        if end is not None and end <= start:
            raise ValueError("end must be greater than start")

    with filesystem_from_path(path).open(str(path), "rb") as f:
        decoder = VideoDecoder(f, seek_mode=seek_mode, device=device)
        if custom_frame_indices is not None:
            indices = list(custom_frame_indices)
        else:
            if end is None:
                end = decoder.metadata.num_frames
                if end is None:
                    raise ValueError("Video frame count is unknown; pass `end` explicitly")
                if start >= end:
                    raise IndexError(f"start={start} is out of range for a video with {end} frames")
            indices = list(range(start, end, stride))
        return decoder.get_frames_at(indices)


def decode_frame_at(
    path: str | AnyPathT,
    index: int,
    seek_mode: SeekMode = "approximate",
    device: str | torch.device | None = None,
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
        "approximate" seeks to the frames to decode by relying on video metadata,
        and returns the same result as "exact" for most videos. "exact" performs
        a full scan of the video first, so instantiating the VideoDecoder is slower.
    device : str | torch.device | None, optional
        Device to decode on, e.g. "cpu" or "cuda". If None (default), uses
        PyTorch's default device, `torch.get_default_device()`, which is the CPU
        unless changed with `torch.set_default_device`.

    Returns
    -------
    torch.Tensor
        The frame, shaped `(C, H, W)` and on `device`.

    Notes
    -----
    Opening a missing `path` raises `FileNotFoundError` and an undecodable
    file raises `ValueError` or `RuntimeError`; the decoder raises `IndexError`
    if `index` is out of range for the video.
    """
    with filesystem_from_path(path).open(str(path), "rb") as f:
        decoder = VideoDecoder(f, seek_mode=seek_mode, device=device)
        return decoder[index]
