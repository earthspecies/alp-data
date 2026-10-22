"""Transforms that cut video rows into clips and set their sampling rate.

Clip columns
------------
Video datasets contain the information:

- `start_frame`, `end_frame`: half-open frame indices in the video's native timebase.
- `frame_stride`: step between decoded frames, in native frames.
- `video_fps`: native frame rate of the video (read only).
- `fps`: effective sampled frame rate, `video_fps / frame_stride`.

Transforms expand one video row into multiple clips and/or edit these columns for the clips.

- `ResampleToFPS` sets `frame_stride` and `fps`.
- `ClipToNumFrames` cuts each clip into clips of a fixed number of sampled frames.
- `ClipToNumSeconds` cuts each clip into clips of a fixed duration.

The clipping transforms are just wrappers on `ExpandByFunction` transforms with the row methods
`clip_to_num_frames` and `clip_to_num_seconds`.

Examples
--------
>>> from alp_data import DatasetConfig
>>> cfg = DatasetConfig(
...     dataset_name="mammalps",
...     split="benchmark-1-val",
...     transformations=[
...         {"type": "clip_to_num_frames", "clip_length": 16, "fps": 5},
...     ],
... )

Or in Python, applied to a dataset's backend:

>>> from alp_data.transforms import ClipToNumFrames
>>> transform = ClipToNumFrames(clip_length=16, fps=5)
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel

from alp_data.backends.protocol import DataBackend

from . import register_transform
from .expand import ExpandByFunction, register_row_method

logger = logging.getLogger("alp_data")

_WARNED_FPS: set[tuple[float, float]] = set()


def _check_positive_int(name: str, value: Any) -> None:  # noqa: ANN401
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


def _check_positive_number(name: str, value: Any) -> None:  # noqa: ANN401
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a number > 0, got {value!r}")


def _stride_for_fps(video_fps: float, fps: float) -> int:
    """Return the decode stride that brings `video_fps` closest to `fps`.

    A warning is logged, once per `(video_fps, fps)` pair, when the resulting rate
    misses `fps` by more than 1%.

    Parameters
    ----------
    video_fps : float
        Native frame rate of the video.
    fps : float
        Target sampling rate.

    Returns
    -------
    int
        The stride, at least 1.

    Raises
    ------
    ValueError
        If `fps` is not positive.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps!r}")
    stride = max(1, round(video_fps / fps))
    effective = video_fps / stride
    if abs(effective - fps) > 1e-2 * fps and (video_fps, fps) not in _WARNED_FPS:
        _WARNED_FPS.add((video_fps, fps))
        logger.warning(
            "Requested fps=%s does not divide video_fps=%s; sampling at %.4g fps (stride %d)",
            fps,
            video_fps,
            effective,
            stride,
        )
    return stride


def _cut_windows(
    start: int, end: int, span: int, hop: int, drop_last: bool
) -> list[tuple[int, int]]:
    """Cut `[start, end)` into windows of `span` frames, `hop` frames apart.

    If `drop_last` is False and the windows do not reach `end`, one more window ending
    exactly at `end` is added. Ranges shorter than `span` give no windows.

    Returns
    -------
    list[tuple[int, int]]
        Half-open `(start, end)` frame windows.
    """
    if end - start < span:
        return []
    windows = [(s, s + span) for s in range(start, end - span + 1, hop)]
    if not drop_last and windows[-1][1] < end:
        windows.append((end - span, end))
    return windows


def _clip_rows(
    windows: list[tuple[int, int]], stride: int, video_fps: float
) -> list[dict[str, Any]]:
    fps = video_fps / stride
    return [
        {"start_frame": s, "end_frame": e, "frame_stride": stride, "fps": fps} for s, e in windows
    ]


@register_row_method()
def clip_to_num_frames(
    row: dict[str, Any],
    clip_length: int,
    fps: float | None = None,
    drop_last: bool = True,
    hop: int | None = None,
) -> list[dict[str, Any]]:
    """Cut a row's clip into consecutive clips of `clip_length` sampled frames.

    Works within the row's current `[start_frame, end_frame)`, so it can be chained
    after other clipping transforms. Each output clip decodes exactly `clip_length`
    frames. Set `clip_length=1` to turn a video dataset into an image dataset.

    Parameters
    ----------
    row : dict[str, Any]
        Row carrying the clip columns.
    clip_length : int
        Number of sampled frames per clip.
    fps : float | None, optional
        Target sampling rate. The stride is `round(video_fps / fps)`; a warning is
        logged when that misses `fps` by more than 1%. If None, the row's current
        `frame_stride` is kept.
    drop_last : bool, optional
        If True (default), frames left over after the last full clip are discarded.
        If False, an extra clip ending on the last frame is added, overlapping its
        predecessor.
    hop : int | None, optional
        Distance between consecutive clip starts, in sampled frames. Defaults to
        `clip_length` (non-overlapping clips).

    Returns
    -------
    list[dict[str, Any]]
        Updates for each clip: `start_frame`, `end_frame`, `frame_stride` and `fps`.
        Empty if the row's clip is shorter than one output clip.

    Notes
    -----
    Raises `ValueError` (from its argument checks) if `clip_length` or `hop` is not a
    positive integer, or `fps` is not positive.
    """
    _check_positive_int("clip_length", clip_length)
    if hop is not None:
        _check_positive_int("hop", hop)
    video_fps = float(row["video_fps"])
    stride = _stride_for_fps(video_fps, fps) if fps is not None else int(row["frame_stride"])
    windows = _cut_windows(
        int(row["start_frame"]),
        int(row["end_frame"]),
        span=clip_length * stride,
        hop=(hop or clip_length) * stride,
        drop_last=drop_last,
    )
    return _clip_rows(windows, stride, video_fps)


@register_row_method()
def clip_to_num_seconds(
    row: dict[str, Any],
    clip_duration: float,
    fps: float | None = None,
    drop_last: bool = True,
    hop_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Cut a row's clip into consecutive clips of `clip_duration` seconds.

    Durations are converted to native frames with `round(seconds * video_fps)`, so
    videos with different frame rates give clips of the same length in time. The
    number of decoded frames per clip is `ceil(span / frame_stride)`, which can vary
    between videos of different frame rates.

    Parameters
    ----------
    row : dict[str, Any]
        Row carrying the clip columns.
    clip_duration : float
        Clip length in seconds.
    fps : float | None, optional
        Target sampling rate, as in `clip_to_num_frames`. If None, the row's current
        `frame_stride` is kept.
    drop_last : bool, optional
        As in `clip_to_num_frames`, by default True.
    hop_duration : float | None, optional
        Distance between consecutive clip starts, in seconds. Defaults to
        `clip_duration` (non-overlapping clips).

    Returns
    -------
    list[dict[str, Any]]
        Updates for each clip: `start_frame`, `end_frame`, `frame_stride` and `fps`.
        Empty if the row's clip is shorter than one output clip.

    Raises
    ------
    ValueError
        If `clip_duration` or `hop_duration` rounds to zero frames.

    Notes
    -----
    Also raises `ValueError` (from its argument checks) if `clip_duration`,
    `hop_duration` or `fps` is not positive.
    """
    _check_positive_number("clip_duration", clip_duration)
    if hop_duration is not None:
        _check_positive_number("hop_duration", hop_duration)
    video_fps = float(row["video_fps"])
    stride = _stride_for_fps(video_fps, fps) if fps is not None else int(row["frame_stride"])
    span = round(clip_duration * video_fps)
    hop = round((hop_duration or clip_duration) * video_fps)
    if span < 1 or hop < 1:
        raise ValueError(
            f"clip_duration={clip_duration} / hop_duration={hop_duration} is shorter "
            f"than one frame at {video_fps} fps"
        )
    windows = _cut_windows(int(row["start_frame"]), int(row["end_frame"]), span, hop, drop_last)
    return _clip_rows(windows, stride, video_fps)


class ClipToNumFramesConfig(BaseModel):
    type: Literal["clip_to_num_frames"]
    clip_length: int
    fps: float | None = None
    drop_last: bool = True
    hop: int | None = None


class ClipToNumFrames(ExpandByFunction):
    """Cut every clip into clips of `clip_length` sampled frames.

    An `ExpandByFunction` with the `clip_to_num_frames` row method; see it for the
    meaning of the parameters.

    Parameters
    ----------
    clip_length : int
        Number of sampled frames per clip.
    fps : float | None, optional
        Target sampling rate. If None, each row's current `frame_stride` is kept.
    drop_last : bool, optional
        Whether to discard frames left over after the last full clip, by default True.
    hop : int | None, optional
        Distance between consecutive clip starts, in sampled frames. Defaults to
        `clip_length`.

    Notes
    -----
    Construction raises `ValueError` (from its argument checks) if `clip_length` or
    `hop` is not a positive integer, or `fps` is not positive.
    """

    def __init__(
        self,
        *,
        clip_length: int,
        fps: float | None = None,
        drop_last: bool = True,
        hop: int | None = None,
    ) -> None:
        _check_positive_int("clip_length", clip_length)
        if hop is not None:
            _check_positive_int("hop", hop)
        if fps is not None:
            _check_positive_number("fps", fps)
        super().__init__(
            method=clip_to_num_frames,
            kwargs={"clip_length": clip_length, "fps": fps, "drop_last": drop_last, "hop": hop},
        )

    @classmethod
    def from_config(cls, cfg: ClipToNumFramesConfig) -> "ClipToNumFrames":
        return cls(**cfg.model_dump(exclude={"type"}))


class ClipToNumSecondsConfig(BaseModel):
    type: Literal["clip_to_num_seconds"]
    clip_duration: float
    fps: float | None = None
    drop_last: bool = True
    hop_duration: float | None = None


class ClipToNumSeconds(ExpandByFunction):
    """Cut every clip into clips of `clip_duration` seconds.

    An `ExpandByFunction` with the `clip_to_num_seconds` row method; see it for the
    meaning of the parameters.

    Parameters
    ----------
    clip_duration : float
        Clip length in seconds.
    fps : float | None, optional
        Target sampling rate. If None, each row's current `frame_stride` is kept.
    drop_last : bool, optional
        Whether to discard frames left over after the last full clip, by default True.
    hop_duration : float | None, optional
        Distance between consecutive clip starts, in seconds. Defaults to
        `clip_duration`.

    Notes
    -----
    Construction raises `ValueError` (from its argument checks) if `clip_duration`,
    `hop_duration` or `fps` is not positive.
    """

    def __init__(
        self,
        *,
        clip_duration: float,
        fps: float | None = None,
        drop_last: bool = True,
        hop_duration: float | None = None,
    ) -> None:
        _check_positive_number("clip_duration", clip_duration)
        if hop_duration is not None:
            _check_positive_number("hop_duration", hop_duration)
        if fps is not None:
            _check_positive_number("fps", fps)
        super().__init__(
            method=clip_to_num_seconds,
            kwargs={
                "clip_duration": clip_duration,
                "fps": fps,
                "drop_last": drop_last,
                "hop_duration": hop_duration,
            },
        )

    @classmethod
    def from_config(cls, cfg: ClipToNumSecondsConfig) -> "ClipToNumSeconds":
        return cls(**cfg.model_dump(exclude={"type"}))


class ResampleToFPSConfig(BaseModel):
    type: Literal["resample_to_fps"]
    fps: float


class ResampleToFPS:
    """Sample every clip at a target frame rate without changing its bounds.

    Sets each row's `frame_stride` to `round(video_fps / fps)` (at least 1) and its
    `fps` to the resulting effective rate, `video_fps / frame_stride`. A warning is
    logged when that misses `fps` by more than 1%.

    Parameters
    ----------
    fps : float
        Target sampling rate.

    Notes
    -----
    Construction raises `ValueError` (from its argument check) if `fps` is not
    positive.
    """

    def __init__(self, *, fps: float) -> None:
        _check_positive_number("fps", fps)
        self.fps = fps

    @classmethod
    def from_config(cls, cfg: ResampleToFPSConfig) -> "ResampleToFPS":
        return cls(**cfg.model_dump(exclude={"type"}))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Set `frame_stride` and `fps` on every row of `backend`.

        Parameters
        ----------
        backend : DataBackend
            The backend wrapping a table with the clip columns.

        Returns
        -------
        tuple[DataBackend, dict]
            The resampled backend (same type as input) and metadata with the number
            of rows (`n_rows`).
        """
        strides = [_stride_for_fps(float(row["video_fps"]), self.fps) for row in backend]
        if strides:
            video_fps = [float(v) for v in backend.unwrap["video_fps"]]
            backend = backend.add_column("frame_stride", strides)
            backend = backend.add_column(
                "fps", [v / s for v, s in zip(video_fps, strides, strict=True)]
            )
        return backend, {"n_rows": len(strides)}


register_transform(ClipToNumFramesConfig, ClipToNumFrames)
register_transform(ClipToNumSecondsConfig, ClipToNumSeconds)
register_transform(ResampleToFPSConfig, ResampleToFPS)
