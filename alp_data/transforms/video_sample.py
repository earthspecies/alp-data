"""Transforms that expand rows corresponding to whole videos into rows corresponding
to individual clips.

Video datasets contain the information:

- `start_frame`, `end_frame`: frame indices, by default referring to native video start/end.
- `frame_stride`: number of native frames to step between the frames to decode.
- `video_fps`: native frame rate of the video (read only).
- `fps`: sampling rate. The dataset sets it from its `frames_per_second` argument
  (defaulting to `video_fps`). The clipping row methods read it to pick the stride and
  write back the effective rate `video_fps / frame_stride`.

`SplitClip` cuts each clip into clips of either a fixed number of sampled frames
(`clip_length`) or a fixed duration (`clip_duration`). It is a wrapper on
`ExpandByFunction` with the row method `split_clip_by_frames` or `split_clip_by_duration`.

Examples
--------
>>> from alp_data import DatasetConfig
>>> cfg = DatasetConfig(
...     dataset_name="mammalps",
...     split="benchmark-1-val",
...     frames_per_second=5,
...     transformations=[
...         {"type": "split_clip", "clip_length": 16},
...     ],
... )

Or in Python, applied to a dataset's backend:

>>> from alp_data.transforms import SplitClip
>>> transform = SplitClip(clip_duration=2.0)
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel

from . import register_transform
from .expand import ExpandByFunction

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


def split_clip_by_frames(
    row: dict[str, Any],
    clip_length: int,
    drop_last: bool = True,
    hop: int | None = None,
) -> list[dict[str, Any]]:
    """Cut a row's clip into consecutive clips of `clip_length` sampled frames.

    Works within the row's current `[start_frame, end_frame)`, so it can be chained
    after other clipping transforms. The stride is `round(video_fps / fps)` from the
    row's `video_fps` and `fps` columns; a warning is logged when the resulting rate
    misses `fps` by more than 1%. Each output clip decodes exactly `clip_length`
    frames. Set `clip_length=1` to turn a video dataset into an image dataset.

    Parameters
    ----------
    row : dict[str, Any]
        Row carrying the clip columns, including `video_fps` and `fps`.
    clip_length : int
        Number of sampled frames per clip.
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
        Updates for each clip: `start_frame`, `end_frame`, `frame_stride` and `fps`
        (the effective rate `video_fps / frame_stride`). Empty if the row's clip is
        shorter than one output clip.

    Notes
    -----
    Raises `ValueError` (from its argument checks) if `clip_length` or `hop` is not a
    positive integer, or the row's `fps` is not positive.
    """
    _check_positive_int("clip_length", clip_length)
    if hop is not None:
        _check_positive_int("hop", hop)
    video_fps = float(row["video_fps"])
    stride = _stride_for_fps(video_fps, float(row["fps"]))
    windows = _cut_windows(
        int(row["start_frame"]),
        int(row["end_frame"]),
        span=clip_length * stride,
        hop=(hop or clip_length) * stride,
        drop_last=drop_last,
    )
    return _clip_rows(windows, stride, video_fps)


def split_clip_by_duration(
    row: dict[str, Any],
    clip_duration: float,
    drop_last: bool = True,
    hop_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Cut a row's clip into consecutive clips of `clip_duration` seconds.

    Durations are converted to native frames with `round(seconds * video_fps)`, so
    videos with different frame rates give clips of the same length in time. The
    number of decoded frames per clip is `ceil(span / frame_stride)`, which can vary
    between videos of different frame rates. The stride is picked from the row's `fps`
    as in `split_clip_by_frames`.

    Parameters
    ----------
    row : dict[str, Any]
        Row carrying the clip columns, including `video_fps` and `fps`.
    clip_duration : float
        Clip length in seconds.
    drop_last : bool, optional
        As in `split_clip_by_frames`, by default True.
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
    `hop_duration` or the row's `fps` is not positive.
    """
    _check_positive_number("clip_duration", clip_duration)
    if hop_duration is not None:
        _check_positive_number("hop_duration", hop_duration)
    video_fps = float(row["video_fps"])
    stride = _stride_for_fps(video_fps, float(row["fps"]))
    span = round(clip_duration * video_fps)
    hop = round((hop_duration or clip_duration) * video_fps)
    if span < 1 or hop < 1:
        raise ValueError(
            f"clip_duration={clip_duration} / hop_duration={hop_duration} is shorter "
            f"than one frame at {video_fps} fps"
        )
    windows = _cut_windows(int(row["start_frame"]), int(row["end_frame"]), span, hop, drop_last)
    return _clip_rows(windows, stride, video_fps)


class SplitClipConfig(BaseModel):
    type: Literal["split_clip"]
    clip_length: int | None = None
    clip_duration: float | None = None
    drop_last: bool = True
    hop: int | None = None
    hop_duration: float | None = None


class SplitClip(ExpandByFunction):
    """Cut every clip into clips of `clip_length` sampled frames or `clip_duration` seconds.

    An `ExpandByFunction` with the `split_clip_by_frames` row method (when `clip_length`
    is given) or the `split_clip_by_duration` row method (when `clip_duration` is given);
    see them for the meaning of the parameters. The sampling rate is read from each
    row's `fps` column, which the dataset sets from its `frames_per_second` argument.

    Parameters
    ----------
    clip_length : int | None, optional
        Number of sampled frames per clip. Exactly one of `clip_length` and
        `clip_duration` must be given.
    clip_duration : float | None, optional
        Clip length in seconds. Exactly one of `clip_length` and `clip_duration` must
        be given.
    drop_last : bool, optional
        Whether to discard frames left over after the last full clip, by default True.
    hop : int | None, optional
        Distance between consecutive clip starts, in sampled frames. Only valid with
        `clip_length`, which it defaults to.
    hop_duration : float | None, optional
        Distance between consecutive clip starts, in seconds. Only valid with
        `clip_duration`, which it defaults to.

    Raises
    ------
    ValueError
        If not exactly one of `clip_length` and `clip_duration` is given, if `hop` is
        given with `clip_duration` or `hop_duration` with `clip_length`, if
        `clip_length` or `hop` is not a positive integer, or if `clip_duration` or
        `hop_duration` is not a positive number.
    """

    def __init__(
        self,
        *,
        clip_length: int | None = None,
        clip_duration: float | None = None,
        drop_last: bool = True,
        hop: int | None = None,
        hop_duration: float | None = None,
    ) -> None:
        if (clip_length is None) == (clip_duration is None):
            raise ValueError(
                "Exactly one of clip_length and clip_duration must be given, got "
                f"clip_length={clip_length!r}, clip_duration={clip_duration!r}"
            )
        if clip_length is not None:
            if hop_duration is not None:
                raise ValueError("hop_duration can only be used with clip_duration; use hop")
            _check_positive_int("clip_length", clip_length)
            if hop is not None:
                _check_positive_int("hop", hop)
            super().__init__(
                method=split_clip_by_frames,
                kwargs={"clip_length": clip_length, "drop_last": drop_last, "hop": hop},
            )
        else:
            if hop is not None:
                raise ValueError("hop can only be used with clip_length; use hop_duration")
            _check_positive_number("clip_duration", clip_duration)
            if hop_duration is not None:
                _check_positive_number("hop_duration", hop_duration)
            super().__init__(
                method=split_clip_by_duration,
                kwargs={
                    "clip_duration": clip_duration,
                    "drop_last": drop_last,
                    "hop_duration": hop_duration,
                },
            )

    @classmethod
    def from_config(cls, cfg: SplitClipConfig) -> "SplitClip":
        return cls(**cfg.model_dump(exclude={"type"}))


register_transform(SplitClipConfig, SplitClip)
