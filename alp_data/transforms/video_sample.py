"""Transforms that expand rows corresponding to whole videos into rows corresponding
to individual clips.

Video datasets contain the information:

- `start_frame`, `end_frame`: frame indices, by default referring to native video start/end.
- `frame_stride`: number of native frames to step between the frames to decode.
- `native_fps`: native frame rate of the video (read only).
- `sampled_fps`: frame rate of decoded video, `native_fps / frame_stride`.

The dataset sets `frame_stride` and `sampled_fps` from its `frames_per_second` argument (by
default stride 1 and `sampled_fps = native_fps`). The clipping row methods use the row's
`frame_stride` as is and only change `start_frame` and `end_frame`.

`validate_video_table` checks that a table has these columns with valid values. Video
datasets call it when loading, and `SplitClip` before cutting clips.

`SplitClip` cuts each clip into clips of either a fixed number of sampled frames
(`clip_length`) or a fixed duration (`clip_duration`). It is a wrapper on
`ExpandByFunction` with the row method `split_clip_by_frames` or `split_clip_by_duration`,
which are also registered under those names for use in `expand_by_function` configs.

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

import math
from typing import Any, Literal

from pydantic import BaseModel

from alp_data.backends.protocol import DataBackend

from . import register_transform
from .expand import ExpandByFunction, register_row_method

VIDEO_COLUMNS = ("start_frame", "end_frame", "frame_stride", "native_fps", "sampled_fps")


def _is_null(value: Any) -> bool:  # noqa: ANN401
    return value is None or (isinstance(value, float) and math.isnan(value))


def validate_video_table(backend: DataBackend) -> None:
    """Check that `backend` is a valid table of video clips ready for any
    video sampling transform.

    First, check that a `video_path` column exists. The table must also have the `VIDEO_COLUMNS`
    (`start_frame`, `end_frame`, `frame_stride`, `native_fps` and `sampled_fps`) with no null
    values, and every row must have `native_fps > 0`, `frame_stride >= 1` and
    `0 <= start_frame < end_frame`.

    Parameters
    ----------
    backend : DataBackend
        The table to check.

    Raises
    ------
    ValueError
        If the table has no `video_path` column, if one of `VIDEO_COLUMNS` is
        missing, or if a row has a null in one of them or values out of range. The
        message names the first offending row.
    """
    if not backend.column_exists("video_path"):
        raise ValueError(
            "Table has no `video_path` column; video clip transforms "
            "can only be applied to video datasets."
        )
    missing = [c for c in VIDEO_COLUMNS if not backend.column_exists(c)]
    if missing:
        raise ValueError(f"Video table is missing columns {missing}")
    for i, row in enumerate(backend):
        nulls = [c for c in VIDEO_COLUMNS if _is_null(row[c])]
        if nulls:
            raise ValueError(f"Video table row {i} has nulls in {nulls}")
        if not row["native_fps"] > 0:
            raise ValueError(
                f"Video table row {i}: native_fps must be > 0, got {row['native_fps']!r}"
            )
        if not row["frame_stride"] >= 1:
            raise ValueError(
                f"Video table row {i}: frame_stride must be >= 1, got {row['frame_stride']!r}"
            )
        if not 0 <= row["start_frame"] < row["end_frame"]:
            raise ValueError(
                f"Video table row {i}: need 0 <= start_frame < end_frame, got "
                f"start_frame={row['start_frame']!r}, end_frame={row['end_frame']!r}"
            )


def _check_positive_int(name: str, value: Any) -> None:  # noqa: ANN401
    """Check that `value`, the argument called `name`, is an integer >= 1.

    Raises
    ------
    ValueError
        If `value` is not an integer >= 1 (booleans are rejected).
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


def _check_positive_number(name: str, value: Any) -> None:  # noqa: ANN401
    """Check that `value`, the argument called `name`, is a number > 0.

    Raises
    ------
    ValueError
        If `value` is not an int or float > 0 (booleans are rejected).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a number > 0, got {value!r}")


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


def _clip_rows(windows: list[tuple[int, int]]) -> list[dict[str, Any]]:
    return [{"start_frame": s, "end_frame": e} for s, e in windows]


@register_row_method
def split_clip_by_frames(
    row: dict[str, Any],
    clip_length: int,
    drop_last: bool = True,
    hop: int | None = None,
) -> list[dict[str, Any]]:
    """Cut a row's clip into consecutive clips of `clip_length` sampled frames.

    Works within the row's current `[start_frame, end_frame)`, so it can be chained
    after other clipping transforms. Frames are sampled with a stride equal to the
    row's `frame_stride`, so each output clip spans `clip_length * frame_stride`
    native frames, and `clip_length` frames are decoded in the final clip.

    Set `clip_length=1` to turn a video dataset into an image dataset.

    Parameters
    ----------
    row : dict[str, Any]
        Row carrying the clip columns.
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
        Updated frame bounds for each clip: the new, calculated `start_frame` and `end_frame`.
        Empty (row dropped) if the row's existing clip is shorter than one requested output clip.

    Notes
    -----
    Neither the arguments nor the row's values are checked here; `SplitClip` checks
    the arguments when it is created and the table with `validate_video_table`.
    """
    stride = int(row["frame_stride"])
    windows = _cut_windows(
        int(row["start_frame"]),
        int(row["end_frame"]),
        span=clip_length * stride,
        hop=(hop or clip_length) * stride,
        drop_last=drop_last,
    )
    return _clip_rows(windows)


@register_row_method
def split_clip_by_duration(
    row: dict[str, Any],
    clip_duration: float,
    drop_last: bool = True,
    hop_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Cut a row's clip into consecutive clips of `clip_duration` seconds.

    Durations are converted to native frames with `round(seconds * native_fps)`, so
    videos with different frame rates give clips of the same length in time. The
    number of decoded frames per clip is `ceil(span / frame_stride)`, which can vary
    between videos of different frame rates.

    Parameters
    ----------
    row : dict[str, Any]
        Row carrying the clip columns.
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
        Updated frame bounds for each clip: the new, calculated `start_frame` and `end_frame`.
        Empty (row dropped) if the row's existing clip is shorter than one requested output clip.

    Raises
    ------
    ValueError
        If `clip_duration` or `hop_duration` rounds to zero frames.

    Notes
    -----
    `clip_duration` and `hop_duration` are not checked to be positive here; `SplitClip`
    checks them when it is created.
    """
    native_fps = float(row["native_fps"])
    span = round(clip_duration * native_fps)
    hop = round((hop_duration or clip_duration) * native_fps)
    if span < 1 or hop < 1:
        raise ValueError(
            f"clip_duration={clip_duration} / hop_duration={hop_duration} is shorter "
            f"than one frame at {native_fps} fps"
        )
    windows = _cut_windows(int(row["start_frame"]), int(row["end_frame"]), span, hop, drop_last)
    return _clip_rows(windows)


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
    is given) or the `split_clip_by_duration` row method (when `clip_duration` is given).
    Frames are sampled with each row's `frame_stride`, which the dataset sets from its
    `frames_per_second` argument.

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

    Notes
    _____
    - Only the start_frame and end_frame columns change per clip. Labels, duration, and path columns
    are inherited for each clip from the whole video.
    - If SplitClip is chained with a subsequent sampling transform, the subsequent transform
    reads the backend *after* each row is cut into clips, so longer videos contribute more clips
    as input to resampling.
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

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Validate `backend` with `validate_video_table`, then cut its clips.

        Parameters
        ----------
        backend : DataBackend
            The table of video clips to cut.

        Returns
        -------
        tuple[DataBackend, dict]
            As in `ExpandByFunction.__call__`.

        Raises
        ------
        ValueError
            If `backend` fails `validate_video_table` or, with `clip_duration`, if
            `clip_duration` or `hop_duration` is shorter than one frame of some row's
            `native_fps`.
        """  # noqa: DOC502 - raised by `validate_video_table` and the row methods
        validate_video_table(backend)
        return super().__call__(backend)

    @classmethod
    def from_config(cls, cfg: SplitClipConfig) -> "SplitClip":
        return cls(**cfg.model_dump(exclude={"type"}))


register_transform(SplitClipConfig, SplitClip)
