import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest
from pydantic import BaseModel

from alp_data import DatasetConfig
from alp_data.backends import PandasBackend, PolarsBackend
from alp_data.backends.protocol import DataBackend
from alp_data.datasets import MammAlps
from alp_data.transforms import SplitClip, SplitClipConfig, transform_from_config
from alp_data.transforms import video_sample as video_sample_module
from alp_data.transforms.expand import ExpandByFunction
from alp_data.transforms.video_sample import split_clip_by_duration, split_clip_by_frames

# Local sample: 2 s, 30 fps, 64x48 clip, see tests/test_read_video.py
SAMPLE_VIDEO = Path("tests/samples/video.mp4").resolve()
LOCAL_SPLIT = "local-video-sample-test"


def _row(
    start: int = 0,
    end: int = 10,
    stride: int = 1,
    video_fps: float = 30.0,
    fps: float | None = None,
) -> dict:
    return {
        "start_frame": start,
        "end_frame": end,
        "frame_stride": stride,
        "video_fps": video_fps,
        "fps": video_fps / stride if fps is None else fps,
    }


def _windows(updates: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(u["start_frame"], u["end_frame"]) for u in updates]


def _make_backend(backend_type: str, video_fps: list[float], fps: list[float]) -> DataBackend:
    n = len(video_fps)
    data = {
        "start_frame": [0] * n,
        "end_frame": [60] * n,
        "frame_stride": [1] * n,
        "video_fps": video_fps,
        "fps": fps,
    }
    if backend_type == "pandas":
        return PandasBackend(pd.DataFrame(data))
    return PolarsBackend(pl.DataFrame(data))


@pytest.fixture(autouse=True)
def reset_fps_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let every test see the once-per-pair fps warning afresh."""
    monkeypatch.setattr(video_sample_module, "_WARNED_FPS", set())


# -- split_clip_by_frames -------------------------------------------------------


def test_split_clip_by_frames_windows() -> None:
    assert _windows(split_clip_by_frames(_row(), clip_length=4)) == [(0, 4), (4, 8)]
    updates = split_clip_by_frames(_row(), clip_length=4, drop_last=False)
    assert _windows(updates) == [(0, 4), (4, 8), (6, 10)]
    updates = split_clip_by_frames(_row(), clip_length=4, hop=2)
    assert _windows(updates) == [(0, 4), (2, 6), (4, 8), (6, 10)]


def test_split_clip_by_frames_within_current_clip() -> None:
    """Test if clipping works inside the row's current bounds, so it can be chained."""
    updates = split_clip_by_frames(_row(start=5, end=15), clip_length=4)
    assert _windows(updates) == [(5, 9), (9, 13)]


def test_split_clip_by_frames_short_clip_dropped() -> None:
    assert split_clip_by_frames(_row(), clip_length=11) == []


def test_split_clip_by_frames_full_length() -> None:
    assert _windows(split_clip_by_frames(_row(), clip_length=10)) == [(0, 10)]


def test_split_clip_by_frames_row_fps_sets_stride() -> None:
    """Test if the row's `fps` picks the stride and spans `clip_length` sampled frames."""
    updates = split_clip_by_frames(_row(end=30, fps=10), clip_length=4)
    assert _windows(updates) == [(0, 12), (12, 24)]
    assert all(u["frame_stride"] == 3 and u["fps"] == 10 for u in updates)


def test_split_clip_by_frames_keeps_row_stride() -> None:
    """Test if a row whose `fps` matches its stride keeps that stride."""
    updates = split_clip_by_frames(_row(end=20, stride=2), clip_length=5)
    assert _windows(updates) == [(0, 10), (10, 20)]
    assert all(u["frame_stride"] == 2 and u["fps"] == 15 for u in updates)


def test_split_clip_by_frames_chaining_keeps_stride() -> None:
    """Test if re-clipping a clipped row derives the same stride from its effective fps."""
    first = split_clip_by_frames(_row(end=60, fps=7), clip_length=5)[0]
    second = split_clip_by_frames({**_row(end=60), **first}, clip_length=2)
    assert all(u["frame_stride"] == first["frame_stride"] == 4 for u in second)


def test_fps_not_dividing_video_fps_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="alp_data"):
        updates = split_clip_by_frames(_row(end=60, fps=7), clip_length=2)
        split_clip_by_frames(_row(end=60, fps=7), clip_length=2)
    assert updates[0]["frame_stride"] == 4
    assert updates[0]["fps"] == 7.5
    warnings = [r for r in caplog.records if "does not divide" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clip_length": 0},
        {"clip_length": -1},
        {"clip_length": 1.5},
        {"clip_length": True},
        {"clip_length": 4, "hop": 0},
    ],
)
def test_split_clip_by_frames_invalid_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        split_clip_by_frames(_row(), **kwargs)
    with pytest.raises(ValueError):
        SplitClip(**kwargs)


def test_split_clip_by_frames_invalid_row_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        split_clip_by_frames(_row(fps=0), clip_length=4)


# -- split_clip_by_duration -----------------------------------------------------


def test_split_clip_by_duration_windows() -> None:
    updates = split_clip_by_duration(_row(end=60), clip_duration=0.5)
    assert _windows(updates) == [(0, 15), (15, 30), (30, 45), (45, 60)]
    updates = split_clip_by_duration(_row(end=30), clip_duration=0.5, hop_duration=0.2)
    assert _windows(updates) == [(0, 15), (6, 21), (12, 27)]
    updates = split_clip_by_duration(_row(end=30), clip_duration=0.4, drop_last=False)
    assert _windows(updates) == [(0, 12), (12, 24), (18, 30)]


def test_split_clip_by_duration_row_fps_sets_stride() -> None:
    """Test if the row's `fps` sets the stride without changing the clip span."""
    updates = split_clip_by_duration(_row(end=30, fps=10), clip_duration=0.5)
    assert _windows(updates) == [(0, 15), (15, 30)]
    assert all(u["frame_stride"] == 3 and u["fps"] == 10 for u in updates)


def test_split_clip_by_duration_same_duration_across_fps() -> None:
    """Test if clips span the same time at different native frame rates."""
    at_30 = split_clip_by_duration(_row(end=60, video_fps=30.0), clip_duration=1.0)
    at_25 = split_clip_by_duration(_row(end=50, video_fps=25.0), clip_duration=1.0)
    assert _windows(at_30) == [(0, 30), (30, 60)]
    assert _windows(at_25) == [(0, 25), (25, 50)]


def test_split_clip_by_duration_shorter_than_one_frame() -> None:
    with pytest.raises(ValueError, match="shorter than one frame"):
        split_clip_by_duration(_row(), clip_duration=0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clip_duration": 0},
        {"clip_duration": -1.0},
        {"clip_duration": 1.0, "hop_duration": 0},
    ],
)
def test_split_clip_by_duration_invalid_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        split_clip_by_duration(_row(), **kwargs)
    with pytest.raises(ValueError):
        SplitClip(**kwargs)


# -- SplitClip ------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "Exactly one"),
        ({"clip_length": 4, "clip_duration": 0.5}, "Exactly one"),
        ({"clip_length": 0, "clip_duration": 0.5}, "Exactly one"),
        ({"clip_length": 4, "hop_duration": 0.5}, "hop_duration"),
        ({"clip_duration": 0.5, "hop": 2}, "hop"),
    ],
)
def test_split_clip_invalid_combinations(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SplitClip(**kwargs)


@pytest.mark.parametrize(
    "kwargs, method, expected",
    [
        ({"clip_length": 4, "hop": 2}, split_clip_by_frames, {"clip_length": 4, "hop": 2}),
        (
            {"clip_duration": 0.5, "hop_duration": 0.2},
            split_clip_by_duration,
            {"clip_duration": 0.5, "hop_duration": 0.2},
        ),
    ],
)
def test_split_clip_dispatch(
    kwargs: dict[str, Any],
    method: object,
    expected: dict[str, Any],
) -> None:
    transform = SplitClip(**kwargs)
    assert isinstance(transform, ExpandByFunction)
    assert transform.method is method
    assert transform.kwargs == {**expected, "drop_last": True}


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_split_clip_reads_fps_per_row(backend_type: str) -> None:
    """Test if each row's own `fps` picks its stride."""
    backend = _make_backend(backend_type, video_fps=[30.0, 25.0], fps=[5.0, 5.0])
    out, meta = SplitClip(clip_length=4)(backend)
    rows = list(out)
    assert meta == {"n_in": 2, "n_out": 5, "n_dropped": 0}
    assert {r["frame_stride"] for r in rows if r["video_fps"] == 30.0} == {6}
    assert {r["frame_stride"] for r in rows if r["video_fps"] == 25.0} == {5}
    assert all(r["fps"] == 5.0 for r in rows)


# -- Config wiring --------------------------------------------------------------


def test_configs_validate_through_dataset_config() -> None:
    cfg = DatasetConfig(
        dataset_name="mammalps",
        split="benchmark-1-val",
        frames_per_second=10,
        transformations=[
            {"type": "split_clip", "clip_length": 16},
            {"type": "split_clip", "clip_duration": 0.5, "hop_duration": 0.25},
        ],
    )
    assert [type(t) for t in cfg.transformations] == [SplitClipConfig, SplitClipConfig]


@pytest.mark.parametrize(
    "config",
    [
        SplitClipConfig(type="split_clip", clip_length=8),
        SplitClipConfig(type="split_clip", clip_length=8, hop=4, drop_last=False),
        SplitClipConfig(type="split_clip", clip_duration=0.5),
        SplitClipConfig(type="split_clip", clip_duration=0.5, hop_duration=0.25),
    ],
)
@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_transform_methods_equivalence(config: BaseModel, backend_type: str) -> None:
    """
    Test that transform_from_config, manual, and from_config produce the same result.
    """
    transforms = [
        SplitClip(**config.model_dump(exclude={"type"})),
        SplitClip.from_config(config),
        transform_from_config(config),
    ]
    results = [
        list(t(_make_backend(backend_type, [30.0, 25.0], [10.0, 10.0]))[0]) for t in transforms
    ]
    assert results[0] == results[1] == results[2]


# -- MammAlps integration (local sample video) ----------------------------------


@pytest.fixture
def local_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a 2-row MammAlps split whose clips point at the local sample video.

    The default `data_root` is pointed at the sample's directory, so the split's
    relative `video_path`s resolve locally.

    Returns
    -------
    str
        The name of the registered split.
    """
    rows = [
        {
            "split": "train",
            "video_path": SAMPLE_VIDEO.name,
            "segmap_path": "",
            "audio_path": "",
            "start_s": 0.0,
            "end_s": 2.0,
            "activity": "foraging",
            "actions": "grazing;none",
            "species": species,
            "num_frames": 60,
            "duration_seconds": 2.0,
            "average_fps": 30.0,
            "width": 64,
            "height": 48,
        }
        for species in ["red_deer", "chamois"]
    ]
    monkeypatch.setattr("alp_data.datasets.mammalps._ROOT", str(SAMPLE_VIDEO.parent))
    csv_path = tmp_path / "local.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    monkeypatch.setitem(MammAlps.info.split_paths, LOCAL_SPLIT, str(csv_path))
    return LOCAL_SPLIT


def _load(
    split: str,
    transformations: list[dict[str, Any]] | None,
    backend: str,
    frames_per_second: float | None = None,
) -> MammAlps:
    cfg = DatasetConfig(
        dataset_name="mammalps",
        split=split,
        backend=backend,
        frames_per_second=frames_per_second,
        transformations=transformations,
    )
    ds, _ = MammAlps.from_config(cfg)
    return ds


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize(
    "frames_per_second, transformations, n_rows, n_frames",
    [
        # stride 3, one 48-frame window per 60-frame clip
        (10, [{"type": "split_clip", "clip_length": 16}], 2, 16),
        # drop_last=False adds a window ending on the last frame
        (10, [{"type": "split_clip", "clip_length": 16, "drop_last": False}], 4, 16),
        # four 15-frame windows per clip
        (None, [{"type": "split_clip", "clip_duration": 0.5}], 8, 15),
        # 15-frame windows sampled at stride 3
        (10, [{"type": "split_clip", "clip_duration": 0.5}], 8, 5),
        # no transform: the whole clip at stride 3
        (10, None, 2, 20),
        # the clip keeps the dataset's stride
        (10, [{"type": "split_clip", "clip_length": 5}], 8, 5),
    ],
)
def test_mammalps_transformed_video_lengths(
    local_split: str,
    backend: str,
    frames_per_second: float | None,
    transformations: list[dict[str, Any]] | None,
    n_rows: int,
    n_frames: int,
) -> None:
    """Test if transformed MammAlps items decode videos of the expected length."""
    ds = _load(local_split, transformations, backend, frames_per_second)
    assert len(ds) == n_rows
    for item in ds:
        assert item["video"].shape == (n_frames, 3, 48, 64)
    assert {item["species"] for item in ds} == {"red_deer", "chamois"}


def test_mammalps_clips_decode_the_right_frames(local_split: str) -> None:
    full = MammAlps(split=local_split)[0]["video"]
    ds = _load(local_split, [{"type": "split_clip", "clip_length": 4}], "polars", 10)
    first, second = ds[0], ds[1]
    assert (first["start_frame"], first["end_frame"]) == (0, 12)
    assert (second["start_frame"], second["end_frame"]) == (12, 24)
    assert (first["video"] == full[0:12:3]).all()
    assert (second["video"] == full[12:24:3]).all()


def test_mammalps_all_clips_too_short(local_split: str) -> None:
    ds = _load(local_split, [{"type": "split_clip", "clip_length": 61}], "polars")
    assert len(ds) == 0
    assert list(ds) == []


# -- MammAlps integration (GCS) -------------------------------------------------


def test_mammalps_gcs_split_clip() -> None:
    ds = _load("benchmark-1-val", [{"type": "split_clip", "clip_length": 16}], "polars", 5)
    assert len(ds) > 0
    rng = random.Random(42)
    for idx in rng.sample(range(len(ds)), 3):
        item = ds[idx]
        assert item["frame_stride"] == 6
        assert item["video"].shape == (16, 3, 256, 256)
