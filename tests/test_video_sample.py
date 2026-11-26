import random
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import polars as pl
import pytest
from pydantic import BaseModel

import alp_data.dataset as alp_data_dataset
from alp_data import Dataset, DatasetConfig, DatasetInfo, dataset_from_config, register_dataset
from alp_data.backends import PandasBackend, PolarsBackend
from alp_data.backends.protocol import DataBackend
from alp_data.datasets import MammAlps
from alp_data.io import read_video_at_frames
from alp_data.transforms import (
    SplitClip,
    SplitClipConfig,
    SubsampleConfig,
    transform_from_config,
    validate_video_table,
)
from alp_data.transforms.expand import ExpandByFunction
from alp_data.transforms.video_sample import split_clip_by_duration, split_clip_by_frames

# Local sample: 2 s, 30 fps, 64x48 clip, see tests/test_read_video.py
SAMPLE_VIDEO = Path("tests/samples/video.mp4").resolve()
LOCAL_SPLIT = "local-video-sample-test"


def _row(
    start: int = 0,
    end: int = 10,
    stride: int = 1,
    native_fps: float = 30.0,
    sampled_fps: float | None = None,
) -> dict:
    return {
        "start_frame": start,
        "end_frame": end,
        "frame_stride": stride,
        "native_fps": native_fps,
        "sampled_fps": native_fps / stride if sampled_fps is None else sampled_fps,
    }


def _windows(updates: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(u["start_frame"], u["end_frame"]) for u in updates]


def _make_backend(backend_type: str, native_fps: list[float], strides: list[int]) -> DataBackend:
    n = len(native_fps)
    data = {
        "video_path": [f"video_{i}.mp4" for i in range(n)],
        "start_frame": [0] * n,
        "end_frame": [60] * n,
        "frame_stride": strides,
        "native_fps": native_fps,
        "sampled_fps": [f / s for f, s in zip(native_fps, strides, strict=True)],
    }
    if backend_type == "pandas":
        return PandasBackend(pd.DataFrame(data))
    return PolarsBackend(pl.DataFrame(data))


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


def test_split_clip_by_frames_row_stride() -> None:
    """Test if clips span `clip_length` frames sampled at the row's `frame_stride`."""
    updates = split_clip_by_frames(_row(end=30, stride=3), clip_length=4)
    assert _windows(updates) == [(0, 12), (12, 24)]


def test_split_clip_by_frames_only_updates_bounds() -> None:
    """Test if the row method leaves `frame_stride` and `sampled_fps` to the parent row."""
    updates = split_clip_by_frames(_row(end=20, stride=2), clip_length=5)
    assert all(u.keys() == {"start_frame", "end_frame"} for u in updates)


def test_split_clip_by_frames_ignores_row_fps() -> None:
    """Test if the stride comes from `frame_stride`, not from recomputing it from `sampled_fps`."""
    updates = split_clip_by_frames(_row(end=30, stride=1, sampled_fps=10), clip_length=4)
    assert _windows(updates) == [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28)]


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
def test_split_clip_invalid_frame_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SplitClip(**kwargs)


# -- split_clip_by_duration -----------------------------------------------------


def test_split_clip_by_duration_windows() -> None:
    updates = split_clip_by_duration(_row(end=60), clip_duration=0.5)
    assert _windows(updates) == [(0, 15), (15, 30), (30, 45), (45, 60)]
    updates = split_clip_by_duration(_row(end=30), clip_duration=0.5, hop_duration=0.2)
    assert _windows(updates) == [(0, 15), (6, 21), (12, 27)]
    updates = split_clip_by_duration(_row(end=30), clip_duration=0.4, drop_last=False)
    assert _windows(updates) == [(0, 12), (12, 24), (18, 30)]


def test_split_clip_by_duration_independent_of_stride() -> None:
    """Test if the row's `frame_stride` does not change the clip span."""
    updates = split_clip_by_duration(_row(end=30, stride=3), clip_duration=0.5)
    assert _windows(updates) == [(0, 15), (15, 30)]
    assert all(u.keys() == {"start_frame", "end_frame"} for u in updates)


def test_split_clip_by_duration_same_duration_across_fps() -> None:
    """Test if clips span the same time at different native frame rates."""
    at_30 = split_clip_by_duration(_row(end=60, native_fps=30.0), clip_duration=1.0)
    at_25 = split_clip_by_duration(_row(end=50, native_fps=25.0), clip_duration=1.0)
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
def test_split_clip_invalid_duration_arguments(kwargs: dict[str, Any]) -> None:
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
def test_split_clip_reads_stride_per_row(backend_type: str) -> None:
    """Test if each row is clipped with its own `frame_stride`, which is kept."""
    backend = _make_backend(backend_type, native_fps=[30.0, 25.0], strides=[6, 5])
    out, meta = SplitClip(clip_length=4)(backend)
    rows = list(out)
    assert meta == {"n_in": 2, "n_out": 5, "n_dropped": 0}
    assert {r["frame_stride"] for r in rows if r["native_fps"] == 30.0} == {6}
    assert {r["frame_stride"] for r in rows if r["native_fps"] == 25.0} == {5}
    assert all(r["sampled_fps"] == 5.0 for r in rows)


# -- validate_video_table -------------------------------------------------------


def _table(backend_type: str, **overrides: list[Any]) -> DataBackend:
    """Build a valid 2-row video table, with `overrides` replacing whole columns.

    Returns
    -------
    DataBackend
        The table, in a backend of type `backend_type`.
    """
    data: dict[str, list[Any]] = {
        "video_path": ["a.mp4", "b.mp4"],
        "start_frame": [0, 10],
        "end_frame": [60, 40],
        "frame_stride": [1, 2],
        "native_fps": [30.0, 30.0],
        "sampled_fps": [30.0, 15.0],
        **overrides,
    }
    if backend_type == "pandas":
        return PandasBackend(pd.DataFrame(data))
    return PolarsBackend(pl.DataFrame(data))


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_validate_video_table_valid(backend_type: str) -> None:
    validate_video_table(_table(backend_type))


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
@pytest.mark.parametrize(
    "column", ["start_frame", "end_frame", "frame_stride", "native_fps", "sampled_fps"]
)
def test_validate_video_table_missing_column(backend_type: str, column: str) -> None:
    backend = _table(backend_type).select_columns(
        [c for c in _table(backend_type).columns if c != column]
    )
    with pytest.raises(ValueError, match=f"missing columns \\['{column}'\\]"):
        validate_video_table(backend)


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
@pytest.mark.parametrize(
    "column, values",
    [
        ("start_frame", [0, None]),
        ("end_frame", [60, None]),
        ("frame_stride", [1, None]),
        ("native_fps", [30.0, None]),
        ("sampled_fps", [30.0, float("nan")]),
    ],
)
def test_validate_video_table_nulls(backend_type: str, column: str, values: list[Any]) -> None:
    """Test if nulls are caught, whether they arrive as `None` or NaN."""
    with pytest.raises(ValueError, match=f"row 1 has nulls in \\['{column}'\\]"):
        validate_video_table(_table(backend_type, **{column: values}))


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"native_fps": [30.0, 0.0]}, "native_fps must be > 0"),
        ({"native_fps": [30.0, -30.0]}, "native_fps must be > 0"),
        ({"frame_stride": [1, 0]}, "frame_stride must be >= 1"),
        ({"start_frame": [0, -1]}, "need 0 <= start_frame < end_frame"),
        ({"start_frame": [0, 40]}, "need 0 <= start_frame < end_frame"),
        ({"start_frame": [0, 50]}, "need 0 <= start_frame < end_frame"),
    ],
)
def test_validate_video_table_out_of_range(
    backend_type: str, overrides: dict[str, list[Any]], match: str
) -> None:
    with pytest.raises(ValueError, match=f"row 1: {match}"):
        validate_video_table(_table(backend_type, **overrides))


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_validate_video_table_rejects_non_video_table(backend_type: str) -> None:
    """Test if a table without `video_path` is rejected before its other columns."""
    backend = _table(backend_type).select_columns(["start_frame", "end_frame"])
    with pytest.raises(ValueError, match="can only be applied to video datasets"):
        validate_video_table(backend)
    with pytest.raises(ValueError, match="can only be applied to video datasets"):
        SplitClip(clip_length=4)(backend)


@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_split_clip_validates_table(backend_type: str) -> None:
    """Test if `SplitClip` rejects an invalid table before cutting any clips."""
    with pytest.raises(ValueError, match="missing columns \\['sampled_fps'\\]"):
        SplitClip(clip_length=4)(
            _table(backend_type).select_columns(
                ["video_path", "start_frame", "end_frame", "frame_stride", "native_fps"]
            )
        )
    with pytest.raises(ValueError, match="frame_stride must be >= 1"):
        SplitClip(clip_duration=0.5)(_table(backend_type, frame_stride=[1, 0]))


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
    results = [list(t(_make_backend(backend_type, [30.0, 25.0], [3, 2]))[0]) for t in transforms]
    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize(
    "config",
    [
        SplitClipConfig(type="split_clip", clip_length=10),
        SplitClipConfig(type="split_clip", clip_length=10, hop=5, drop_last=False),
        SplitClipConfig(type="split_clip", clip_duration=0.25),
        SplitClipConfig(type="split_clip", clip_duration=0.25, hop_duration=0.1),
    ],
)
@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_split_clip_then_subsample(config: SplitClipConfig, backend_type: str) -> None:
    """Test that sampling by label after `SplitClip` keeps the requested share of clips.

    Regression test: pandas used to give all clips of a video the parent's index label,
    so label-based sampling picked a label and got every clip of that video.
    """
    data = {
        "video": ["a", "b"],
        "video_path": ["a.mp4", "b.mp4"],
        "label": ["x", "x"],
        "start_frame": [0, 0],
        "end_frame": [40, 40],
        "frame_stride": [1, 1],
        "native_fps": [30.0, 30.0],
        "sampled_fps": [30.0, 30.0],
    }
    backend = (
        PandasBackend(pd.DataFrame(data))
        if backend_type == "pandas"
        else (PolarsBackend(pl.DataFrame(data)))
    )
    clips, _ = transform_from_config(config)(backend)
    n_clips = len(clips)

    subsampled, _ = transform_from_config(
        SubsampleConfig(type="subsample", property="label", ratios={"x": 0.25})
    )(clips)
    rows = list(subsampled)
    assert len(rows) == int(n_clips * 0.25)
    # Sampling without replacement must give distinct clips
    assert len({(r["video"], r["start_frame"]) for r in rows}) == len(rows)

    upsampled = clips.upsample_by_column("label", {"x": 2 * n_clips})
    assert len(upsampled) == 2 * n_clips
    # A target below the group size samples without replacement: clips must be distinct
    downsampled = list(clips.upsample_by_column("label", {"x": n_clips // 2}))
    assert len(downsampled) == n_clips // 2
    assert len({(r["video"], r["start_frame"]) for r in downsampled}) == len(downsampled)


# -- Registered dataset from YAML (local sample video) -------------------------

EXAMPLE_DATASET = "example_video_dataset"
EXAMPLE_SPLIT = "example_split"


@pytest.fixture
def example_video_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> type[Dataset]:
    """Register a minimal video dataset with one row pointing at the sample video.

    It is registered in a copy of the dataset registry, so it is gone after the test.

    Returns
    -------
    type[Dataset]
        The registered dataset class.
    """
    csv_path = tmp_path / f"{EXAMPLE_SPLIT}.csv"
    pd.DataFrame(
        [
            {
                "video_path": str(SAMPLE_VIDEO),
                "label": "red_deer",
                "start_frame": 0,
                "end_frame": 60,
                "native_fps": 30.0,
                "frame_stride": 1,
                "sampled_fps": 30.0,
            }
        ]
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(
        "alp_data.dataset._dataset_registry", dict(alp_data_dataset._dataset_registry)
    )

    @register_dataset
    class ExampleVideoDataset(Dataset):
        info = DatasetInfo(
            name=EXAMPLE_DATASET,
            owner="test",
            split_paths={EXAMPLE_SPLIT: str(csv_path)},
            version="0.1.0",
            description="One-row video dataset for tests.",
            sources=["tests/samples/video.mp4"],
            license="none",
        )

        def __init__(self, split: str = EXAMPLE_SPLIT, backend: str = "polars") -> None:
            super().__init__(backend=backend)
            self.split = split
            self._load()

        @property
        def available_splits(self) -> list[str]:
            return list(self.info.split_paths)

        @property
        def columns(self) -> list[str]:
            return list(self._data.columns)

        def _load(self) -> None:
            self._data = self._backend_class.from_csv(self.info.split_paths[self.split])

        @classmethod
        def from_config(cls, cfg: DatasetConfig) -> tuple["ExampleVideoDataset", dict[str, Any]]:
            ds = cls(split=cfg.split, backend=cfg.backend)
            if cfg.transformations:
                return ds, ds.apply_transformations(cfg.transformations)
            return ds, {}

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return self._process(self._data[idx])

        def __iter__(self) -> Iterator[dict[str, Any]]:
            for row in self._data:
                yield self._process(row)

        def __str__(self) -> str:
            return EXAMPLE_DATASET

        def _process(self, row: dict[str, Any]) -> dict[str, Any]:
            row["video"] = read_video_at_frames(
                row["video_path"],
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                stride=int(row["frame_stride"]),
            )
            return row

    return ExampleVideoDataset


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_from_yaml_with_transformations(
    example_video_dataset: type[Dataset], tmp_path: Path, backend: str
) -> None:
    """Test if a registered dataset loaded from YAML applies `split_clip`."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        f"""
dataset:
    dataset_name: {EXAMPLE_DATASET}
    split: {EXAMPLE_SPLIT}
    backend: {backend}
    transformations:
        - type: split_clip
          clip_duration: 0.1
"""
    )
    ds, metadata = dataset_from_config(yaml_path)

    # 0.1 s at 30 fps is 3 frames, so the 60-frame clip gives 20 clips
    assert isinstance(ds, example_video_dataset)
    assert metadata
    assert len(ds) == 20
    full = read_video_at_frames(str(SAMPLE_VIDEO), start_frame=0, end_frame=60)
    for i, item in enumerate(ds):
        assert (item["start_frame"], item["end_frame"]) == (3 * i, 3 * i + 3)
        assert item["label"] == "red_deer"
        assert item["video"].shape == (3, 3, 48, 64)
        assert (item["video"] == full[3 * i : 3 * i + 3]).all()


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
            # Columns added by `mammalps_frame_info.py`
            "start_frame": 0,
            "end_frame": 60,
            "native_fps": 30.0,
            "frame_stride": 1,
            "sampled_fps": 30.0,
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
