"""Test suite for the MammAlps camera-trap video dataset."""

import hashlib
import random
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch

from alp_data import DatasetConfig
from alp_data.datasets import MammAlps

# Local sample: 2 s, 30 fps, 64x48 clip, see tests/test_read_video.py
SAMPLE_VIDEO = Path("tests/samples/video.mp4").resolve()
LOCAL_SPLIT = "local-test"
CSV_COLUMNS = [
    "split",
    "video_path",
    "segmap_path",
    "audio_path",
    "start_s",
    "end_s",
    "activity",
    "actions",
    "species",
    "num_frames",
    "duration_seconds",
    "average_fps",
    "width",
    "height",
]
CLIP_COLUMNS = ["start_frame", "end_frame", "frame_stride", "video_fps", "fps"]

# --- Dataset snapshot (GCS) ---
# To regenerate:
#   from alp_data.datasets import MammAlps
#   for s in MammAlps.info.split_paths: print(s, len(MammAlps(split=s)))
EXPECTED_LEN = {
    "benchmark-1-all": 6135,
    "benchmark-1-train": 4205,
    "benchmark-1-val": 686,
    "benchmark-1-test": 1244,
}

# To regenerate:
#   import hashlib
#   from alp_data.datasets import MammAlps
#   video = MammAlps(split="benchmark-1-train", backend="pandas")[0]["video"]
#   print(hashlib.sha256(video.contiguous().numpy().tobytes()).hexdigest())
EXPECTED_FIRST_ITEM_VIDEO_SHA256 = (
    "49698b08ae3c06a053050bd1b9c65fcf202fbc9fafa47d01d32533d348a48914"
)


def _write_split(path: Path, video_paths: list[str]) -> None:
    rows = [
        {
            "split": "train",
            "video_path": video_path,
            "segmap_path": "",
            "audio_path": "",
            "start_s": 0.0,
            "end_s": 2.0,
            "activity": "foraging",
            "actions": "grazing;none",
            "species": "red_deer",
            "num_frames": 60,
            "duration_seconds": 2.0,
            "average_fps": 30.0,
            "width": 64,
            "height": 48,
        }
        for video_path in video_paths
    ]
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False)


@pytest.fixture
def local_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a 3-row split whose clips all point at the local sample video.

    The default `data_root` is pointed at the sample's directory, so the split's
    relative `video_path`s resolve locally.

    Returns
    -------
    str
        The name of the registered split.
    """
    monkeypatch.setattr("alp_data.datasets.mammalps._ROOT", str(SAMPLE_VIDEO.parent))
    csv_path = tmp_path / "local.csv"
    _write_split(csv_path, [SAMPLE_VIDEO.name] * 3)
    monkeypatch.setitem(MammAlps.info.split_paths, LOCAL_SPLIT, str(csv_path))
    return LOCAL_SPLIT


# -- Local tests (no network) -------------------------------------------------


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_clip_columns_added(local_split: str, backend: str) -> None:
    """Test if loading adds clip columns covering each whole clip."""
    ds = MammAlps(split=local_split, backend=backend)
    assert set(CSV_COLUMNS + CLIP_COLUMNS) <= set(ds.columns)

    row = ds._data[0]
    assert row["start_frame"] == 0
    assert row["end_frame"] == 60
    assert row["frame_stride"] == 1
    assert row["video_fps"] == 30.0
    assert row["fps"] == 30.0


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_getitem(local_split: str, backend: str) -> None:
    ds = MammAlps(split=local_split, backend=backend)
    item = ds[0]
    assert isinstance(item["video"], torch.Tensor)
    assert item["video"].shape == (60, 3, 48, 64)
    assert item["video"].dtype == torch.uint8
    assert item["species"] == "red_deer"


def test_length_and_iteration(local_split: str) -> None:
    ds = MammAlps(split=local_split)
    assert len(ds) == 3
    items = list(ds)
    assert len(items) == 3
    assert all(item["video"].shape == (60, 3, 48, 64) for item in items)


def test_index_error_handling(local_split: str) -> None:
    ds = MammAlps(split=local_split)
    with pytest.raises(IndexError):
        ds[len(ds)]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_edited_clip_columns_are_respected(local_split: str, backend: str) -> None:
    """Test if edited clip columns select the decoded frames, as transforms do."""
    ds = MammAlps(split=local_split, backend=backend)
    full = ds[0]["video"]

    ds._data = ds._data.add_column("start_frame", [10, 0, 50])
    ds._data = ds._data.add_column("end_frame", [20, 60, 60])
    ds._data = ds._data.add_column("frame_stride", [1, 3, 5])

    torch.testing.assert_close(ds[0]["video"], full[10:20])
    torch.testing.assert_close(ds[1]["video"], full[0:60:3])
    torch.testing.assert_close(ds[2]["video"], full[50:60:5])


def test_output_take_and_give(local_split: str) -> None:
    ds = MammAlps(split=local_split, output_take_and_give={"video": "frames", "species": "label"})
    item = ds[0]
    assert set(item) == {"frames", "label"}
    assert item["frames"].shape == (60, 3, 48, 64)
    assert item["label"] == "red_deer"


def test_data_root_handling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test if video paths are resolved relative to a local mirror passed as `data_root`."""
    relative = "mammalps/v0.1.0/benchmark_1/clips/clip_c0.mp4"
    mirror = tmp_path / "mirror"
    (mirror / relative).parent.mkdir(parents=True)
    shutil.copy(SAMPLE_VIDEO, mirror / relative)

    csv_path = tmp_path / "remote.csv"
    _write_split(csv_path, [relative])
    monkeypatch.setitem(MammAlps.info.split_paths, LOCAL_SPLIT, str(csv_path))

    ds = MammAlps(split=LOCAL_SPLIT, data_root=str(mirror))
    assert ds[0]["video"].shape == (60, 3, 48, 64)


def test_invalid_split() -> None:
    with pytest.raises(LookupError, match="Invalid split"):
        MammAlps(split="nonexistent")


def test_load_from_config(local_split: str) -> None:
    cfg = DatasetConfig(dataset_name="mammalps", split=local_split, backend="pandas")
    ds, meta = MammAlps.from_config(cfg)
    assert isinstance(ds, MammAlps)
    assert meta == {}
    assert len(ds) == 3


def test_load_from_config_streaming_unsupported(local_split: str) -> None:
    cfg = DatasetConfig(dataset_name="mammalps", split=local_split, streaming=True)
    with pytest.raises(ValueError, match="does not support streaming"):
        MammAlps.from_config(cfg)


def test_str_representation(local_split: str) -> None:
    text = str(MammAlps(split=local_split))
    assert "mammalps (v0.1.0)" in text
    assert "Rows: 3" in text
    assert "Sources: https://zenodo.org/records/15040900, " in text


# -- GCS tests ----------------------------------------------------------------


@pytest.fixture(scope="module")
def ds_val() -> MammAlps:
    return MammAlps(split="benchmark-1-val", backend="polars")


def test_info_property() -> None:
    assert MammAlps.info.name == "mammalps"
    assert MammAlps.info.version == "0.1.0"
    assert set(EXPECTED_LEN) <= set(MammAlps.info.split_paths)


@pytest.mark.parametrize("split", list(EXPECTED_LEN))
def test_length(split: str) -> None:
    assert len(MammAlps(split=split)) == EXPECTED_LEN[split]


def test_columns_property(ds_val: MammAlps) -> None:
    assert set(CSV_COLUMNS + CLIP_COLUMNS) <= set(ds_val.columns)


def test_check_video(ds_val: MammAlps) -> None:
    """Test if sampled clips decode to `num_frames` frames of 256x256 RGB."""
    rng = random.Random(42)
    for idx in rng.sample(range(len(ds_val)), 3):
        item = ds_val[idx]
        assert item["video"].shape == (item["num_frames"], 3, 256, 256)
        assert item["video"].dtype == torch.uint8


def test_load_from_config_gcs() -> None:
    cfg = DatasetConfig(dataset_name="mammalps", split="benchmark-1-val")
    ds, _ = MammAlps.from_config(cfg)
    assert len(ds) == EXPECTED_LEN["benchmark-1-val"]


def test_reference_item_stability() -> None:
    """Check that a canonical item (index 0 of the train split) is bitwise-stable.

    We hash the raw uint8 frame buffer. This catches changes to the split's order,
    the clip columns' defaults, the decoder's frame selection, or the files on GCS.
    Decoded bytes can depend on the FFmpeg build; if this fails for a legitimate
    reason, regenerate `EXPECTED_FIRST_ITEM_VIDEO_SHA256` (see top of file).
    """
    ds = MammAlps(split="benchmark-1-train", backend="pandas")
    item = ds[0]

    # Defensive checks, so a hash failure is easier to interpret
    assert "video" in item, "[0] missing 'video' key"
    video = item["video"]
    assert isinstance(video, torch.Tensor), "[0] video is not a torch.Tensor"
    assert video.dtype == torch.uint8, f"[0] video dtype is {video.dtype}, expected uint8"
    assert video.shape == (item["num_frames"], 3, 256, 256)

    h = hashlib.sha256(video.contiguous().numpy().tobytes()).hexdigest()
    assert h == EXPECTED_FIRST_ITEM_VIDEO_SHA256, f"[0] video sha256 mismatch: {h}"
