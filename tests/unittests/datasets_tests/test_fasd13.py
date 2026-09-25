"""
Unit tests for FASD13 dataset.

Run with:
    pytest -q test_fasd13.py

Note on split choice: FASD13 contains 8-hour recordings (HG is 72 h across 9
files), so a single item there is multiple GB of float32 once decoded. Every
test that actually reads audio is therefore pinned to `AS`, the smallest
sub-dataset (12 files, 0.20 h total), at fixed indices.
"""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import pytest

from alp_data import DatasetConfig
from alp_data.datasets import FASD13
from alp_data.datasets.fasd13 import SUBDATASET_CODES
from alp_data.utils import create_hash


EXPECTED_LEN_ALL = 109
EXPECTED_LEN_AS = 12
EXPECTED_FIRST_AS_AUDIO_SHA256 = (
    "77aadf4a911c9b58073a4fe13d98a5075db3a18241ed6d0345886aef92816fe6"
)
AS_ANNOTATIONS_SHA256 = "d7005afd70b68e5b43578cc0fb4a16049f689156bb6924105c448b42efbd01db"

SELECTION_TABLE_COLUMNS = [
    "Selection",
    "Begin Time (s)",
    "End Time (s)",
    "Q",
    "event_index",
]
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds_all() -> FASD13:
    """Load the full FASD13 manifest (metadata only -- do not read audio here)."""
    return FASD13(split="all", backend="pandas")


@pytest.fixture(scope="module")
def ds() -> FASD13:
    """Load the AS sub-dataset, the smallest split, for audio-level testing."""
    return FASD13(split="AS", sample_rate=32000, backend="pandas")


def test_ds_not_empty(ds_all: FASD13, ds: FASD13):
    """Both the full benchmark and a sub-dataset should have the published sizes."""
    assert len(ds_all) == EXPECTED_LEN_ALL
    assert len(ds) == EXPECTED_LEN_AS


def test_available_splits(ds_all: FASD13):
    """`all` plus the 13 sub-dataset codes should be available."""
    assert set(ds_all.available_splits) == {"all", *SUBDATASET_CODES}


def test_invalid_split_raises():
    """An unknown split should fail loudly at load time."""
    with pytest.raises(LookupError):
        FASD13(split="not_a_split")


def test_subdataset_split_is_consistent(ds: FASD13):
    """A sub-dataset split should only contain rows from that sub-dataset."""
    assert set(ds._data.unwrap["subdataset"]) == {"AS"}


def test_get_available_labels(ds: FASD13):
    """The class is nameless, so `Q` is the annotation column: statuses, not names."""
    labels = ds.get_available_labels()
    assert isinstance(labels, list), "get_available_labels should return a list"
    assert labels == ["POS"], f"AS has no UNK events, so expected ['POS'], got {labels}"
    assert FASD13(split="MS", backend="pandas").get_available_labels() == ["POS", "UNK"]


def test_check_audio(ds: FASD13):
    """Basic audio integrity checks on a few deterministic items."""
    for idx in (0, 5, 11):
        item = ds[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert audio.dtype == np.float32, f"[{idx}] audio dtype is {audio.dtype}, expected float32"
        assert audio.ndim == 1, f"[{idx}] audio is not mono, shape={audio.shape}"
        assert audio.size >= 10, f"[{idx}] audio too short (size={audio.size})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"
        assert item["sample_rate"] == 32000, f"[{idx}] unexpected sample rate"


def test_presampled_columns_exist(ds: FASD13):
    """Pre-resampled path columns should be present in the loaded data."""
    assert "16khz_path" in ds.columns
    assert "32khz_path" in ds.columns
    assert ds.available_sample_rates == [16000, 32000]


def test_load_presampled_16khz():
    """Loading at 16 kHz should read the 16 kHz mirror, not resample the 32 kHz one."""
    ds_16 = FASD13(split="AS", sample_rate=16000, backend="pandas")
    item_16 = ds_16[0]
    audio_16 = item_16["audio"]

    assert isinstance(audio_16, np.ndarray)
    assert audio_16.dtype == np.float32
    assert item_16["sample_rate"] == 16000

    # Same recording, half the rate: exactly half the samples.
    ds_32 = FASD13(split="AS", sample_rate=32000, backend="pandas")
    assert ds_32[0]["audio"].size == 2 * audio_16.size


def test_audio_fp_points_at_the_originals(ds_all: FASD13):
    """`audio_fp` must be the native recording, not one of the mirrors."""
    rows = ds_all._data.unwrap
    assert rows["audio_fp"].str.startswith("audio_native/").all()
    assert rows["16khz_path"].str.startswith("audio_16k/").all()
    assert rows["32khz_path"].str.startswith("audio_32k/").all()
    # All three views must name the same recording.
    stems = rows["audio_fp"].str.replace("audio_native/", "", regex=False)
    assert stems.equals(rows["32khz_path"].str.replace("audio_32k/", "", regex=False))


def test_native_read_returns_native_rate():
    """`sample_rate=None` should return the recording at its own rate."""
    ds_native = FASD13(split="AS", sample_rate=None, backend="pandas")
    row = ds_native._data[0]
    item = ds_native[0]
    assert item["sample_rate"] == int(row["native_sample_rate"]) == 22050
    assert item["audio"].dtype == np.float32
    assert abs(item["audio"].size / item["sample_rate"] - float(row["audio_duration"])) < 0.05


def test_ha_flac_in_wav_originals_are_readable():
    """
    Every HA original must decode in full.

    HA ships FLAC bitstreams under a .wav extension, and two of them hold
    frames libsndfile rejects. Those two are published transcoded to plain PCM,
    bit-identical to the 32 kHz mirror. If the transcode is ever lost, these
    reads raise LibsndfileError again.
    """
    ds_native = FASD13(split="HA", sample_rate=None, backend="pandas")
    for idx in (7, 11):
        row = ds_native._data[idx]
        item = ds_native[idx]
        assert item["sample_rate"] == 32000
        assert abs(item["audio"].size / 32000 - float(row["audio_duration"])) < 0.05
        assert not np.all(item["audio"] == 0)


def test_backends_agree(ds: FASD13):
    """The polars and pandas backends should return identical audio."""
    ds_polars = FASD13(split="AS", sample_rate=32000, backend="polars")
    assert len(ds_polars) == len(ds)
    assert np.array_equal(ds_polars[0]["audio"], ds[0]["audio"])


def test_streaming_iteration():
    """Streaming mode should yield processed rows and refuse `len()`."""
    ds_stream = FASD13(split="AS", sample_rate=32000, streaming=True)
    with pytest.raises(NotImplementedError):
        len(ds_stream)

    item = next(iter(ds_stream))
    assert isinstance(item["audio"], np.ndarray)
    assert item["audio"].dtype == np.float32


def test_check_selection_table(ds: FASD13):
    """Selection tables should be Raven-shaped with sane, in-bounds event times."""
    for idx in (0, 5, 11):
        item = ds[idx]
        assert "selection_table" in item, f"[{idx}] missing 'selection_table' key"
        st = item["selection_table"]

        assert isinstance(st, pd.DataFrame), f"[{idx}] selection_table is not a DataFrame"
        assert list(st.columns) == SELECTION_TABLE_COLUMNS, (
            f"[{idx}] unexpected selection table columns: {list(st.columns)}"
        )
        assert len(st) > 0, f"[{idx}] selection table is empty"

        assert set(st["Q"].astype(str)) <= {"POS", "UNK", "NEG"}, f"[{idx}] unexpected Q values"
        assert "Label" not in st.columns, f"[{idx}] Label marked UNK events as positive"
        assert (st["End Time (s)"] >= st["Begin Time (s)"]).all(), f"[{idx}] negative-length event"

        duration = item["audio"].size / item["sample_rate"]
        assert st["End Time (s)"].max() <= duration + 1e-3, f"[{idx}] event past end of audio"


def _windowed(ds: FASD13, idx: int, start: float, end: float) -> dict:
    """Process one row as if a caller had attached window columns to it."""
    row = dict(ds._data[idx])
    row["window_start_sec"], row["window_end_sec"] = start, end
    return ds._process(row)


def test_windowed_read_returns_only_the_window(ds: FASD13):
    """A windowed row should read just that segment, not the whole recording."""
    full = ds[0]["audio"]
    item = _windowed(ds, 0, 10.0, 20.0)

    assert item["sample_rate"] == 32000
    assert abs(item["audio"].size / 32000 - 10.0) < 0.05, "window is not 10 s of audio"
    assert item["audio"].size < full.size, "windowed read returned the whole recording"
    np.testing.assert_allclose(item["audio"], full[10 * 32000 : 20 * 32000], atol=1e-6)


def test_windowed_selection_table_is_window_relative(ds: FASD13):
    """Events should be filtered to the window and shifted onto the returned audio."""
    full_st = ds[0]["selection_table"]
    # Pick an event far enough in that a 2 s lead-in stays inside the recording.
    event = full_st[full_st["Begin Time (s)"] >= 2.0].iloc[0]
    start = float(event["Begin Time (s)"]) - 2.0
    end = start + 6.0

    st = _windowed(ds, 0, start, end)["selection_table"]

    assert len(st) > 0, "the window was built around an event, so it cannot be empty"
    assert (st["Begin Time (s)"] >= 0).all(), "event begins before the window"
    assert (st["End Time (s)"] <= 6.0 + 1e-6).all(), "event ends after the window"

    # The event the window was built around should sit ~2 s in, and keep its identity.
    kept = st[st["event_index"] == event["event_index"]]
    assert len(kept) == 1, "the target event was dropped from its own window"
    assert abs(float(kept.iloc[0]["Begin Time (s)"]) - 2.0) < 1e-6

    # Events outside the window are gone.
    assert len(st) < len(full_st), "no events were filtered out of the window"


def test_unwindowed_selection_table_stays_absolute(ds: FASD13):
    """Without window columns the full table is returned with absolute times."""
    item = ds[0]
    st = item["selection_table"]

    assert len(st) == int(ds._data[0]["n_events"]), "unwindowed read dropped events"
    duration = item["audio"].size / item["sample_rate"]
    assert st["End Time (s)"].max() <= duration + 1e-3
    # Times span the recording rather than a window rebased onto zero.
    assert st["End Time (s)"].max() > 10.0


def test_windowed_read_on_long_recording():
    """
    Windowed reads must work on the pathological files, not just the small ones.

    HG holds 8-hour recordings: ~3.7 GB of float32 apiece at 32 kHz. Cutting an
    N-shot support clip or chunking a query region downstream is only possible
    because the segment is streamed rather than downloaded whole. If this test
    starts taking minutes or exhausting memory, the windowed path has regressed
    to a full-file read.
    """
    ds_hg = FASD13(split="HG", sample_rate=32000, backend="pandas")
    row = dict(ds_hg._data[0])
    assert row["audio_duration"] > 3600, "expected HG to hold multi-hour recordings"

    row["window_start_sec"], row["window_end_sec"] = 100.0, 110.0
    item = ds_hg._process(row)

    assert abs(item["audio"].size / 32000 - 10.0) < 0.05
    assert item["audio"].dtype == np.float32


def _shot_end_times(ds: FASD13, idx: int, n_shots: int = 5) -> list[float]:
    """Derive shot boundaries the way a client is expected to.

    Mirrors the recipe in the class docstring: the dataset deliberately does
    not expose this, so the test derives it from the selection table the same
    way a consumer would.
    """
    st = pd.read_csv(StringIO(ds._data[idx]["selection_table"]), sep="\t")
    known = st[st["Q"] != "UNK"]
    return [float(t) for t in sorted(known["End Time (s)"])[:n_shots]]


def test_shot_boundaries_are_not_stored(ds_all: FASD13):
    """
    Shot boundaries must be derivable, never stored.

    A stored copy goes stale under windowed reads, where table times are
    re-based but a manifest column would not be -- the two would end up in
    different coordinate systems on the same row.
    """
    rows = ds_all._data.unwrap
    assert "shot_end_times" not in rows.columns
    assert "n_shots_available" not in rows.columns

    ends = _shot_end_times(ds_all, 0)
    assert 1 <= len(ends) <= 5
    assert ends == sorted(ends), "shot end times should be ascending"


def test_shot_boundaries_exclude_unk():
    """UNK events must not be usable as shots, per the protocol."""
    ms = FASD13(split="MS", backend="pandas")          # 632 UNK events
    st = pd.read_csv(StringIO(ms._data[0]["selection_table"]), sep="\t")
    pos_ends = sorted(float(t) for t in st[st["Q"] == "POS"]["End Time (s)"])
    assert _shot_end_times(ms, 0) == pos_ends[:5]

    unk_ends = {float(t) for t in st[st["Q"] == "UNK"]["End Time (s)"]}
    assert not (set(_shot_end_times(ms, 0)) & unk_ends), "an UNK event was used as a shot"


def test_shot_boundaries_stay_absolute_under_windowing(ds: FASD13):
    """Deriving from the row must not be perturbed by a windowed read."""
    before = _shot_end_times(ds, 0)
    _windowed(ds, 0, 5.0, 15.0)
    assert _shot_end_times(ds, 0) == before


def test_per_row_licensing(ds_all: FASD13):
    """Licensing is per sub-dataset, so every row must carry its own license."""
    rows = ds_all._data.unwrap
    assert rows["license"].astype(str).str.len().gt(0).all()
    assert rows["subdataset"].nunique() == len(SUBDATASET_CODES)
    # Mixed licensing is the whole reason it lives on the row rather than in info.
    assert rows["license"].nunique() > 1


def test_from_config():
    """`from_config` should build the same dataset as direct instantiation."""
    config = DatasetConfig(dataset_name="fasd13", split="AS", sample_rate=32000, backend="pandas")
    ds_cfg, meta = FASD13.from_config(config)

    assert isinstance(ds_cfg, FASD13)
    assert meta == {}
    assert len(ds_cfg) == EXPECTED_LEN_AS
    assert ds_cfg.sample_rate == 32000


def test_output_take_and_give():
    """`output_take_and_give` should rename and filter the returned keys."""
    ds_mapped = FASD13(
        split="AS",
        sample_rate=32000,
        backend="pandas",
        output_take_and_give={"audio": "waveform", "sound_name": "filename"},
    )
    item = ds_mapped[0]
    assert set(item.keys()) == {"waveform", "filename"}
    assert isinstance(item["waveform"], np.ndarray)


def test_reference_item_stability(ds: FASD13):
    """
    Check that a canonical item (index 0 of AS) is bitwise-stable.

    We hash the raw float32 audio buffer. This catches:
    - sample rate changes (resampling -> different samples)
    - channel handling changes (stereo->mono logic changed)
    - dtype changes
    - ordering changes in the split (if a different recording moved to idx 0)

    If this fails for a legitimate/intentional reason, recompute the hash below
    and update EXPECTED_FIRST_AS_AUDIO_SHA256.

    We do the same for the manifest.
    """
    item = ds[0]

    # audio presence/type checks (defensive, so the hash failure message is clearer)
    assert "audio" in item, "[0] missing 'audio' key"
    audio = item["audio"]
    assert isinstance(audio, np.ndarray), "[0] audio is not a numpy array"
    assert audio.dtype == np.float32, f"[0] audio dtype is {audio.dtype}, expected float32"

    h = create_hash(audio.tobytes())
    assert h == EXPECTED_FIRST_AS_AUDIO_SHA256, (
        "First item's audio hash changed.\n"
        f"Got    {h}\n"
        f"Expect {EXPECTED_FIRST_AS_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace EXPECTED_FIRST_AS_AUDIO_SHA256 with the new hash."
    )

    csv_bytes = (
        ds._data.unwrap.sort_index(axis=0).sort_index(axis=1).to_csv(index=True).encode("utf-8")
    )
    h = create_hash(csv_bytes)
    assert h == AS_ANNOTATIONS_SHA256, (
        "Annotation's hash changed.\n"
        f"Got    {h}\n"
        f"Expect {AS_ANNOTATIONS_SHA256}\n\n"
        "If this is an intentional dataset/content update, "
        "replace AS_ANNOTATIONS_SHA256 with the new hash."
    )


def test_str(ds_all: FASD13):
    """`str()` should summarise the dataset without touching audio."""
    text = str(ds_all)
    assert "fasd13" in text
    assert "0.1.0" in text
    assert "AS" in text
