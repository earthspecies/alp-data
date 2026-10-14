"""
Unit tests for SpanishCarrionCrowsChunks dataset.

These cover the chunk layout and the row expansion, both of which are pure
functions of the recording table and so run without touching GCS. The audio
path (`_process`) is not covered here: it needs the real recordings, and
`data_root` is not yet settled for this dataset.

Run with:
    pytest -q test_spanish_carrion_crows_chunks.py
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest
import soundfile as sf

from alp_data.backends import PolarsBackend
from alp_data.datasets.spanish_carrion_crows_chunks import (
    SpanishCarrionCrowsChunks,
    SpanishCarrionCrowsChunksConfig,
)

SR = 32000

# Small, readable layout: 10 s chunks overlapping by 2 s, so the hop is 8 s.
CHUNK_SEC = 10
OVERLAP_SEC = 2
HOP_SEC = CHUNK_SEC - OVERLAP_SEC


@pytest.fixture
def make_chunker():
    """_make builds a dataset only for accessing chunking functions.

    Constructed without ``__init__`` so we can test `_chunk_index`
    and `_expand_to_chunks` without loading the table of recordings.
    """

    def _make(
        chunk_duration_sec: float = CHUNK_SEC,
        chunk_overlap_sec: float = OVERLAP_SEC,
        sample_rate: int = SR,
    ) -> SpanishCarrionCrowsChunks:
        ds = object.__new__(SpanishCarrionCrowsChunks)
        ds.chunk_duration_sec = chunk_duration_sec
        ds.chunk_overlap_sec = chunk_overlap_sec
        ds.sample_rate = sample_rate
        return ds

    return _make


def _recordings(
    durations: list[float],
    paths: list[str] | None = None,
    audio_sr: int = 46875,
) -> PolarsBackend:
    """Build a recording table shaped like the real info CSV."""
    n = len(durations)
    if paths is None:
        paths = [f"spanish-carrion-crows/18/AW/rec{i}.wav" for i in range(n)]
    return PolarsBackend(
        pl.DataFrame(
            {
                "year": [2018] * n,
                "territory": ["AW"] * n,
                "focal_individual": [f"2018_AW_{i}" for i in range(n)],
                "audio_fp": paths,
                "audio_duration": list(durations),
                "audio_sr": [audio_sr] * n,
                "selection_table": ["Begin Time (s)\tEnd Time (s)\tAnnotation\n"] * n,
            }
        )
    )


# --- chunk layout -----------------------------------------------------------


def test_recording_shorter_than_one_chunk_yields_one(make_chunker):
    """A recording under one chunk long is a single chunk ending at EOF."""
    index = make_chunker()._chunk_index(np.array([5.0]))

    assert len(index["chunk_idx"]) == 1
    assert index["start_sample"][0] == 0
    assert index["end_sample"][0] == 5 * SR


def test_recording_of_exactly_one_chunk_yields_one(make_chunker):
    """The boundary case does not spill into a second, empty chunk."""
    index = make_chunker()._chunk_index(np.array([float(CHUNK_SEC)]))

    assert len(index["chunk_idx"]) == 1
    assert index["end_sample"][0] == CHUNK_SEC * SR


def test_final_chunk_is_truncated_at_eof(make_chunker):
    """A recording that does not divide evenly gets a short final chunk."""
    # 19 s at hop 8 s: chunks start at 0, 8 and 16 s; the last holds only 3 s.
    index = make_chunker()._chunk_index(np.array([19.0]))

    assert len(index["chunk_idx"]) == 3
    assert list(index["start_sample"]) == [0, 8 * SR, 16 * SR]
    assert list(index["end_sample"]) == [10 * SR, 18 * SR, 19 * SR]


def test_chunks_cover_the_whole_recording(make_chunker):
    """Every sample of every recording falls inside at least one chunk."""
    durations = [5.0, 10.0, 19.0, 123.456]
    index = make_chunker()._chunk_index(np.array(durations))

    for row, duration in enumerate(durations):
        mask = index["row_index"] == row
        starts, ends = index["start_sample"][mask], index["end_sample"][mask]
        assert starts[0] == 0, "first chunk must start at the beginning"
        assert ends[-1] == int(np.floor(duration * SR)), "last chunk must reach EOF"
        # Consecutive chunks overlap, so each starts before its predecessor ends.
        assert np.all(starts[1:] < ends[:-1]), "gap between consecutive chunks"


def test_hop_is_exact_and_overlap_recoverable_from_spans(make_chunker):
    """Consecutive chunks step by the hop and overlap by exactly the margin."""
    index = make_chunker()._chunk_index(np.array([100.0, 60.0]))
    same_recording = index["row_index"][1:] == index["row_index"][:-1]

    hops = np.diff(index["start_sample"])[same_recording]
    assert np.all(hops == HOP_SEC * SR)

    overlaps = (index["end_sample"][:-1] - index["start_sample"][1:])[same_recording]
    assert np.all(overlaps == OVERLAP_SEC * SR)


def test_chunk_idx_restarts_per_recording(make_chunker):
    """`chunk_idx` counts within a recording,
        while `row_index` identifies the recording.
    """
    index = make_chunker()._chunk_index(np.array([19.0, 5.0, 26.0]))

    assert list(index["row_index"]) == [0, 0, 0, 1, 2, 2, 2]
    assert list(index["chunk_idx"]) == [0, 1, 2, 0, 0, 1, 2]


def test_seconds_agree_exactly_with_samples(make_chunker):
    """The two units cannot drift: seconds are derived from the sample grid."""
    index = make_chunker()._chunk_index(np.array([19.0, 123.456]))

    assert np.all(index["start_sec"] * SR == index["start_sample"])
    assert np.all(index["end_sec"] * SR == index["end_sample"])


@pytest.mark.parametrize("overlap", [CHUNK_SEC, CHUNK_SEC + 1])
def test_overlap_at_least_chunk_duration_is_rejected(make_chunker, overlap):
    """Overlap duration should be strictly smaller than the chunk duration."""
    with pytest.raises(ValueError, match="chunk_overlap_sec"):
        make_chunker(chunk_overlap_sec=overlap)._chunk_index(np.array([100.0]))


# --- row expansion ----------------------------------------------------------


def test_expansion_has_one_row_per_chunk(make_chunker):
    """The expanded table lines up with the layout it was built from."""
    chunker = make_chunker()
    durations = [5.0, 19.0, 26.0]
    expanded = chunker._expand_to_chunks(_recordings(durations))

    assert len(expanded) == len(chunker._chunk_index(np.array(durations))["chunk_idx"])
    assert len(expanded) == 1 + 3 + 3


def test_expansion_drops_the_selection_table(make_chunker):
    """Labels are not carried through; repeating them per chunk is costly."""
    expanded = make_chunker()._expand_to_chunks(_recordings([19.0]))

    assert "selection_table" not in expanded.columns
    # The parent recording's other columns survive.
    assert {"year", "territory", "focal_individual", "audio_fp", "audio_sr"} <= set(expanded.columns)


def test_expansion_carries_parent_columns_to_every_chunk(make_chunker):
    """Each chunk keeps the metadata of the recording it came from."""
    expanded = make_chunker()._expand_to_chunks(_recordings([19.0, 5.0]))
    rows = [expanded[i] for i in range(len(expanded))]

    assert [row["audio_fp"] for row in rows] == [
        "spanish-carrion-crows/18/AW/rec0.wav",
        "spanish-carrion-crows/18/AW/rec0.wav",
        "spanish-carrion-crows/18/AW/rec0.wav",
        "spanish-carrion-crows/18/AW/rec1.wav",
    ]


def test_chunk_id_encodes_path_span_and_rate(make_chunker):
    """Check that the id format is as expected.
    The id will travel with the detections from BirdCODE so it is
    an easy way of attaching the detections to the section of the
    recording that was being analyzed. If we need to stitch together
    detections later, then the id itself can be used to determine
    the overlaps within a file."""
    expanded = make_chunker()._expand_to_chunks(_recordings([19.0]))
    ids = [expanded[i]["chunk_id"] for i in range(len(expanded))]

    assert ids == [
        f"spanish-carrion-crows/18/AW/rec0.wav#0-{10 * SR}@{SR}",
        f"spanish-carrion-crows/18/AW/rec0.wav#{8 * SR}-{18 * SR}@{SR}",
        f"spanish-carrion-crows/18/AW/rec0.wav#{16 * SR}-{19 * SR}@{SR}",
    ]


def test_chunk_ids_are_unique(make_chunker):
    """Ids must be unique. Large-scale inference keys stores results with this id."""
    expanded = make_chunker()._expand_to_chunks(_recordings([19.0, 5.0, 26.0, 100.0]))
    ids = [expanded[i]["chunk_id"] for i in range(len(expanded))]

    assert len(set(ids)) == len(ids)


def test_expanded_spans_match_the_layout(make_chunker):
    """Double-check that the values that feed read_audio and the chunk_id are matched."""
    chunker = make_chunker()
    index = chunker._chunk_index(np.array([19.0, 26.0]))
    expanded = chunker._expand_to_chunks(_recordings([19.0, 26.0]))
    rows = [expanded[i] for i in range(len(expanded))]

    assert [row["chunk_start_sample"] for row in rows] == list(index["start_sample"])
    assert [row["chunk_end_sample"] for row in rows] == list(index["end_sample"])
    assert [row["chunk_start_sec"] for row in rows] == list(index["start_sec"])
    assert [row["chunk_end_sec"] for row in rows] == list(index["end_sec"])


# --- configuration ----------------------------------------------------------


def test_config_defaults_are_complete():
    """A config naming only the dataset still carries every chunking field.

    Without the registered config class these would be ``extra`` keys on the
    base `DatasetConfig`, absent from a dump whenever the YAML omitted them,
    and `from_config` would raise rather than fall back to a default.
    """
    cfg = SpanishCarrionCrowsChunksConfig()

    assert cfg.dataset_name == "spanish-carrion-crows-chunks"
    assert cfg.sample_rate == 32000
    assert cfg.chunk_duration_sec == 900
    assert cfg.chunk_overlap_sec == 5
    assert cfg.split == "all"


def test_config_is_registered_for_the_dataset_name():
    """`dataset_from_config` must find this class, not the base config."""
    from alp_data.dataset import _custom_config_registry, _dataset_registry

    assert _dataset_registry["spanish-carrion-crows-chunks"] is SpanishCarrionCrowsChunks
    assert (
        _custom_config_registry["spanish-carrion-crows-chunks"] is SpanishCarrionCrowsChunksConfig
    )


# --- splice positions -------------------------------------------------------

#: A recording long enough to produce three chunks, the last one truncated.
SPLICE_DURATION_SEC = 19


@pytest.fixture
def ramp_recording(tmp_path):
    """Write a 32 kHz WAV in which every sample's value *is* its own index.

    Sample ``i`` holds the value ``i``, so a chunk's position in the recording
    is read straight off its audio: a misplaced splice reports where it
    actually landed rather than merely that it was wrong.

    Written at `SR` so `_process` does no resampling: the point is where the
    read starts and stops, and a resampler would smear the value-to-index
    correspondence this relies on.
    """
    total = SPLICE_DURATION_SEC * SR
    # FLOAT, not the usual PCM_16: these are sample indices rather than a
    # signal in [-1, 1], and integer PCM would clip every one of them.
    sf.write(tmp_path / "ramp.wav", np.arange(total, dtype=np.float32), SR, subtype="FLOAT")
    return tmp_path


@pytest.fixture
def splice_dataset(make_chunker, ramp_recording):
    """A dataset serving chunks of the ramp recording from a local directory."""
    ds = make_chunker()
    ds.output_take_and_give = None
    ds.data_root = ramp_recording
    ds._data = ds._expand_to_chunks(
        _recordings([float(SPLICE_DURATION_SEC)], paths=["ramp.wav"], audio_sr=SR)
    )
    return ds


def test_process_splices_at_the_expected_samples(splice_dataset):
    """Each chunk's audio begins and ends at the samples in the _data dataframe."""
    expected = [(0, 10 * SR), (8 * SR, 18 * SR), (16 * SR, SPLICE_DURATION_SEC * SR)]
    assert len(splice_dataset._data) == len(expected)

    for i, (start, end) in enumerate(expected):
        audio = splice_dataset[i]["audio"]

        assert int(audio[0]) == start, f"chunk {i} began at the wrong sample"
        assert int(audio[-1]) == end - 1, f"chunk {i} ended at the wrong sample"
        assert len(audio) == end - start, f"chunk {i} has the wrong length"


def test_process_splice_matches_the_chunk_id(splice_dataset):
    """The audio delivered agrees with the span advertised in the chunk id."""
    for i in range(len(splice_dataset._data)):
        item = splice_dataset[i]
        span = item["chunk_id"].rsplit("#", 1)[1].split("@")[0]
        start, end = (int(part) for part in span.split("-"))

        assert int(item["audio"][0]) == start
        assert len(item["audio"]) == end - start


# --- Dataset snapshot -------------------------------------------------------
#
# Code to generate the snapshot:
#
# import hashlib
# from alp_data.datasets import SpanishCarrionCrowsChunks

# ds = SpanishCarrionCrowsChunks(
#     sample_rate=SNAPSHOT_SR,
#     chunk_duration_sec=SNAPSHOT_CHUNK_SEC,
#     chunk_overlap_sec=SNAPSHOT_OVERLAP_SEC,
#     backend="polars",
# )
# print("len(ds) =", len(ds))

# df = ds._data.unwrap
# csv_bytes = df.sort(df.columns).write_csv().encode("utf-8")
# print("chunk table sha256:", hashlib.sha256(csv_bytes).hexdigest())

# audio = ds[SNAPSHOT_AUDIO_INDEX]["audio"]
# print("dtype:", audio.dtype, "shape:", audio.shape)
# print("audio sha256:", hashlib.sha256(audio.tobytes()).hexdigest())
# quit()

# ---------------------------------------------------------------------------

SNAPSHOT_SR = 32000
SNAPSHOT_CHUNK_SEC = 900
SNAPSHOT_OVERLAP_SEC = 5

#: Which item the audio snapshot hashes. Item 0 is a recording shorter than one
#: chunk and item 1 is the first chunk of the next, both of which start at zero
#: and so read from the beginning of their file. Item 2 is the first that seeks
#: into a recording, which is the path the whole chunking scheme rests on.
SNAPSHOT_AUDIO_INDEX = 2

EXPECTED_LEN_ALL = 18018
EXPECTED_CHUNK_TABLE_SHA256 = "9e106320d90c28dbc98cccbeb3b6a4e343beac40ca2dee571372836bda4b8f5f"
EXPECTED_SNAPSHOT_AUDIO_SHA256 = "e500ddc584346d2f6ae203e393695e3f790d3f5c8a998abd5a68fa75219c1b40"
# ---------------------------------------------------------------------------

needs_table_snapshot = pytest.mark.skipif(
    not EXPECTED_CHUNK_TABLE_SHA256 or not EXPECTED_LEN_ALL,
    reason="chunk-table snapshot not recorded yet; see the generator block above",
)
needs_audio_snapshot = pytest.mark.skipif(
    not EXPECTED_SNAPSHOT_AUDIO_SHA256,
    reason="audio snapshot not recorded yet; needs `data_root` resolved",
)


@pytest.fixture(scope="module")
def live_ds() -> SpanishCarrionCrowsChunks:
    """Load the real dataset from GCS at the snapshotted chunk parameters."""
    return SpanishCarrionCrowsChunks(
        sample_rate=SNAPSHOT_SR,
        chunk_duration_sec=SNAPSHOT_CHUNK_SEC,
        chunk_overlap_sec=SNAPSHOT_OVERLAP_SEC,
        backend="polars",
    )


@needs_table_snapshot
def test_snapshot_length(live_ds: SpanishCarrionCrowsChunks):
    """The corpus expands to the expected number of chunks."""
    assert len(live_ds) == EXPECTED_LEN_ALL


@needs_table_snapshot
def test_snapshot_chunk_table(live_ds: SpanishCarrionCrowsChunks):
    """Check that the expanded table is bitwise stable.

    One hash over every chunk's row, so this catches a change to the recordings
    themselves (a recording added, removed, or re-timed) and a change to the
    layout or the id format alike. Rows are sorted first so the hash does not
    depend on the order the expansion happened to emit them in.

    If this fails for a legitimate reason — a dataset version bump, or a
    deliberate change to the chunking — regenerate with the block above and
    replace EXPECTED_CHUNK_TABLE_SHA256. Do check which of the two it was: a
    layout change silently invalidates the chunk ids in any run already stored.
    """
    df = live_ds._data.unwrap
    digest = hashlib.sha256(df.sort(df.columns).write_csv().encode("utf-8")).hexdigest()

    assert digest == EXPECTED_CHUNK_TABLE_SHA256, (
        "Chunk table hash changed.\n"
        f"Got    {digest}\n"
        f"Expect {EXPECTED_CHUNK_TABLE_SHA256}\n\n"
        "If this is an intentional dataset or chunking update, replace "
        "EXPECTED_CHUNK_TABLE_SHA256 with the new hash."
    )


@needs_audio_snapshot
def test_snapshot_seeked_chunk_audio(live_ds: SpanishCarrionCrowsChunks):
    """A chunk read from partway into a recording is bitwise stable.

    Hashing the raw float32 buffer catches a changed splice position, a changed
    resampler or rate, a change in channel handling, and a reordering that put a
    different recording at this index. `SNAPSHOT_AUDIO_INDEX` is chosen so the
    read seeks rather than starting at the beginning of its file.
    """
    audio = live_ds[SNAPSHOT_AUDIO_INDEX]["audio"]
    assert audio.dtype == np.float32, f"audio dtype is {audio.dtype}, expected float32"

    digest = hashlib.sha256(audio.tobytes()).hexdigest()
    assert digest == EXPECTED_SNAPSHOT_AUDIO_SHA256, (
        f"Audio hash changed for item {SNAPSHOT_AUDIO_INDEX}.\n"
        f"Got    {digest}\n"
        f"Expect {EXPECTED_SNAPSHOT_AUDIO_SHA256}\n\n"
        "If this is an intentional dataset or pipeline update, replace "
        "EXPECTED_SNAPSHOT_AUDIO_SHA256 with the new hash."
    )
