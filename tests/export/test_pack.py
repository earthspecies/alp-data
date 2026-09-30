"""End-to-end tests for `alp_data.export.pack`."""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import soundfile as sf
import yaml

from alp_data.dataset import ChainedDatasetConfig, ConcatConfig, dataset_from_config
from alp_data.export import pack, to_hf
from alp_data.export.columns import (
    OFFSET_COL,
    SHA256_COL,
    SHARD_COL,
    SIZE_COL,
    SOURCE_INDEX_COL,
)
from alp_data.export.serializers import decode_audio, decode_value
from alp_data.io.packed_media_store import PackedMediaStore
from tests.export.pack_test_dataset import PackTestConfig, make_source


@pytest.fixture
def source(tmp_path: Path) -> PackTestConfig:
    return make_source(tmp_path / "src", n=5)


def _load(out: Path) -> tuple[dict[str, Any], pl.DataFrame, PackedMediaStore]:
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    table = pl.read_parquet(out / "table.parquet")
    store = PackedMediaStore(out / "media", [s["name"] for s in cfg["shards"]])
    return cfg, table, store


def test_pack_writes_the_expected_layout(source: PackTestConfig, tmp_path: Path) -> None:
    out = pack(source, tmp_path / "pack", samples_per_shard=2)
    assert out == tmp_path / "pack"
    assert sorted(p.name for p in out.iterdir()) == ["config.yaml", "media", "table.parquet"]
    assert sorted(p.name for p in (out / "media").iterdir()) == [
        "shard-00000.tar",
        "shard-00001.tar",
        "shard-00002.tar",
    ]


def test_table_has_one_row_per_source_row_and_bookkeeping_columns(
    source: PackTestConfig, tmp_path: Path
) -> None:
    pack(source, tmp_path / "pack", samples_per_shard=2)
    _, table, _ = _load(tmp_path / "pack")
    assert table.height == 5
    assert table[SOURCE_INDEX_COL].to_list() == [0, 1, 2, 3, 4]
    assert table[SHARD_COL].to_list() == [0, 0, 1, 1, 2]
    for col in (OFFSET_COL, SIZE_COL, SHA256_COL):
        assert col in table.columns
    assert "audio" not in table.columns
    assert {"label", "score", "sample_rate", "labels", "selection_table", "targets"} <= set(
        table.columns
    )


def test_audio_round_trips_exactly(source: PackTestConfig, tmp_path: Path) -> None:
    pack(source, tmp_path / "pack", samples_per_shard=2)
    _, table, store = _load(tmp_path / "pack")
    ds, _ = dataset_from_config(source)
    for row in table.iter_rows(named=True):
        data = store.read(row[SHARD_COL], row[OFFSET_COL], row[SIZE_COL])
        audio, sr = decode_audio(data)
        expected = ds[row[SOURCE_INDEX_COL]]
        assert sr == expected["sample_rate"]
        np.testing.assert_array_equal(audio, expected["audio"])


def test_opaque_columns_round_trip(source: PackTestConfig, tmp_path: Path) -> None:
    pack(source, tmp_path / "pack")
    cfg, table, _ = _load(tmp_path / "pack")
    assert cfg["opaque_columns"] == {"selection_table": "dataframe", "targets": "ndarray"}
    ds, _ = dataset_from_config(source)
    row = table.row(3, named=True)
    expected = ds[3]
    pd.testing.assert_frame_equal(
        decode_value(row["selection_table"], "dataframe"), expected["selection_table"]
    )
    np.testing.assert_array_equal(decode_value(row["targets"], "ndarray"), expected["targets"])
    assert row["labels"] == expected["labels"]


def test_config_yaml_records_source_format_and_provenance(
    source: PackTestConfig, tmp_path: Path
) -> None:
    pack(source, tmp_path / "pack", samples_per_shard=2, audio_format="wav")
    cfg, _, _ = _load(tmp_path / "pack")
    assert cfg["source"]["dataset_name"] == "pack_test"
    assert cfg["source"]["csv_path"] == source.csv_path
    assert cfg["name"] == "pack_test"
    assert cfg["version"] == "0.1.0"
    assert cfg["split"] == "train"
    assert cfg["audio_format"] == "wav"
    assert cfg["audio_key"] == "audio"
    assert cfg["sample_rate_key"] == "sample_rate"
    assert cfg["samples_per_shard"] == 2
    assert cfg["num_rows"] == 5
    assert cfg["num_skipped"] == 0
    assert [s["name"] for s in cfg["shards"]] == [f"shard-{i:05d}.tar" for i in range(3)]
    assert all(s["size"] > 0 and len(s["sha256"]) == 64 for s in cfg["shards"])
    assert cfg["alp_data_version"]
    assert "alp_data_commit" in cfg
    assert cfg["created_at"]


def test_output_take_and_give_is_frozen_into_the_pack(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=3)
    source.output_take_and_give = {"audio": "waveform", "sample_rate": "sr", "label": "y"}
    pack(source, tmp_path / "pack")
    cfg, table, _ = _load(tmp_path / "pack")
    assert cfg["audio_key"] == "waveform"
    assert cfg["sample_rate_key"] == "sr"
    assert set(table.columns) == {
        "y",
        "sr",
        SOURCE_INDEX_COL,
        SHARD_COL,
        OFFSET_COL,
        SIZE_COL,
        SHA256_COL,
    }


def test_transformations_are_applied_before_packing(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=6)
    source.transformations = [
        {"type": "filter", "mode": "include", "property": "label", "values": ["species_0"]}
    ]
    pack(source, tmp_path / "pack")
    _, table, _ = _load(tmp_path / "pack")
    assert table.height == 3
    assert set(table["label"].to_list()) == {"species_0"}


def test_identical_audio_in_one_shard_shares_a_blob(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=4, duplicate_of={2: 0})
    pack(source, tmp_path / "pack", samples_per_shard=4)
    _, table, _ = _load(tmp_path / "pack")
    rows = table.rows(named=True)
    assert rows[0][OFFSET_COL] == rows[2][OFFSET_COL]
    assert rows[0][SHA256_COL] == rows[2][SHA256_COL]
    assert rows[1][OFFSET_COL] != rows[0][OFFSET_COL]


def test_corrupt_row_raises_by_default(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=4, corrupt={1})
    with pytest.raises(sf.LibsndfileError):
        pack(source, tmp_path / "pack")
    assert not (tmp_path / "pack" / "config.yaml").exists()


def test_corrupt_row_is_skipped_and_recorded_on_request(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=4, corrupt={1})
    out = pack(source, tmp_path / "pack", on_error="skip")
    cfg, table, _ = _load(out)
    assert table.height == 3
    assert table[SOURCE_INDEX_COL].to_list() == [0, 2, 3]
    assert cfg["num_rows"] == 3
    assert cfg["num_skipped"] == 1
    errors = pl.read_parquet(out / "pack_errors.parquet")
    assert errors[SOURCE_INDEX_COL].to_list() == [1]
    assert errors["error"][0]


def test_rerun_skips_finished_shards(tmp_path: Path) -> None:
    # Row 3 sits in shard 1. The first run finishes shard 0, then fails on it.
    source = make_source(tmp_path / "src", n=5, corrupt={3})
    out = tmp_path / "pack"
    with pytest.raises(sf.LibsndfileError):
        pack(source, out, samples_per_shard=2)
    shard0 = out / "media" / "shard-00000.tar"
    assert shard0.exists()
    assert not (out / "media" / "shard-00001.tar").exists()
    before = shard0.stat().st_mtime_ns

    make_source(tmp_path / "src", n=5)  # repair the corrupt file
    pack(source, out, samples_per_shard=2)
    assert shard0.stat().st_mtime_ns == before
    cfg, table, _ = _load(out)
    assert table.height == 5
    assert len(cfg["shards"]) == 3
    assert cfg["opaque_columns"] == {"selection_table": "dataframe", "targets": "ndarray"}


def test_rerun_on_a_complete_pack_is_a_no_op(source: PackTestConfig, tmp_path: Path) -> None:
    out = pack(source, tmp_path / "pack", samples_per_shard=2)
    before = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    pack(source, out, samples_per_shard=2)
    after = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    assert before == after


def test_chained_config_warns_and_packs_as_concatenation(tmp_path: Path) -> None:
    a = make_source(tmp_path / "a", n=2)
    b = make_source(tmp_path / "b", n=3)
    chain = ChainedDatasetConfig(datasets=[a, b])
    with pytest.warns(UserWarning, match="concatenat"):
        pack(chain, tmp_path / "pack")
    cfg, table, _ = _load(tmp_path / "pack")
    assert table.height == 5
    assert cfg["source"]["dataset_name"] == "concatenated_dataset"


def test_concat_config_packs_directly(tmp_path: Path) -> None:
    a = make_source(tmp_path / "a", n=2)
    b = make_source(tmp_path / "b", n=3)
    concat = ConcatConfig(datasets=[a, b])
    pack(concat, tmp_path / "pack")
    cfg, table, _ = _load(tmp_path / "pack")
    assert table.height == 5
    assert "_source_dataset" in table.columns
    # The concat's own per-child row index must survive next to the export index.
    assert table["_source_index"].to_list() == [0, 1, 0, 1, 2]
    assert table[SOURCE_INDEX_COL].to_list() == [0, 1, 2, 3, 4]
    # And the frozen source must keep the children's custom config fields.
    assert cfg["source"]["datasets"][0]["csv_path"] == a.csv_path


def test_encoding_type_errors_are_not_skippable(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=2, unpackable=True)
    with pytest.raises(TypeError, match="cannot be packed"):
        pack(source, tmp_path / "pack", on_error="skip")


def test_skip_tolerates_a_corrupt_first_row(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=3, corrupt={0})
    out = pack(source, tmp_path / "pack", on_error="skip")
    cfg, table, _ = _load(out)
    assert table[SOURCE_INDEX_COL].to_list() == [1, 2]
    assert cfg["num_skipped"] == 1


def test_resume_does_not_mislabel_free_text_columns(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=5, corrupt={3}, multiline_notes=True)
    out = tmp_path / "pack"
    with pytest.raises(sf.LibsndfileError):
        pack(source, out, samples_per_shard=2)
    make_source(tmp_path / "src", n=5, multiline_notes=True)
    pack(source, out, samples_per_shard=2)
    cfg, table, _ = _load(out)
    assert cfg["opaque_columns"] == {"selection_table": "dataframe", "targets": "ndarray"}
    assert table["notes"][0] == "line one\tcol\nline two 0"
    shard0 = out / "media" / "shard-00000.tar"
    assert cfg["shards"][0]["size"] == shard0.stat().st_size
    assert cfg["shards"][0]["sha256"] == hashlib.sha256(shard0.read_bytes()).hexdigest()


def test_finalise_tolerates_a_missing_parts_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Object stores have no empty directories: an export that wrote no parts
    must not fail when the runner removes the parts prefix."""
    from fsspec.implementations.local import LocalFileSystem

    import alp_data.export.runner as runner

    class NoEmptyDirsFS(LocalFileSystem):
        cachable = False  # codespell:ignore cachable

        def makedirs(self, path: str, exist_ok: bool = False) -> None:
            if str(path).rstrip("/").endswith(runner.PARTS_DIR):
                return
            super().makedirs(path, exist_ok=exist_ok)

    monkeypatch.setattr(runner, "filesystem_from_path", lambda _p: NoEmptyDirsFS())
    source = make_source(tmp_path / "src", n=2)
    hf_dir = to_hf(source, tmp_path / "hf")
    assert (hf_dir / "README.md").exists()
    assert not (hf_dir / runner.PARTS_DIR).exists()


def test_pack_rejects_a_live_dataset_object(source: PackTestConfig, tmp_path: Path) -> None:
    ds, _ = dataset_from_config(source)
    with pytest.raises(TypeError, match="config"):
        pack(ds, tmp_path / "pack")


def test_parallel_pack_matches_single_process(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=7)
    pack(source, tmp_path / "one", samples_per_shard=2, num_workers=1)
    pack(source, tmp_path / "two", samples_per_shard=2, num_workers=3)
    _, table_one, _ = _load(tmp_path / "one")
    _, table_two, _ = _load(tmp_path / "two")
    assert table_one.equals(table_two)
    for i in range(4):
        one = (tmp_path / "one" / "media" / f"shard-{i:05d}.tar").read_bytes()
        two = (tmp_path / "two" / "media" / f"shard-{i:05d}.tar").read_bytes()
        assert one == two


def test_source_config_survives_pickling_for_workers(source: PackTestConfig) -> None:
    clone = pickle.loads(pickle.dumps(source))
    assert clone == source
    assert os.path.exists(clone.csv_path)
