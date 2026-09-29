"""Tests for `to_hf`, which rewrites a pack as Hugging Face style parquet."""

import io
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from alp_data.dataset import dataset_from_config
from alp_data.export import pack, to_hf
from alp_data.export.columns import BOOKKEEPING_COLS
from tests.export.pack_test_dataset import make_source


def test_to_hf_writes_parquet_shards_with_embedded_audio(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=5)
    out = pack(source, tmp_path / "pack", samples_per_shard=2)
    hf_dir = to_hf(out, tmp_path / "hf", samples_per_shard=3)

    files = sorted(p.name for p in hf_dir.iterdir() if p.suffix == ".parquet")
    assert files == ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
    assert (hf_dir / "README.md").exists()
    table = pa.concat_tables([pq.read_table(hf_dir / f) for f in files])
    assert table.num_rows == 5
    audio_type = table.schema.field("audio").type
    assert pa.types.is_struct(audio_type)
    assert audio_type.field("bytes").type == pa.binary()
    assert audio_type.field("path").type == pa.string()
    assert not set(BOOKKEEPING_COLS) & set(table.column_names)


def test_to_hf_declares_the_audio_feature_in_parquet_metadata(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=2)
    out = pack(source, tmp_path / "pack")
    hf_dir = to_hf(out, tmp_path / "hf")
    schema = pq.read_schema(hf_dir / "train-00000-of-00001.parquet")
    meta = json.loads(schema.metadata[b"huggingface"])
    assert meta["info"]["features"]["audio"] == {"_type": "Audio", "sampling_rate": 16000}


def test_to_hf_audio_bytes_decode_to_the_source_audio(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=3)
    out = pack(source, tmp_path / "pack")
    hf_dir = to_hf(out, tmp_path / "hf")
    rows = pq.read_table(hf_dir / "train-00000-of-00001.parquet").to_pylist()
    src, _ = dataset_from_config(source)
    for i, row in enumerate(rows):
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        np.testing.assert_array_equal(audio, src[i]["audio"])
        assert row["audio"]["path"] == f"{i:09d}.flac"
        assert row["label"] == src[i]["label"]
        assert row["labels"] == src[i]["labels"]


def test_to_hf_uses_the_packed_split_name(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=2)
    source.split = "validation"
    out = pack(source, tmp_path / "pack")
    hf_dir = to_hf(out, tmp_path / "hf")
    assert sorted(p.name for p in hf_dir.iterdir()) == [
        "README.md",
        "validation-00000-of-00001.parquet",
    ]


# --- to_hf from a config: the same loop as pack, a different writer ---------


def _hf_tables(hf_dir: Path) -> pa.Table:
    files = sorted(p for p in hf_dir.iterdir() if p.suffix == ".parquet")
    return pa.concat_tables([pq.read_table(f) for f in files])


def test_to_hf_from_config_writes_one_file_per_shard(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=5)
    hf_dir = to_hf(source, tmp_path / "hf", samples_per_shard=2)
    names = sorted(p.name for p in hf_dir.iterdir())
    assert names == [
        "README.md",
        "train-00000-of-00003.parquet",
        "train-00001-of-00003.parquet",
        "train-00002-of-00003.parquet",
    ]
    table = _hf_tables(hf_dir)
    assert table.num_rows == 5
    assert not set(BOOKKEEPING_COLS) & set(table.column_names)


def test_to_hf_from_config_matches_the_source(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=4)
    hf_dir = to_hf(source, tmp_path / "hf", samples_per_shard=3)
    src, _ = dataset_from_config(source)
    rows = _hf_tables(hf_dir).to_pylist()
    for i, row in enumerate(rows):
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        np.testing.assert_array_equal(audio, src[i]["audio"])
        assert sr == src[i]["sample_rate"]
        assert row["audio"]["path"] == f"{i:09d}.flac"
        assert row["labels"] == src[i]["labels"]
        assert row["label"] == src[i]["label"]
    schema = pq.read_schema(hf_dir / "train-00000-of-00002.parquet")
    meta = json.loads(schema.metadata[b"huggingface"])
    assert meta["info"]["features"]["audio"] == {"_type": "Audio", "sampling_rate": 16000}


def test_to_hf_readme_records_the_source_config(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=2)
    hf_dir = to_hf(source, tmp_path / "hf")
    readme = (hf_dir / "README.md").read_text()
    assert "dataset_name: pack_test" in readme
    assert source.csv_path in readme
    assert "alp_data_version" in readme


def test_to_hf_from_config_skips_and_records_bad_rows(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=4, corrupt={2})
    hf_dir = to_hf(source, tmp_path / "hf", on_error="skip")
    assert _hf_tables(hf_dir).num_rows == 3
    errors = pl.read_ndjson(hf_dir / "export_errors.jsonl")
    assert errors["_source_index"].to_list() == [2]
    assert not (hf_dir / "export_errors.parquet").exists()


def test_to_hf_from_config_resumes_after_a_failure(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=5, corrupt={3})
    out = tmp_path / "hf"
    with pytest.raises(Exception):
        to_hf(source, out, samples_per_shard=2)
    first = out / "train-00000-of-00003.parquet"
    assert first.exists()
    before = first.stat().st_mtime_ns
    make_source(tmp_path / "src", n=5)
    to_hf(source, out, samples_per_shard=2)
    assert first.stat().st_mtime_ns == before
    assert _hf_tables(out).num_rows == 5


def test_to_hf_parallel_matches_single_process(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=7)
    to_hf(source, tmp_path / "one", samples_per_shard=2, num_workers=1)
    to_hf(source, tmp_path / "two", samples_per_shard=2, num_workers=3)
    assert _hf_tables(tmp_path / "one").equals(_hf_tables(tmp_path / "two"))


def test_to_hf_rejects_a_live_dataset(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=2)
    ds, _ = dataset_from_config(source)
    with pytest.raises(TypeError, match="config"):
        to_hf(ds, tmp_path / "hf")
