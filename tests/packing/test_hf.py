"""Tests for `to_hf`, which rewrites a pack as Hugging Face style parquet."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from alp_data.dataset import dataset_from_config
from alp_data.packing import pack, to_hf
from alp_data.packing.columns import BOOKKEEPING_COLS
from tests.packing.pack_test_dataset import make_source


def test_to_hf_writes_parquet_shards_with_embedded_audio(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src", n=5)
    out = pack(source, tmp_path / "pack", samples_per_shard=2)
    hf_dir = to_hf(out, tmp_path / "hf", rows_per_file=3)

    files = sorted(p.name for p in hf_dir.iterdir())
    assert files == ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
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
        import io

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
    assert [p.name for p in hf_dir.iterdir()] == ["validation-00000-of-00001.parquet"]
