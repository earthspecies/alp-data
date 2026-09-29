"""Rewrite a pack as Hugging Face style parquet.

The Hub's native audio layout is parquet with the encoded audio embedded in a
struct of `bytes` and `path`, and the column declared as an `Audio` feature in
the parquet schema metadata. `to_hf` streams a pack into that shape so the
result can be uploaded as files or loaded with `datasets.load_dataset`.
"""

from __future__ import annotations

import json
import math

import pyarrow as pa
import pyarrow.parquet as pq

from alp_data.io import AnyPathT, anypath, filesystem_from_path
from alp_data.packing.columns import (
    BOOKKEEPING_COLS,
    OFFSET_COL,
    SHARD_COL,
    SIZE_COL,
    SOURCE_INDEX_COL,
)
from alp_data.packing.packed_dataset import PackedDataset
from alp_data.packing.serializers import decode_audio
from alp_data.packing.shards import member_name


def to_hf(
    pack_path: str | AnyPathT, out_dir: str | AnyPathT, *, rows_per_file: int = 1000
) -> AnyPathT:
    """Write a pack as Hugging Face style parquet files.

    Parameters
    ----------
    pack_path : str | AnyPathT
        A pack directory written by `pack`.
    out_dir : str | AnyPathT
        Destination directory. Files are named `<split>-NNNNN-of-MMMMM.parquet`.
    rows_per_file : int
        Rows per parquet file.

    Returns
    -------
    AnyPathT
        `out_dir`, as an `anypath`.

    Notes
    -----
    Audio bytes are copied as stored in the pack, so the encoding is whatever
    `pack` used. The parquet metadata declares only the audio column as an
    `Audio` feature; every other column is inferred by the reader. Opaque
    columns (TSV strings, array structs) are passed through unchanged.
    """
    ds = PackedDataset(pack_path)
    out = anypath(str(out_dir))
    fs = filesystem_from_path(out)
    fs.makedirs(str(out), exist_ok=True)

    ext = ds.pack_config["audio_format"]
    sample_rate = ds.pack_config.get("sample_rate") or _first_sample_rate(ds)
    features = {"audio": {"_type": "Audio", "sampling_rate": sample_rate}}
    metadata = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

    num_rows = len(ds._data)
    num_files = math.ceil(num_rows / rows_per_file)
    for file_idx in range(num_files):
        start, stop = file_idx * rows_per_file, min((file_idx + 1) * rows_per_file, num_rows)
        columns: dict[str, list] = {ds.audio_key: []}
        for row_idx in range(start, stop):
            row = ds._data[row_idx]
            data = ds._store.read(int(row[SHARD_COL]), int(row[OFFSET_COL]), int(row[SIZE_COL]))
            columns[ds.audio_key].append(
                {"bytes": data, "path": member_name(int(row[SOURCE_INDEX_COL]), ext)}
            )
            for key, value in row.items():
                if key in BOOKKEEPING_COLS:
                    continue
                columns.setdefault(key, []).append(value)
        arrays = {
            name: pa.array(values, type=audio_type if name == ds.audio_key else None)
            for name, values in columns.items()
        }
        arrow = pa.table(arrays).replace_schema_metadata(metadata)
        name = f"{ds.split}-{file_idx:05d}-of-{num_files:05d}.parquet"
        with fs.open(str(out / name), "wb") as f:
            pq.write_table(arrow, f)
    return out


def _first_sample_rate(ds: PackedDataset) -> int:
    """Sample rate of the first row: from its table value, else by decoding its blob.

    Parameters
    ----------
    ds : PackedDataset
        The pack being converted.

    Returns
    -------
    int
        Sample rate in Hz.
    """
    row = ds._data[0]
    if ds.sample_rate_key is not None:
        return int(row[ds.sample_rate_key])
    data = ds._store.read(int(row[SHARD_COL]), int(row[OFFSET_COL]), int(row[SIZE_COL]))
    return int(decode_audio(data)[1])
