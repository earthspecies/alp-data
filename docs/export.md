# `alp_data.export` Module

`alp_data.export` freezes a configured dataset into files. Two formats share one loop:

| Format | Function | Read back with | Use it for |
|---|---|---|---|
| **Pack** | `pack(config, out)` | `PackedDataset` | Moving data between clusters, training from a frozen snapshot |
| **Hugging Face parquet** | `to_hf(config, out)` | `datasets.load_dataset` | Publishing to the Hub |

Both build the dataset from its config, call `ds[i]` for every row, encode the audio, and write what comes back. They take the same arguments (`samples_per_shard`, `audio_format`, `num_workers`, `on_error`), both resume, and both take a **config**, never a live dataset object. A pack can also be converted to Hugging Face parquet after the fact with `to_hf(pack_path, out)`, which copies blobs without decoding.

## What is a packed dataset?

A **pack** is a configured dataset frozen into a few large files:

```
<pack>/
  config.yaml            frozen source config, audio format, provenance, shard list
  table.parquet          one row per sample: every output key except the audio, plus bookkeeping
  pack_errors.parquet    only when rows were skipped
  media/
    shard-00000.tar      one encoded audio blob per row
    shard-00001.tar
```

`pack` builds the dataset from its config, calls `ds[i]` for every row, and stores what comes back. The audio array becomes one FLAC (or WAV) blob inside a tar shard. Every other key becomes a column of the table. Nothing about the source dataset class changes, so whatever its `_process` does is frozen into the pack: windowing, label parsing, sample rate, transforms, `output_take_and_give`.

`PackedDataset` reads a pack back as a normal map-style dataset. Reading sample `i` is one table lookup and one range read into a shard. There is no manifest to consult and no re-decoding logic to keep in sync with the source.

Use a pack when you need to:

- move a dataset with millions of small files to another cluster as a handful of large ones,
- train from a frozen, reproducible snapshot of a dataset at a stated version, split, and sample rate,
- publish a dataset to the Hugging Face Hub (see [`to_hf`](#hugging-face)).

## Packing

```python
from alp_data import DatasetConfig, pack

config = DatasetConfig(
    dataset_name="beans",
    split="validation",
    sample_rate=16000,
    transformations=[
        {"type": "filter", "mode": "exclude", "property": "source_dataset", "values": ["esc50"]},
    ],
)
pack(config, "gs://my-bucket/packs/beans-validation-16k", samples_per_shard=1000, num_workers=8)
```

`pack` takes a **config**, never a live dataset object. Workers rebuild the dataset from the config, and the config is what gets written into `config.yaml`. A dataset built in a notebook with programmatic transforms cannot be packed unless those transforms are in its config.

| Argument | Meaning |
|---|---|
| `samples_per_shard` | Rows per tar shard. Shard `j` holds source rows `j*N` to `(j+1)*N`. A shard is also the unit of parallelism, so with `num_workers > 1` the value is lowered when needed so every worker gets a shard. Pick it from your clip length: 1,000 three-second clips is about 50 MB of FLAC, 1,000 twenty-minute recordings is about 20 GB. |
| `audio_format` | `"flac"` (16-bit PCM, default) or `"wav"` (float32, bit-exact, four times larger). A row whose audio exceeds `[-1, 1]`, which 16-bit FLAC would clip, is stored as float32 WAV regardless; `config.yaml` counts such rows under `num_rows_lossless_fallback` and the table records each row's format in `_audio_format`. Datasets that resample on the fly produce such rows (see issue #325). |
| `num_workers` | Spawned worker processes. Each rebuilds the dataset from the config and packs whole shards. |
| `on_error` | `"raise"` (default) stops on the first row that fails after retries. `"skip"` drops the row, records it in `pack_errors.parquet`, and counts it in `config.yaml` as `num_skipped`. |
| `audio_key`, `sample_rate_key` | Output keys holding the audio and sample rate. Resolved from the config's `output_take_and_give` by default. |

`ConcatConfig` packs directly. A `ChainedDatasetConfig` is packed as a concatenation with a warning, because a pack is one table. Load several packs into a chain instead when you need a chain.

### Resuming

Shard boundaries are fixed before any work starts and each shard is written atomically. Rerunning `pack` with the same arguments skips finished shards and completes the rest. The plan (row count, rows per shard) is recorded when an export starts, and a rerun with a different plan is refused rather than reusing shards that hold the wrong rows. A pack whose `config.yaml` exists is left untouched.

### What ends up in the table

Scalars, strings, and lists of scalars are native parquet columns. Two kinds of value are stored opaquely and rebuilt on read; `config.yaml` lists them under `opaque_columns`:

- a `pandas.DataFrame` (selection tables) is written as arrow IPC bytes, so dtypes survive,
- a `numpy.ndarray` is written as a struct of raw bytes, dtype, and shape.

Six bookkeeping columns are added: `_export_index`, `_shard`, `_offset`, `_size`, `_sha256`, `_audio_format`. The index is not called `_source_index` because `ConcatenatedDataset` already uses that name for the row within a child dataset, and that column survives packing. `PackedDataset` strips them from returned items. Identical audio within one shard is stored once and shares an offset.

## Reading

```python
from alp_data import PackedDataset

ds = PackedDataset("gs://my-bucket/packs/beans-validation-16k")
item = ds[0]          # same keys the source dataset returned
len(ds), ds.info.version, ds.split
```

`PackedDataset` takes no `sample_rate`: it is frozen in the pack. Transforms, concatenation, and chaining work as on any other dataset, because the table is a normal pandas or polars backend. `output_take_and_give` on a `PackedDataset` applies on top of the one frozen into the pack.

From YAML, a pack sits next to any other dataset:

```yaml
chain:
  datasets:
    - dataset_name: packed_dataset
      path: gs://my-bucket/packs/beans-validation-16k
    - dataset_name: inaturalist
      split: train
      sample_rate: 16000
```

The store that does the range reads opens one handle per shard, lazily, and pickles without them, so `DataLoader(num_workers > 0)` with the `spawn` start method works as it does for every other dataset.

## Hugging Face

The Hub's native audio layout is parquet with the encoded audio embedded as a struct of `bytes` and `path`. `to_hf` writes that shape from any dataset config, through the same loop as `pack`:

```python
from alp_data import DatasetConfig, to_hf

config = DatasetConfig(dataset_name="beans", split="validation", sample_rate=16000)
to_hf(config, "./beans-hf", samples_per_shard=1000, num_workers=8)
```

or from an existing pack, copying its blobs without decoding:

```python
to_hf("gs://my-bucket/packs/beans-validation-16k", "./beans-hf")
```

The result is a directory of `<split>-NNNNN-of-MMMMM.parquet` files whose schema metadata declares the audio column as an `Audio` feature, plus a `README.md` with the source config and provenance. Skipped rows, if any, are listed in `export_errors.jsonl`, a name the Hub's parquet loader ignores. Upload the directory to a dataset repository, or load it with `datasets.load_dataset("parquet", data_dir="./beans-hf")`. This does not add `datasets` as a dependency of `alp-data`.

## Design notes

- Tar is a container, not a format. The reader seeks to a byte offset and never opens the archive as a tar. Tar is kept so `tar -tvf` and `tar -xf` work when a human needs them.
- Shards are not compressed. Audio is already compressed, and compressing the archive would break range reads.
- There is no `unpack`. Blobs are re-encoded outputs, possibly windowed, not the original files.
- The pack format is described in full in the API reference for `alp_data.export`.
