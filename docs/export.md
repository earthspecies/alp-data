# `alp_data.export` Module

## What is a packed dataset?

A **pack** is a configured dataset frozen into a few large files:

```
<pack>/
  config.yaml            frozen source config, audio format, provenance, shard list
  table.parquet          one row per sample: every output key except the audio
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
| `samples_per_shard` | Rows per tar shard. Shard `j` holds source rows `j*N` to `(j+1)*N`. Pick it from your clip length: 1,000 three-second clips is about 50 MB of FLAC, 1,000 twenty-minute recordings is about 20 GB. |
| `audio_format` | `"flac"` (16-bit PCM, default) or `"wav"` (float32, bit-exact, four times larger). |
| `num_workers` | Spawned worker processes. Each rebuilds the dataset from the config and packs whole shards. |
| `on_error` | `"raise"` (default) stops on the first row that fails after retries. `"skip"` drops the row, records it in `pack_errors.parquet`, and counts it in `config.yaml` as `num_skipped`. |
| `audio_key`, `sample_rate_key` | Output keys holding the audio and sample rate. Resolved from the config's `output_take_and_give` by default. |

`ConcatConfig` packs directly. A `ChainedDatasetConfig` is packed as a concatenation with a warning, because a pack is one table. Load several packs into a chain instead when you need a chain.

### Resuming

Shard boundaries are fixed before any work starts and each shard is written atomically. Rerunning `pack` with the same arguments skips finished shards and completes the rest. A pack whose `config.yaml` exists is left untouched.

### What ends up in the table

Scalars, strings, and lists of scalars are native parquet columns. Two kinds of value are stored opaquely and rebuilt on read; `config.yaml` lists them under `opaque_columns`:

- a `pandas.DataFrame` (selection tables) is written as a TSV string,
- a `numpy.ndarray` is written as a struct of raw bytes, dtype, and shape.

Five bookkeeping columns are added: `_source_index`, `_shard`, `_offset`, `_size`, `_sha256`. `PackedDataset` strips them from returned items. Identical audio within one shard is stored once and shares an offset.

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

The Hub's native audio layout is parquet with the encoded audio embedded as a struct of `bytes` and `path`. `to_hf` streams a pack into that shape:

```python
from alp_data import to_hf

to_hf("gs://my-bucket/packs/beans-validation-16k", "./beans-hf", rows_per_file=1000)
```

The result is a directory of `<split>-NNNNN-of-MMMMM.parquet` files whose schema metadata declares the audio column as an `Audio` feature. Upload the files to a dataset repository, or load them with `datasets.load_dataset("parquet", data_dir="./beans-hf")`. This does not add `datasets` as a dependency of `alp-data`.

## Design notes

- Tar is a container, not a format. The reader seeks to a byte offset and never opens the archive as a tar. Tar is kept so `tar -tvf` and `tar -xf` work when a human needs them.
- Shards are not compressed. Audio is already compressed, and compressing the archive would break range reads.
- There is no `unpack`. Blobs are re-encoded outputs, possibly windowed, not the original files.
- The pack format is described in full in the API reference for `alp_data.export`.
