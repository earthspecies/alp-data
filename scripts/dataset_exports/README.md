# Dataset exports

`export_dataset.py` freezes a configured dataset with `alp_data.export`, as a
pack (tar shards + parquet table, read back with `PackedDataset`) or as Hugging
Face parquet, and can verify the result against the live dataset afterwards.

```
uv run python scripts/dataset_exports/export_dataset.py \
    --config scripts/dataset_exports/configs/beans_validation_16k.yaml \
    --out gs://esp-ci-cd-tests/esp-data-tests/exports/beans/validation-16k \
    --num-workers 32 --verify 200 --verify-workers 4 \
    --summary ~/outputs/exports/beans_validation_16k.json
```

- `--config` is a dataset YAML with a `dataset:`, `concat:` or `chain:` key,
  the same format `dataset_from_config` reads. Put transforms and the sample
  rate there: they are what gets frozen.
- `--out` is any `anypath` target. Exports resume: rerun the same command after
  an interruption and finished shards are skipped.
- `--verify N` compares `N` random exported samples with the live dataset key
  by key. For a pack it also times packed versus live reads, and
  `--verify-workers K` repeats the read through `K` spawned processes, which
  is what a `DataLoader` does. For Hugging Face output it decodes the embedded
  bytes and matches rows to the source through the audio `path`.
- `--from-pack gs://.../pack --format hf` converts an existing pack instead of
  exporting a config, copying blobs without decoding.
- `--import-module` imports a module before anything else, so a dataset
  registered outside `alp_data` can be exported (workers import it too).
- `--summary` writes timings, sizes, and verification results as JSON.

On the cluster, `jobs/export_dataset.sh` wraps this. Every knob is an
environment variable:

```
CONFIG=scripts/dataset_exports/configs/beans_validation_16k.yaml \
OUT=gs://esp-ci-cd-tests/esp-data-tests/exports/beans/validation-16k \
WORKERS=32 sbatch jobs/export_dataset.sh

# Hugging Face parquet straight from a config
FORMAT=hf CONFIG=... OUT=gs://.../beans-hf sbatch jobs/export_dataset.sh

# or from a pack that already exists
FROM_PACK=gs://.../beans/validation-16k OUT=gs://.../beans-hf sbatch jobs/export_dataset.sh
```

Configs in `configs/`: `beans_validation_16k` (62k short clips), `fasd13_all_16k`
(109 windowed long recordings with selection tables), `hawaiian_birds_all_16k`
(635 recordings with selection tables).
