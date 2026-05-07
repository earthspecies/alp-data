# esp-data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**esp-data** gives you unified access to dozens of bioacoustic datasets, including recordings from birds, marine mammals, primates, insects, anurans, and multi-taxon benchmarks. Every built-in dataset shares a common `Dataset` interface, with streaming, configurable transforms, and consistent loading regardless of source format.

## Why esp-data

Bioacoustic recordings live in many places: Zenodo, OSF, GBIF, institutional repositories. Each dataset arrives with its own format, manifest schema, audio organization, sampling rate, and licensing posture. Researchers wanting to listen across datasets first have to write a custom loader per dataset, then a custom mixer for combining them.

esp-data removes that scaffolding. Every built-in dataset surfaces the same interface (`for sample in ds`, `ds[i]`, `len(ds)`), with `audio` and `sample_rate` keys returned for every sample. Dataset-specific keys carry labels, annotations, and other metadata. Sample-rate harmonization, label derivation, and split-aware loading are wired in. Adding a new dataset means writing one class, not rebuilding plumbing.

You can stand up a multi-dataset benchmark, compare models across taxa, or stream species-specific audio for transfer learning without first becoming a data-engineering specialist.

## What's included

esp-data ships with 30+ built-in datasets across:

- **Birds** — large benchmarks including BirdSet (6,800+ training hours, 10,000 species) and WABAD (1,192 species, 72 sites); aggregator corpora like Xeno-Canto; and site- or species-specific recordings spanning arctic species, Hawaiian soundscapes, the Powdermill dawn chorus, and individual-ID datasets for chiffchaff, little owl, and tree pipit.
- **Marine mammals** — the Watkins Marine Mammal Sound Database (~13,700 clips across ~50 cetacean and pinniped species), DCLDE 2026 killer whale annotations, dolphin whistle and click corpora, and humpback / blue / fin whale phonations from Barkley Canyon.
- **Primates** — gelada vocal sequences, gibbon solos, infant marmoset vocalizations, and macaque coo calls.
- **Insects, anurans, and other mammals** — InsectSet459 (459 Orthoptera and Cicadidae species), AnuraSetStrong (42 frog species, 27 hours of expert annotations), and giant otter vocalization types.
- **Multi-taxon benchmarks and aggregators** — BEANS and BeansZero (the canonical bioacoustic and zero-shot benchmarks), AnimalSpeak (1M+ audio–caption pairs), AnimalSoundArchive, iNaturalist audio, AudioSet / AudioSetStrong, and Voxaboxen (overlapping vocalization detection).

Most datasets are openly licensed (CC-BY, CC-BY-NC, CC0, public domain). License and source metadata are available per-dataset via `Dataset.info` — provenance matters.

## Quickstart

```python
from esp_data import Beans

# Load 'train' split of the BEANS bioacoustic benchmark at 16kHz.
# Resampling is done on the fly via librosa.
beans = Beans(split="train", sample_rate=16000)

print(len(beans))

# Iterate
for sample in beans:
    print(sample["audio"].shape)
    break

# Indexed access
sample = beans[0]
print(sample["audio"].shape)

# Streaming mode (lower memory; len() not available)
beans_streaming = Beans(split="train", streaming=True)
for sample in beans_streaming:
    print(sample["audio"].shape)
    break
```

Datasets and transforms can also be loaded from a YAML config:

```yaml
# config.yaml
dataset:
  dataset_name: beans
  split: train
  sample_rate: 16000
  transformations:
    - type: filter
      property: source_dataset
      mode: include
      values: ["watkins"]
```

```python
from esp_data import dataset_from_config

ds, transform_metadata = dataset_from_config("config.yaml")

# ds now only contains "watkins" samples from the BEANS train split, resampled to 16kHz.
```

## Installation
```sh
git clone https://github.com/earthspecies/esp-data.git
cd esp-data
pip install -e .  # or uv sync
```

## Development

This repository uses [uv](https://docs.astral.sh/uv/) for dependency management.

```sh
# Install all dependencies including dev tools
uv sync --dev

# Set up pre-commit hooks (uses prek; pass --overwrite to replace existing pre-commit lib hooks)
uv run prek install

# Run tests (Note: some tests require authentication via google or cloudflare; see test files for details)
uv run pytest
```


## Highlights

- **Source-flexible loading** — built-in datasets stream from a public Cloudflare R2 bucket; the I/O module can also read from local paths, GCS, S3, and other R2 buckets, with CSV / JSON-Lines / Parquet manifests supported at the backend layer.
- **Iterate or random-access indexing** — every Dataset supports `for sample in ds` and `ds[i]`.
- **Streaming mode** — process datasets larger than memory with `streaming=True`.
- **Configurable transforms** — filter rows, select columns, derive labels, deduplicate, balance, upsample long tails, subsample by ratio.
- **Combine datasets** — `ConcatenatedDataset` merges multiple datasets with configurable column-merge strategies; `ChainedDataset` iterates over them sequentially with streaming support.
- **Pluggable backends** — pandas or polars under the hood, selectable per-Dataset.


## License

esp-data is released under the MIT License. See [LICENSE](LICENSE) for the full text.

The datasets accessed through esp-data are governed by their own licenses, set by their original creators — independent of esp-data's code license. Most are openly licensed (CC-BY, CC-BY-NC, CC0, public domain), but terms vary per dataset and may include attribution, share-alike, or non-commercial restrictions. Per-dataset license and source metadata are available via `Dataset.info`.

## Contributing

For improvements, bug reports, or proposals, open an issue or pull request.
