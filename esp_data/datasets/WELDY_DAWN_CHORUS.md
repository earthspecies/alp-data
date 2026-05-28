# Weldy NW Dawn Chorus

A WABAD-style strong-detection dataset wrapping Weldy et al. 2024,
*"Audio tagging of avian dawn chorus recordings in California, Oregon, and
Washington"* (Zenodo [10895837](https://zenodo.org/records/10895837),
CC-BY-4.0).

- **Source:** 1,575 × 5-min mono soundscape recordings (32 kHz, two
  omnidirectional Song Meter 4 mics averaged) from 525 sites on US federal
  forests in CA/OR/WA, May–August 2022 dawn chorus.
- **Annotations:** **multi-label per 2-s window**, sonotype-aware
  (`song`/`call`/`drum`/`engine`/…), covering **58 bird + 2 mammal + 6
  biotic-aggregate + 8 non-biotic** sound types. Labels are eBird-2021 codes
  with sonotype suffixes (e.g. `herthr_song_1`, `chbchi_call_1`, `rain_1`).
- **Annotation coverage:** 156 files fully annotated (53,509 window-labels,
  ~12 h), 215 partially annotated (5,500 labels), the rest unlabeled.

## Splits

| split | recordings | annotations |
|---|--:|--:|
| `all` | 1,575 | 59,009 |
| `complete` | 156 | 53,509 |
| `partial` | 215 | 5,500 |
| `labeled` | 371 | 59,009 |
| `unlabeled` | ~1,204 | 0 |

## GCS layout

```
gs://esp-data-ingestion/weldy_dawn_chorus/v0.1.0/
├── recordings/<file>.wav      # 32 kHz stereo originals
├── audio_16k/<file>.wav       # 16 kHz mono PCM16
├── audio_32k/<file>.wav       # 32 kHz mono PCM16
├── annotations/               # raw Weldy TSVs (provenance)
├── metadata/                  # acoustic_files / annotation_metadata /
│                              # environmental_characteristics + weldy_species_taxonomy
├── all.csv  complete.csv  partial.csv  labeled.csv  unlabeled.csv
└── species_labels.csv
```

## Row schema

`fn`, `file`, `site`, `replicate`, `date`, `annotation_status`
(`complete`/`partial`/`no`), `audio_fp`, `16khz_path`, `32khz_path`,
`audio_duration`, `sample_rate`, `n_events`, `selection_table`.

### `selection_table` columns (per 2-s window, multi-label)

`Begin Time (s)`, `End Time (s)`, **`Species`** (GBIF canonical), `Species Code`
(eBird_2021), `Common Name`, **`Sonotype`** (`song`/`call`/`drum`/`engine`/…),
**`Category`** (`species` / `non-biotic` / `biotic-aggregate` / `method`),
`Label` (raw Weldy code), `clip_complete` (TRUE/FALSE — TRUE for the
fully-annotated set; partial windows carry the dataset's `clip_complete` flag).

Method tokens (`complete`, `empty`) are kept in the selection table for
faithful reproduction — filter on `Category != "method"` for sound events.

## Taxonomy

Species (~60 unique eBird codes with real binomials) → GBIF canonical via
`esp_data.discover.gbif_taxonomy.GBIFConverter`. Aggregates (`Aves`, `Insecta`,
`Setophaga spp.`) and non-biotic / method tokens have empty `Species` (common
name retained in `Common Name`). The sonotype dimension is independent of the
species dimension and kept in `Sonotype`.

## Reproduction

```bash
# 1. Stage audio + TSV annotations to GCS (dev VM; ~20 GB, ≤8 streams, NFS)
WORK=/mnt/home/weldy_staging \
    bash scripts/data_preprocessing_scripts/weldy_dawn_chorus_stage.sh

# 2. Resample to 16 kHz + 32 kHz mono (Slurm cpu)
ssh slurm-login 'cd ~/esp-data-dev && sbatch jobs/resample_weldy_dawn_chorus.sh'

# 3. Resolve eBird → GBIF taxonomy
uv run python scripts/data_preprocessing_scripts/weldy_dawn_chorus_taxonomy.py

# 4. Build split CSVs + species_labels
uv run python scripts/data_preprocessing_scripts/weldy_dawn_chorus_build_csv.py
```

## Usage

```python
from esp_data.datasets import WeldyDawnChorus
ds = WeldyDawnChorus(split="complete", sample_rate=16000)   # rigorous label set
item = ds[0]
item["audio"]            # np.float32 mono @ 16 kHz
item["selection_table"]  # multi-label DataFrame (sonotype-aware)
```

## Notes / caveats

- **Multi-label / sonotype-aware**: a single 2-s window can carry multiple
  labels (e.g. a bird singing while it rains); each is a separate row in the
  embedded selection table.
- **Vocabulary is aspirational**: the dataset defines 260 labels in
  `annotation_metadata.tsv` but only ~115 are *used* (n_annotations > 0); of
  those, 30 appear at a single site (potentially fragile for geographic
  generalisation). Of the 60 species in the labelled set, 28 have >1 used
  sonotype variant, 21 have ≥2 variants with ≥5 examples each.
- **Stereo → mono**: 16 kHz / 32 kHz versions are mono (averaged); the
  ``recordings/`` originals retain stereo.
