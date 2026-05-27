# Dartmouth Avian Soundscapes

Large-scale, stratified, fully annotated acoustic **forest soundscape** dataset
of avian vocalisations from eastern North America, wrapped as a WABAD-style
strong-detection dataset (one row per recording, each carrying a Raven-style
selection table of annotated events).

- **Source:** Zenodo record [20038954](https://zenodo.org/records/20038954)
  (DOI: 10.5281/zenodo.20038954), Panwar et al., Dartmouth College.
- **License:** CC-BY-4.0
- **Audio:** 1,302 × 10-minute mono recordings, originally 32 kHz FLAC.
- **Annotations:** ~183,000 expert-annotated vocalisations, 96 bird species
  (plus a few mammal / unknown labels), produced in Raven Pro 1.6 over three
  review rounds across four northeastern US protected areas (Acadia NP, Hubbard
  Brook Experimental Forest, Katahdin Woods and Waters NM,
  Marsh-Billings-Rockefeller NHP), 2022–2023 breeding seasons.

## Splits

The collection ships as three sub-datasets, each exposed as a split, plus a
combined `all`:

| split  | sub-dataset  | recordings | description                                  |
|--------|--------------|-----------:|----------------------------------------------|
| `acad` | DatasetACAD  | 396        | Acadia NP, post-sunrise clear-day recordings |
| `mabi` | DatasetMABI  | 477        | Marsh-Billings-Rockefeller NHP               |
| `simr` | DatasetSIMR  | 429        | point-count survey recordings (all 4 parks)  |
| `all`  | —            | 1,302      | all three combined                           |

## GCS layout

```
gs://esp-data-ingestion/dartmouth-avian-soundscapes/v0.1.0/
├── recordings/<DatasetID>/*.flac     # lossless originals, 32 kHz
├── audio_16k/<DatasetID>/*.wav       # pre-resampled 16 kHz PCM16
├── audio_32k/<DatasetID>/*.wav       # 32 kHz PCM16 (re-encoded)
├── annotations/<DatasetID>/*.txt     # raw Raven Pro tables (provenance)
├── metadata/                         # recording/site/species/data_dictionary + ReadMe
│   └── species_taxonomy.csv          # AOU code -> GBIF canonical + taxonomy
├── all.csv  acad.csv  mabi.csv  simr.csv
└── species_labels.csv                # resolved canonical-name vocabulary
```

## Row schema (split CSVs)

| column            | description                                               |
|-------------------|-----------------------------------------------------------|
| `fn`              | recording stem                                            |
| `item_id`         | dataset ItemID                                            |
| `subdataset`      | `acad` / `mabi` / `simr`                                  |
| `audio_fp`        | original FLAC path (relative to the v0.1.0 root)          |
| `16khz_path`      | pre-resampled 16 kHz WAV path                             |
| `32khz_path`      | pre-resampled 32 kHz WAV path                             |
| `audio_duration`  | seconds (from FLAC header)                                |
| `sample_rate`     | original sample rate (Hz)                                 |
| `site_id`,`park_id` | recording location identifiers                          |
| `year`,`date`,`time` | collection time (YYYY / YYYYMMDD / HHMM)               |
| `latitude`,`longitude` | site coordinates (WGS84)                             |
| `n_events`        | number of annotated events                                |
| `selection_table` | embedded tab-separated table (see below)                  |

### `selection_table` columns

`Begin Time (s)`, `End Time (s)`, `Low Freq (Hz)`, `High Freq (Hz)`,
`Species` (GBIF canonical name), `Species Code` (raw 4-letter AOU code),
`Common Name`, `Background` (incidental Raven notes — other species, elevated
noise, occasional event-type labels).

## Taxonomy

Species are labelled in the source with 4-letter AOU codes. We map each code to
its eBird-2021 scientific name (`species_metadata.csv`) and then to the
**GBIF canonical name** via `esp_data.discover.gbif_taxonomy.GBIFConverter`
(against the GBIF animals backbone). The canonical name populates `Species`;
the original code is preserved in `Species Code`. Codes without a scientific
name (`????` unknown, `UNMA` unknown mammal, `UNWO` unknown woodpecker) map to
`Species = "Unknown"`. One manual correction is applied: Cooper's Hawk
`Astur cooperii` → `Accipiter cooperii` (the GBIF backbone predates the
*Accipiter*→*Astur* split).

## Reproduction

```bash
# 1. Stage raw audio + annotations + metadata to GCS (dev VM, background)
bash scripts/data_preprocessing_scripts/dartmouth_avian_soundscapes_stage.sh

# 2. Resolve AOU codes -> GBIF canonical names
uv run python scripts/data_preprocessing_scripts/dartmouth_avian_soundscapes_taxonomy.py

# 3. Build split CSVs + species_labels.csv + QC report
uv run python scripts/data_preprocessing_scripts/dartmouth_avian_soundscapes_build_csv.py

# 4. Resample to 16 kHz / 32 kHz WAV (Slurm cpu partition)
ssh slurm-login 'mkdir -p ~/logs && cd ~/esp-data-dev && sbatch jobs/resample_dartmouth_avian_soundscapes.sh'
```

## Usage

```python
from esp_data.datasets import DartmouthAvianSoundscapes

ds = DartmouthAvianSoundscapes(split="all", sample_rate=16000)
item = ds[0]
item["audio"]            # np.float32 mono waveform @ 16 kHz
item["selection_table"]  # pandas DataFrame of annotated events
```
