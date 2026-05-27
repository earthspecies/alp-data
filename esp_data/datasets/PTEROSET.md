# PteroSet

A strongly annotated passive-acoustic-monitoring dataset for **tropical bird
monitoring** in Colombia, wrapped as a WABAD-style strong-detection dataset
(one row per recording, each carrying a Raven-style selection table of
annotated events).

- **Source:** Zenodo record [19137071](https://zenodo.org/records/19137071),
  code at [microsoft/PteroSet](https://github.com/microsoft/PteroSet),
  paper arXiv:2605.20578.
- **License:** CC-BY-4.0
- **Audio:** 563 mono recordings (~73.6 h), originally **192 kHz** WAV
  (10-second clips every 30 min over 24-hour cycles).
- **Annotations:** 15,372 expert Raven Pro events; every event is a bird
  vocalisation (`Identification = AVEVOC`), of which 6,702 are identified to
  one of 168 species. Recorded at two Colombian regions (Putumayo, Magdalena),
  2023-2025.

## Splits

Recordings are grouped by monitoring project, each a split, plus `all`:

| split  | project | recordings | region                |
|--------|---------|-----------:|-----------------------|
| `map1` | MAP1    | 46         | Magdalena (Pivijay)   |
| `ppa1` | PPA1    | 108        | Putumayo (Puerto Asís)|
| `ppa2` | PPA2    | 137        | Putumayo              |
| `ppa3` | PPA3    | 151        | Putumayo              |
| `ppa4` | PPA4    | 121        | Putumayo              |
| `all`  | —       | 563        | both                  |

## GCS layout

```
gs://esp-data-ingestion/pteroset/v0.1.0/
├── recordings/*.wav            # 192 kHz originals (flat)
├── audio_16k/*.wav             # pre-resampled 16 kHz PCM16
├── audio_32k/*.wav             # pre-resampled 32 kHz PCM16
├── annotations/*.txt           # raw Raven Pro selection tables (provenance)
├── metadata/                   # metadata.csv, species.csv, annotations_*.json
│   └── species_taxonomy.csv    # PteroSet code -> GBIF canonical + taxonomy
├── all.csv  map1.csv  ppa1.csv  ppa2.csv  ppa3.csv  ppa4.csv
└── species_labels.csv          # resolved canonical-name vocabulary
```

## Row schema (split CSVs)

`fn`, `site`, `project`, `audio_fp` (192 kHz WAV, relative to v0.1.0 root),
`16khz_path`, `32khz_path`, `audio_duration`, `sample_rate`, `date`, `country`,
`department`, `municipality`, `latitude`, `longitude`, `zone`, `land_cover`,
`n_events`, `selection_table`.

### `selection_table` columns

`Begin Time (s)`, `End Time (s)`, `Low Freq (Hz)`, `High Freq (Hz)`,
`Species` (GBIF canonical name; `Unknown` when not identified to species),
`Species Code` (raw PteroSet code from the Raven `Determination` column),
`Identification` (Raven `ID`, e.g. `AVEVOC`), `Type` (Raven `Tipo`, e.g. `BIO`).

## Taxonomy

PteroSet codes (first three letters of genus + species epithet, e.g.
`MEGCHO` = *Megascops choliba*) are mapped to scientific names via `species.csv`
and then to the **GBIF canonical name** via
`esp_data.discover.gbif_taxonomy.GBIFConverter`. The canonical name populates
`Species`; the original code is kept in `Species Code`. Events with an empty or
`INDETE` (indeterminate) `Determination` map to `Species = "Unknown"`.

## Reproduction

```bash
# 1. Stage audio + annotations + metadata to GCS (dev VM; 86 GB parallel download)
WORK=/mnt/home/pteroset_staging bash scripts/data_preprocessing_scripts/pteroset_stage.sh

# 2. Resolve PteroSet codes -> GBIF canonical names
uv run python scripts/data_preprocessing_scripts/pteroset_taxonomy.py

# 3. Build split CSVs + species_labels.csv + QC report
uv run python scripts/data_preprocessing_scripts/pteroset_build_csv.py

# 4. Resample to 16 kHz / 32 kHz WAV (Slurm cpu partition)
ssh slurm-login 'mkdir -p ~/logs && cd ~/esp-data-dev && sbatch jobs/resample_pteroset.sh'
```

## Usage

```python
from esp_data.datasets import PteroSet

ds = PteroSet(split="all", sample_rate=16000)  # or map1 / ppa1 / ppa2 / ppa3 / ppa4
item = ds[0]
item["audio"]            # np.float32 mono waveform @ 16 kHz
item["selection_table"]  # pandas DataFrame of annotated events
```
