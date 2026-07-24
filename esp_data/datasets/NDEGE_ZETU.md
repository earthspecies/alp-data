# Ndege Zetu

**"Ndege Zetu: comparing bird species monitoring approaches in the Mt Kenya
ecosystem"** (wa Maina et al. / DeKUT-DSAIL, 2025). Dryad
[10.5061/dryad.d51c5b0c7](https://doi.org/10.5061/dryad.d51c5b0c7); primary
article [Phil Trans R Soc B 2024.0057](https://doi.org/10.1098/rstb.2024.0057);
code Zenodo 13342281 / GitHub `DeKUT-DSAIL/ndege-zetu`.

Autonomous-recording-unit (ARU) bird soundscapes from two Mt Kenya sites —
Dedan Kimathi University Wildlife Conservancy (**DeKUWC**) and Mt Kenya National
Park (**MKNP**) — annotated by expert ornithologists.

- **This is a WEAK, clip-level dataset — not detection.** Each ~1-minute
  recording carries foreground + background species presence, with **no time
  localization** (no centerpoints, no boxes). ("Point counts" in the source are
  a spatial field-survey method; that survey data is not ingested here.)
- **3,893 recordings, 100 species** (common names mapped to scientific via the
  dataset's own `Kenya-Species-List.csv`). ~30% have ≥1 species; the rest are
  negatives.
- **Audio:** 16 kHz mono MP3, ~60 s (**Nyquist 8 kHz** — high-pitched species
  are frequency-capped). Original MP3 at `audio/`; 16 kHz (native) and 32 kHz
  (upsampled) WAV mirrors at `audio_16k/` / `audio_32k/`.
- **License:** CC0-1.0.

## Row schema (WABAD-shaped, weak; one row per recording)

| column | notes |
|---|---|
| `sound_name` | MP3 basename |
| `site` | `dekuwc` / `mknp` |
| `aru_batch` | `dekuwc_2016` (reuse of Dryad `69g60`) / `dekuwc_2017` / `mknp_2017_2018` |
| `split` | `train` / `val` / `test` |
| **`audio_duration_sec`** | exact per-recording duration (seconds) |
| `audio_fp` | `audio/<basename>.mp3` (16 kHz original) |
| `16khz_path` / `32khz_path` | pre-resampled WAV mirrors |
| `foreground_species` / `background_species` | `;`-joined scientific names |
| `n_species` | foreground + background count |
| `remarks` | annotator notes (~5%) |
| `source_dataset` / `license` | `ndege_zetu` / `CC0-1.0` |
| `selection_table` | full-clip weak boxes: `Selection, Begin Time (s)=0, End Time (s)=duration, Low Freq (Hz)=0, High Freq (Hz)=8000, Species, Presence` |

`annotation_columns = ["Species"]`. The `selection_table` holds one **full-clip**
weak box per species (a `Presence` column marks foreground vs background), so
the recording flows through `window_annotations` — but every window inherits the
whole-clip label (no localization). Vocabulary in `ndege_zetu_labels.csv`.

## Splits
Deterministic per-recording hashed ~80/10/10 (seed 42), proportional across
sites: `train` ~3,103 · `val` ~373 · `test` ~417 · `all` 3,893. ~30% of each
split carry ≥1 species.

## GCS layout
```
gs://esp-data-ingestion/ndege-zetu/v0.1.0/
├── ndege_zetu_{all,train,val,test}.csv
├── ndege_zetu_labels.csv
├── audio/<basename>.mp3          (16 kHz original)
├── audio_16k/<stem>.wav
└── audio_32k/<stem>.wav
```

## Reproduction
```bash
# zip downloaded from Dryad via OAuth (POST https://datadryad.org/oauth/token) to NFS.
sbatch scripts/data_preprocessing_scripts/ndege_zetu/build_ndege_zetu.sh
```

## Usage
```python
from esp_data.datasets import NdegeZetu
ds = NdegeZetu(split="val", sample_rate=16000)   # source is 16 kHz
item = ds[0]
# item["audio"]; item["audio_duration_sec"]; item["foreground_species"];
# item["selection_table"] = weak full-clip boxes w/ Presence column.
```

## Notes / caveats
- **Weak, not localized** — best used with weak multi-label species templates
  (`species_scientific` / multilabel), not the strong `species_timestamps` /
  `species_bboxes` SED tasks. Windowing inherits the clip label (noisy).
- **Sparse + imbalanced** — ~70% of recordings have no species (large negative
  pool → downsample); Brown Woodland Warbler (*Phylloscopus umbrovirens*)
  dominates the positives.
- **8 kHz Nyquist** — 16 kHz source caps high-frequency species content.
- **2016 subset** (`aru_batch == dekuwc_2016`, 2,701 clips) is a re-use of Dryad
  `69g60`; the `aru_batch` column lets you exclude it if it overlaps other data.
- **Leakage:** check the 100 Kenyan species against `beans_zero_heldout_taxa.yml`
  before training.
```
