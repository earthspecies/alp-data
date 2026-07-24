# Baringo Soundscapes (western Kenya)

**"A collection of annotated soundscape recordings from western Kenya"**
(Kahl, Reers, Cherutich, Jacot, Klinck; 2024). Zenodo
[10.5281/zenodo.10943500](https://doi.org/10.5281/zenodo.10943500),
**CC-BY-4.0**. Same Cornell/BirdCLEF team as CEB.

35 ~1-hour soundscape recordings (32 h total) from Baringo County, Kenya
(AudioMoth + SWIFT, 2021–22), expert-annotated with **centerpoint** bird-call
labels — 10,294 calls, each marking a call's *center time* (zero-width event) —
for **176 species** (eBird codes → scientific via `species.csv`).

- **Audio:** source is 32 kHz FLAC (Nyquist 16 kHz — native stack fit), but the
  source FLACs trip a libsndfile decoder bug, so they are re-encoded to clean
  **16 kHz + 32 kHz WAV** mirrors (`16khz_path` / `32khz_path`; `audio_fp` → the
  32 kHz WAV). `audio_duration` gives the exact per-recording length.
- **Centerpoints, not boxes:** the `selection_table` events have
  `Begin Time (s) == End Time (s) == center`. This drives `window_annotations`
  (a species is "in" a window when its center falls inside); center→box
  expansion for timestamp tasks is left to the consumer.
- **License:** CC-BY-4.0.

## Row schema (WABAD-shaped; one row per recording)

| column | notes |
|---|---|
| `sound_name` | FLAC basename (`KEN_###_YYYYMMDD_HHMMSS.flac`) |
| `split` | `all` (single split — see provenance) |
| **`audio_duration`** | exact recording duration (seconds) |
| `audio_fp` / `32khz_path` | `audio_32k/<stem>.wav` (clean re-encoded 32 kHz) |
| `16khz_path` | `audio_16k/<stem>.wav` (downsampled mirror) |
| `n_events` / `n_species` | centerpoint count / distinct species |
| `source_dataset` / `license` | `baringo_soundscapes` / `CC-BY-4.0` |
| `selection_table` | TSV: `Selection, Begin Time (s), End Time (s), Low Freq (Hz)=0, High Freq (Hz)=16000, Species, eBird_Code` (Begin==End==center) |

`annotation_columns = ["Species"]`. Vocabulary (scientific + eBird code +
common) in `baringo_soundscapes_labels.csv`.

## Split
Single **`all`** (all 35 recordings, 10,294 centerpoints). See provenance —
this is intended as a held-out soundscape SED eval.

## Distribution
Labels/recording 110–603 (mean 294). Heavy imbalance: Common Bulbul
(*Pycnonotus barbatus*) 702, Red-cheeked Cordon-bleu 508, … 13 singletons,
39 species ≤ 5 labels.

## GCS layout
```
gs://esp-data-ingestion/baringo-soundscapes/v0.1.0/
├── baringo_soundscapes_all.csv
├── baringo_soundscapes_labels.csv
├── audio_16k/<stem>.wav
└── audio_32k/<stem>.wav          (clean re-encoded 32 kHz)
```

## Reproduction
```bash
# annotations.csv/species.csv + soundscape_data.zip downloaded from Zenodo to NFS.
sbatch scripts/data_preprocessing_scripts/baringo_soundscapes/build_baringo_soundscapes.sh
```

## Usage
```python
from esp_data.datasets import BaringoSoundscapes
ds = BaringoSoundscapes(split="all", sample_rate=32000)   # native
item = ds[0]
# item["audio_duration"]; item["selection_table"] = centerpoints (Begin==End).
```

## Notes / caveats
- **Centerpoints** — supports windowed species-presence and center-tolerance
  localization, NOT precise box/timestamp training (a zero-width event renders
  `t–t`). Expand centers to a nominal ±δ box if used for timestamp tasks.
- **Provenance / leakage:** partly used as **2023 BirdCLEF test data**, and the
  176 species overlap the BEANS-Zero held-out taxa broadly (1 species-level +
  19 genus-level: Apalis, Cinnyris, Cisticola, Pycnonotus, Ploceus, Zosterops,
  …). Keep as **eval**; any training use must go behind the
  species/genus/family held-out filter.
- **~1-hour recordings** → windowing essential.
```
