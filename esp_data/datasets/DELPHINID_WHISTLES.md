# Delphinid Whistles

**"Bounding-box detection data for delphinid whistles"** (Ferguson, Alongi,
Sugarman, Pettis Schallert, Lyn; 2025). Dryad
[10.5061/dryad.z34tmpgq6](https://doi.org/10.5061/dryad.z34tmpgq6),
primary article JASA 157(6):4613
([10.1121/10.0036942](https://doi.org/10.1121/10.0036942)).

Time+frequency **bounding-box** detection of bottlenose dolphin
(*Tursiops truncatus*) whistles across four acoustic environments — two
aquarium and two open ocean. Annotated with DeepAcoustics (a DeepSqueak-derived
YOLO spectrogram detector); annotations shipped as Raven selection tables
(`.txt`) + `.mat` Calls. **8,576 whistle boxes over 96 recordings.**

- **Labels:** single species, `Tursiops truncatus` (source `Annotation` = "Whistle").
- **Audio:** 48 kHz mono; whistle boxes annotated on a 0–24 kHz view. Original
  48 kHz at `audio/`, pre-resampled 16 kHz / 32 kHz mirrors at
  `audio_16k/` / `audio_32k/`. **32 kHz recommended.**
- **License:** CC0-1.0 (public domain — no redistribution constraints).

## Sites

| site slug | source | audio layout |
|---|---|---|
| `aquarium_imms` | Institute for Marine Mammal Studies, Gulfport MS | merged multi-hour wav(s) |
| `aquarium_oceanografic` | Oceanogràfic Foundation, Valencia (labelled "Valencia") | merged multi-hour wav(s) |
| `openocean_dclde2011` | DCLDE 2011 workshop | many native wavs, one multi-file `.txt` |
| `openocean_swfsc` | NOAA SWFSC towed array (labelled "TowedArray") | many native wavs, one multi-file `.txt` |

Recording granularity = one **audio file**. Open-ocean selection tables are
multi-file: events are assigned to their `Begin File` and timed by
`File Offset (s)` (within-file begin) + duration; aquarium tables use
`Begin/End Time (s)` on the merged wav. Open-ocean wavs with **no whistles are
kept as pure-negative recordings** (8 of them).

## Row schema (WABAD-shaped, one row per recording)

| column | notes |
|---|---|
| `sound_name` | wav basename |
| `site` | one of the four slugs above |
| `split` | `train` / `val` / `test` |
| `audio_fp` | `audio/<site>/<basename>` (48 kHz original) |
| `16khz_path` / `32khz_path` | pre-resampled mirrors |
| `n_events` | number of whistle boxes in the selection table |
| `source_dataset` | `delphinid_whistles` |
| `license` | `CC0-1.0` |
| `selection_table` | TSV: `Selection, Begin Time (s), End Time (s), Low Freq (Hz), High Freq (Hz), Species` |

`annotation_columns = ["Species"]`; `Species` is always `Tursiops truncatus`.
Times in the selection table are **within-file** (recording-relative).

## Splits
`train` (75 recs / 6,450 boxes) · `val` (12 recs, seeded 15% open-ocean holdout)
· `test` (9 recs / 1,551 boxes, the dataset's native held-out set) · `all` (96).
Aquarium has only 2 merged training recordings per site, so it stays entirely
in `train` (val is carved from the many open-ocean files only).

## Frequency distribution vs the 16 kHz Nyquist ceiling
Boxes annotated up to 24 kHz. Across all 8,576 boxes: Low Freq median ~6.7 kHz
(~99% start below 16 kHz), High Freq median ~16.6 kHz. **At a 16 kHz ceiling:
~47% fully below, ~52% straddle (top clipped), ~1% fully above** — aquarium
~62% fully below, open ocean only ~30%.

Implications for training:
- **Time-only detection** (`species_timestamps` / `species_event_times` /
  `species_query_timestamps` + time-pixel): robust across all sites — the
  whistle contour is in band for ~99% of boxes.
- **Frequency-bbox / pixel-bbox** (`species_bboxes` / `species_pixel_bboxes`):
  **filter to `High Freq (Hz) <= 16000`** (drops straddling boxes; keeps ~47%,
  aquarium-skewed) rather than clamping, to avoid ceiling-pinned targets.
- The data is natively 48 kHz / 24 kHz-content: if the encoder ever moves to
  48 kHz, ~100% of boxes become fully captured.

## GCS layout
```
gs://esp-data-ingestion/delphinid-whistles/v0.1.0/
├── delphinid_whistles_{all,train,val,test}.csv
├── delphinid_whistles_labels.csv
├── audio/<site>/<basename>.wav        (48 kHz original)
├── audio_16k/<site>/<basename>.wav
└── audio_32k/<site>/<basename>.wav
```

## Reproduction
```bash
# zips downloaded from Dryad via OAuth (POST https://datadryad.org/oauth/token
# with client_id/secret/grant_type=client_credentials -> bearer; then
# Authorization: Bearer on /api/v2/files/{id}/download) to NFS staging.
sbatch scripts/data_preprocessing_scripts/delphinid_whistles/build_delphinid_whistles.sh
```

## Usage
```python
from esp_data.datasets import DelphinidWhistles
ds = DelphinidWhistles(split="test", sample_rate=32000)
item = ds[0]
# item["audio"] @ 32 kHz; item["selection_table"] has
# [Selection, Begin Time (s), End Time (s), Low Freq (Hz), High Freq (Hz), Species]
```

## Notes / caveats
- **Nyquist clip** is the key caveat — see the frequency section. Time-only SED
  is the primary contribution; freq/pixel-bbox is a filtered, aquarium-weighted bonus.
- **Leakage:** *Tursiops truncatus* is not a BEANS-Zero held-out taxon (safe to
  train), but check recording overlap with `dinardo_dolphin_whistles` and
  `superwhales_detection` (both Tursiops; SuperWhales also draws on DCLDE/SWFSC
  sources) before using any Tursiops split as an eval.
- **Annotation completeness:** DeepAcoustics ground truth is exhaustive per
  analysed file, so empty windows are valid negatives (unlike DataSED's partial
  labelling) — the 8 no-whistle open-ocean recordings are genuine negatives.
```
