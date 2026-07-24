# DataSED

**DataSED — Dataset for Sound Event Detection of environmental noise**
(Fredianelli, Artuso, Pompei, Licitra, Iannace, Akbaba; 2025).
Zenodo record [15346092](https://zenodo.org/records/15346092),
DOI [10.5281/zenodo.15346092](https://doi.org/10.5281/zenodo.15346092).

A WABAD-shaped strong (time-localised) sound-event-detection dataset of
**717 non-synthesised WAV recordings** (44.1 kHz, mono, PCM_16, ~17 h
total, 2.3–285 s each) gathered from sound-level measurements and online
repositories across urban→rural environments in Italy.

- **Labels:** human-annotated by experts. Each source row is one event
  with absolute begin/end seconds. **Not exhaustively labelled** —
  unlabelled spans are *not* guaranteed negative; treat only annotated
  spans as positives.
- **Audio:** original 44.1 kHz WAV at `audio/`, with pre-resampled 16 kHz
  and 32 kHz mono mirrors at `audio_16k/` / `audio_32k/`. 32 kHz is the
  recommended default (broadband environmental sound).
- **License:** CC-BY-NC-SA-4.0 (**non-commercial** and **share-alike**).
  Keep private; any redistributed derivative inherits the share-alike
  obligation.

## Two annotation schemes (exposed as split families)

| scheme | recordings | events | classes | notes |
|---|---:|---:|---:|---|
| **poly** (headline) | 703 | 4,034 | 21 | overlapping multi-class events, realistic; excludes `Wind turbine` |
| **mono** | 717 | 4,309 | 22 | one class at a time (non-overlapping) |

## Row schema (one row per recording, WABAD-shaped)

| column | notes |
|---|---|
| `sound_name` | recording basename, e.g. `S-0001.wav` |
| `audio_fp` | relative path to original 44.1 kHz WAV (`audio/S-0001.wav`) |
| `16khz_path` / `32khz_path` | relative paths to pre-resampled mirrors |
| `split` | `train` / `val` (recording-level, seeded 90/10, shared across schemes) |
| `n_events` | number of events in the selection table |
| `source_dataset` | `datased` |
| `license` | `CC-BY-NC-SA-4.0` |
| `selection_table` | tab-separated Raven blob: `Selection`, `Begin Time (s)`, `End Time (s)`, `Label` |

`annotation_columns = ["Label"]`. The 22-class vocabulary is in
`datased_labels.csv` (`Label`, `label_snake`). At load time the loader
parses the `selection_table` into a pandas DataFrame and drops events
beginning at/after the (optionally windowed) audio end.

## Splits
`poly_all` · `poly_train` · `poly_val` · `mono_all` · `mono_train` ·
`mono_val`. Default is `poly_train`. Train/val is a single seeded (42)
recording-level 90/10 split shared across both schemes, so no recording
crosses between train and val.

## GCS layout
```
gs://esp-data-ingestion/datased/v0.1.0/
├── datased_poly_all.csv   datased_poly_train.csv   datased_poly_val.csv
├── datased_mono_all.csv   datased_mono_train.csv   datased_mono_val.csv
├── datased_labels.csv
├── audio/S-0001.wav ...          (original 44.1 kHz)
├── audio_16k/S-0001.wav ...      (pre-resampled 16 kHz)
└── audio_32k/S-0001.wav ...      (pre-resampled 32 kHz)
```

## Reproduction
```bash
# 1. Download the 4.5 GB zip from Zenodo to NFS (md5 44e093f675fc44cfb8a11b68456b72d7).
# 2. Build audio mirrors + manifests on Slurm (avoids OOM on the dev VM):
sbatch scripts/data_preprocessing_scripts/datased/build_datased.sh
```
The build extracts the WAVs, uploads originals, writes 16k/32k mirrors,
and (re)builds + uploads the manifests. Metadata-only rebuilds:
```bash
uv run python scripts/data_preprocessing_scripts/datased/build_datased.py \
    manifests --csv-dir <csv-dir> --out-dir <out-dir>
```

## Usage
```python
from esp_data.datasets import DataSED
ds = DataSED(split="poly_val", sample_rate=32000)
item = ds[0]
# item["audio"] is the recording at 32 kHz; item["selection_table"] is a
# DataFrame of [Selection, Begin Time (s), End Time (s), Label] events.
```

## Notes / caveats
- **Partial labelling** — do not mine unlabelled time as hard negatives.
- **Class-name anomaly** — a few source events have `end_time` beyond the
  recording duration; the loader clips events past the audio end.
- **22 vs 21 classes** — `Wind turbine` appears only in the monophonic
  scheme (14 recordings are monophonic-only).
