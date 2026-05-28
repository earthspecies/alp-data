# DORI (Phase 1)

Southern Resident Killer Whale & marine-mammal passive-acoustic clips, curated
from 30+ years of public hydrophone archives via positive-unlabelled active
learning (Nestor et al. 2026, arXiv:2602.09295; collection
[DORI-SRKW/dori](https://huggingface.co/collections/DORI-SRKW/dori)).

**Phase 1** ingests the three HuggingFace-hosted sources as clip-level
classification (one ~15 s clip per row). SanctSound (~32k labels, audio not in
the HF repos — only in the NOAA/NCEI archive) is **deferred to Phase 2**.

- **Labels:** predominantly **human-generated** (DORI project) or inherited from
  source programs; a minority are model pseudo-labels. Multi-dimensional:
  species (→ GBIF canonical), orca **ecotype** (SRKW/transient/offshore/NRKW),
  SRKW **call-type** (S1, S44, clicks…), and **presence** (positives vs ONC
  unlabelled negatives).
- **Audio:** the DORI source recordings are long (ONC ≈ 5–7 min / 32 MB each);
  we **crop on download** to the labelled ~15 s window and keep only that, then
  pre-resample to 16 kHz and 32 kHz. Original-rate cropped windows are kept as
  FLAC under `recordings/`.
- **License:** mixed — CC-BY-4.0 / CC-BY-NC-SA-4.0 (Orcasound); see the
  per-row `license` column.

## Splits
`all`, `train`, `test` (the collection's partition), per-source `onc`,
`orcasound`, `ooi`, and `onc_benchmark`. ONC also contributes
~1/5-of-positives **unlabelled negatives** (`is_negative` / `presence=0`).

`onc_benchmark` is the **expert-labelled presence/absence test set** (385 *full*
~5-min recordings, not cropped 15 s clips): `presence` = expert `mammal_present`
(159 present / 226 absent), with three amateur-annotator columns
(`amateur_1/2/3`). License CC-BY-4.0.

## GCS layout
```
gs://esp-data-ingestion/dori/v0.1.0/
├── recordings/<source>/<clip_id>.flac   # cropped window, source sample rate
├── audio_16k/<source>/<clip_id>.wav
├── audio_32k/<source>/<clip_id>.wav
├── metadata/dori_phase1_manifest.csv
├── all.csv  train.csv  test.csv  onc.csv  orcasound.csv  ooi.csv
└── species_labels.csv
```

## Row schema
`clip_id`, `source`, `audio_fp`, `16khz_path`, `32khz_path`, `audio_duration`,
`species` (GBIF canonical; empty for negatives / non-species), `species_common`,
`ecotype`, `call_type`, `presence` (1/0), `label_source`, `license`,
`is_negative`.

## Taxonomy
Species common names → GBIF canonical (orca → *Orcinus orca*, humpback →
*Megaptera novaeangliae*, Pacific white-sided dolphin → *Lagenorhynchus
obliquidens*, fin → *Balaenoptera physalus*, false killer whale → *Pseudorca
crassidens*, sperm → *Physeter macrocephalus*, …). "sea lion" is genus-ambiguous
and `noise`/`uncertain`/`multiple classes` are non-species → empty `species`
(common name retained). Ecotype and call-type are kept as separate columns.

## Reproduction
```bash
# 1. Build the manifest (HF-hosted sources + ONC negatives + GBIF)
uv run --with huggingface_hub --with pandas python \
    scripts/data_preprocessing_scripts/dori_build_manifest.py
# 2. Crop-on-download audio -> GCS (Slurm; needs HF internet + GCS)
ssh slurm-login 'cd ~/esp-data-dev && sbatch jobs/dori_crop_download.sh'
# 3. Build split CSVs
uv run python scripts/data_preprocessing_scripts/dori_build_csv.py
```

## Usage
```python
from esp_data.datasets import DORI
ds = DORI(split="all", sample_rate=16000)   # or train/test/onc/orcasound/ooi
item = ds[0]            # item["audio"], item["species"], item["ecotype"], ...
```

## Notes / caveats
- DORI human labels for ONC are by an **amateur** labeller (Jaccard ~0.7 vs an
  expert; species/ecotype not expert-validated). The `onc_benchmark` split is
  the expert-verified counterpart (presence/absence).
- Phase 2 TODO: SanctSound audio (from NOAA/NCEI); fix the OOI FLAC-decode loss.
