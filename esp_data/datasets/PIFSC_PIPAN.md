# PIFSC PIPAN

Annotations from the NOAA Pacific Islands Passive Acoustic Network
(PIPAN) 10 kHz decimated HARP archive (DOI
[10.25921/Z787-9Y54](https://doi.org/10.25921/Z787-9Y54)), labelled by
Allen et al. (2021,
[doi:10.3389/fmars.2021.607321](https://doi.org/10.3389/fmars.2021.607321)).

**Phase 1** ingests the public Allen 2021 annotations — humpback whale
song (`Mn`) plus negative / non-target classes (`Background`, `Other`,
`Vessel`, `Fish`, `Device`) — across 16 PIFSC HARP deployments from 2005
to 2018 (Wake Atoll, Saipan, Tinian, Hawaiian Islands, Palmyra Atoll,
Cross Seamount, Pearl & Hermes, Kingman Reef, etc.).

**Phase 2 (deferred)** will add:
- Allen 2024 Bryde's whale biotwang annotations (Saipan + Tinian
  HARP deployments, Front. Mar. Sci. 11:1394695)
- Additional NOAA Perch 2.0 multi-species labels (orca, common minke,
  sei, blue, fin, anthropogenic, unknown) — from the same DOI but
  not yet in the public bucket as of 2026-06.

- **Labels:** **human-generated** by the Allen / PIFSC bioacoustics
  group across multiple audit rounds (`audit_name` enum: `initial`,
  `validation`, `postpub`, `testexpand`, `segments`, plus four
  model-output review rounds `model1..4`).
- **Audio:** the source archive is 10 kHz decimated FLAC-compressed
  XWAV. PIPAN records continuously then archives 75-second
  sub-chunks separated by deployment-specific duty-cycle gaps. We
  parse the XWAV foreign-metadata at build time and persist exact
  `begin_in_file_s` / `end_in_file_s` so consumers never see XWAV.
- **License:** CC0-1.0.

## Row schema (one row per analyst event)

| column | type | notes |
|---|---|---|
| `audio_path` | str | absolute `gs://` to the source 10 kHz FLAC |
| `16khz_path` / `32khz_path` | str | absolute `gs://` to pre-resampled WAV; empty when not pre-resampled (~77% of files lack mirrors as of Phase 1) |
| `deployment` | str | one of 16 site slugs (saipan / wake / hawaii / kauai / palmyra_wt / …) |
| `xwav_subchunk_index` | int | 0-based subchunk inside the source XWAV |
| `begin_in_subchunk_s` / `end_in_subchunk_s` | float | event time relative to the subchunk start; equal for weak (subchunk-level) labels |
| `begin_in_file_s` / `end_in_file_s` | float | event time in the decoded continuous audio stream; the loader uses these as the window |
| `begin_utc` / `end_utc` | str | ISO-8601 UTC |
| `label` | enum | `Mn` (humpback song), `Background`, `Other`, `Vessel`, `Fish`, `Device` |
| `label_is_strong` | bool | True ⇒ tight time bounds; False ⇒ event-present-in-subchunk only |
| `implicit_negatives` | bool | True ⇒ unlabelled time in this subchunk is *known* to not contain this label |
| `audit_name` | str | annotation effort that produced the row |
| `coarse_call_type` | str | `song` / `noise` / `fish` / `background` / `other` |
| `species` | str | `Megaptera novaeangliae` for `Mn`; empty otherwise |
| `canonical_name`, `gbifID`, `kingdom`..`species_common` | str | GBIF taxonomy fields, populated only for `Mn` |
| `license` | str | `CC0-1.0` |
| `source_dataset` | str | `pifsc_pipan` |

## Splits
`all`, `train`, `val`. Train/val is **deployment-stratified 90/10 at
the audio-file level** — events from one FLAC stay in one split, but
every deployment contributes to both. Seeded with `42`.

## GCS layout
```
gs://esp-data-ingestion/pifsc-pipan/v0.1.0/
├── pifsc_pipan_all.csv          (38,857 events)
├── pifsc_pipan_train.csv        (~35k)
├── pifsc_pipan_val.csv          (~3.9k)
├── pifsc_pipan_labels.csv       (label vocabulary)
└── pifsc_pipan_xwav_index.csv   (per-FLAC subchunk summary; side artifact)
```

Audio physically lives at
`gs://esp-data-ingestion/superwhale/v0.1.0/raw/pifsc/audio/pipan_10/...`
(and the partial 16k/32k mirrors at `raw/audio_{16k,32k}/pifsc/...`);
the manifest holds absolute `gs://` paths so no `data_root` is needed.

## Label distribution (Allen 2021 vocabulary)

| label | count | strong | weak | description |
|---|---:|---:|---:|---|
| `Mn` | 32,189 | 30,894 | 1,295 | humpback whale song (*Megaptera novaeangliae*) |
| `Background` | 5,928 | 0 | 5,928 | environmental sound only |
| `Other` | 306 | 306 | 0 | acoustic event not in the other categories |
| `Vessel` | 242 | 0 | 242 | vessel noise |
| `Fish` | 98 | 0 | 98 | fish sound not otherwise identified |
| `Device` | 94 | 0 | 94 | recording-equipment noise |

## Deployments

Top deployments (by event count): hawaii (7,867) · kauai (7,036) ·
wake (4,316) · saipan (3,422) · phr_a (3,412) · crosssm (2,716) ·
tinian (2,361) · phr_b (2,059) · kingman (1,752) · laddsm_s (1,309) ·
equator (863) · palmyra_wt (611) · laddsm_d (512) · pagan (343) ·
palmyra_ns (239) · howland (39).

## Reproduction

```bash
# 1. Build the manifest (parses XWAV headers, joins GBIF taxonomy)
uv run python scripts/data_preprocessing_scripts/pifsc_pipan/build_pifsc_pipan.py \
    --out-dir ~/pifsc_pipan_staging --workers 8 --upload
```

The build is metadata-only: no audio re-processing, ~15 min on the
dev VM (5,489 unique FLAC headers parsed in parallel). All audio
references stay in the existing
`gs://esp-data-ingestion/superwhale/v0.1.0/raw/pifsc/...` mirror.

## Usage

```python
from esp_data.datasets import PIFSCPipan
ds = PIFSCPipan(split="val", sample_rate=16000)
item = ds[0]
# item["audio"] is the windowed event audio at 16 kHz,
# item["label"], item["label_is_strong"], item["audit_name"] carry provenance.
```

## Notes / caveats

- **Strong vs weak.** All `Background`, `Vessel`, `Fish`, `Device`
  events are subchunk-level (`label_is_strong=False`,
  `begin_in_subchunk_s == end_in_subchunk_s`). Treat these as
  "label present somewhere in the subchunk" — the loader widens
  zero-width events to ~10 s.
- **Pre-resampled coverage.** ~23% of unique FLACs have parallel
  16/32 kHz mirrors; the rest resample on the fly (cheap upsample
  from 10 kHz). Run a separate Slurm batch to fill in the
  pre-resampled mirror if needed for throughput-critical training.
- **Phase 2 sourcing.** Bryde's + Perch additional species are NOT
  in the public bucket (last updated 2023-08-18). To extend,
  source from the Allen 2024 paper or contact PIFSC.
- **Overlap with SuperWhales.** The legacy
  `superwhales_detection.py:217` entry (1,268 audio files / 16,338
  events) is a strict subset of the events here and remains
  available for backwards compatibility.
