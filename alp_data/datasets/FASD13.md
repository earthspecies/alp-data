# FASD13

**"Fewshot Animal Sound Detection 13"** (Hoffman, Robinson, Miron, Baglione,
Canestrari, Elias, Trapote, Effenberger, Cusimano, Hagiwara, Keen, Pietquin;
2025). Zenodo [10.5281/zenodo.15843741](https://doi.org/10.5281/zenodo.15843741),
primary article *Synthetic data enables context-aware bioacoustic sound event
detection* ([arXiv:2503.00296](https://arxiv.org/abs/2503.00296)).

The **evaluation** benchmark from the DRASDIC paper: 13 bioacoustics
sub-datasets, **109 recordings, ~143 h, 11,315 annotated events**, each
recording paired with onsets/offsets of *one* predetermined target category.
Deliberately eval-only — it complements the DCASE few-shot train/val sets.

> Not to be confused with `alp_data.datasets.drasdic` (`DRASDIC`), which is the
> **synthetic training** corpus from the same paper. FASD13 shares no audio
> with it.

- **Labels:** a single, *nameless* positive class per recording. In few-shot
  detection the target is defined by the support examples, not by a name, so
  `Label` is the constant `"target"`; the `detection_target` column records
  what kind of category each sub-dataset annotates.
- **Audio:** native rates 16–96 kHz, some sources stereo. Pre-resampled mono
  mirrors at `audio_16k/` and `audio_32k/`. **32 kHz recommended.**
- **License:** **mixed, per sub-dataset** — see the table below and the
  per-row `license` / `citation` columns. `MS` requires citing two specific
  papers. `GS` and `HG` are non-commercial.

## Sub-datasets

| code | name | files | dur (h) | events | recording type | location | taxa | detection target | license |
|---|---|---|---|---|---|---|---|---|---|
| `AS` | AnuraSet | 12 | 0.20 | 162 | TPAM | Brazil | Anura | Species | CC-BY-1.0 |
| `CC` | Carrion Crow | 10 | 10.00 | 2200 | On-body | Spain | *Corvus corone* + *Clamator glandarius* | Species+Life Stage | CC-BY-SA-4.0 |
| `GS` | Gunshot | 7 | 38.33 | 85 | TPAM | Gabon | *Homo sapiens* | Production Mechanism | CC-BY-NC-4.0 |
| `HA` | Hawaiian Birds | 12 | 1.10 | 628 | TPAM | Hawaii, USA | Aves | Species | CC-BY-4.0 |
| `HG` | Hainan Gibbons | 9 | 72.00 | 483 | TPAM | Hainan, China | *Nomascus hainanus* | Species | CC-BY-NC-SA-4.0 |
| `HW` | Humpback Whale | 10 | 2.79 | 1565 | UPAM | North Pacific | *Megaptera novaeangliae* | Species | public domain |
| `JS` | Jumping Spider | 4 | 0.23 | 924 | Substrate | Laboratory | *Habronattus* | Sound Type | CC-BY-SA-4.0 |
| `KD` | Katydid | 12 | 2.00 | 883 | TPAM | Panamá | Tettigoniidae | Species | public domain |
| `MS` | Marmoset | 10 | 1.67 | 1369 | Laboratory | Laboratory | *Callithrix jacchus* | Call Type | custom (cite) |
| `PM` | Powdermill | 4 | 6.42 | 2032 | TPAM | Pennsylvania, USA | Passeriformes | Species | public domain |
| `RG` | Ruffed Grouse | 2 | 1.50 | 34 | TPAM | Pennsylvania, USA | *Bonasa umbellus* | Species | public domain |
| `RS` | Rana Sierrae | 7 | 1.87 | 552 | UPAM | California, USA | *Rana sierrae* | Species | public domain |
| `RW` | Right Whale | 10 | 5.00 | 398 | UPAM | Gulf of St. Lawrence | *Eubalaena glacialis* | Species | CC-BY-4.0 |

`CC` and `JS` are published for the first time in FASD13; the other 11 are
re-releases (see `LICENSE.txt` on GCS for the original citations).

Splits are `all` plus the 13 codes. Note the extreme duration imbalance: `HG`
(72 h) and `GS` (38 h) are 77 % of the benchmark between them, so cap or
stratify per sub-dataset before pooling.

## The N-shot protocol

From `drasdic/data/test.py` (earthspecies/drasdic), following Nolasco et al.
2023:

1. Order events by **end time**; exclude `UNK` from shot selection.
2. The **support region** is the audio up through the end of the Nth event.
3. The **query region** is everything after it — that, and only that, is scored.
4. `UNK` events are neither positives nor negatives and are **masked out of
   scoring**.

Each row carries `shot_end_times` (the end times of the first 5 non-`UNK`
events) so any N ≤ 5 can be derived without re-reading the annotations.

## Row schema (WABAD-shaped, one row per recording)

| column | notes |
|---|---|
| `sound_name` | wav basename |
| `subdataset` / `subdataset_name` | e.g. `PM` / `Powdermill` |
| `split` | same as `subdataset` |
| `audio_duration` | seconds |
| `audio_fp` | `audio_32k/<CODE>/<stem>.wav` (default originals pointer) |
| `16khz_path` / `32khz_path` | pre-resampled mono mirrors |
| `native_sample_rate` / `channels` | as shipped by Zenodo |
| `n_events` / `n_pos` / `n_unk` | annotation counts |
| `n_shots_available` / `shot_end_times` | N-shot protocol support |
| `taxa` / `detection_target` / `recording_type` / `location` | sub-dataset metadata |
| `license` / `citation` | **per sub-dataset** |
| `selection_table` | Raven-shaped TSV (below) |

Selection table columns: `Selection`, `Begin Time (s)`, `End Time (s)`, `Q`,
`Label`, `event_index`. `Q` ∈ {`POS`, `UNK`, `NEG`}; `event_index` numbers the
non-`UNK` events in end-time order (UNK gets `-1`). There are no frequency
bounds — FASD13 is time-only, so frequency-bbox tasks are not available.

## Few-shot support clips

`support_16k/` and `support_32k/` hold **5 pre-cut 10 s support clips per
recording** (545 clips total), indexed by `fasd13_support.csv`. Clip *k* is
built from event *k* alone — placed one third of the way into the window,
mirroring DRASDIC's `subselect_support_fixed` — so an N-shot episode just takes
clips 1..N and one build serves the 1-, 4- and 5-shot configs.

`fasd13_support.csv` columns: `subdataset`, `sound_name`, `shot_index`,
`support_16khz_path`, `support_32khz_path`, `window_start_sec`,
`window_end_sec`, `clip_duration`, `event_times`, `n_events_in_clip`,
`n_unk_in_clip`. `event_times` is pre-rendered in the wording NatureLM was
trained on (`"1.7s-2.9s, 5.8s-6.6s"`).

Two caveats worth knowing:

- **Near-duplicate support clips.** Where events are dense (notably `JS`, 924
  events in 350 s), the first 5 events fall within a fraction of a second of
  each other, so the 5 clips overlap almost completely. This is inherent to
  "the first N events" and matches what DRASDIC itself sees.
- **Support region bound.** Clips are bounded by the end of the Nth event, never by the recording length, so they can never contain query-region audio (the earlier unbounded version leaked 442 query events into 250 of 545 clips). Where the 5-event prefix is shorter than 10 s the whole prefix is used, giving a shorter clip.
- **`UNK` inside support clips.** Clips keep contiguous audio, so an `UNK`
  event can be audible but is deliberately absent from `event_times`.
  `n_unk_in_clip` flags these. (DRASDIC instead excises `UNK` samples, which
  breaks contiguity.)

## Multi-audio mode

When a row carries `support_16khz_paths` / `support_32khz_paths` (set by the
few-shot episode transform), `FASD13` returns an `audios` list — the support
clips followed by the query window — instead of a single `audio`, matching the
`BeansProMultiAudio` convention. Otherwise it behaves like WABAD: one `audio`
per recording, honouring `window_start_sec` / `window_end_sec` for lazy
windowed reads.

## Build

```
sbatch scripts/data_preprocessing_scripts/fasd13/build_fasd13.sh     # download → resample → manifests → support → upload
sbatch scripts/data_preprocessing_scripts/fasd13/validate_fasd13.sh  # cross-check GCS + smoke-load
```

GCS root: `gs://esp-data-ingestion/fasd13/v0.1.0/`.
