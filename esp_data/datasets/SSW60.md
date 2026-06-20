# SSW60 (Sapsucker Woods 60)

A fine-grained **audiovisual** bird dataset of 60 species from Sapsucker
Woods (Ithaca, NY), from Van Horn et al. (2022,
[arXiv:2207.10664](https://arxiv.org/abs/2207.10664); code at
[visipedia/ssw60](https://github.com/visipedia/ssw60)). SSW60 is
esp-data's first **multimodal** dataset — it loads audio, image, *and*
video, including the aligned audio+visual pairs that each video provides.

All four source collections share the same 60-class taxonomy, which we
GBIF-link at build time:

- **audio** — 3,861 Macaulay Library focal recordings (22.05 kHz mono).
- **video** — 5,400 expert-curated clips; each clip is intrinsically
  audio+visual, so its audio track is the only *instance-level*
  audio↔visual pairing in the dataset.
- **image** — 21,600 iNat2021 images + 10,221 NABirds images (31,821
  total).

Cross-modal alignment across the four collections is at the
**species-label level** (the shared 60-class `label`); audio_ml is the
paper's "unpaired audio".

- **License:** research-only. The SSW60 terms forbid redistribution and
  non-research use. The media are staged on internal
  `gs://esp-data-ingestion` for ESP research only — not redistributed.
- **Phase 2 (deferred):** extracting the audio of `reliable_audio` videos
  into standalone audio training clips. For now only the 3,861 stand-alone
  Macaulay clips are exposed as `modality=audio`; video rows still expose
  their own audio track at load time.

## Modality dispatch

Each row carries a `modality` column. `SSW60.__getitem__` dispatches on it:

| modality | loads | fields populated |
|---|---|---|
| `audio` | 22.05 kHz WAV (16 kHz / 32 kHz mirror if available, else resampled on the fly), mono | `audio`, `sample_rate` |
| `image` | image via Pillow, optionally resized to `image_size` | `image` (HWC uint8) |
| `video` | frames via PyAV + the aligned audio track from the same file | `video_frames` (T,H,W,C uint8), `audio`, `sample_rate`, `fps` |

Video decoding requires the optional **`video`** extra (PyAV /
`av`); `read_video` imports it lazily and raises a clear error if it is
missing. Audio-only installs are unaffected.

## Splits

| split | rows | notes |
|---|---:|---|
| `all` | 41,081 | every asset, all modalities |
| `audio_all` / `audio_train` / `audio_test` | 3,861 / 2,597 / 1,264 | Macaulay audio |
| `video_all` / `video_train` / `video_test` | 5,400 / 3,462 / 1,938 | expert videos |
| `image_all` / `image_train` / `image_test` / `image_val` | 31,820 / 23,049 / 8,171 / 600 | iNat2021 + NABirds |

> One NABirds asset (`13e5d907…`, *Setophaga ruticilla*) is listed in the
> upstream `images_nabirds.csv` but its JPG is absent from the tarball, so
> it is dropped from the manifest (31,821 → 31,820).

## Row schema (unified manifest)

| column | type | notes |
|---|---|---|
| `asset_id` | str | Macaulay / iNat / NABirds id |
| `modality` | str | `audio` / `video` / `image` |
| `label` | int | 0-based SSW60 class index (0–59) |
| `species_code` | str | eBird / Clements species code |
| `canonical_name` | str | GBIF canonical scientific name |
| `species_common` | str | English common name |
| `family`, `order` | str | source taxonomy from `taxa.csv` |
| `kingdom`, `phylum`, `class`, `genus`, `gbifID`, `taxonKey` | str | GBIF taxonomy |
| `split` | str | source split (`train` / `test` / `validation`) |
| `audio_path`, `16khz_path`, `32khz_path` | str | absolute `gs://` (audio rows) |
| `image_path` | str | absolute `gs://` (image rows) |
| `video_path` | str | absolute `gs://` (video rows) |
| `fps`, `frame_count`, `duration_seconds`, `frame_height`, `frame_width` | float | video metadata |
| `reliable_audio` | bool | whether the video's audio is curator-reliable |

## GCS layout

```
gs://esp-data-ingestion/ssw60/v0.1.0/
├── audio/<asset_id>.wav   audio_16k/…   audio_32k/…
├── video/<asset_id>.mp4
├── images_inat/<asset_id>.jpg   images_nabirds/<asset_id>.jpg
├── ssw60_all.csv                (unified, modality column)
├── ssw60_audio_{all,train,test}.csv
├── ssw60_video_{all,train,test}.csv
├── ssw60_images_{all,train,test,val}.csv
└── taxa_gbif.csv
```

Built by `scripts/data_preprocessing_scripts/ssw60/build_ssw60.py`
(`jobs/build_ssw60.sh`, Slurm `cpu`).

## References

- Van Horn, Qian, Wilber, Adams, Mac Aodha, Belongie (2022). "Exploring
  Fine-Grained Audiovisual Categorization with the SSW60 Dataset", ECCV.
  [arXiv:2207.10664](https://arxiv.org/abs/2207.10664).
