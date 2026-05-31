# /// script
# requires-python = ">=3.10"
# dependencies = ["soundfile", "google-cloud-storage"]
# ///
"""One-off sample-rate audit of resampled outputs."""

import io

import soundfile as sf
from google.cloud import storage

paths = [
    "synthetic/synthetic_sed_scenes_32k/audio/scene_0.wav",
    "synthetic/synthetic_sed_diarization_32k/audio/diar_scene_0.wav",
    "synthetic/animalspeak_pseudovox_32k/XC552898-ZOOM0038_source0_clip3.wav",
    "synthetic/cropped/wabad/audio_32k/CAT_20210304_071800__crop_0000000_0019543.wav",
]

c = storage.Client()
b = c.bucket("foundation-model-data")
for p in paths:
    blob = b.blob(p)
    if not blob.exists():
        print(f"{p}: MISSING")
        continue
    blob.reload()
    raw = blob.download_as_bytes()
    info = sf.info(io.BytesIO(raw))
    print(
        f"{p}: sr={info.samplerate} dur={info.duration:.2f}s ch={info.channels} "
        f"bytes={len(raw)} created={blob.time_created}"
    )
