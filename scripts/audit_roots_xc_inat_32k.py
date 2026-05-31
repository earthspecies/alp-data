# /// script
# requires-python = ">=3.10"
# dependencies = ["gcsfs", "google-cloud-storage"]
# ///
"""Audit XC and iNat 32 kHz mirrors against ROOTS audio_paths.

For a sample of rows from each XC/iNat T1 v2 split, compute the joined
audio URI, then enumerate plausible 32 kHz mirror candidates and check
which (if any) actually exist on GCS. Used to decide the prefix +
extension rewrite rule before patching ROOTS' `_prefer_presampled`.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import gcsfs

XC_ROOT = "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/audio"
INAT_ROOT = "gs://esp-ml-datasets/inaturalist/v0.1.0/raw"
T1_ROOT = "gs://foundation-model-data/synthetic/train_final/T1_with_captions_train_v2"

XC_SPLITS = (
    "acoustic_caption_field_notes_xc_train_unseen_v2_clean.jsonl",
    "cat_snr_mcq_custom_bins_xc_train_unseen_v1_clean.jsonl",
    "cat_snr_mcq_xc_train_unseen_v2_clean.jsonl",
    "snr_binary_xc_train_unseen_v1_clean.jsonl",
    "snr_oe_xc_train_unseen_v1_clean.jsonl",
    "voc_desc_mcq_field_notes_xc_train_unseen_v1_clean.jsonl",
)
INAT_SPLITS = (
    "acoustic_caption_field_notes_inat_train_unseen_v2_clean.jsonl",
    "cat_snr_mcq_inat_train_unseen_v2_clean.jsonl",
    "voc_desc_mcq_field_notes_inat_train_unseen_v1_clean.jsonl",
)

SAMPLE_PER_SPLIT = 5


def xc_candidates(joined: str) -> list[str]:
    # joined like .../raw/audio/Eurasian Reed Warbler/XC470988-….mp3
    if "/raw/audio/" not in joined:
        return []
    base = joined.replace("/raw/audio/", "/raw/audio_32k/", 1)
    cands: list[str] = []
    # Mirror with same relative path under audio_32k, original extension and .wav
    cands.append(base)
    cands.append(re.sub(r"\.[^./]+$", ".wav", base))
    # Mirror may be flat — drop species subdir
    tail = base.rsplit("/", 1)[-1]
    cands.append(f"gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/audio_32k/{tail}")
    cands.append(
        re.sub(r"\.[^./]+$", ".wav", f"gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/audio_32k/{tail}")
    )
    # Dedupe preserving order
    seen, dedup = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def inat_candidates(joined: str) -> list[str]:
    # joined like .../raw/audio_16k/audio/674197.wav or .../raw/audio_16k/413/inat_41396773.wav
    cands: list[str] = []
    # Direct swap audio_16k -> audio_32k
    if "/raw/audio_16k/" in joined:
        cands.append(joined.replace("/raw/audio_16k/", "/raw/audio_32k/", 1))
    # Direct swap audio_16k -> audio_32khz (flat)
    if "/raw/audio_16k/audio/" in joined:
        # flat audio_32khz/<id>.wav
        tail = joined.rsplit("/audio_16k/audio/", 1)[-1]
        cands.append(f"gs://esp-ml-datasets/inaturalist/v0.1.0/raw/audio_32khz/{tail}")
    if "/raw/audio_16k/" in joined and "/raw/audio_16k/audio/" not in joined:
        # buckets: audio_16k/413/inat_<id>.wav -> audio_32khz/inat_<id>.wav?
        tail = joined.rsplit("/audio_16k/", 1)[-1].split("/", 1)[-1]
        cands.append(f"gs://esp-ml-datasets/inaturalist/v0.1.0/raw/audio_32khz/{tail}")
    seen, dedup = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def main() -> None:
    fs = gcsfs.GCSFileSystem()
    print("==== XC ====", flush=True)
    xc_pattern_hits = Counter()
    for split in XC_SPLITS:
        path = f"{T1_ROOT}/{split}"
        print(f"\n-- {split}", flush=True)
        with fs.open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= SAMPLE_PER_SPLIT:
                    break
                row = json.loads(line)
                rel = row["audio_paths"][0]
                joined = f"{XC_ROOT}/{rel}"
                cands = xc_candidates(joined)
                hit_index = None
                for j, c in enumerate(cands):
                    if fs.exists(c):
                        hit_index = j
                        break
                tag = f"cand{hit_index}" if hit_index is not None else "MISS"
                xc_pattern_hits[tag] += 1
                print(f"  rel={rel[:80]}", flush=True)
                if hit_index is not None:
                    print(f"     -> {cands[hit_index]}", flush=True)
                else:
                    print(f"     MISS; candidates: {cands}", flush=True)
    print(f"\nXC pattern hits: {dict(xc_pattern_hits)}", flush=True)

    print("\n==== iNat ====", flush=True)
    inat_pattern_hits = Counter()
    for split in INAT_SPLITS:
        path = f"{T1_ROOT}/{split}"
        print(f"\n-- {split}", flush=True)
        with fs.open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= SAMPLE_PER_SPLIT:
                    break
                row = json.loads(line)
                rel = row["audio_paths"][0]
                joined = f"{INAT_ROOT}/{rel}"
                cands = inat_candidates(joined)
                hit_index = None
                for j, c in enumerate(cands):
                    if fs.exists(c):
                        hit_index = j
                        break
                tag = f"cand{hit_index}" if hit_index is not None else "MISS"
                inat_pattern_hits[tag] += 1
                print(f"  rel={rel[:80]}", flush=True)
                if hit_index is not None:
                    print(f"     -> {cands[hit_index]}", flush=True)
                else:
                    print(f"     MISS; candidates: {cands}", flush=True)
    print(f"\niNat pattern hits: {dict(inat_pattern_hits)}", flush=True)


if __name__ == "__main__":
    main()
