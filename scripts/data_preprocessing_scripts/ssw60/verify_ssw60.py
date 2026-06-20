"""Post-upload verification for the SSW60 multimodal ingest.

Two checks:

1. Existence audit — every media path referenced by ``ssw60_all.csv``
   resolves to an object on GCS. Done by listing each GCS media prefix
   once and checking set-membership (cheap; avoids 100k per-object stats).
2. Decode smoke — load one sample per modality through the ``SSW60``
   dataset and assert the expected fields/shapes (audio waveform, image
   HWC uint8, video frames + aligned audio). Requires the ``video`` extra.

Run via jobs/verify_ssw60.sh (Slurm cpu).
"""

from __future__ import annotations

import numpy as np

from esp_data.datasets import SSW60
from esp_data.io import filesystem

GCS_ROOT = "gs://esp-data-ingestion/ssw60/v0.1.0"


def _ids_in_prefix(fs: object, prefix: str, ext: str) -> set[str]:
    """Return the set of asset-id stems present under a GCS prefix.

    Parameters
    ----------
    fs : fsspec.AbstractFileSystem
        A GCS filesystem.
    prefix : str
        GCS directory (without trailing slash).
    ext : str
        File extension including the dot (e.g. ``".wav"``).

    Returns
    -------
    set[str]
        The id stems of objects with the given extension under the prefix.
    """
    out = set()
    for p in fs.ls(prefix):
        name = str(p).rsplit("/", 1)[-1]
        if name.endswith(ext):
            out.add(name[: -len(ext)])
    return out


def existence_audit() -> None:
    """Assert every manifest media path resolves to an object on GCS."""
    fs = filesystem("gcs")
    ds = SSW60(split="all", sample_rate=None, backend="pandas")
    rows = list(ds._data)

    prefixes = {
        "audio": _ids_in_prefix(fs, f"{GCS_ROOT}/audio", ".wav"),
        "audio_16k": _ids_in_prefix(fs, f"{GCS_ROOT}/audio_16k", ".wav"),
        "audio_32k": _ids_in_prefix(fs, f"{GCS_ROOT}/audio_32k", ".wav"),
        "video": _ids_in_prefix(fs, f"{GCS_ROOT}/video", ".mp4"),
        "images_inat": _ids_in_prefix(fs, f"{GCS_ROOT}/images_inat", ".jpg"),
        "images_nabirds": _ids_in_prefix(fs, f"{GCS_ROOT}/images_nabirds", ".jpg"),
    }
    for k, v in prefixes.items():
        print(f"  GCS {k}: {len(v)} objects")

    missing = []
    for r in rows:
        aid, mod = r["asset_id"], r["modality"]
        if mod == "audio":
            for key in ("audio", "audio_16k", "audio_32k"):
                if aid not in prefixes[key]:
                    missing.append((aid, key))
        elif mod == "video":
            if aid not in prefixes["video"]:
                missing.append((aid, "video"))
        elif mod == "image":
            sub = "images_nabirds" if "images_nabirds" in r["image_path"] else "images_inat"
            if aid not in prefixes[sub]:
                missing.append((aid, sub))
    print(f"  audited {len(rows)} rows; missing: {len(missing)}")
    assert not missing, f"missing media: {missing[:10]}"
    print("  existence audit PASSED")


def decode_smoke() -> None:
    """Decode one sample per modality and assert the expected fields."""
    a = SSW60(split="audio_test", sample_rate=16000, backend="pandas")[0]
    assert a["modality"] == "audio" and a["audio"].ndim == 1 and a["sample_rate"] == 16000
    print(f"  audio: {a['audio'].shape} @ {a['sample_rate']} Hz — {a['canonical_name']}")

    im = SSW60(split="image_test", backend="pandas")[0]
    assert im["modality"] == "image" and im["image"].dtype == np.uint8 and im["image"].ndim == 3
    print(f"  image: {im['image'].shape} {im['image'].dtype} — {im['canonical_name']}")

    v = SSW60(split="video_test", sample_rate=16000, backend="pandas", max_frames=8)[0]
    assert v["modality"] == "video" and v["video_frames"].ndim == 4
    assert v["video_frames"].shape[0] <= 8
    aud = v["audio"]
    aud_desc = "none" if aud is None else f"{aud.shape} @ {v['sample_rate']} Hz"
    print(
        f"  video: frames {v['video_frames'].shape}, aligned audio {aud_desc} "
        f"— {v['canonical_name']}"
    )
    print("  decode smoke PASSED")


def main() -> None:
    """Run the existence audit then the decode smoke test."""
    print("=== existence audit ===", flush=True)
    existence_audit()
    print("\n=== decode smoke ===", flush=True)
    decode_smoke()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
