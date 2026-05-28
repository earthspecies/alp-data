"""
Build the DORI ONC-Benchmark expert test split.

ONC-Benchmark (DORI-SRKW/ONC-Benchmark) is 385 full hydrophone recordings with
an expert ``mammal_present`` (0/1) label plus three amateur-annotator labels.
Presence is over the *whole* recording, so these are ingested uncropped (one
clip = one recording), unlike the cropped 15 s Phase-1 clips.

Two modes:
  --mode manifest : build the whole-file fetch manifest (segment 0..0 -> whole
                    file) for dori_crop_download.py, upload to GCS metadata.
  --mode csv      : after audio is uploaded, build onc_benchmark.csv (standard
                    DORI columns + presence = mammal_present + amateur labels),
                    keeping only clips present in GCS.

Usage:
    uv run --with huggingface_hub --with pandas python \
        scripts/data_preprocessing_scripts/dori_build_onc_benchmark.py --mode manifest
"""

from __future__ import annotations

import argparse
import os
import subprocess

import pandas as pd

REPO = "DORI-SRKW/ONC-Benchmark"
META_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/metadata.csv"
GCS_ROOT = "gs://esp-data-ingestion/dori/v0.1.0"
MANIFEST_GCS = f"{GCS_ROOT}/metadata/onc_benchmark_manifest.csv"


def _stem(fn: str) -> str:
    return os.path.splitext(os.path.basename(str(fn)))[0]


def build_manifest(out: str) -> None:
    m = pd.read_csv(META_URL)
    df = pd.DataFrame(
        {
            "clip_id": m["filename"].map(_stem),
            "source": "onc_benchmark",
            "repo_id": REPO,
            "repo_path": m["filename"].astype(str),
            "segment_start": 0.0,
            "segment_end": 0.0,  # 0,0 -> dori_crop_download reads the whole file
            "mammal_present": m["mammal_present"].astype(int),
            "amateur_1": m.get("amateur_1"),
            "amateur_2": m.get("amateur_2"),
            "amateur_3": m.get("amateur_3"),
        }
    )
    df.to_csv(out, index=False)
    os.system(f"gsutil -q cp {out} {MANIFEST_GCS}")
    print(f"manifest: {len(df)} clips -> {out} + {MANIFEST_GCS}")
    print("mammal_present:\n", df["mammal_present"].value_counts().to_string())


def gcs_present() -> set[str]:
    r = subprocess.run(
        ["gsutil", "ls", f"{GCS_ROOT}/audio_32k/onc_benchmark/"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    return {ln.rsplit("/", 1)[-1][:-4] for ln in r.stdout.splitlines() if ln.endswith(".wav")}


def build_csv(manifest: str, out: str) -> None:
    m = pd.read_csv(manifest, keep_default_na=False)
    present = gcs_present()
    print(f"present in GCS: {len(present)} / {len(m)}")
    m = m[m["clip_id"].astype(str).isin(present)].reset_index(drop=True)
    cid = m["clip_id"].astype(str)
    out_df = pd.DataFrame(
        {
            "clip_id": cid,
            "source": "onc_benchmark",
            "audio_fp": "recordings/onc_benchmark/" + cid + ".flac",
            "16khz_path": "audio_16k/onc_benchmark/" + cid + ".wav",
            "32khz_path": "audio_32k/onc_benchmark/" + cid + ".wav",
            "audio_duration": "",  # whole recording; computed at load if needed
            "species": "",
            "species_common": "",
            "ecotype": "",
            "call_type": "",
            "presence": m["mammal_present"].astype(int),
            "label_source": "DORI (expert)",
            "license": "CC BY 4.0",
            "is_negative": (m["mammal_present"].astype(int) == 0),
            "amateur_1": m.get("amateur_1", ""),
            "amateur_2": m.get("amateur_2", ""),
            "amateur_3": m.get("amateur_3", ""),
        }
    )
    out_df.to_csv(out, index=False)
    os.system(f"gsutil -q cp {out} {GCS_ROOT}/onc_benchmark.csv")
    n_present = int(out_df["presence"].sum())
    print(
        f"onc_benchmark.csv: {len(out_df)} clips "
        f"(present={n_present}, absent={len(out_df) - n_present}) -> {GCS_ROOT}/onc_benchmark.csv"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    _dm = os.path.expanduser("~/dori_staging/onc_benchmark_manifest.csv")
    ap.add_argument("--mode", choices=["manifest", "csv"], required=True)
    ap.add_argument("--manifest", default=_dm)
    ap.add_argument("--out", default=os.path.expanduser("~/dori_staging/onc_benchmark.csv"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    if args.mode == "manifest":
        build_manifest(args.manifest)
    else:
        build_csv(args.manifest, args.out)


if __name__ == "__main__":
    main()
